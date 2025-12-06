# based heavily on https://github.com/KohakuBlueleaf/LyCORIS/blob/main/lycoris/modules/loha.py
# Adapted for ai-toolkit with ToolkitModuleMixin integration

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from toolkit.network_mixins import ToolkitModuleMixin

from typing import TYPE_CHECKING, Union, List

try:
    from optimum.quanto import QBytesTensor, QTensor
except ImportError:  # fallback stubs if optimum.quanto is unavailable
    class QBytesTensor:
        pass
    class QTensor:
        pass

if TYPE_CHECKING:
    from toolkit.lora_special import LoRASpecialNetwork


class HadaWeight(torch.autograd.Function):
    """Hadamard product weight computation with custom backward."""
    @staticmethod
    def forward(ctx, w1d, w1u, w2d, w2u, scale=torch.tensor(1)):
        ctx.save_for_backward(w1d, w1u, w2d, w2u, scale)
        diff_weight = ((w1u @ w1d) * (w2u @ w2d)) * scale
        return diff_weight

    @staticmethod
    def backward(ctx, grad_out):
        (w1d, w1u, w2d, w2u, scale) = ctx.saved_tensors
        grad_out = grad_out * scale
        temp = grad_out * (w2u @ w2d)
        grad_w1u = temp @ w1d.T
        grad_w1d = w1u.T @ temp

        temp = grad_out * (w1u @ w1d)
        grad_w2u = temp @ w2d.T
        grad_w2d = w2u.T @ temp

        del temp
        return grad_w1d, grad_w1u, grad_w2d, grad_w2u, None


class HadaWeightTucker(torch.autograd.Function):
    """Hadamard product with Tucker decomposition for conv layers."""
    @staticmethod
    def forward(ctx, t1, w1d, w1u, t2, w2d, w2u, scale=torch.tensor(1)):
        ctx.save_for_backward(t1, w1d, w1u, t2, w2d, w2u, scale)

        rebuild1 = torch.einsum("i j ..., j r, i p -> p r ...", t1, w1d, w1u)
        rebuild2 = torch.einsum("i j ..., j r, i p -> p r ...", t2, w2d, w2u)

        return rebuild1 * rebuild2 * scale

    @staticmethod
    def backward(ctx, grad_out):
        (t1, w1d, w1u, t2, w2d, w2u, scale) = ctx.saved_tensors
        grad_out = grad_out * scale

        temp = torch.einsum("i j ..., j r -> i r ...", t2, w2d)
        rebuild = torch.einsum("i j ..., i r -> r j ...", temp, w2u)

        grad_w = rebuild * grad_out
        del rebuild

        grad_w1u = torch.einsum("r j ..., i j ... -> r i", temp, grad_w)
        grad_temp = torch.einsum("i j ..., i r -> r j ...", grad_w, w1u.T)
        del grad_w, temp

        grad_w1d = torch.einsum("i r ..., i j ... -> r j", t1, grad_temp)
        grad_t1 = torch.einsum("i j ..., j r -> i r ...", grad_temp, w1d.T)
        del grad_temp

        temp = torch.einsum("i j ..., j r -> i r ...", t1, w1d)
        rebuild = torch.einsum("i j ..., i r -> r j ...", temp, w1u)

        grad_w = rebuild * grad_out
        del rebuild

        grad_w2u = torch.einsum("r j ..., i j ... -> r i", temp, grad_w)
        grad_temp = torch.einsum("i j ..., i r -> r j ...", grad_w, w2u.T)
        del grad_w, temp

        grad_w2d = torch.einsum("i r ..., i j ... -> r j", t2, grad_temp)
        grad_t2 = torch.einsum("i j ..., j r -> i r ...", grad_temp, w2d.T)
        del grad_temp
        return grad_t1, grad_w1d, grad_w1u, grad_t2, grad_w2d, grad_w2u, None


def make_hada_weight(w1d, w1u, w2d, w2u, scale):
    """Compute Hadamard product weight: (w1u @ w1d) ⊙ (w2u @ w2d) * scale"""
    return HadaWeight.apply(w1d, w1u, w2d, w2u, scale)


def make_hada_weight_tucker(t1, w1d, w1u, t2, w2d, w2u, scale):
    """Compute Hadamard product weight with Tucker decomposition for conv layers."""
    return HadaWeightTucker.apply(t1, w1d, w1u, t2, w2d, w2u, scale)


class LohaModule(ToolkitModuleMixin, nn.Module):
    """
    LoHa (Low-rank Hadamard Product) adapter module.

    LoHa decomposes weight updates as:
    ΔW = (A₁ × B₁) ⊙ (A₂ × B₂)

    where ⊙ is element-wise (Hadamard) product and × is matrix multiplication.

    This is particularly effective for:
    - Fine-grained detail capture (faces, textures)
    - Persona training
    - Cases where element-wise interactions matter
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=0.,
        rank_dropout=0.,
        module_dropout=0.,
        network: 'LoRASpecialNetwork' = None,
        # LoHa-specific parameters
        use_tucker: bool = False,  # Tucker decomposition for conv layers
        use_scalar: bool = False,  # Trainable scalar for weight diff
        weight_decompose: bool = False,  # DoRA-style weight decomposition
        wd_on_out: bool = True,  # Weight decomposition direction (True = output dim)
        rs_lora: bool = False,  # Rank-stabilized scaling (sqrt)
        rank_dropout_scale: bool = False,  # Scale by dropout rate
        bypass_mode: bool = False,  # Use bypass forward mode
        **kwargs,
    ):
        """
        Initialize LoHa module.

        Args:
            lora_name: Name for this LoHa module
            org_module: Original module to adapt (Linear or Conv2d)
            multiplier: LoHa weight multiplier
            lora_dim: Rank for low-rank decomposition
            alpha: Scaling factor (if 0 or None, defaults to lora_dim)
            dropout: Dropout rate applied to weight delta (zeros random elements)
            rank_dropout: Dropout applied to entire ranks (zeros entire rows)
            module_dropout: Probability to skip this module during training
            network: Parent LoRASpecialNetwork
            use_tucker: Use Tucker decomposition for conv layers
            use_scalar: Use trainable scalar (initialized to 1.0 for immediate effect)
            weight_decompose: Use DoRA-style weight decomposition
            wd_on_out: Direction for weight decomposition
            rs_lora: Use rank-stabilized scaling
            rank_dropout_scale: Scale output by dropout rate
            bypass_mode: Use bypass forward mode
        """
        ToolkitModuleMixin.__init__(self, network=network)
        torch.nn.Module.__init__(self)

        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.can_merge_in = True
        self.bypass_mode = bypass_mode
        self.rank_dropout_scale = rank_dropout_scale

        # Determine module type
        self.shape = org_module.weight.shape
        self.tucker = False

        if org_module.__class__.__name__ == 'Conv2d':
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            out_dim = org_module.out_channels
            self.shape = (out_dim, in_dim, *k_size)
            self.tucker = use_tucker and any(i != 1 for i in k_size)

            if self.tucker:
                w_shape = (out_dim, in_dim, *k_size)
            else:
                w_shape = (out_dim, in_dim * torch.tensor(k_size).prod().item())

            self.op = F.conv2d
            self.extra_args = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups
            }
        else:  # Linear
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            w_shape = (out_dim, in_dim)

            self.op = F.linear
            self.extra_args = {}

        # Create LoHa parameters
        if self.tucker:
            # Tucker decomposition for conv
            self.hada_t1 = nn.Parameter(torch.empty(lora_dim, lora_dim, *w_shape[2:]))
            self.hada_w1_a = nn.Parameter(torch.empty(lora_dim, w_shape[0]))  # out_dim, 1-mode
            self.hada_w1_b = nn.Parameter(torch.empty(lora_dim, w_shape[1]))  # in_dim, 2-mode

            self.hada_t2 = nn.Parameter(torch.empty(lora_dim, lora_dim, *w_shape[2:]))
            self.hada_w2_a = nn.Parameter(torch.empty(lora_dim, w_shape[0]))  # out_dim, 1-mode
            self.hada_w2_b = nn.Parameter(torch.empty(lora_dim, w_shape[1]))  # in_dim, 2-mode
        else:
            # Standard LoHa
            self.hada_w1_a = nn.Parameter(torch.empty(w_shape[0], lora_dim))
            self.hada_w1_b = nn.Parameter(torch.empty(lora_dim, w_shape[1]))

            self.hada_w2_a = nn.Parameter(torch.empty(w_shape[0], lora_dim))
            self.hada_w2_b = nn.Parameter(torch.empty(lora_dim, w_shape[1]))

        # Weight decomposition (DoRA-style)
        self.weight_decompose = weight_decompose
        self.wd_on_out = wd_on_out
        if self.weight_decompose:
            if isinstance(org_module.weight, (QTensor, QBytesTensor)):
                org_weight = org_module.weight.dequantize().cpu().float()
            else:
                org_weight = org_module.weight.cpu().clone().float()
            self.dora_norm_dims = org_weight.dim() - 1
            if self.wd_on_out:
                self.dora_scale = nn.Parameter(
                    torch.norm(
                        org_weight.reshape(org_weight.shape[0], -1),
                        dim=1, keepdim=True,
                    ).reshape(org_weight.shape[0], *[1] * self.dora_norm_dims)
                ).float()
            else:
                self.dora_scale = nn.Parameter(
                    torch.norm(
                        org_weight.transpose(0, 1).reshape(org_weight.shape[1], -1),
                        dim=1, keepdim=True,
                    ).reshape(org_weight.shape[1], *[1] * self.dora_norm_dims).transpose(0, 1)
                ).float()

        # Dropout settings
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        # Scale calculation
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha

        self.rs_lora = rs_lora
        r_factor = lora_dim
        if self.rs_lora:
            r_factor = math.sqrt(r_factor)

        self.scale = alpha / r_factor
        # Warning: rs_lora with high alpha can cause very large scales.
        # Without rs_lora: scale = alpha / rank (e.g., 64/64 = 1.0)
        # With rs_lora: scale = alpha / sqrt(rank) (e.g., 64/8 = 8.0 for rank=64)
        # If using rs_lora, consider reducing alpha to ~sqrt(rank) to maintain similar scaling.
        if self.rs_lora and alpha > 2 * math.sqrt(lora_dim):
            warnings.warn(
                f"LoHA '{lora_name}': rs_lora=True with alpha={alpha} and rank={lora_dim} "
                f"produces scale={self.scale:.2f}. This may cause training instability. "
                f"Consider alpha <= {2 * math.sqrt(lora_dim):.1f} (2*sqrt(rank)) for stability.",
                UserWarning
            )
        # Note: alpha buffer is stored for state_dict compatibility with LyCORIS format.
        # The actual scaling uses self.scale during forward pass.
        # Buffer value = alpha * (lora_dim / r_factor) to match LyCORIS conventions.
        self.register_buffer("alpha", torch.tensor(alpha * (lora_dim / r_factor)))

        # Trainable scalar (initialized to 1.0 for immediate effect, matching LokrModule)
        if use_scalar:
            self.scalar = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)

        # Initialize weights
        if self.tucker:
            torch.nn.init.normal_(self.hada_t1, std=0.1)
            torch.nn.init.normal_(self.hada_t2, std=0.1)
        torch.nn.init.normal_(self.hada_w1_b, std=1)
        torch.nn.init.normal_(self.hada_w1_a, std=0.1)
        torch.nn.init.normal_(self.hada_w2_b, std=1)
        if use_scalar:
            torch.nn.init.normal_(self.hada_w2_a, std=0.1)
        else:
            torch.nn.init.constant_(self.hada_w2_a, 0)

        self.multiplier = multiplier
        self.org_module = [org_module]

    def apply_to(self):
        """Hook this module's forward into the original module."""
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def get_weight(self, shape=None):
        """Compute the LoHa weight delta."""
        # Get reference tensor for device/dtype
        ref_tensor = self.hada_w1_b

        # Prepare scale tensor
        scale = self.scale
        if torch.is_tensor(scale):
            scale = scale.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        else:
            scale = torch.tensor(scale, device=ref_tensor.device, dtype=ref_tensor.dtype)

        if self.tucker:
            weight = make_hada_weight_tucker(
                self.hada_t1, self.hada_w1_b, self.hada_w1_a,
                self.hada_t2, self.hada_w2_b, self.hada_w2_a,
                scale
            )
        else:
            weight = make_hada_weight(
                self.hada_w1_b, self.hada_w1_a,
                self.hada_w2_b, self.hada_w2_a,
                scale
            )

        if shape is not None:
            weight = weight.reshape(shape)

        if self.training and self.dropout and self.dropout > 0:
            weight = F.dropout(weight, p=self.dropout, training=True)

        # Apply rank dropout during training
        if self.training and self.rank_dropout:
            drop = (torch.rand(weight.size(0), device=weight.device) > self.rank_dropout).to(weight.dtype)
            drop = drop.view(-1, *[1] * len(weight.shape[1:]))
            if self.rank_dropout_scale:
                # Add epsilon guard to prevent divide-by-zero if all ranks drop
                drop /= drop.mean().clamp(min=1e-6)
            weight = weight * drop

        return weight

    def get_merged_weight_with_dora(self, orig_weight, multiplier=1.0, device=None, dtype=None):
        """Get merged weight with DoRA normalization applied."""
        if device is None:
            device = orig_weight.device
        if dtype is None:
            dtype = orig_weight.dtype

        if orig_weight.device != device or orig_weight.dtype != dtype:
            orig_weight = orig_weight.to(device=device, dtype=dtype)

        # Apply scalar before dtype cast to keep intermediate math in fp32,
        # then cast once at the boundary (matches _call_forward pattern)
        loha_weight = self.get_weight(self.shape) * self.scalar
        if loha_weight.device != device or loha_weight.dtype != dtype:
            loha_weight = loha_weight.to(device=device, dtype=dtype)

        if isinstance(multiplier, (int, float)):
            multiplier = torch.tensor(multiplier, device=device, dtype=dtype)
        elif multiplier.device != device or multiplier.dtype != dtype:
            multiplier = multiplier.to(device=device, dtype=dtype)

        merged_weight = orig_weight + loha_weight * multiplier

        if self.weight_decompose:
            dora_scale = self.dora_scale.to(dtype=merged_weight.dtype, device=merged_weight.device)
            # Use practical epsilon for numerical stability in DoRA normalization.
            # torch.finfo(dtype).eps can be too small (~6e-8 for fp16), causing issues
            # with very small norms. Use max(finfo.eps, 1e-6) for robustness.
            eps = max(torch.finfo(merged_weight.dtype).eps, 1e-6)
            if self.wd_on_out:
                merged_weight_norm = torch.norm(
                    merged_weight.reshape(merged_weight.shape[0], -1),
                    dim=1, keepdim=True
                ).reshape(merged_weight.shape[0], *[1] * self.dora_norm_dims)
                merged_weight = merged_weight * (dora_scale / (merged_weight_norm + eps))
            else:
                merged_weight_norm = torch.norm(
                    merged_weight.transpose(0, 1).reshape(merged_weight.shape[1], -1),
                    dim=1, keepdim=True
                ).reshape(merged_weight.shape[1], *[1] * self.dora_norm_dims).transpose(0, 1)
                merged_weight = merged_weight * (dora_scale / (merged_weight_norm + eps))

        return merged_weight

    @torch.no_grad()
    def merge_in(self, merge_weight=1.0):
        """Merge LoHa weights into the original module."""
        if not self.can_merge_in:
            return

        org_sd = self.org_module[0].state_dict()
        weight_key = "weight._data" if 'weight._data' in org_sd else "weight"
        if weight_key == "weight._data":
            # quantized weight merge not supported yet
            return

        orig_dtype = org_sd[weight_key].dtype
        weight = org_sd[weight_key].float()

        if self.weight_decompose:
            merged_weight = self.get_merged_weight_with_dora(weight, merge_weight)
        else:
            loha_weight = self.get_weight(self.shape) * self.scalar
            merged_weight = weight + (loha_weight * merge_weight).to(weight.device, dtype=weight.dtype)

        org_sd[weight_key] = merged_weight.to(orig_dtype)
        self.org_module[0].load_state_dict(org_sd)

    def get_orig_weight(self):
        """Get the original module's weight."""
        weight = self.org_module[0].weight
        if isinstance(weight, (QTensor, QBytesTensor)):
            return weight.dequantize().data.detach()
        else:
            return weight.data.detach()

    def get_orig_bias(self):
        """Get the original module's bias if it exists."""
        if hasattr(self.org_module[0], 'bias') and self.org_module[0].bias is not None:
            if isinstance(self.org_module[0].bias, (QTensor, QBytesTensor)):
                return self.org_module[0].bias.dequantize().data.detach()
            else:
                return self.org_module[0].bias.data.detach()
        return None

    def _call_forward(self, x):
        """Internal forward computation."""
        if isinstance(x, (QTensor, QBytesTensor)):
            x = x.dequantize()

        orig_dtype = x.dtype
        device = x.device

        orig_weight = self.get_orig_weight()
        if orig_weight.device != device or orig_weight.dtype != orig_dtype:
            orig_weight = orig_weight.to(device=device, dtype=orig_dtype)

        multiplier = self.network_ref().torch_multiplier
        # Note: Per-sample multipliers are not supported in weight-merging mode.
        # When multiplier is a tensor with multiple values (e.g., for slider training),
        # we average them since the weight is shared across the batch.
        # Use bypass_mode=True for true per-sample multiplier support.
        if multiplier.numel() > 1:
            multiplier = torch.mean(multiplier)
        # Ensure multiplier matches target device/dtype to avoid dtype promotion
        if isinstance(multiplier, (int, float)):
            multiplier = torch.tensor(multiplier, device=device, dtype=orig_dtype)
        else:
            if multiplier.device != device or multiplier.dtype != orig_dtype:
                multiplier = multiplier.to(device=device, dtype=orig_dtype)

        if self.weight_decompose:
            weight = self.get_merged_weight_with_dora(orig_weight, multiplier, device=device, dtype=orig_dtype)
        else:
            loha_weight = self.get_weight(self.shape) * self.scalar
            if loha_weight.device != device or loha_weight.dtype != orig_dtype:
                loha_weight = loha_weight.to(device=device, dtype=orig_dtype)
            weight = orig_weight + loha_weight * multiplier

        bias = self.get_orig_bias()
        if bias is not None:
            bias = bias.to(device=device, dtype=orig_dtype)

        output = self.op(
            x,
            weight.view(self.shape),
            bias,
            **self.extra_args
        )
        return output

    def _bypass_forward_diff(self, x, scale=1.0):
        """Compute just the LoHA delta output for bypass mode.

        Args:
            x: Input tensor of shape [batch, ...]
            scale: Either a scalar or a tensor of shape [batch] for per-sample scaling

        Note: This method applies the same dropout/rank_dropout as the standard path
        via get_weight(), ensuring consistency between bypass and weight-merge modes.
        Bias is handled by org_forward() in _bypass_forward(), not here.
        """
        device = x.device
        dtype = x.dtype

        # get_weight() already applies rank_dropout and element dropout during training
        diff_weight = self.get_weight(self.shape) * self.scalar
        if diff_weight.device != device or diff_weight.dtype != dtype:
            diff_weight = diff_weight.to(device=device, dtype=dtype)

        # Compute LoHA output (no bias - that's in org_forward)
        diff_output = self.op(x, diff_weight.view(self.shape), None, **self.extra_args)

        # Apply per-sample scaling if scale is a tensor
        if isinstance(scale, torch.Tensor) and scale.numel() > 1:
            # Reshape scale for broadcasting: [batch] -> [batch, 1, 1, ...]
            scale = scale.view(-1, *([1] * (diff_output.dim() - 1)))
            if scale.device != device:
                scale = scale.to(device=device, dtype=dtype)

        return diff_output * scale

    def _bypass_forward(self, x, scale=1.0):
        """Bypass forward: Y = W_base(X) + scale * ΔW(X) instead of Y = (W_base + ΔW)(X).

        This mode supports per-sample multipliers since scaling is applied to output,
        not to weights. Useful for:
        - Quantized models where we can't easily merge weights
        - Slider training with per-sample multipliers
        """
        return self.org_forward(x) + self._bypass_forward_diff(x, scale=scale)

    def forward(self, x, *args, **kwargs):
        """Forward pass with LoHa adaptation."""
        # Module dropout during training
        if self.module_dropout and self.training:
            if torch.rand(1).item() < self.module_dropout:
                return self.org_forward(x, *args, **kwargs)

        # Bypass mode: compute base and delta separately
        # This mode supports per-sample multipliers (no averaging needed)
        if self.bypass_mode:
            multiplier = self.network_ref().torch_multiplier
            # Ensure multiplier matches input device/dtype to avoid promotion
            if isinstance(multiplier, (int, float)):
                multiplier = torch.tensor(multiplier, device=x.device, dtype=x.dtype)
            elif multiplier.device != x.device or multiplier.dtype != x.dtype:
                multiplier = multiplier.to(device=x.device, dtype=x.dtype)
            return self._bypass_forward(x, scale=multiplier)

        return self._call_forward(x)
