# based heavily on https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/lokr.py

import math

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


def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    '''
    return a tuple of two value of input dimension decomposed by the number closest to factor
    second value is higher or equal than first value.

    In LoRA with Kroneckor Product, first value is a value for weight scale.
    secon value is a value for weight.

    Becuase of non-commutative property, A⊗B ≠ B⊗A. Meaning of two matrices is slightly different.

    examples)
    factor
        -1               2                4               8               16               ...
    127 -> 127, 1   127 -> 127, 1    127 -> 127, 1   127 -> 127, 1   127 -> 127, 1
    128 -> 16, 8    128 -> 64, 2     128 -> 32, 4    128 -> 16, 8    128 -> 16, 8
    250 -> 125, 2   250 -> 125, 2    250 -> 125, 2   250 -> 125, 2   250 -> 125, 2
    360 -> 45, 8    360 -> 180, 2    360 -> 90, 4    360 -> 45, 8    360 -> 45, 8
    512 -> 32, 16   512 -> 256, 2    512 -> 128, 4   512 -> 64, 8    512 -> 32, 16
    1024 -> 32, 32  1024 -> 512, 2   1024 -> 256, 4  1024 -> 128, 8  1024 -> 64, 16
    '''

    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        return m, n
    if factor == -1:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


def make_weight_cp(t, wa, wb):
    rebuild2 = torch.einsum('i j k l, i p, j r -> p r k l',
                            t, wa, wb)  # [c, d, k1, k2]
    return rebuild2


def make_kron(w1, w2, scale):
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    rebuild = torch.kron(w1, w2)

    return rebuild*scale


class LokrModule(ToolkitModuleMixin, nn.Module):
    # Class-level flag to show dropout warning only once
    _dropout_warning_shown = False

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
        use_cp=False,
        decompose_both=False,
        network: 'LoRASpecialNetwork' = None,
        factor: int = -1,  # factorization factor
        # Advanced LyCORIS parameters
        weight_decompose: bool = False,  # DoRA-style weight decomposition
        wd_on_out: bool = True,  # Weight decomposition direction (True = output dim)
        use_scalar: bool = False,  # Trainable scalar for weight diff
        rs_lora: bool = False,  # Rank-stabilized scaling (sqrt)
        unbalanced_factorization: bool = False,  # Swap factorization dimensions
        use_tucker: bool = False,  # Tucker decomposition for conv (alias for use_cp)
        **kwargs,
    ):
        """ if alpha == 0 or None, alpha is rank (no scaling). """
        ToolkitModuleMixin.__init__(self, network=network)
        torch.nn.Module.__init__(self)
        factor = int(factor)
        self.lora_name = lora_name
        # Clamp/auto-adjust rank to a safe maximum (prevents accidental 1e10 full-rank allocations)
        # For full-rank requests, pass a large number or -1; we'll cap to min(in_dim, out_dim).
        self.lora_dim = lora_dim
        self.cp = False
        self.use_w1 = False
        self.use_w2 = False
        self.can_merge_in = True

        # Advanced LyCORIS parameters
        self.weight_decompose = weight_decompose
        self.wd_on_out = wd_on_out
        self.use_scalar_param = use_scalar
        self.rs_lora = rs_lora

        # Handle use_tucker as alias for use_cp
        if use_tucker:
            use_cp = True

        self.shape = org_module.weight.shape
        if org_module.__class__.__name__ == 'Conv2d':
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            out_dim = org_module.out_channels

            max_rank = min(in_dim, out_dim)
            if self.lora_dim < 0 or self.lora_dim > max_rank:
                self.lora_dim = max_rank
            lora_dim = self.lora_dim

            # Handle unbalanced factorization
            if unbalanced_factorization:
                in_n, in_m = factorization(in_dim, factor)
                out_k, out_l = factorization(out_dim, factor)
            else:
                in_m, in_n = factorization(in_dim, factor)
                out_l, out_k = factorization(out_dim, factor)
            # ((a, b), (c, d), *k_size)
            shape = ((out_l, out_k), (in_m, in_n), *k_size)

            self.cp = use_cp and k_size != (1, 1)
            if decompose_both and lora_dim < max(shape[0][0], shape[1][0])/2:
                self.lokr_w1_a = nn.Parameter(
                    torch.empty(shape[0][0], lora_dim))
                self.lokr_w1_b = nn.Parameter(
                    torch.empty(lora_dim, shape[1][0]))
            else:
                self.use_w1 = True
                self.lokr_w1 = nn.Parameter(torch.empty(
                    shape[0][0], shape[1][0]))  # a*c, 1-mode

            if lora_dim >= max(shape[0][1], shape[1][1])/2:
                self.use_w2 = True
                self.lokr_w2 = nn.Parameter(torch.empty(
                    shape[0][1], shape[1][1], *k_size))
            elif self.cp:
                self.lokr_t2 = nn.Parameter(torch.empty(
                    lora_dim, lora_dim, shape[2], shape[3]))
                self.lokr_w2_a = nn.Parameter(
                    torch.empty(lora_dim, shape[0][1]))  # b, 1-mode
                self.lokr_w2_b = nn.Parameter(
                    torch.empty(lora_dim, shape[1][1]))  # d, 2-mode
            else:  # Conv2d not cp
                # bigger part. weight and LoRA. [b, dim] x [dim, d*k1*k2]
                self.lokr_w2_a = nn.Parameter(
                    torch.empty(shape[0][1], lora_dim))
                self.lokr_w2_b = nn.Parameter(torch.empty(
                    lora_dim, shape[1][1]*shape[2]*shape[3]))
                # w1 ⊗ (w2_a x w2_b) = (a, b)⊗((c, dim)x(dim, d*k1*k2)) = (a, b)⊗(c, d*k1*k2) = (ac, bd*k1*k2)

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

            max_rank = min(in_dim, out_dim)
            if self.lora_dim < 0 or self.lora_dim > max_rank:
                self.lora_dim = max_rank
            lora_dim = self.lora_dim

            # Handle unbalanced factorization
            if unbalanced_factorization:
                in_n, in_m = factorization(in_dim, factor)
                out_k, out_l = factorization(out_dim, factor)
            else:
                in_m, in_n = factorization(in_dim, factor)
                out_l, out_k = factorization(out_dim, factor)
            # ((a, b), (c, d)), out_dim = a*c, in_dim = b*d
            shape = ((out_l, out_k), (in_m, in_n))

            # smaller part. weight scale
            if decompose_both and lora_dim < max(shape[0][0], shape[1][0])/2:
                self.lokr_w1_a = nn.Parameter(
                    torch.empty(shape[0][0], lora_dim))
                self.lokr_w1_b = nn.Parameter(
                    torch.empty(lora_dim, shape[1][0]))
            else:
                self.use_w1 = True
                self.lokr_w1 = nn.Parameter(torch.empty(
                    shape[0][0], shape[1][0]))  # a*c, 1-mode

            if lora_dim < max(shape[0][1], shape[1][1])/2:
                # bigger part. weight and LoRA. [b, dim] x [dim, d]
                self.lokr_w2_a = nn.Parameter(
                    torch.empty(shape[0][1], lora_dim))
                self.lokr_w2_b = nn.Parameter(
                    torch.empty(lora_dim, shape[1][1]))
                # w1 ⊗ (w2_a x w2_b) = (a, b)⊗((c, dim)x(dim, d)) = (a, b)⊗(c, d) = (ac, bd)
            else:
                self.use_w2 = True
                self.lokr_w2 = nn.Parameter(
                    torch.empty(shape[0][1], shape[1][1]))

            self.op = F.linear
            self.extra_args = {}

        self.dropout = dropout
        if dropout and not LokrModule._dropout_warning_shown:
            print("[WARN] LoKr hasn't implemented normal dropout yet. This warning will only show once.")
            LokrModule._dropout_warning_shown = True
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 or (isinstance(alpha, (int, float)) and alpha < 0) else alpha
        if self.use_w2 and self.use_w1:
            # use scale = 1
            alpha = self.lora_dim

        # Apply rank-stabilized scaling if enabled
        if self.rs_lora:
            self.scale = alpha / math.sqrt(self.lora_dim)
        else:
            self.scale = alpha / self.lora_dim
        self.register_buffer('alpha', torch.tensor(alpha))  # treat as constant

        # Initialize trainable scalar if enabled
        if self.use_scalar_param:
            self.scalar = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('scalar', torch.tensor(1.0))

        # Initialize DoRA scale if weight decomposition is enabled
        if self.weight_decompose:
            if isinstance(org_module.weight, (QTensor, QBytesTensor)):
                org_weight = org_module.weight.dequantize().cpu().float()
            else:
                org_weight = org_module.weight.cpu().clone().float()
            self.dora_norm_dims = org_weight.dim() - 1
            if self.wd_on_out:
                # Compute norm along all dimensions except the output dimension (dim 0)
                self.dora_scale = nn.Parameter(
                    torch.norm(
                        org_weight.reshape(org_weight.shape[0], -1),
                        dim=1, keepdim=True,
                    ).reshape(org_weight.shape[0], *[1] * self.dora_norm_dims)
                ).float()
            else:
                # Compute norm along output dimension
                self.dora_scale = nn.Parameter(
                    torch.norm(
                        org_weight.transpose(0, 1).reshape(org_weight.shape[1], -1),
                        dim=1, keepdim=True,
                    ).reshape(org_weight.shape[1], *[1] * self.dora_norm_dims).transpose(0, 1)
                ).float()

        if self.use_w2:
            torch.nn.init.constant_(self.lokr_w2, 0)
        else:
            if self.cp:
                torch.nn.init.kaiming_uniform_(self.lokr_t2, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            torch.nn.init.constant_(self.lokr_w2_b, 0)

        if self.use_w1:
            torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        else:
            torch.nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.lokr_w1_b, a=math.sqrt(5))

        self.multiplier = multiplier
        self.org_module = [org_module]
        weight = make_kron(
            self.lokr_w1 if self.use_w1 else self.lokr_w1_a@self.lokr_w1_b,
            (self.lokr_w2 if self.use_w2
             else make_weight_cp(self.lokr_t2, self.lokr_w2_a, self.lokr_w2_b) if self.cp
             else self.lokr_w2_a@self.lokr_w2_b),
            torch.tensor(self.multiplier * self.scale)
        )
        assert torch.sum(torch.isnan(weight)) == 0, "weight is nan"

    # Same as locon.py
    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def get_weight(self, orig_weight=None, multiplier=None):
        scale = self.scale
        if self.use_scalar_param:
            scale = scale * self.scalar

        # Get reference tensor for device/dtype
        ref_tensor = self.lokr_w1 if self.use_w1 else self.lokr_w1_a

        # Preserve autograd graph: if scale is a tensor (e.g., when using trainable scalar),
        # keep it and just move/cast; only wrap with torch.tensor when it's a plain number.
        if torch.is_tensor(scale):
            scale = scale.to(device=ref_tensor.device, dtype=ref_tensor.dtype, non_blocking=True)
        else:
            scale = torch.tensor(scale, device=ref_tensor.device, dtype=ref_tensor.dtype)

        weight = make_kron(
            self.lokr_w1 if self.use_w1 else self.lokr_w1_a@self.lokr_w1_b,
            (self.lokr_w2 if self.use_w2
             else make_weight_cp(self.lokr_t2, self.lokr_w2_a, self.lokr_w2_b) if self.cp
             else self.lokr_w2_a@self.lokr_w2_b),
            scale
        )

        if self.training and self.dropout and self.dropout > 0:
            weight = F.dropout(weight, p=self.dropout, training=True)
        if orig_weight is not None:
            weight = weight.reshape(orig_weight.shape)
        if self.training and self.rank_dropout:
            keep = torch.rand(weight.size(0), device=weight.device) >= self.rank_dropout
            weight *= keep.view(-1, *[1] * len(weight.shape[1:]))
        return weight

    def get_merged_weight_with_dora(self, orig_weight, multiplier=1.0, device=None, dtype=None):
        """Get merged weight with DoRA normalization applied."""
        if device is None:
            device = orig_weight.device
        if dtype is None:
            dtype = orig_weight.dtype

        # Ensure orig_weight is on correct device/dtype only if needed
        if orig_weight.device != device or orig_weight.dtype != dtype:
            orig_weight = orig_weight.to(device=device, dtype=dtype)

        lokr_weight = self.get_weight(orig_weight)
        if lokr_weight.device != device or lokr_weight.dtype != dtype:
            lokr_weight = lokr_weight.to(device=device, dtype=dtype)

        # Ensure multiplier is a tensor on the correct device
        if isinstance(multiplier, (int, float)):
            multiplier = torch.tensor(multiplier, device=device, dtype=dtype)
        elif multiplier.device != device:
            multiplier = multiplier.to(device=device)

        # Compute the merged weight
        merged_weight = orig_weight + lokr_weight * multiplier

        if self.weight_decompose:
            # Apply DoRA normalization
            dora_scale = self.dora_scale.to(dtype=merged_weight.dtype, device=merged_weight.device, non_blocking=True)
            if self.wd_on_out:
                # Normalize along all dims except output dim
                merged_weight_norm = torch.norm(
                    merged_weight.reshape(merged_weight.shape[0], -1),
                    dim=1, keepdim=True
                ).reshape(merged_weight.shape[0], *[1] * self.dora_norm_dims)
                merged_weight = merged_weight * (dora_scale / (merged_weight_norm + 1e-8))
            else:
                # Normalize along output dim
                merged_weight_norm = torch.norm(
                    merged_weight.transpose(0, 1).reshape(merged_weight.shape[1], -1),
                    dim=1, keepdim=True
                ).reshape(merged_weight.shape[1], *[1] * self.dora_norm_dims).transpose(0, 1)
                merged_weight = merged_weight * (dora_scale / (merged_weight_norm + 1e-8))

        return merged_weight

    @torch.no_grad()
    def merge_in(self, merge_weight=1.0):
        if not self.can_merge_in:
            return

        # extract weight from org_module
        org_sd = self.org_module[0].state_dict()
        # todo find a way to merge in weights when doing quantized model
        weight_key = "weight._data" if 'weight._data' in org_sd else "weight"
        if weight_key == "weight._data":
            # quantized weight merge not supported yet
            return

        orig_dtype = org_sd[weight_key].dtype
        weight = org_sd[weight_key].float()

        if self.weight_decompose:
            merged_weight = self.get_merged_weight_with_dora(weight, merge_weight)
        else:
            scale = self.scale
            # handle trainable scaler method locon does
            if self.use_scalar_param:
                scale = scale * self.scalar

            lokr_weight = self.get_weight(weight)

            merged_weight = (
                weight
                + (lokr_weight * merge_weight).to(weight.device, dtype=weight.dtype)
            )

        # set weight to org_module
        org_sd[weight_key] = merged_weight.to(orig_dtype)
        self.org_module[0].load_state_dict(org_sd)

    def get_orig_weight(self):
        weight = self.org_module[0].weight
        if isinstance(weight, QTensor) or isinstance(weight, QBytesTensor):
            return weight.dequantize().data.detach()
        else:
            return weight.data.detach()

    def get_orig_bias(self):
        if hasattr(self.org_module[0], 'bias') and self.org_module[0].bias is not None:
            if isinstance(self.org_module[0].bias, QTensor) or isinstance(self.org_module[0].bias, QBytesTensor):
                return self.org_module[0].bias.dequantize().data.detach()
            else:
                return self.org_module[0].bias.data.detach()
        return None

    def _call_forward(self, x):
        if isinstance(x, QTensor) or isinstance(x, QBytesTensor):
            x = x.dequantize()

        orig_dtype = x.dtype
        device = x.device

        orig_weight = self.get_orig_weight()

        # Move orig_weight to correct device/dtype only if needed
        if orig_weight.device != device or orig_weight.dtype != orig_dtype:
            orig_weight = orig_weight.to(device=device, dtype=orig_dtype, non_blocking=True)

        multiplier = self.network_ref().torch_multiplier
        # we do not currently support split batch multipliers for lokr. Just do a mean
        multiplier = torch.mean(multiplier)
        if multiplier.device != device:
            multiplier = multiplier.to(device=device)

        # Get merged weight (with DoRA if enabled)
        if self.weight_decompose:
            weight = self.get_merged_weight_with_dora(orig_weight, multiplier, device=device, dtype=orig_dtype)
        else:
            lokr_weight = self.get_weight(orig_weight)
            # Ensure lokr_weight is on correct device/dtype
            if lokr_weight.device != device or lokr_weight.dtype != orig_dtype:
                lokr_weight = lokr_weight.to(device=device, dtype=orig_dtype)
            weight = orig_weight + lokr_weight * multiplier

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
