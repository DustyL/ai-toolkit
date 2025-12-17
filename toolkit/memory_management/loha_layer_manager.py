"""
LoHA Layer Memory Manager for AI Toolkit Layer Offloading.

This module provides CPU-GPU weight offloading for LoHA (Low-rank Hadamard Product)
modules from the LyCORIS library. LoHA uses 4 core parameters (hada_w1_a, hada_w1_b,
hada_w2_a, hada_w2_b) plus optional Tucker tensors (hada_t1, hada_t2).

The key difference from Linear offloading:
- LoHA uses nn.Parameter directly, not nn.Linear submodules
- Weight computation: ΔW = (w1_a @ w1_b) ⊙ (w2_a @ w2_b) (Hadamard product)
- Gradients must be computed for all 4+ parameters

Architecture follows the per-module state pattern from manager_modules.py.
"""

import torch
import torch.nn as nn
import warnings
from typing import TYPE_CHECKING, Optional

from .manager_modules import (
    ModuleOffloadState,
    _ensure_cpu_pinned,
    _is_quantized_tensor,
    BaseLayerMemoryManager,
)

if TYPE_CHECKING:
    from .manager import MemoryManager


# =============================================================================
# LoHA Module Detection
# =============================================================================

def _is_loha_module(module: nn.Module) -> bool:
    """
    Detect LoHA modules by presence of characteristic parameters.

    LoHA modules have hada_w1_a, hada_w1_b, hada_w2_a, hada_w2_b as nn.Parameter.
    """
    return (
        hasattr(module, 'hada_w1_a') and isinstance(module.hada_w1_a, nn.Parameter) and
        hasattr(module, 'hada_w1_b') and isinstance(module.hada_w1_b, nn.Parameter) and
        hasattr(module, 'hada_w2_a') and isinstance(module.hada_w2_a, nn.Parameter) and
        hasattr(module, 'hada_w2_b') and isinstance(module.hada_w2_b, nn.Parameter)
    )


def _is_lokr_module(module: nn.Module) -> bool:
    """
    Detect LoKr modules by presence of characteristic parameters.

    LoKr modules have lokr_w1 or lokr_w1_a, and lokr_w2 or lokr_w2_a.
    """
    has_w1 = (
        (hasattr(module, 'lokr_w1') and isinstance(module.lokr_w1, nn.Parameter)) or
        (hasattr(module, 'lokr_w1_a') and isinstance(module.lokr_w1_a, nn.Parameter))
    )
    has_w2 = (
        (hasattr(module, 'lokr_w2') and isinstance(module.lokr_w2, nn.Parameter)) or
        (hasattr(module, 'lokr_w2_a') and isinstance(module.lokr_w2_a, nn.Parameter))
    )
    return has_w1 and has_w2


# =============================================================================
# Extended Offload State for LoHA
# =============================================================================

class LoHAModuleOffloadState(ModuleOffloadState):
    """
    Extended per-module state for LoHA layer offloading.

    Adds buffers for LoHA's 4 core parameters plus optional Tucker tensors.
    """

    def __init__(self, device: torch.device):
        super().__init__(device)

        # Forward buffers for 4 core parameters
        self.w1_a_buffer: Optional[torch.Tensor] = None
        self.w1_b_buffer: Optional[torch.Tensor] = None
        self.w2_a_buffer: Optional[torch.Tensor] = None
        self.w2_b_buffer: Optional[torch.Tensor] = None

        # Forward buffers for Tucker tensors (optional)
        self.t1_buffer: Optional[torch.Tensor] = None
        self.t2_buffer: Optional[torch.Tensor] = None

        # Scale buffer
        self.scale_buffer: Optional[torch.Tensor] = None
        self.scalar_buffer: Optional[torch.Tensor] = None

        # Backward buffers
        self.w1_a_bwd: Optional[torch.Tensor] = None
        self.w1_b_bwd: Optional[torch.Tensor] = None
        self.w2_a_bwd: Optional[torch.Tensor] = None
        self.w2_b_bwd: Optional[torch.Tensor] = None
        self.t1_bwd: Optional[torch.Tensor] = None
        self.t2_bwd: Optional[torch.Tensor] = None

        # Gradient buffers for 4 core parameters
        self.grad_w1_a_buffer: Optional[torch.Tensor] = None
        self.grad_w1_b_buffer: Optional[torch.Tensor] = None
        self.grad_w2_a_buffer: Optional[torch.Tensor] = None
        self.grad_w2_b_buffer: Optional[torch.Tensor] = None

        # Gradient buffers for Tucker tensors (optional)
        self.grad_t1_buffer: Optional[torch.Tensor] = None
        self.grad_t2_buffer: Optional[torch.Tensor] = None

        # Gradient buffer for scalar (when trainable)
        self.grad_scalar_buffer: Optional[torch.Tensor] = None

    def destroy(self):
        """Clean up CUDA resources explicitly."""
        if self._destroyed:
            return

        # Clear LoHA-specific buffers first
        self.w1_a_buffer = None
        self.w1_b_buffer = None
        self.w2_a_buffer = None
        self.w2_b_buffer = None
        self.t1_buffer = None
        self.t2_buffer = None
        self.scale_buffer = None
        self.scalar_buffer = None

        self.w1_a_bwd = None
        self.w1_b_bwd = None
        self.w2_a_bwd = None
        self.w2_b_bwd = None
        self.t1_bwd = None
        self.t2_bwd = None

        self.grad_w1_a_buffer = None
        self.grad_w1_b_buffer = None
        self.grad_w2_a_buffer = None
        self.grad_w2_b_buffer = None
        self.grad_t1_buffer = None
        self.grad_t2_buffer = None
        self.grad_scalar_buffer = None

        # Call parent cleanup
        super().destroy()


# =============================================================================
# LoHA Autograd Function with Per-Module State
# =============================================================================

class _BouncingLoHAFn(torch.autograd.Function):
    """
    LoHA forward/backward with per-module CPU-GPU weight bouncing.

    This implements the full LoHA computation with offloading:

    Forward:
        1. Transfer 4 core params (+ optional Tucker tensors) from CPU → GPU
        2. Compute diff_weight: (w1_a @ w1_b) ⊙ (w2_a @ w2_b) * scale
        3. Apply LoHA delta to base forward

    Backward:
        1. Transfer parameters back for gradient computation
        2. Compute gradients for all 4+ parameters using Hadamard product rules
        3. Transfer gradients GPU → CPU

    Gradient formulas (non-Tucker case):
        grad_w1_a = (grad_out * (w2_a @ w2_b)) @ w1_b.T
        grad_w1_b = w1_a.T @ (grad_out * (w2_a @ w2_b))
        grad_w2_a = (grad_out * (w1_a @ w1_b)) @ w2_b.T
        grad_w2_b = w2_a.T @ (grad_out * (w1_a @ w1_b))
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        # CPU parameters
        w1_a_cpu: torch.Tensor,
        w1_b_cpu: torch.Tensor,
        w2_a_cpu: torch.Tensor,
        w2_b_cpu: torch.Tensor,
        t1_cpu: Optional[torch.Tensor],
        t2_cpu: Optional[torch.Tensor],
        scale_cpu: torch.Tensor,
        scalar_cpu: torch.Tensor,
        # Module info
        org_forward,
        device: torch.device,
        state: LoHAModuleOffloadState,
        # LoHA config
        shape: tuple,
        has_tucker: bool,
        training: bool,
        rank_dropout: float,
        rank_dropout_scale: bool,
    ):
        """
        Forward pass with CPU→GPU parameter transfer.
        """
        target_dtype = (
            x.dtype
            if x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            else torch.bfloat16
        )

        # CPU fallback path
        if device.type != "cuda":
            if not hasattr(_BouncingLoHAFn, '_cpu_warning_shown'):
                warnings.warn(
                    f"LoHA layer offloading running on CPU device '{device}' - "
                    "this is significantly slower than CUDA path",
                    RuntimeWarning
                )
                _BouncingLoHAFn._cpu_warning_shown = True

            # Compute on CPU directly
            diff_weight = _compute_loha_diff_weight(
                w1_a_cpu, w1_b_cpu, w2_a_cpu, w2_b_cpu,
                t1_cpu, t2_cpu, scale_cpu, scalar_cpu, shape,
                has_tucker, training, rank_dropout, rank_dropout_scale, target_dtype
            )

            # Save for backward
            ctx.save_for_backward(
                x, w1_a_cpu, w1_b_cpu, w2_a_cpu, w2_b_cpu,
                t1_cpu if t1_cpu is not None else torch.tensor(0),
                t2_cpu if t2_cpu is not None else torch.tensor(0),
                scale_cpu, scalar_cpu
            )
            ctx.device = device
            ctx.target_dtype = target_dtype
            ctx.state = None
            ctx.shape = shape
            ctx.has_tucker = has_tucker
            ctx.org_forward = org_forward

            return diff_weight, None

        # CUDA path with per-module state
        ts = state.transfer_stream

        # Transfer parameters to GPU on dedicated stream
        with torch.cuda.stream(ts):
            # Wait for any previous compute
            if state.compute_forward_start is not None:
                ts.wait_event(state.compute_forward_start)

            # Transfer core parameters
            state.w1_a_buffer = w1_a_cpu.to(device, dtype=target_dtype, non_blocking=True)
            state.w1_b_buffer = w1_b_cpu.to(device, dtype=target_dtype, non_blocking=True)
            state.w2_a_buffer = w2_a_cpu.to(device, dtype=target_dtype, non_blocking=True)
            state.w2_b_buffer = w2_b_cpu.to(device, dtype=target_dtype, non_blocking=True)

            # Transfer Tucker tensors if present
            if has_tucker and t1_cpu is not None and t2_cpu is not None:
                state.t1_buffer = t1_cpu.to(device, dtype=target_dtype, non_blocking=True)
                state.t2_buffer = t2_cpu.to(device, dtype=target_dtype, non_blocking=True)
            else:
                state.t1_buffer = None
                state.t2_buffer = None

            # Transfer scale
            state.scale_buffer = scale_cpu.to(device, dtype=target_dtype, non_blocking=True)
            state.scalar_buffer = scalar_cpu.to(device, dtype=target_dtype, non_blocking=True)

            # Signal transfer complete
            state.transfer_forward_done.record(ts)

        # Compute stream waits for transfer
        torch.cuda.current_stream().wait_event(state.transfer_forward_done)
        state.compute_forward_start.record()

        # Compute diff_weight on GPU
        diff_weight = _compute_loha_diff_weight(
            state.w1_a_buffer, state.w1_b_buffer,
            state.w2_a_buffer, state.w2_b_buffer,
            state.t1_buffer, state.t2_buffer,
            state.scale_buffer, state.scalar_buffer, shape,
            has_tucker, training, rank_dropout, rank_dropout_scale, target_dtype
        )

        # Save for backward
        ctx.save_for_backward(
            x, w1_a_cpu, w1_b_cpu, w2_a_cpu, w2_b_cpu,
            t1_cpu if t1_cpu is not None else torch.tensor(0),
            t2_cpu if t2_cpu is not None else torch.tensor(0),
            scale_cpu, scalar_cpu
        )
        ctx.device = device
        ctx.target_dtype = target_dtype
        ctx.state = state
        ctx.shape = shape
        ctx.has_tucker = has_tucker
        ctx.org_forward = org_forward

        return diff_weight, None

    @staticmethod
    def backward(ctx, grad_diff_weight, _):
        """
        Backward pass with gradient computation and GPU→CPU transfer.
        """
        (
            x, w1_a_cpu, w1_b_cpu, w2_a_cpu, w2_b_cpu,
            t1_cpu, t2_cpu, scale_cpu, scalar_cpu
        ) = ctx.saved_tensors

        device = ctx.device
        target_dtype = ctx.target_dtype
        state = ctx.state
        shape = ctx.shape
        has_tucker = ctx.has_tucker

        # Handle placeholder tensor for Tucker
        if t1_cpu.numel() == 1:
            t1_cpu = None
            t2_cpu = None

        # CPU fallback path
        if device.type != "cuda" or state is None:
            grads = _compute_loha_grads_cpu(
                grad_diff_weight, x,
                w1_a_cpu, w1_b_cpu, w2_a_cpu, w2_b_cpu,
                t1_cpu, t2_cpu, scale_cpu, scalar_cpu,
                has_tucker, target_dtype
            )
            # grads = (grad_w1_a, grad_w1_b, grad_w2_a, grad_w2_b, grad_t1, grad_t2, None, grad_scalar)
            # Return gradients matching forward signature order
            return (
                None,  # x
                grads[0],  # grad_w1_a
                grads[1],  # grad_w1_b
                grads[2],  # grad_w2_a
                grads[3],  # grad_w2_b
                grads[4],  # grad_t1
                grads[5],  # grad_t2
                None,  # scale (not trainable)
                grads[7],  # grad_scalar
                None,  # org_forward
                None,  # device
                None,  # state
                None,  # shape
                None,  # has_tucker
                None,  # training
                None,  # rank_dropout
                None,  # rank_dropout_scale
            )

        # CUDA path with per-module state
        transfer_stream = state.transfer_stream
        transfer_grad_stream = state.transfer_grad_stream

        # Transfer parameters for backward on dedicated stream
        with torch.cuda.stream(transfer_stream):
            if state.compute_backward_start is not None:
                transfer_stream.wait_event(state.compute_backward_start)

            state.w1_a_bwd = w1_a_cpu.to(device, dtype=target_dtype, non_blocking=True)
            state.w1_b_bwd = w1_b_cpu.to(device, dtype=target_dtype, non_blocking=True)
            state.w2_a_bwd = w2_a_cpu.to(device, dtype=target_dtype, non_blocking=True)
            state.w2_b_bwd = w2_b_cpu.to(device, dtype=target_dtype, non_blocking=True)

            if has_tucker and t1_cpu is not None and t2_cpu is not None:
                state.t1_bwd = t1_cpu.to(device, dtype=target_dtype, non_blocking=True)
                state.t2_bwd = t2_cpu.to(device, dtype=target_dtype, non_blocking=True)

            state.transfer_backward_done.record(transfer_stream)

        # Compute stream waits for weight transfer
        torch.cuda.current_stream().wait_event(state.transfer_backward_done)
        state.compute_backward_start.record()

        # Wait for any previous gradient transfer
        if state.transfer_grad_done is not None:
            torch.cuda.current_stream().wait_event(state.transfer_grad_done)

        # Compute gradients on GPU
        scale_gpu = scale_cpu.to(device, dtype=target_dtype)
        scalar_gpu = scalar_cpu.to(device, dtype=target_dtype)
        grad_out = grad_diff_weight.to(device, dtype=target_dtype)

        if has_tucker and state.t1_bwd is not None:
            # Tucker decomposition gradients
            grads = _compute_loha_tucker_grads_gpu(
                grad_out,
                state.w1_a_bwd, state.w1_b_bwd,
                state.w2_a_bwd, state.w2_b_bwd,
                state.t1_bwd, state.t2_bwd,
                scale_gpu, scalar_gpu
            )
            state.grad_w1_a_buffer, state.grad_w1_b_buffer = grads[0], grads[1]
            state.grad_w2_a_buffer, state.grad_w2_b_buffer = grads[2], grads[3]
            state.grad_t1_buffer, state.grad_t2_buffer = grads[4], grads[5]

            # Compute scalar gradient for Tucker case
            # weight_before_scalar = rebuild1 * rebuild2 * scale
            rebuild1 = torch.einsum("i j ..., j r, i p -> p r ...", state.t1_bwd, state.w1_b_bwd, state.w1_a_bwd)
            rebuild2 = torch.einsum("i j ..., j r, i p -> p r ...", state.t2_bwd, state.w2_b_bwd, state.w2_a_bwd)
            weight_before_scalar = rebuild1 * rebuild2 * scale_gpu
            state.grad_scalar_buffer = (grad_out * weight_before_scalar).sum()
        else:
            # Standard LoHA gradients
            grads = _compute_loha_grads_gpu(
                grad_out,
                state.w1_a_bwd, state.w1_b_bwd,
                state.w2_a_bwd, state.w2_b_bwd,
                scale_gpu, scalar_gpu
            )
            state.grad_w1_a_buffer, state.grad_w1_b_buffer = grads[0], grads[1]
            state.grad_w2_a_buffer, state.grad_w2_b_buffer = grads[2], grads[3]
            state.grad_t1_buffer = None
            state.grad_t2_buffer = None

            # Compute scalar gradient for standard case
            # weight_before_scalar = (w1_a @ w1_b) * (w2_a @ w2_b) * scale
            w1_prod = state.w1_a_bwd @ state.w1_b_bwd
            w2_prod = state.w2_a_bwd @ state.w2_b_bwd
            weight_before_scalar = w1_prod * w2_prod * scale_gpu
            state.grad_scalar_buffer = (grad_out * weight_before_scalar).sum()

        state.compute_backward_done.record()

        # Transfer gradients back to CPU on dedicated stream
        with torch.cuda.stream(transfer_grad_stream):
            transfer_grad_stream.wait_event(state.compute_backward_done)

            grad_w1_a = None
            grad_w1_b = None
            grad_w2_a = None
            grad_w2_b = None
            grad_t1 = None
            grad_t2 = None
            grad_scalar = None

            if getattr(w1_a_cpu, "requires_grad", False):
                grad_w1_a = state.grad_w1_a_buffer.to("cpu", non_blocking=True)
            if getattr(w1_b_cpu, "requires_grad", False):
                grad_w1_b = state.grad_w1_b_buffer.to("cpu", non_blocking=True)
            if getattr(w2_a_cpu, "requires_grad", False):
                grad_w2_a = state.grad_w2_a_buffer.to("cpu", non_blocking=True)
            if getattr(w2_b_cpu, "requires_grad", False):
                grad_w2_b = state.grad_w2_b_buffer.to("cpu", non_blocking=True)

            if has_tucker:
                if t1_cpu is not None and getattr(t1_cpu, "requires_grad", False):
                    grad_t1 = state.grad_t1_buffer.to("cpu", non_blocking=True)
                if t2_cpu is not None and getattr(t2_cpu, "requires_grad", False):
                    grad_t2 = state.grad_t2_buffer.to("cpu", non_blocking=True)

            # Transfer scalar gradient if scalar is trainable
            if getattr(scalar_cpu, "requires_grad", False):
                grad_scalar = state.grad_scalar_buffer.to("cpu", non_blocking=True)

            state.transfer_grad_done.record(transfer_grad_stream)

        # Return gradients for all inputs
        # Order: x, w1_a, w1_b, w2_a, w2_b, t1, t2, scale, scalar, org_forward, device, state, ...
        return (
            None,  # x (no grad needed, handled by original forward)
            grad_w1_a,
            grad_w1_b,
            grad_w2_a,
            grad_w2_b,
            grad_t1,
            grad_t2,
            None,  # scale (buffer, not trainable)
            grad_scalar,  # scalar gradient (trainable when use_scalar=True)
            None,  # org_forward
            None,  # device
            None,  # state
            None,  # shape
            None,  # has_tucker
            None,  # training
            None,  # rank_dropout
            None,  # rank_dropout_scale
        )


# =============================================================================
# Helper Functions for LoHA Computation
# =============================================================================

def _compute_loha_diff_weight(
    w1_a, w1_b, w2_a, w2_b, t1, t2, scale, scalar, shape,
    has_tucker, training, rank_dropout, rank_dropout_scale, dtype
):
    """
    Compute LoHA diff weight: (w1_a @ w1_b) ⊙ (w2_a @ w2_b) * scale * scalar
    """
    if has_tucker and t1 is not None and t2 is not None:
        # Tucker decomposition: use einsum
        rebuild1 = torch.einsum("i j ..., j r, i p -> p r ...", t1, w1_b, w1_a)
        rebuild2 = torch.einsum("i j ..., j r, i p -> p r ...", t2, w2_b, w2_a)
        weight = rebuild1 * rebuild2 * scale
    else:
        # Standard LoHA: (w1_a @ w1_b) ⊙ (w2_a @ w2_b)
        weight = (w1_a @ w1_b) * (w2_a @ w2_b) * scale

    # Apply scalar
    weight = weight * scalar

    # Reshape to target shape
    if shape is not None:
        weight = weight.reshape(shape)

    # Apply rank dropout during training
    if training and rank_dropout > 0:
        drop = (torch.rand(weight.size(0), device=weight.device) > rank_dropout).to(weight.dtype)
        drop = drop.view(-1, *[1] * len(weight.shape[1:]))
        if rank_dropout_scale:
            drop = drop / (drop.mean() + 1e-8)
        weight = weight * drop

    return weight


def _compute_loha_grads_gpu(
    grad_out, w1_a, w1_b, w2_a, w2_b, scale, scalar
):
    """
    Compute gradients for standard LoHA (non-Tucker).

    ΔW = (w1_a @ w1_b) ⊙ (w2_a @ w2_b) * scale * scalar

    Gradients:
        grad_w1_a = (grad_out * (w2_a @ w2_b)) @ w1_b.T * scale * scalar
        grad_w1_b = w1_a.T @ (grad_out * (w2_a @ w2_b)) * scale * scalar
        grad_w2_a = (grad_out * (w1_a @ w1_b)) @ w2_b.T * scale * scalar
        grad_w2_b = w2_a.T @ (grad_out * (w1_a @ w1_b)) * scale * scalar
    """
    # Flatten if needed for matmul
    go_flat = grad_out.reshape(-1, grad_out.shape[-1]) if grad_out.ndim > 2 else grad_out

    # Apply scale and scalar
    go_scaled = go_flat * scale * scalar

    # Compute intermediate products
    w1_prod = w1_a @ w1_b  # (out_dim, in_dim)
    w2_prod = w2_a @ w2_b  # (out_dim, in_dim)

    # Gradients for w1
    temp1 = go_scaled * w2_prod
    grad_w1_a = temp1 @ w1_b.T
    grad_w1_b = w1_a.T @ temp1

    # Gradients for w2
    temp2 = go_scaled * w1_prod
    grad_w2_a = temp2 @ w2_b.T
    grad_w2_b = w2_a.T @ temp2

    return grad_w1_a, grad_w1_b, grad_w2_a, grad_w2_b


def _compute_loha_tucker_grads_gpu(
    grad_out, w1_a, w1_b, w2_a, w2_b, t1, t2, scale, scalar
):
    """
    Compute gradients for Tucker-decomposed LoHA.

    Uses einsum-based gradient computation matching LyCORIS HadaWeightTucker.backward.
    """
    go_scaled = grad_out * scale * scalar

    # Gradients for first component (w1_a, w1_b, t1)
    temp = torch.einsum("i j ..., j r -> i r ...", t2, w2_b)
    rebuild = torch.einsum("i j ..., i r -> r j ...", temp, w2_a)

    grad_w = rebuild * go_scaled

    grad_w1_a = torch.einsum("r j ..., i j ... -> r i", temp, grad_w)
    grad_temp = torch.einsum("i j ..., i r -> r j ...", grad_w, w1_a.T)

    grad_w1_b = torch.einsum("i r ..., i j ... -> r j", t1, grad_temp)
    grad_t1 = torch.einsum("i j ..., j r -> i r ...", grad_temp, w1_b.T)

    # Gradients for second component (w2_a, w2_b, t2)
    temp = torch.einsum("i j ..., j r -> i r ...", t1, w1_b)
    rebuild = torch.einsum("i j ..., i r -> r j ...", temp, w1_a)

    grad_w = rebuild * go_scaled

    grad_w2_a = torch.einsum("r j ..., i j ... -> r i", temp, grad_w)
    grad_temp = torch.einsum("i j ..., i r -> r j ...", grad_w, w2_a.T)

    grad_w2_b = torch.einsum("i r ..., i j ... -> r j", t2, grad_temp)
    grad_t2 = torch.einsum("i j ..., j r -> i r ...", grad_temp, w2_b.T)

    return grad_w1_a, grad_w1_b, grad_w2_a, grad_w2_b, grad_t1, grad_t2


def _compute_loha_grads_cpu(
    grad_out, x, w1_a, w1_b, w2_a, w2_b, t1, t2, scale, scalar, has_tucker, dtype
):
    """
    Compute LoHA gradients on CPU (fallback path).
    """
    go = grad_out.to(dtype=dtype)
    scale_typed = scale.to(dtype)
    scalar_typed = scalar.to(dtype)

    if has_tucker and t1 is not None and t2 is not None:
        w1_a_typed = w1_a.to(dtype)
        w1_b_typed = w1_b.to(dtype)
        w2_a_typed = w2_a.to(dtype)
        w2_b_typed = w2_b.to(dtype)
        t1_typed = t1.to(dtype)
        t2_typed = t2.to(dtype)

        grads = _compute_loha_tucker_grads_gpu(
            go, w1_a_typed, w1_b_typed, w2_a_typed, w2_b_typed,
            t1_typed, t2_typed, scale_typed, scalar_typed
        )
        grad_w1_a, grad_w1_b, grad_w2_a, grad_w2_b, grad_t1, grad_t2 = grads

        # Compute scalar gradient for Tucker case
        rebuild1 = torch.einsum("i j ..., j r, i p -> p r ...", t1_typed, w1_b_typed, w1_a_typed)
        rebuild2 = torch.einsum("i j ..., j r, i p -> p r ...", t2_typed, w2_b_typed, w2_a_typed)
        weight_before_scalar = rebuild1 * rebuild2 * scale_typed
        grad_scalar = (go * weight_before_scalar).sum()
    else:
        w1_a_typed = w1_a.to(dtype)
        w1_b_typed = w1_b.to(dtype)
        w2_a_typed = w2_a.to(dtype)
        w2_b_typed = w2_b.to(dtype)

        grads = _compute_loha_grads_gpu(
            go, w1_a_typed, w1_b_typed, w2_a_typed, w2_b_typed,
            scale_typed, scalar_typed
        )
        grad_w1_a, grad_w1_b, grad_w2_a, grad_w2_b = grads
        grad_t1 = None
        grad_t2 = None

        # Compute scalar gradient for standard case
        w1_prod = w1_a_typed @ w1_b_typed
        w2_prod = w2_a_typed @ w2_b_typed
        weight_before_scalar = w1_prod * w2_prod * scale_typed
        grad_scalar = (go * weight_before_scalar).sum()

    # Apply requires_grad check
    if not getattr(w1_a, "requires_grad", False):
        grad_w1_a = None
    if not getattr(w1_b, "requires_grad", False):
        grad_w1_b = None
    if not getattr(w2_a, "requires_grad", False):
        grad_w2_a = None
    if not getattr(w2_b, "requires_grad", False):
        grad_w2_b = None
    if t1 is not None and not getattr(t1, "requires_grad", False):
        grad_t1 = None
    if t2 is not None and not getattr(t2, "requires_grad", False):
        grad_t2 = None
    if not getattr(scalar, "requires_grad", False):
        grad_scalar = None

    return grad_w1_a, grad_w1_b, grad_w2_a, grad_w2_b, grad_t1, grad_t2, None, grad_scalar


# =============================================================================
# LoHA Layer Memory Manager
# =============================================================================

class LoHALayerMemoryManager(BaseLayerMemoryManager):
    """
    Memory manager for LoHA layers with per-module state.

    Handles the unique structure of LoHA modules which use raw nn.Parameter
    instead of nn.Linear submodules.
    """

    def __init__(
        self,
        module: nn.Module,
        manager: "MemoryManager",
    ):
        super().__init__(module, manager)

        # Check module type
        if not _is_loha_module(module):
            raise ValueError(f"Module {module} is not a LoHA module")

        # Create extended per-module state
        self._state = LoHAModuleOffloadState(manager.process_device)

        # Detect Tucker mode
        self.has_tucker = hasattr(module, 'hada_t1') and module.hada_t1 is not None

        # Move parameters to CPU with pinned memory
        self._move_params_to_cpu()

        # Save original forward
        self._original_forward = module.forward

        # Capture state and module references for closure
        state = self._state
        loha_module = self.module
        has_tucker = self.has_tucker
        device = manager.process_device

        def _mm_forward(x, *args, **kwargs):
            """
            Offloaded forward for LoHA module.

            Computes diff_weight with offloading, then applies it using
            the original LoHA forward logic.
            """
            # Get training state
            training = loha_module.training
            rank_dropout = getattr(loha_module, 'rank_dropout', 0.0)
            rank_dropout_scale = getattr(loha_module, 'rank_dropout_scale', False)
            shape = getattr(loha_module, 'shape', None)

            # Get CPU parameters
            w1_a_cpu = loha_module.hada_w1_a
            w1_b_cpu = loha_module.hada_w1_b
            w2_a_cpu = loha_module.hada_w2_a
            w2_b_cpu = loha_module.hada_w2_b

            t1_cpu = getattr(loha_module, 'hada_t1', None) if has_tucker else None
            t2_cpu = getattr(loha_module, 'hada_t2', None) if has_tucker else None

            # Get scale and scalar
            scale_cpu = torch.tensor(
                loha_module.scale,
                dtype=w1_b_cpu.dtype,
                device=w1_b_cpu.device
            )
            scalar_cpu = loha_module.scalar

            # Get original forward for base computation
            org_forward = getattr(loha_module, 'org_forward', None)

            # Compute diff_weight with offloading
            diff_weight, _ = _BouncingLoHAFn.apply(
                x,
                w1_a_cpu, w1_b_cpu, w2_a_cpu, w2_b_cpu,
                t1_cpu, t2_cpu,
                scale_cpu, scalar_cpu,
                org_forward, device, state,
                shape, has_tucker, training,
                rank_dropout, rank_dropout_scale
            )

            # Apply module dropout during training
            if loha_module.module_dropout and training:
                if torch.rand(1) < loha_module.module_dropout:
                    return org_forward(x, *args, **kwargs) if org_forward else x

            # Bypass mode
            if getattr(loha_module, 'bypass_mode', None):
                return _apply_bypass_forward(loha_module, x, diff_weight)

            # Standard forward: base + delta
            if org_forward is not None:
                base = org_forward(x, *args, **kwargs)
            else:
                base = x

            # Get base weight for delta computation
            base_weight = loha_module._current_weight().to(x.device) if hasattr(loha_module, '_current_weight') else None

            if base_weight is not None:
                # Weight decompose (DoRA) handling
                if getattr(loha_module, 'wd', False):
                    new_weight = loha_module.apply_weight_decompose(
                        base_weight + diff_weight.to(base_weight.dtype),
                        loha_module.multiplier
                    )
                else:
                    new_weight = base_weight + diff_weight.to(base_weight.dtype) * loha_module.multiplier

                delta_weight = new_weight - base_weight
                delta = loha_module.op(x, delta_weight, None, **loha_module.kw_dict)
                return base + delta
            else:
                # Fallback: just add diff_weight effect
                return base + loha_module.op(x, diff_weight, None, **loha_module.kw_dict) * loha_module.multiplier

        # Replace forward
        module.forward = _mm_forward
        module._memory_management_device = manager.process_device

    def _move_params_to_cpu(self):
        """Move all LoHA parameters to CPU with pinned memory."""
        module = self.module

        # Core parameters
        with torch.no_grad():
            module.hada_w1_a.data = _ensure_cpu_pinned(module.hada_w1_a.data)
            module.hada_w1_b.data = _ensure_cpu_pinned(module.hada_w1_b.data)
            module.hada_w2_a.data = _ensure_cpu_pinned(module.hada_w2_a.data)
            module.hada_w2_b.data = _ensure_cpu_pinned(module.hada_w2_b.data)

            # Mark as memory managed
            module.hada_w1_a._is_memory_managed = True
            module.hada_w1_b._is_memory_managed = True
            module.hada_w2_a._is_memory_managed = True
            module.hada_w2_b._is_memory_managed = True

            # Tucker tensors
            if self.has_tucker:
                if hasattr(module, 'hada_t1') and module.hada_t1 is not None:
                    module.hada_t1.data = _ensure_cpu_pinned(module.hada_t1.data)
                    module.hada_t1._is_memory_managed = True
                if hasattr(module, 'hada_t2') and module.hada_t2 is not None:
                    module.hada_t2.data = _ensure_cpu_pinned(module.hada_t2.data)
                    module.hada_t2._is_memory_managed = True

            # Scalar (if it's a trainable nn.Parameter)
            if hasattr(module, 'scalar') and isinstance(module.scalar, nn.Parameter):
                module.scalar.data = _ensure_cpu_pinned(module.scalar.data)
                module.scalar._is_memory_managed = True

    def detach(self):
        """Detach memory management and restore original forward."""
        # Restore original forward
        if self._original_forward is not None:
            self.module.forward = self._original_forward

        # Move parameters back to manager device (if available)
        # For now, keep on CPU as that's the offloaded state

        # Clean up per-module state
        if self._state is not None:
            self._state.destroy()
            self._state = None

        # Remove management markers
        for param_name in ['hada_w1_a', 'hada_w1_b', 'hada_w2_a', 'hada_w2_b', 'hada_t1', 'hada_t2', 'scalar']:
            param = getattr(self.module, param_name, None)
            if param is not None and hasattr(param, "_is_memory_managed"):
                delattr(param, "_is_memory_managed")

        if hasattr(self.module, "_memory_management_device"):
            delattr(self.module, "_memory_management_device")

        if hasattr(self.module, "_layer_memory_manager"):
            delattr(self.module, "_layer_memory_manager")


def _apply_bypass_forward(module, x, diff_weight):
    """Apply bypass forward mode for LoHA."""
    multiplier = getattr(module, 'multiplier', 1.0)
    op = getattr(module, 'op', None)
    kw_dict = getattr(module, 'kw_dict', {})
    org_forward = getattr(module, 'org_forward', None)

    if op is not None:
        # Cast multiplier to match diff_weight dtype/device to avoid promotion
        if isinstance(multiplier, torch.Tensor):
            multiplier = multiplier.to(device=diff_weight.device, dtype=diff_weight.dtype)
        diff_out = op(x, diff_weight * multiplier, None, **kw_dict)
        if hasattr(module, 'drop'):
            diff_out = module.drop(diff_out)
        if org_forward is not None:
            return org_forward(x) + diff_out
        return diff_out
    return x
