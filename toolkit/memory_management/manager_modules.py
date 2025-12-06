"""
This code was heavily inspired by the work of Lodestone-Rock, pretty much all credit goes
to them. The original code can be found here:
https://github.com/lodestone-rock/RamTorch/blob/main/ramtorch/modules/linear.py

Modified for AI Toolkit with per-module state management to fix:
- P0: Global state shared across layers (now per-module)
- P0: Backward event reuse safety (now per-module events)
- P1: Forced serialization (now modules can overlap)
- P1: Cleanup API (now has detach/destroy methods)
- P1: Quantized gradient handling (now raises clear error)
- P1: Dtype asymmetry (now consistent between forward/backward)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import TYPE_CHECKING, Optional, Tuple
from torch.overrides import has_torch_function_unary  # torchao detection

if TYPE_CHECKING:
    from .manager import MemoryManager


# =============================================================================
# Per-Module State Management (replaces global _DEVICE_STATE)
# =============================================================================

class ModuleOffloadState:
    """
    Per-module state for layer offloading.

    Each managed module gets its own streams, events, and buffers.
    This eliminates cross-layer contention and enables proper overlap.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._destroyed = False

        if device.type == "cuda":
            with torch.cuda.device(device):
                # Per-module streams (dedicated for this layer)
                self.transfer_stream = torch.cuda.Stream(device=device)
                self.transfer_grad_stream = torch.cuda.Stream(device=device)

                # Per-module events (no cross-layer sharing)
                # Using enable_timing=False for minimal overhead
                self.transfer_forward_done = torch.cuda.Event(enable_timing=False)
                self.compute_forward_start = torch.cuda.Event(enable_timing=False)
                self.transfer_backward_done = torch.cuda.Event(enable_timing=False)
                self.transfer_grad_done = torch.cuda.Event(enable_timing=False)
                self.compute_backward_start = torch.cuda.Event(enable_timing=False)
                self.compute_backward_done = torch.cuda.Event(enable_timing=False)

                # Per-module buffers (no ping-pong needed with per-module state)
                # Single buffer per module is sufficient since we have dedicated events
                self.w_buffer: Optional[torch.Tensor] = None
                self.b_buffer: Optional[torch.Tensor] = None
                self.w_bwd_buffer: Optional[torch.Tensor] = None
                self.w_grad_buffer: Optional[torch.Tensor] = None
                self.b_grad_buffer: Optional[torch.Tensor] = None
        else:
            # CPU path - no CUDA resources needed
            self.transfer_stream = None
            self.transfer_grad_stream = None
            self.transfer_forward_done = None
            self.compute_forward_start = None
            self.transfer_backward_done = None
            self.transfer_grad_done = None
            self.compute_backward_start = None
            self.compute_backward_done = None
            self.w_buffer = None
            self.b_buffer = None
            self.w_bwd_buffer = None
            self.w_grad_buffer = None
            self.b_grad_buffer = None

    def destroy(self):
        """Clean up CUDA resources explicitly."""
        if self._destroyed:
            return
        self._destroyed = True

        # Clear buffer references (allows GC to free GPU memory)
        self.w_buffer = None
        self.b_buffer = None
        self.w_bwd_buffer = None
        self.w_grad_buffer = None
        self.b_grad_buffer = None

        # Synchronize streams before cleanup to ensure no in-flight work
        if self.transfer_stream is not None:
            try:
                self.transfer_stream.synchronize()
            except Exception:
                pass
        if self.transfer_grad_stream is not None:
            try:
                self.transfer_grad_stream.synchronize()
            except Exception:
                pass

        # Note: CUDA streams and events are reference-counted by PyTorch
        # They will be cleaned up when references are dropped
        self.transfer_stream = None
        self.transfer_grad_stream = None
        self.transfer_forward_done = None
        self.compute_forward_start = None
        self.transfer_backward_done = None
        self.transfer_grad_done = None
        self.compute_backward_start = None
        self.compute_backward_done = None

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        self.destroy()


# =============================================================================
# Legacy Global State (deprecated, kept for backwards compatibility)
# =============================================================================

_DEVICE_STATE = {}
_LEGACY_MODE_WARNING_SHOWN = False


def _get_device_state(device: torch.device):
    """
    Get or initialize per-device state.

    DEPRECATED: This global state approach causes cross-layer contention.
    Use ModuleOffloadState for per-module state instead.
    """
    global _LEGACY_MODE_WARNING_SHOWN
    if not _LEGACY_MODE_WARNING_SHOWN:
        warnings.warn(
            "Using legacy global device state for layer offloading. "
            "This may cause performance issues. Consider updating to per-module state.",
            DeprecationWarning,
            stacklevel=3
        )
        _LEGACY_MODE_WARNING_SHOWN = True

    if isinstance(device, str):
        device = torch.device(device)

    # CPU path needs no CUDA state
    if device.type != "cuda":
        if device not in _DEVICE_STATE:
            _DEVICE_STATE[device] = {}
        return _DEVICE_STATE[device]

    if device not in _DEVICE_STATE:
        with torch.cuda.device(device):
            _DEVICE_STATE[device] = {
                # streams & events
                "transfer_stream": torch.cuda.Stream(device=device),
                "transfer_grad_stream": torch.cuda.Stream(device=device),
                "transfer_forward_finished_event": torch.cuda.Event(),
                "compute_forward_start_event": torch.cuda.Event(),
                "transfer_backward_finished_event": torch.cuda.Event(),
                "transfer_weight_backward_finished_event": torch.cuda.Event(),
                "compute_backward_start_event": torch.cuda.Event(),
                "compute_backward_finished_event": torch.cuda.Event(),
                # ping-pong buffers
                "w_buffers": [None, None],
                "b_buffers": [None, None],
                "w_bwd_buffers": [None, None],
                # device-side staging for grads to be sent to CPU
                "w_grad_buffers": [None, None],
                "b_grad_buffers": [None, None],
                # clocks
                "forward_clk": 0,
                "backward_clk": 0,
            }
    return _DEVICE_STATE[device]


def clear_device_state(device: Optional[torch.device] = None):
    """
    Clear global device state for cleanup.

    Args:
        device: Specific device to clear, or None to clear all devices.
    """
    global _DEVICE_STATE

    if device is not None:
        if device in _DEVICE_STATE:
            state = _DEVICE_STATE[device]
            # Clear buffers
            for key in ['w_buffers', 'b_buffers', 'w_bwd_buffers', 'w_grad_buffers', 'b_grad_buffers']:
                if key in state and state[key] is not None:
                    for i in range(len(state[key])):
                        state[key][i] = None
            # Synchronize streams before removal
            if 'transfer_stream' in state and state['transfer_stream'] is not None:
                try:
                    state['transfer_stream'].synchronize()
                except Exception:
                    pass
            if 'transfer_grad_stream' in state and state['transfer_grad_stream'] is not None:
                try:
                    state['transfer_grad_stream'].synchronize()
                except Exception:
                    pass
            del _DEVICE_STATE[device]
    else:
        # Clear all devices
        for dev in list(_DEVICE_STATE.keys()):
            clear_device_state(dev)

    # Try to free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Utility Functions
# =============================================================================

def _is_ao_quantized_tensor(t: Optional[torch.Tensor]) -> bool:
    """Detect torchao wrapper tensors."""
    if t is None:
        return False
    try:
        if has_torch_function_unary(t):
            return t.__class__.__module__.startswith("torchao.")
    except Exception:
        pass
    for attr in (
        "_scale",
        "_scales",
        "_zero_point",
        "_zp",
        "_block_size",
        "_group_size",
        "_pack_dim",
    ):
        if hasattr(t, attr):
            return True
    return False


def _is_quantized_tensor(t: Optional[torch.Tensor]) -> bool:
    """Check if tensor is quantized (torch or torchao)."""
    if t is None:
        return False
    # torch quantized tensors
    try:
        if torch.is_quantized(t):
            return True
    except Exception:
        pass
    # torchao quantized wrappers
    if _is_ao_quantized_tensor(t):
        return True
    # packed/int formats (weight-only)
    return not t.dtype.is_floating_point


def _ensure_cpu_pinned(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Move tensor to CPU with pinned memory for fast async transfers."""
    if t is None:
        return None
    if t.device.type != "cpu":
        try:
            t = t.to("cpu", copy=True)
        except Exception:
            t = t.to("cpu")
    # Don't attempt to pin quantized tensors; many backends don't support it
    if _is_quantized_tensor(t):
        return t
    if torch.cuda.is_available():
        try:
            t = t.pin_memory()
        except RuntimeError:
            pass
    return t


def _move_params_to_cpu_and_pin(module: nn.Module):
    """Force parameters to CPU (+pinned) so we can 'bounce' them per forward/backward."""
    with torch.no_grad():
        if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
            module.weight.data = _ensure_cpu_pinned(module.weight.data).detach()
        if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
            if module.bias is not None:
                module.bias.data = _ensure_cpu_pinned(module.bias.data).detach()


def _check_quantized_training(weight_cpu: torch.Tensor, operation: str = "forward"):
    """
    Check for unsupported quantized weight training.

    Raises RuntimeError if quantized weights have requires_grad=True,
    as gradients cannot be properly computed for quantized parameters.
    """
    if _is_quantized_tensor(weight_cpu) and getattr(weight_cpu, "requires_grad", False):
        raise RuntimeError(
            f"Layer offloading does not support training with quantized weights "
            f"(requires_grad=True on quantized tensor during {operation}). "
            f"Either disable quantization for training, or disable layer_offloading, "
            f"or freeze the quantized weights (requires_grad=False)."
        )


def _materialize_weight(cpu_w: torch.Tensor, device: torch.device, target_dtype: torch.dtype) -> torch.Tensor:
    """
    Materialize weight on GPU with consistent dtype handling.

    This function is used for BOTH forward and backward to ensure
    consistent dtype treatment (fixes P1 dtype asymmetry).

    Args:
        cpu_w: Weight tensor on CPU
        device: Target device
        target_dtype: Target dtype for computation

    Returns:
        Weight tensor on device with correct dtype
    """
    if _is_quantized_tensor(cpu_w):
        # Quantized path: dequantize on GPU, cast to target dtype
        w_q_gpu = cpu_w.to(device, non_blocking=True)
        try:
            w_fp_gpu = w_q_gpu.dequantize()
        except Exception:
            w_fp_gpu = w_q_gpu.to(dtype=torch.float32, non_blocking=True)
        if w_fp_gpu.dtype != target_dtype:
            w_fp_gpu = w_fp_gpu.to(target_dtype, non_blocking=True)
        return w_fp_gpu

    # Float path: transfer and cast to target dtype for consistency
    w_gpu = cpu_w.to(device, non_blocking=True)
    if w_gpu.dtype != target_dtype:
        w_gpu = w_gpu.to(target_dtype, non_blocking=True)
    return w_gpu


# =============================================================================
# Autograd Functions with Per-Module State
# =============================================================================

class _BouncingLinearFnPerModule(torch.autograd.Function):
    """
    Linear layer with per-module CPU-GPU weight bouncing.

    Uses dedicated streams and events per module to avoid cross-layer
    contention and enable proper compute/transfer overlap.
    """

    @staticmethod
    def forward(ctx, x, weight_cpu, bias_cpu, device: torch.device, state: ModuleOffloadState):
        # Check for unsupported quantized training
        _check_quantized_training(weight_cpu, "forward")

        # Choose compute dtype to match activations
        target_dtype = (
            x.dtype
            if x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            else torch.bfloat16
        )

        # CPU fallback path
        if device.type != "cuda":
            if not hasattr(_BouncingLinearFnPerModule, '_cpu_warning_shown'):
                warnings.warn(
                    f"Layer offloading running on CPU device '{device}' - "
                    "this is significantly slower than CUDA path",
                    RuntimeWarning
                )
                _BouncingLinearFnPerModule._cpu_warning_shown = True

            w_mat = _materialize_weight(weight_cpu, torch.device("cpu"), target_dtype)
            out = F.linear(x.to("cpu"), w_mat, bias_cpu)
            ctx.save_for_backward(x.to("cpu"), weight_cpu, bias_cpu)
            ctx.device = torch.device("cpu")
            ctx.target_dtype = target_dtype
            ctx.state = None
            return out.to(x.device)

        # CUDA path with per-module state
        ts = state.transfer_stream

        # Transfer weights to GPU on dedicated stream
        with torch.cuda.stream(ts):
            # Wait for any previous compute that might use these buffers
            if state.compute_forward_start is not None:
                ts.wait_event(state.compute_forward_start)

            # Transfer and materialize weight
            state.w_buffer = _materialize_weight(weight_cpu, device, target_dtype)
            state.b_buffer = (
                bias_cpu.to(device, non_blocking=True) if bias_cpu is not None else None
            )

            # Signal transfer complete
            state.transfer_forward_done.record(ts)

        # Compute stream waits for transfer, then computes
        torch.cuda.current_stream().wait_event(state.transfer_forward_done)
        state.compute_forward_start.record()

        out = F.linear(x, state.w_buffer, state.b_buffer)

        # Save for backward
        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.device = device
        ctx.target_dtype = target_dtype
        ctx.state = state

        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device = ctx.device
        target_dtype = ctx.target_dtype
        state = ctx.state

        # Check for unsupported quantized training in backward
        _check_quantized_training(weight_cpu, "backward")

        # CPU fallback path
        if device.type != "cuda" or state is None:
            go_cpu = grad_out.to("cpu")
            x_cpu = x.to("cpu")
            w_mat = _materialize_weight(weight_cpu, torch.device("cpu"), target_dtype)

            grad_input = go_cpu @ w_mat

            grad_weight = None
            if getattr(weight_cpu, "requires_grad", False) and weight_cpu.dtype.is_floating_point:
                grad_weight = go_cpu.flatten(0, -2).T @ x_cpu.flatten(0, -2)

            grad_bias = None
            if bias_cpu is not None and getattr(bias_cpu, "requires_grad", False):
                grad_bias = go_cpu.sum(dim=tuple(range(go_cpu.ndim - 1)))

            return grad_input.to(grad_out.device), grad_weight, grad_bias, None, None

        # CUDA path with per-module state
        transfer_stream = state.transfer_stream
        transfer_grad_stream = state.transfer_grad_stream

        # Transfer weights for backward on dedicated stream
        with torch.cuda.stream(transfer_stream):
            if state.compute_backward_start is not None:
                transfer_stream.wait_event(state.compute_backward_start)

            state.w_bwd_buffer = _materialize_weight(weight_cpu, device, target_dtype)
            state.transfer_backward_done.record(transfer_stream)

        # Compute stream waits for weight transfer
        torch.cuda.current_stream().wait_event(state.transfer_backward_done)
        state.compute_backward_start.record()

        # Compute gradient w.r.t. input
        grad_input = grad_out.to(dtype=target_dtype) @ state.w_bwd_buffer

        # Wait for any previous gradient transfer to complete before reusing buffer
        if state.transfer_grad_done is not None:
            torch.cuda.current_stream().wait_event(state.transfer_grad_done)

        # Compute gradients if needed
        grad_weight = None
        grad_bias = None

        if getattr(weight_cpu, "requires_grad", False) and weight_cpu.dtype.is_floating_point:
            state.w_grad_buffer = grad_out.flatten(0, -2).T @ x.flatten(0, -2)

        if bias_cpu is not None and getattr(bias_cpu, "requires_grad", False):
            reduce_dims = tuple(range(grad_out.ndim - 1))
            state.b_grad_buffer = grad_out.sum(dim=reduce_dims)

        state.compute_backward_done.record()

        # Transfer gradients to CPU on dedicated stream
        with torch.cuda.stream(transfer_grad_stream):
            transfer_grad_stream.wait_event(state.compute_backward_done)

            if getattr(weight_cpu, "requires_grad", False) and weight_cpu.dtype.is_floating_point:
                grad_weight = state.w_grad_buffer.to("cpu", non_blocking=True)

            if bias_cpu is not None and getattr(bias_cpu, "requires_grad", False):
                grad_bias = state.b_grad_buffer.to("cpu", non_blocking=True)

            state.transfer_grad_done.record(transfer_grad_stream)

        return grad_input.to(dtype=grad_out.dtype), grad_weight, grad_bias, None, None


class _BouncingConv2dFnPerModule(torch.autograd.Function):
    """
    Conv2d layer with per-module CPU-GPU weight bouncing.

    Uses dedicated streams and events per module to avoid cross-layer
    contention and enable proper compute/transfer overlap.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        weight_cpu,
        bias_cpu,
        device: torch.device,
        state: ModuleOffloadState,
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        groups: int,
    ):
        # Check for unsupported quantized training
        _check_quantized_training(weight_cpu, "forward")

        target_dtype = (
            x.dtype
            if x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            else torch.bfloat16
        )

        # CPU fallback path
        if device.type != "cuda":
            if not hasattr(_BouncingConv2dFnPerModule, '_cpu_warning_shown'):
                warnings.warn(
                    f"Layer offloading running on CPU device '{device}' - "
                    "this is significantly slower than CUDA path",
                    RuntimeWarning
                )
                _BouncingConv2dFnPerModule._cpu_warning_shown = True

            w_mat = _materialize_weight(weight_cpu, torch.device("cpu"), target_dtype)
            out = F.conv2d(x.to("cpu"), w_mat, bias_cpu, stride, padding, dilation, groups)
            ctx.save_for_backward(x.to("cpu"), weight_cpu, bias_cpu)
            ctx.meta = ("cpu", stride, padding, dilation, groups, target_dtype, None)
            return out.to(x.device)

        # CUDA path with per-module state
        ts = state.transfer_stream

        with torch.cuda.stream(ts):
            if state.compute_forward_start is not None:
                ts.wait_event(state.compute_forward_start)

            state.w_buffer = _materialize_weight(weight_cpu, device, target_dtype)
            state.b_buffer = (
                bias_cpu.to(device, non_blocking=True) if bias_cpu is not None else None
            )
            state.transfer_forward_done.record(ts)

        torch.cuda.current_stream().wait_event(state.transfer_forward_done)
        state.compute_forward_start.record()

        out = F.conv2d(x, state.w_buffer, state.b_buffer, stride, padding, dilation, groups)

        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.meta = (device, stride, padding, dilation, groups, target_dtype, state)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device, stride, padding, dilation, groups, target_dtype, state = ctx.meta

        # Check for unsupported quantized training
        _check_quantized_training(weight_cpu, "backward")

        # CPU fallback path
        if device == "cpu" or (isinstance(device, torch.device) and device.type != "cuda") or state is None:
            go = grad_out.to("cpu")
            x_cpu = x.to("cpu")
            w_cpu = _materialize_weight(weight_cpu, torch.device("cpu"), target_dtype)

            from torch.nn.grad import conv2d_input, conv2d_weight

            grad_input = conv2d_input(
                x_cpu.shape, w_cpu, go,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
            )

            grad_weight = None
            if getattr(weight_cpu, "requires_grad", False) and weight_cpu.dtype.is_floating_point:
                grad_weight = conv2d_weight(
                    x_cpu, w_cpu.shape, go,
                    stride=stride, padding=padding, dilation=dilation, groups=groups,
                )

            grad_bias = None
            if bias_cpu is not None and getattr(bias_cpu, "requires_grad", False):
                grad_bias = go.sum(dim=(0, 2, 3))

            return (
                grad_input.to(grad_out.device),
                grad_weight,
                grad_bias,
                None, None, None, None, None, None,
            )

        # CUDA path with per-module state
        transfer_stream = state.transfer_stream
        transfer_grad_stream = state.transfer_grad_stream

        with torch.cuda.stream(transfer_stream):
            if state.compute_backward_start is not None:
                transfer_stream.wait_event(state.compute_backward_start)

            state.w_bwd_buffer = _materialize_weight(weight_cpu, device, target_dtype)
            state.transfer_backward_done.record(transfer_stream)

        torch.cuda.current_stream().wait_event(state.transfer_backward_done)
        state.compute_backward_start.record()

        from torch.nn.grad import conv2d_input, conv2d_weight

        grad_input = conv2d_input(
            x.shape, state.w_bwd_buffer, grad_out.to(dtype=target_dtype),
            stride=stride, padding=padding, dilation=dilation, groups=groups,
        )

        if state.transfer_grad_done is not None:
            torch.cuda.current_stream().wait_event(state.transfer_grad_done)

        grad_weight = None
        grad_bias = None

        if getattr(weight_cpu, "requires_grad", False) and weight_cpu.dtype.is_floating_point:
            state.w_grad_buffer = conv2d_weight(
                x, weight_cpu.shape, grad_out,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
            )

        if bias_cpu is not None and getattr(bias_cpu, "requires_grad", False):
            state.b_grad_buffer = grad_out.sum(dim=(0, 2, 3))

        state.compute_backward_done.record()

        with torch.cuda.stream(transfer_grad_stream):
            transfer_grad_stream.wait_event(state.compute_backward_done)

            if getattr(weight_cpu, "requires_grad", False) and weight_cpu.dtype.is_floating_point:
                grad_weight = state.w_grad_buffer.to("cpu", non_blocking=True)

            if bias_cpu is not None and getattr(bias_cpu, "requires_grad", False):
                grad_bias = state.b_grad_buffer.to("cpu", non_blocking=True)

            state.transfer_grad_done.record(transfer_grad_stream)

        return (
            grad_input.to(dtype=grad_out.dtype),
            grad_weight,
            grad_bias,
            None, None, None, None, None, None,
        )


# =============================================================================
# Legacy Autograd Functions (for backwards compatibility)
# =============================================================================

class _BouncingLinearFn(torch.autograd.Function):
    """
    Legacy Linear layer bouncing using global per-device state.

    DEPRECATED: Use _BouncingLinearFnPerModule for better performance.
    """

    @staticmethod
    def forward(ctx, x, weight_cpu, bias_cpu, device: torch.device):
        target_dtype = (
            x.dtype
            if x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            else torch.bfloat16
        )

        def _materialize_linear_weight(cpu_w, dev):
            return _materialize_weight(cpu_w, dev, target_dtype)

        if device.type != "cuda":
            out = F.linear(
                x.to("cpu"),
                _materialize_linear_weight(weight_cpu, torch.device("cpu")),
                bias_cpu,
            )
            ctx.save_for_backward(x.to("cpu"), weight_cpu, bias_cpu)
            ctx.device = torch.device("cpu")
            return out.to(x.device)

        state = _get_device_state(device)
        ts = state["transfer_stream"]
        w_bufs, b_bufs = state["w_buffers"], state["b_buffers"]
        ev_tx_f = state["transfer_forward_finished_event"]
        ev_cu_s = state["compute_forward_start_event"]
        idx = state["forward_clk"]

        with torch.cuda.stream(ts):
            ts.wait_event(ev_cu_s)
            w_bufs[idx] = _materialize_linear_weight(weight_cpu, device)
            b_bufs[idx] = (
                bias_cpu.to(device, non_blocking=True) if bias_cpu is not None else None
            )
            state["forward_clk"] ^= 1
            ev_tx_f.record()

        torch.cuda.current_stream().wait_event(ev_tx_f)
        ev_cu_s.record()
        out = F.linear(x, w_bufs[idx], b_bufs[idx])

        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.device = device
        ctx.target_dtype = target_dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device = ctx.device
        target_dtype = getattr(ctx, "target_dtype", grad_out.dtype)

        if device.type != "cuda":
            go_cpu = grad_out.to("cpu")
            x_cpu = x.to("cpu")
            w_mat = _materialize_weight(weight_cpu, torch.device("cpu"), target_dtype)
            grad_input = go_cpu @ w_mat
            grad_weight = (
                go_cpu.flatten(0, -2).T @ x_cpu.flatten(0, -2)
                if getattr(weight_cpu, "requires_grad", False)
                and weight_cpu.dtype.is_floating_point
                else None
            )
            grad_bias = (
                go_cpu.sum(dim=tuple(range(go_cpu.ndim - 1)))
                if (bias_cpu is not None and getattr(bias_cpu, "requires_grad", False))
                else None
            )
            return grad_input.to(grad_out.device), grad_weight, grad_bias, None

        state = _get_device_state(device)
        transfer_stream = state["transfer_stream"]
        transfer_grad_stream = state["transfer_grad_stream"]

        w_bwd_buffers = state["w_bwd_buffers"]
        w_grad_buffers = state["w_grad_buffers"]
        b_grad_buffers = state["b_grad_buffers"]

        ev_tx_b = state["transfer_backward_finished_event"]
        ev_tx_w_bwd_done = state["transfer_weight_backward_finished_event"]
        ev_cu_b_start = state["compute_backward_start_event"]
        ev_cu_b_finish = state["compute_backward_finished_event"]

        idx = state["backward_clk"]

        with torch.cuda.stream(transfer_stream):
            transfer_stream.wait_event(ev_cu_b_start)
            w_bwd_buffers[idx] = _materialize_weight(weight_cpu, device, target_dtype)
            state["backward_clk"] ^= 1
            ev_tx_b.record()

        torch.cuda.current_stream().wait_event(ev_tx_b)
        ev_cu_b_start.record()

        grad_input = grad_out.to(dtype=target_dtype) @ w_bwd_buffers[idx]

        torch.cuda.current_stream().wait_event(ev_tx_w_bwd_done)

        grad_weight = None
        grad_bias = None
        if (
            getattr(weight_cpu, "requires_grad", False)
            and weight_cpu.dtype.is_floating_point
        ):
            w_grad_buffers[idx] = grad_out.flatten(0, -2).T @ x.flatten(0, -2)
        if bias_cpu is not None and getattr(bias_cpu, "requires_grad", False):
            reduce_dims = tuple(range(grad_out.ndim - 1))
            b_grad_buffers[idx] = grad_out.sum(dim=reduce_dims)

        ev_cu_b_finish.record()

        with torch.cuda.stream(transfer_grad_stream):
            transfer_grad_stream.wait_event(ev_cu_b_finish)
            if (
                getattr(weight_cpu, "requires_grad", False)
                and weight_cpu.dtype.is_floating_point
            ):
                grad_weight = w_grad_buffers[idx].to("cpu", non_blocking=True)
            if bias_cpu is not None and getattr(bias_cpu, "requires_grad", False):
                grad_bias = b_grad_buffers[idx].to("cpu", non_blocking=True)
            state["transfer_weight_backward_finished_event"].record()

        return grad_input.to(dtype=grad_out.dtype), grad_weight, grad_bias, None


class _BouncingConv2dFn(torch.autograd.Function):
    """
    Legacy Conv2d layer bouncing using global per-device state.

    DEPRECATED: Use _BouncingConv2dFnPerModule for better performance.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        weight_cpu,
        bias_cpu,
        device: torch.device,
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        groups: int,
    ):
        target_dtype = (
            x.dtype
            if x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            else torch.bfloat16
        )

        def _materialize_conv_weight(cpu_w, dev):
            return _materialize_weight(cpu_w, dev, target_dtype)

        if device.type != "cuda":
            out = F.conv2d(
                x.to("cpu"),
                _materialize_conv_weight(weight_cpu, torch.device("cpu")),
                bias_cpu,
                stride,
                padding,
                dilation,
                groups,
            )
            ctx.save_for_backward(x.to("cpu"), weight_cpu, bias_cpu)
            ctx.meta = ("cpu", stride, padding, dilation, groups, target_dtype)
            return out.to(x.device)

        state = _get_device_state(device)
        ts = state["transfer_stream"]
        w_bufs, b_bufs = state["w_buffers"], state["b_buffers"]
        ev_tx_f = state["transfer_forward_finished_event"]
        ev_cu_s = state["compute_forward_start_event"]
        idx = state["forward_clk"]

        with torch.cuda.stream(ts):
            ts.wait_event(ev_cu_s)
            w_bufs[idx] = _materialize_conv_weight(weight_cpu, device)
            b_bufs[idx] = (
                bias_cpu.to(device, non_blocking=True) if bias_cpu is not None else None
            )
            state["forward_clk"] ^= 1
            ev_tx_f.record()

        torch.cuda.current_stream().wait_event(ev_tx_f)
        ev_cu_s.record()
        out = F.conv2d(x, w_bufs[idx], b_bufs[idx], stride, padding, dilation, groups)

        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.meta = (device, stride, padding, dilation, groups, target_dtype)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device, stride, padding, dilation, groups, target_dtype = ctx.meta

        if (
            isinstance(device, torch.device) and device.type != "cuda"
        ) or device == "cpu":
            go = grad_out.to("cpu")
            x_cpu = x.to("cpu")
            w_cpu = _materialize_weight(weight_cpu, torch.device("cpu"), target_dtype)
            from torch.nn.grad import conv2d_input, conv2d_weight

            grad_input = conv2d_input(
                x_cpu.shape,
                w_cpu,
                go,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            grad_weight = (
                conv2d_weight(
                    x_cpu,
                    w_cpu.shape,
                    go,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                if getattr(weight_cpu, "requires_grad", False)
                and weight_cpu.dtype.is_floating_point
                else None
            )
            grad_bias = (
                go.sum(dim=(0, 2, 3))
                if (bias_cpu is not None and getattr(bias_cpu, "requires_grad", False))
                else None
            )
            return (
                grad_input.to(grad_out.device),
                grad_weight,
                grad_bias,
                None,
                None,
                None,
                None,
                None,
            )

        state = _get_device_state(device)
        transfer_stream = state["transfer_stream"]
        transfer_grad_stream = state["transfer_grad_stream"]

        w_bwd_buffers = state["w_bwd_buffers"]
        w_grad_buffers = state["w_grad_buffers"]
        b_grad_buffers = state["b_grad_buffers"]

        ev_tx_b = state["transfer_backward_finished_event"]
        ev_tx_w_bwd_done = state["transfer_weight_backward_finished_event"]
        ev_cu_b_start = state["compute_backward_start_event"]
        ev_cu_b_finish = state["compute_backward_finished_event"]

        idx = state["backward_clk"]

        with torch.cuda.stream(transfer_stream):
            transfer_stream.wait_event(ev_cu_b_start)
            w_bwd_buffers[idx] = _materialize_weight(weight_cpu, device, target_dtype)
            state["backward_clk"] ^= 1
            ev_tx_b.record()

        torch.cuda.current_stream().wait_event(ev_tx_b)
        ev_cu_b_start.record()

        from torch.nn.grad import conv2d_input, conv2d_weight

        grad_input = conv2d_input(
            x.shape,
            w_bwd_buffers[idx],
            grad_out.to(dtype=target_dtype),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        torch.cuda.current_stream().wait_event(ev_tx_w_bwd_done)

        grad_weight = None
        grad_bias = None
        if (
            getattr(weight_cpu, "requires_grad", False)
            and weight_cpu.dtype.is_floating_point
        ):
            w_grad_buffers[idx] = conv2d_weight(
                x,
                weight_cpu.shape,
                grad_out,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        if bias_cpu is not None and getattr(bias_cpu, "requires_grad", False):
            b_grad_buffers[idx] = grad_out.sum(dim=(0, 2, 3))

        ev_cu_b_finish.record()

        with torch.cuda.stream(transfer_grad_stream):
            transfer_grad_stream.wait_event(ev_cu_b_finish)
            if (
                getattr(weight_cpu, "requires_grad", False)
                and weight_cpu.dtype.is_floating_point
            ):
                grad_weight = w_grad_buffers[idx].to("cpu", non_blocking=True)
            if bias_cpu is not None and getattr(bias_cpu, "requires_grad", False):
                grad_bias = b_grad_buffers[idx].to("cpu", non_blocking=True)
            state["transfer_weight_backward_finished_event"].record()

        return (
            grad_input.to(dtype=grad_out.dtype),
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
        )


# =============================================================================
# Layer Memory Managers
# =============================================================================

class BaseLayerMemoryManager:
    """Base class for per-layer memory management."""

    def __init__(
        self,
        module: nn.Module,
        manager: "MemoryManager",
    ):
        self.module: nn.Module = module
        self.manager: "MemoryManager" = manager
        self._state: Optional[ModuleOffloadState] = None
        self._original_forward = None

    @classmethod
    def attach(cls, module: nn.Module, manager: "MemoryManager"):
        if hasattr(module, "_layer_memory_manager"):
            return
        module._layer_memory_manager = cls(module, manager)

        # Mark parameters as memory managed
        for param in module.parameters(recurse=False):
            param._is_memory_managed = True

    def detach(self):
        """
        Detach memory management from the module and clean up resources.
        """
        # Restore original forward method
        if self._original_forward is not None:
            if hasattr(self.module, "ara_lora_ref"):
                self.module.ara_lora_ref().org_forward = self._original_forward
            else:
                self.module.forward = self._original_forward

        # Clean up per-module state
        if self._state is not None:
            self._state.destroy()
            self._state = None

        # Remove management markers
        for param in self.module.parameters(recurse=False):
            if hasattr(param, "_is_memory_managed"):
                delattr(param, "_is_memory_managed")

        if hasattr(self.module, "_memory_management_device"):
            delattr(self.module, "_memory_management_device")

        if hasattr(self.module, "_layer_memory_manager"):
            delattr(self.module, "_layer_memory_manager")


class LinearLayerMemoryManager(BaseLayerMemoryManager):
    """Memory manager for Linear layers with per-module state."""

    def __init__(
        self,
        module: nn.Module,
        manager: "MemoryManager",
    ):
        super().__init__(module, manager)

        # Create per-module state
        self._state = ModuleOffloadState(manager.process_device)

        # Move params to CPU + pin memory for fast H2D
        _move_params_to_cpu_and_pin(self.module)

        # Hijack forward
        if hasattr(self.module, "ara_lora_ref"):
            self._original_forward = getattr(self.module.ara_lora_ref(), "org_forward")
        else:
            self._original_forward = getattr(self.module, "forward")

        # Capture state reference for closure
        state = self._state

        def _mm_forward(x, *args, **kwargs):
            # Ensure we only use expected signature (Linear: x)
            if args or kwargs:
                # Fall back to original if a custom signature is used
                return self._original_forward(x, *args, **kwargs)

            weight_cpu = self.module.weight
            bias_cpu = getattr(self.module, "bias", None)
            device = self.manager.process_device

            # Use per-module autograd function
            return _BouncingLinearFnPerModule.apply(x, weight_cpu, bias_cpu, device, state)

        if hasattr(self.module, "ara_lora_ref"):
            self.module.ara_lora_ref().org_forward = _mm_forward
        else:
            self.module.forward = _mm_forward

        self.module._memory_management_device = self.manager.process_device


class ConvLayerMemoryManager(BaseLayerMemoryManager):
    """Memory manager for Conv2d layers with per-module state."""

    def __init__(
        self,
        module: nn.Module,
        manager: "MemoryManager",
    ):
        super().__init__(module, manager)

        # Create per-module state
        self._state = ModuleOffloadState(manager.process_device)

        # Move params to CPU + pin memory for fast H2D
        _move_params_to_cpu_and_pin(self.module)

        # Cache static conv attributes from the module
        stride = (
            self.module.stride
            if isinstance(self.module.stride, tuple)
            else (self.module.stride, self.module.stride)
        )
        padding = (
            self.module.padding
            if isinstance(self.module.padding, tuple)
            else (self.module.padding, self.module.padding)
        )
        dilation = (
            self.module.dilation
            if isinstance(self.module.dilation, tuple)
            else (self.module.dilation, self.module.dilation)
        )
        groups = self.module.groups

        # Hijack forward
        if hasattr(self.module, "ara_lora_ref"):
            self._original_forward = getattr(self.module.ara_lora_ref(), "org_forward")
        else:
            self._original_forward = getattr(self.module, "forward")

        # Capture state reference for closure
        state = self._state

        def _mm_forward(x, *args, **kwargs):
            # Support the typical Conv2d(x) call; if user passes uncommon extras, fallback.
            if args or kwargs:
                return self._original_forward(x, *args, **kwargs)

            weight_cpu = self.module.weight
            bias_cpu = getattr(self.module, "bias", None)
            device = self.manager.process_device

            return _BouncingConv2dFnPerModule.apply(
                x, weight_cpu, bias_cpu, device, state, stride, padding, dilation, groups
            )

        if hasattr(self.module, "ara_lora_ref"):
            self.module.ara_lora_ref().org_forward = _mm_forward
        else:
            self.module.forward = _mm_forward

        self.module._memory_management_device = self.manager.process_device
