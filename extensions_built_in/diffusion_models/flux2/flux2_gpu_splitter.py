"""
GPU Model Splitter for FLUX.2

Distributes FLUX.2 transformer blocks across multiple GPUs for memory-efficient
training of large models without gradient checkpointing.

FLUX.2 Architecture:
- 8 double blocks (DoubleStreamBlock): Process image and text streams separately
- 48 single blocks (SingleStreamBlock): Process concatenated img+txt stream
- Total: ~12B transformer parameters

Features:
- Deterministic block splits for 2, 4, and 8 GPUs (even distribution)
- User-configurable splits via gpu_split_double/gpu_split_single
- Configurable per-block synchronization (default: off for performance)
- Configurable output device routing
- Optional CUDA stream-based transfers for overlapped compute/transfer
- Proper validation of split device state across .to() operations
- Debug logging support (set FLUX_SPLITTER_DEBUG=1 to enable)
"""

import os
from functools import partial
from typing import Optional, List, TYPE_CHECKING
import torch
from torch import Tensor
import logging

if TYPE_CHECKING:
    from .src.model import Flux2

logger = logging.getLogger(__name__)

# Import stream manager (optional feature)
try:
    from .cuda_stream_manager import get_transfer_manager, DeviceTransferManager
    STREAM_MANAGER_AVAILABLE = True
except ImportError:
    STREAM_MANAGER_AVAILABLE = False
    get_transfer_manager = lambda: None


# Default block splits for common GPU configurations
# FLUX.2 has 8 double blocks and 48 single blocks
DEFAULT_SPLITS = {
    2: {
        "double": [4, 4],      # Even split: 4 blocks per GPU
        "single": [24, 24],    # Even split: 24 blocks per GPU
    },
    4: {
        "double": [2, 2, 2, 2],      # Even split: 2 blocks per GPU
        "single": [12, 12, 12, 12],  # Even split: 12 blocks per GPU
    },
    8: {
        "double": [1, 1, 1, 1, 1, 1, 1, 1],  # 1 block per GPU
        "single": [6, 6, 6, 6, 6, 6, 6, 6],  # 6 blocks per GPU
    },
}


# Global debug flag (can be set via set_gpu_splitter_debug or FLUX_SPLITTER_DEBUG env var)
_gpu_splitter_debug = os.environ.get("FLUX_SPLITTER_DEBUG", "").lower() in ("1", "true", "yes")

# Track if we've logged the initial debug state
_debug_state_logged = False


def set_gpu_splitter_debug(enabled: bool = True):
    """Enable/disable debug logging for GPU splitter."""
    global _gpu_splitter_debug
    _gpu_splitter_debug = enabled
    if enabled:
        logger.info("[GPU Splitter] Debug logging ENABLED")


def _log_debug(msg: str):
    """Log debug message if debug mode is enabled."""
    global _debug_state_logged
    if _gpu_splitter_debug:
        if not _debug_state_logged:
            logger.info("[GPU Splitter] Debug mode active (set FLUX_SPLITTER_DEBUG=0 to disable)")
            _debug_state_logged = True
        logger.info(f"[GPU Splitter] {msg}")


def _move_tensor_to_device(tensor: Tensor, device: torch.device) -> Tensor:
    """
    Move tensor to device if not already there.

    Uses non_blocking=True for efficiency. For CPU->GPU transfers, PyTorch
    handles synchronization automatically. The caller should synchronize
    if needed before using the result.
    """
    if tensor.device != device:
        return tensor.to(device, non_blocking=True)
    return tensor


def _move_tuple_to_device(t: tuple, device: torch.device) -> tuple:
    """Move all tensors in a tuple to device. Handles None items gracefully."""
    return tuple(
        _move_tensor_to_device(item, device) if item is not None else None
        for item in t
    )


def _synchronize_device(device: torch.device):
    """Synchronize a specific CUDA device to ensure all transfers are complete."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def split_gpu_double_block_forward(
    self,
    img: Tensor,
    txt: Tensor,
    pe: Tensor,
    pe_ctx: Tensor,
    mod_img: tuple,
    mod_txt: tuple,
    sync_per_block: bool = False,
) -> tuple[Tensor, Tensor]:
    """
    Wrapped forward for DoubleStreamBlock that handles device transfers.

    Moves all inputs to the block's assigned GPU before calling the original forward.
    Synchronization is configurable - by default, we do NOT sync per-block to allow
    pipelining. The event-based sync in the streamed path handles this properly.

    Args:
        sync_per_block: If True, synchronize device after transfers. Default False
                       for better pipelining. Only enable if not using stream transfers.
    """
    device = self._split_device

    # Move main tensors
    img = _move_tensor_to_device(img, device)
    txt = _move_tensor_to_device(txt, device)
    pe = _move_tensor_to_device(pe, device)
    pe_ctx = _move_tensor_to_device(pe_ctx, device)

    # Move modulation tuples - each is ((shift, scale, gate), (shift, scale, gate))
    mod_img = tuple(_move_tuple_to_device(m, device) for m in mod_img)
    mod_txt = tuple(_move_tuple_to_device(m, device) for m in mod_txt)

    # Optionally synchronize to ensure all transfers are complete
    # By default OFF - the stream manager handles this via events
    if sync_per_block:
        _synchronize_device(device)
        _log_debug(f"Synced double block on {device}")

    return self._pre_gpu_split_forward(img, txt, pe, pe_ctx, mod_img, mod_txt)


def split_gpu_single_block_forward(
    self,
    x: Tensor,
    pe: Tensor,
    mod: tuple,
    sync_per_block: bool = False,
) -> Tensor:
    """
    Wrapped forward for SingleStreamBlock that handles device transfers.

    Moves all inputs to the block's assigned GPU before calling the original forward.
    If this is the last block, routes output back to the configured output device.

    Args:
        sync_per_block: If True, synchronize device after transfers. Default False.
    """
    device = self._split_device

    # Move main tensors
    x = _move_tensor_to_device(x, device)
    pe = _move_tensor_to_device(pe, device)

    # Move modulation tuple - (shift, scale, gate)
    mod = _move_tuple_to_device(mod, device)

    # Optionally synchronize to ensure all transfers are complete
    if sync_per_block:
        _synchronize_device(device)
        _log_debug(f"Synced single block on {device}")

    x_out = self._pre_gpu_split_forward(x, pe, mod)

    # Route output to specified device (used for last block to return to output device)
    if hasattr(self, "_split_output_device"):
        output_device = self._split_output_device
        if x_out.device != output_device:
            x_out = x_out.to(output_device, non_blocking=True)
            if sync_per_block:
                _synchronize_device(output_device)

    return x_out


# =============================================================================
# Stream-Based Forward Wrappers (Optional - for overlapped compute/transfer)
# =============================================================================

def split_gpu_double_block_forward_streamed(
    self,
    img: Tensor,
    txt: Tensor,
    pe: Tensor,
    pe_ctx: Tensor,
    mod_img: tuple,
    mod_txt: tuple,
) -> tuple[Tensor, Tensor]:
    """
    Stream-based forward for DoubleStreamBlock with overlapped transfers.

    Uses DeviceTransferManager for fine-grained event-based synchronization
    instead of blocking device synchronization. Falls back to non-streamed
    path if no manager is active.
    """
    device = self._split_device
    mgr = get_transfer_manager()

    if mgr is None:
        # Fallback to blocking behavior - use model's configured sync_per_block
        # (defaults to False if not set, which is correct for pipelining)
        sync_per_block = getattr(self, '_sync_per_block', False)
        _log_debug(f"Streamed wrapper fallback: using sync_per_block={sync_per_block}")
        return split_gpu_double_block_forward(
            self, img, txt, pe, pe_ctx, mod_img, mod_txt, sync_per_block=sync_per_block
        )

    # Transfer main tensors on the transfer stream
    img, txt, pe, pe_ctx = mgr.transfer_tensors(
        (img, txt, pe, pe_ctx),
        target_device=device,
    )

    # Try to get pre-staged modulation tensors
    staged_mods = mgr.get_staged_modulation_tensors(device, "double")
    if staged_mods is not None:
        mod_img, mod_txt = staged_mods
    else:
        # Fallback: transfer modulation tensors
        mod_img = mgr.transfer_nested_tuples(mod_img, device)
        mod_txt = mgr.transfer_nested_tuples(mod_txt, device)

    # Wait for transfers to complete before compute
    mgr.wait_for_transfer(device)

    # Run the actual forward
    img_out, txt_out = self._pre_gpu_split_forward(
        img, txt, pe, pe_ctx, mod_img, mod_txt
    )

    # Mark compute complete (allows next device's transfer to start)
    mgr.mark_compute_complete(device)

    return img_out, txt_out


def split_gpu_single_block_forward_streamed(
    self,
    x: Tensor,
    pe: Tensor,
    mod: tuple,
) -> Tensor:
    """
    Stream-based forward for SingleStreamBlock with overlapped transfers.
    """
    device = self._split_device
    mgr = get_transfer_manager()

    if mgr is None:
        # Fallback to blocking behavior - use model's configured sync_per_block
        sync_per_block = getattr(self, '_sync_per_block', False)
        _log_debug(f"Streamed wrapper fallback (single): using sync_per_block={sync_per_block}")
        return split_gpu_single_block_forward(self, x, pe, mod, sync_per_block=sync_per_block)

    # Transfer main tensors
    x, pe = mgr.transfer_tensors((x, pe), target_device=device)

    # Try to get pre-staged modulation tensors
    staged_mod = mgr.get_staged_modulation_tensors(device, "single")
    if staged_mod is not None:
        mod = staged_mod
    else:
        # Fallback: transfer modulation tuple
        mod = mgr.transfer_nested_tuples(mod, device)

    # Wait for transfers
    mgr.wait_for_transfer(device)

    # Compute
    x_out = self._pre_gpu_split_forward(x, pe, mod)

    # Mark compute complete
    mgr.mark_compute_complete(device)

    # Handle output routing for last block
    if hasattr(self, "_split_output_device"):
        output_device = self._split_output_device
        if x_out.device != output_device:
            x_out, = mgr.transfer_tensors((x_out,), target_device=output_device)
            mgr.wait_for_transfer(output_device)

    return x_out


def new_device_to_flux2(self: "Flux2", *args, **kwargs):
    """
    Custom .to() method that respects GPU splitting.

    When a model is split across GPUs, calling .to(device) should:
    - Move global layers (embeddings, modulation, final_layer) to target device
    - Move transformer blocks to their assigned split devices

    This prevents .to() from accidentally moving all parameters to one device.

    Supports:
    - .to(device) - move to device, blocks go to split devices
    - .to(dtype) - change dtype only, preserve device placement
    - .to(device, dtype) - both
    - .to(device=..., dtype=...) - kwargs form
    - .to("cpu") - moves ALL layers to CPU, clears split device assignments

    Validation:
    - After moving back to CUDA from CPU, validates that split device topology
      is still valid (same number of GPUs available)
    """
    # Parse device and dtype from args/kwargs robustly
    device = None
    dtype = None
    remaining_args = []

    # Handle kwargs first
    if 'device' in kwargs:
        device = kwargs.pop('device')
    if 'dtype' in kwargs:
        dtype = kwargs.pop('dtype')

    # Handle positional args
    for arg in args:
        if isinstance(arg, torch.device):
            if device is None:
                device = arg
            else:
                remaining_args.append(arg)
        elif isinstance(arg, str) and ('cuda' in arg.lower() or 'cpu' in arg.lower()):
            # NOTE: Only 'cuda' and 'cpu' device strings are parsed here.
            # 'meta' and other device types (mps, xpu) pass through as remaining_args
            # and won't trigger GPU split topology restoration logic.
            if device is None:
                device = torch.device(arg)
            else:
                remaining_args.append(arg)
        elif isinstance(arg, torch.dtype):
            if dtype is None:
                dtype = arg
            else:
                remaining_args.append(arg)
        else:
            remaining_args.append(arg)

    args = tuple(remaining_args)
    has_device = device is not None

    # Normalize device to torch.device for comparison
    if device is not None and isinstance(device, str):
        device = torch.device(device)

    # Check if moving to CPU - this requires special handling
    moving_to_cpu = has_device and device is not None and device.type == "cpu"

    # Check if moving from CPU to CUDA
    moving_to_cuda = (
        has_device
        and device is not None
        and device.type == "cuda"
        and hasattr(self, '_split_devices')
    )

    # Validate GPU topology when moving back to CUDA
    if moving_to_cuda:
        available_gpus = torch.cuda.device_count()
        required_gpus = len(self._split_devices) if hasattr(self, '_split_devices') else 0

        if required_gpus > available_gpus:
            # Cannot restore - clear split metadata and warn
            _log_debug(f"WARNING: Model was split across {required_gpus} GPUs but only "
                       f"{available_gpus} are available. Clearing stale split metadata.")
            print(f"[GPU Splitter] WARNING: Cannot restore GPU split - required {required_gpus} GPUs "
                  f"but only {available_gpus} available. Falling back to single-device mode.")
            # Clear stale split metadata
            if hasattr(self, '_split_devices'):
                delattr(self, '_split_devices')
            if hasattr(self, '_use_stream_transfers'):
                delattr(self, '_use_stream_transfers')
            if hasattr(self, '_sync_per_block'):
                delattr(self, '_sync_per_block')
            if hasattr(self, '_output_device'):
                delattr(self, '_output_device')
            # Reset all blocks to single-device mode
            for block in self.double_blocks:
                _reset_block(block)
            for block in self.single_blocks:
                _reset_block(block)
            # NOTE: We intentionally keep self.to as this wrapper (not restoring _pre_gpu_split_to)
            # so future .to() calls still get smart device handling. The wrapper gracefully handles
            # the "no split" case via the has_device branch below.
            # Fall through to standard .to() behavior
            moving_to_cuda = False  # Treat as normal move, no split restoration
        else:
            # Validate each GPU index in _split_devices
            for i, split_dev in enumerate(self._split_devices):
                if split_dev.type == 'cuda' and split_dev.index >= available_gpus:
                    _log_debug(f"WARNING: Split device {split_dev} (index {split_dev.index}) "
                               f"exceeds available GPUs ({available_gpus})")
                    print(f"[GPU Splitter] WARNING: Stale split device {split_dev} detected. "
                          f"Clearing split metadata.")
                    # Clear and fall back
                    if hasattr(self, '_split_devices'):
                        delattr(self, '_split_devices')
                    for block in self.double_blocks:
                        _reset_block(block)
                    for block in self.single_blocks:
                        _reset_block(block)
                    moving_to_cuda = False
                    break

    # Helper to call .to() with or without device argument
    def to_with_optional_device(module, target_device):
        """Call module.to() with device and/or dtype as appropriate."""
        call_kwargs = dict(kwargs)
        if has_device and target_device is not None:
            call_kwargs['device'] = target_device
        if dtype is not None:
            call_kwargs['dtype'] = dtype
        if call_kwargs:
            return module.to(*args, **call_kwargs)
        elif args:
            return module.to(*args)
        return module

    # Move global input layers to target device (GPU 0) or just change dtype
    self.pe_embedder = to_with_optional_device(self.pe_embedder, device)
    self.img_in = to_with_optional_device(self.img_in, device)
    self.time_in = to_with_optional_device(self.time_in, device)
    self.guidance_in = to_with_optional_device(self.guidance_in, device)
    self.txt_in = to_with_optional_device(self.txt_in, device)

    # Move modulation layers to target device (they're lightweight and used globally)
    self.double_stream_modulation_img = to_with_optional_device(self.double_stream_modulation_img, device)
    self.double_stream_modulation_txt = to_with_optional_device(self.double_stream_modulation_txt, device)
    self.single_stream_modulation = to_with_optional_device(self.single_stream_modulation, device)

    # Move transformer blocks
    #
    # Block device assignment has three modes:
    #   1. moving_to_cpu=True: Stash original _split_device in _original_split_device,
    #      then move all blocks to CPU. Preserves split topology for later restoration.
    #   2. moving_to_cuda=True: Restore the multi-GPU split topology from _original_split_device.
    #      Only active when _split_devices exists and GPU count is sufficient.
    #   3. has_device=True with moving_to_cuda=False: Topology was cleared (GPU count shrank)
    #      or never had a split. Behaves like normal .to(device), no split assumptions.
    #
    # _original_split_device lifecycle:
    #   - Created on first .to(cpu) to remember where block belonged in the split
    #   - Restored on .to(cuda) when topology is still valid
    #   - Cleared by _reset_block() when topology fails permanently
    #
    for i, block in enumerate(self.double_blocks):
        if moving_to_cpu:
            # CPU mode: all blocks go to CPU, stash original split device
            if not hasattr(block, "_original_split_device"):
                block._original_split_device = getattr(block, "_split_device", device)
            block._split_device = device
            block_device = device
        elif moving_to_cuda:
            # Restoring a valid split topology from CPU
            if hasattr(block, "_original_split_device"):
                block._split_device = block._original_split_device
            # Guard against missing _split_device and ensure consistent state
            block_device = getattr(block, "_split_device", device)
            block._split_device = block_device
        elif has_device:
            # Split metadata was cleared (topology failure) or never had a split:
            # restore _original_split_device if available, else use target device
            if hasattr(block, "_original_split_device"):
                block._split_device = block._original_split_device
                block_device = block._split_device
            else:
                # No split history - just move to target device
                block._split_device = device
                block_device = device
        else:
            # dtype-only: no device change
            block_device = None
        self.double_blocks[i] = to_with_optional_device(block, block_device)

    for i, block in enumerate(self.single_blocks):
        if moving_to_cpu:
            if not hasattr(block, "_original_split_device"):
                block._original_split_device = getattr(block, "_split_device", device)
            block._split_device = device
            block_device = device
        elif moving_to_cuda:
            # Restoring a valid split topology from CPU
            if hasattr(block, "_original_split_device"):
                block._split_device = block._original_split_device
            # Guard against missing _split_device and ensure consistent state
            block_device = getattr(block, "_split_device", device)
            block._split_device = block_device
        elif has_device:
            # Split metadata was cleared or never had a split:
            # restore _original_split_device if available, else use target device
            if hasattr(block, "_original_split_device"):
                block._split_device = block._original_split_device
                block_device = block._split_device
            else:
                block._split_device = device
                block_device = device
        else:
            block_device = None
        self.single_blocks[i] = to_with_optional_device(block, block_device)

    # Move output layer to target device
    self.final_layer = to_with_optional_device(self.final_layer, device)

    # Update _split_output_device to match where final_layer now lives
    if has_device and device is not None:
        last_block = self.single_blocks[-1]
        if moving_to_cpu:
            if not hasattr(last_block, "_original_split_output_device"):
                last_block._original_split_output_device = getattr(last_block, "_split_output_device", device)
            last_block._split_output_device = device
        elif moving_to_cuda:
            # Restoring valid split topology: use original split output device if available
            if hasattr(last_block, "_original_split_output_device"):
                last_block._split_output_device = last_block._original_split_output_device
            # Guard against missing _split_output_device and ensure consistent state
            output_dev = getattr(last_block, "_split_output_device", device)
            last_block._split_output_device = output_dev
        else:
            # Split metadata was cleared or never had a split: set to target device
            last_block._split_output_device = device

    _log_debug(f"Model .to() completed: device={device}, dtype={dtype}, "
               f"has_split={hasattr(self, '_split_devices')}")

    return self


def _reset_block(block):
    """Reset a block's GPU split state, restoring original forward if present."""
    orig_forward = getattr(block, "_pre_gpu_split_forward", None)
    if orig_forward is not None:
        block.forward = orig_forward  # restore original first
    for attr in ("_original_split_device", "_original_split_output_device",
                 "_pre_gpu_split_forward", "_split_device", "_split_output_device",
                 "_sync_per_block"):
        if hasattr(block, attr):
            delattr(block, attr)


def _apply_deterministic_split(
    transformer: "Flux2",
    gpu_ids: List[int],
    double_split: List[int],
    single_split: List[int],
    use_stream_transfers: bool = False,
    sync_per_block: bool = False,
    output_device: Optional[torch.device] = None,
):
    """
    Apply deterministic block distribution based on explicit per-GPU counts.

    Args:
        transformer: The Flux2 model to split
        gpu_ids: List of GPU IDs to use
        double_split: Number of double blocks per GPU (must sum to 8)
        single_split: Number of single blocks per GPU (must sum to 48)
        use_stream_transfers: If True, use stream-based forward wrappers for overlapped transfers
        sync_per_block: If True, synchronize after each block's transfers (default False for perf)
        output_device: Device where final output should go (default: gpu_ids[0])
    """
    # Validate split sums match actual block counts
    num_double = len(transformer.double_blocks)
    num_single = len(transformer.single_blocks)
    if sum(double_split) != num_double:
        raise ValueError(
            f"gpu_split_double sums to {sum(double_split)}, but model has {num_double} double blocks. "
            f"Split must sum to exactly {num_double}."
        )
    if sum(single_split) != num_single:
        raise ValueError(
            f"gpu_split_single sums to {sum(single_split)}, but model has {num_single} single blocks. "
            f"Split must sum to exactly {num_single}."
        )

    print(f"[GPU Splitter] Using deterministic split:")
    print(f"  Double blocks per GPU: {double_split}")
    print(f"  Single blocks per GPU: {single_split}")
    if use_stream_transfers:
        print(f"  Stream-based transfers: ENABLED")
    if sync_per_block:
        print(f"  Per-block sync: ENABLED (may reduce throughput)")

    # Determine output device
    if output_device is None:
        output_device = torch.device(f"cuda:{gpu_ids[0]}")
    print(f"  Output device: {output_device}")

    # Select forward wrapper based on stream transfer setting
    if use_stream_transfers and STREAM_MANAGER_AVAILABLE:
        double_forward_fn = split_gpu_double_block_forward_streamed
        single_forward_fn = split_gpu_single_block_forward_streamed
    else:
        # Bind sync_per_block to the non-streamed wrappers
        double_forward_fn = lambda self, img, txt, pe, pe_ctx, mod_img, mod_txt: \
            split_gpu_double_block_forward(self, img, txt, pe, pe_ctx, mod_img, mod_txt, sync_per_block)
        single_forward_fn = lambda self, x, pe, mod: \
            split_gpu_single_block_forward(self, x, pe, mod, sync_per_block)

    block_distribution = {gpu_id: {"double": 0, "single": 0} for gpu_id in gpu_ids}

    # Assign double blocks
    block_idx = 0
    for gpu_idx, count in enumerate(double_split):
        device = torch.device(f"cuda:{gpu_ids[gpu_idx]}")
        for _ in range(count):
            if block_idx >= len(transformer.double_blocks):
                break
            block = transformer.double_blocks[block_idx]
            block._pre_gpu_split_forward = block.forward
            block.forward = partial(double_forward_fn, block)
            block._split_device = device
            block._sync_per_block = sync_per_block  # For streamed wrapper fallback
            block_distribution[gpu_ids[gpu_idx]]["double"] += 1
            block_idx += 1

    # Assign single blocks
    block_idx = 0
    for gpu_idx, count in enumerate(single_split):
        device = torch.device(f"cuda:{gpu_ids[gpu_idx]}")
        for _ in range(count):
            if block_idx >= len(transformer.single_blocks):
                break
            block = transformer.single_blocks[block_idx]
            block._pre_gpu_split_forward = block.forward
            block.forward = partial(single_forward_fn, block)
            block._split_device = device
            block._sync_per_block = sync_per_block  # For streamed wrapper fallback
            block_distribution[gpu_ids[gpu_idx]]["single"] += 1
            block_idx += 1

    # Set last single block to route output to output_device
    # This ensures the final tensor ends up on the correct device for loss computation
    transformer.single_blocks[-1]._split_output_device = output_device
    _log_debug(f"Last block output device set to {output_device}")

    # Log distribution
    print("[GPU Splitter] Block distribution:")
    for gpu_id in gpu_ids:
        counts = block_distribution[gpu_id]
        print(f"  GPU {gpu_id}: {counts['double']} double blocks, {counts['single']} single blocks")
    print(f"  Final output routed to: {output_device}")


def _apply_greedy_split(
    transformer: "Flux2",
    gpu_ids: List[int],
    other_module_params: float,
    other_module_param_count_scale: float,
    use_stream_transfers: bool = False,
    sync_per_block: bool = False,
    output_device: Optional[torch.device] = None,
):
    """
    Apply greedy parameter-based block distribution (legacy algorithm).

    Falls back to this when no deterministic split is available.
    """
    print(f"[GPU Splitter] Using greedy parameter-based split")
    if use_stream_transfers:
        print(f"  Stream-based transfers: ENABLED")
    if sync_per_block:
        print(f"  Per-block sync: ENABLED")

    # Determine output device
    if output_device is None:
        output_device = torch.device(f"cuda:{gpu_ids[0]}")
    print(f"  Output device: {output_device}")

    # Select forward wrapper based on stream transfer setting
    if use_stream_transfers and STREAM_MANAGER_AVAILABLE:
        double_forward_fn = split_gpu_double_block_forward_streamed
        single_forward_fn = split_gpu_single_block_forward_streamed
    else:
        double_forward_fn = lambda self, img, txt, pe, pe_ctx, mod_img, mod_txt: \
            split_gpu_double_block_forward(self, img, txt, pe, pe_ctx, mod_img, mod_txt, sync_per_block)
        single_forward_fn = lambda self, x, pe, mod: \
            split_gpu_single_block_forward(self, x, pe, mod, sync_per_block)

    # Scale other module params
    other_module_params = other_module_params * other_module_param_count_scale

    # Calculate total params for distribution
    total_params = sum(p.numel() for p in transformer.parameters()) + other_module_params
    params_per_gpu = total_params / len(gpu_ids)

    print(f"[GPU Splitter] Total params: {total_params/1e9:.2f}B, Target per GPU: {params_per_gpu/1e9:.2f}B")

    current_gpu_idx = 0
    current_gpu_params = other_module_params  # GPU 0 starts with non-block params

    block_distribution = {gpu_id: {"double": 0, "single": 0} for gpu_id in gpu_ids}

    # Apply splitting to double blocks
    for double_block in transformer.double_blocks:
        device = torch.device(f"cuda:{gpu_ids[current_gpu_idx]}")

        double_block._pre_gpu_split_forward = double_block.forward
        double_block.forward = partial(double_forward_fn, double_block)
        double_block._split_device = device
        double_block._sync_per_block = sync_per_block  # For streamed wrapper fallback

        block_distribution[gpu_ids[current_gpu_idx]]["double"] += 1
        current_gpu_params += sum(p.numel() for p in double_block.parameters())

        if current_gpu_params > params_per_gpu and current_gpu_idx < len(gpu_ids) - 1:
            current_gpu_idx += 1
            current_gpu_params = 0

    # Apply splitting to single blocks
    for single_block in transformer.single_blocks:
        device = torch.device(f"cuda:{gpu_ids[current_gpu_idx]}")

        single_block._pre_gpu_split_forward = single_block.forward
        single_block.forward = partial(single_forward_fn, single_block)
        single_block._split_device = device
        single_block._sync_per_block = sync_per_block  # For streamed wrapper fallback

        block_distribution[gpu_ids[current_gpu_idx]]["single"] += 1
        current_gpu_params += sum(p.numel() for p in single_block.parameters())

        if current_gpu_params > params_per_gpu and current_gpu_idx < len(gpu_ids) - 1:
            current_gpu_idx += 1
            current_gpu_params = 0

    # Set last single block to route output to output_device
    # This ensures the final tensor ends up on the correct device for loss computation
    transformer.single_blocks[-1]._split_output_device = output_device
    _log_debug(f"Last block output device set to {output_device}")

    # Log distribution
    print("[GPU Splitter] Block distribution:")
    for gpu_id in gpu_ids:
        counts = block_distribution[gpu_id]
        print(f"  GPU {gpu_id}: {counts['double']} double blocks, {counts['single']} single blocks")
    print(f"  Final output routed to: {output_device}")


def add_model_gpu_splitter_to_flux2(
    transformer: "Flux2",
    # User-specified splits (takes precedence if provided)
    gpu_split_double: Optional[List[int]] = None,
    gpu_split_single: Optional[List[int]] = None,
    # Stream-based transfers for overlapped compute/transfer
    use_stream_transfers: bool = False,
    # Per-block synchronization (default off for better pipelining)
    sync_per_block: bool = False,
    # Output device configuration
    output_device: Optional[torch.device] = None,
    # Legacy greedy algorithm parameters (fallback)
    other_module_params: Optional[float] = 25e9,
    other_module_param_count_scale: Optional[float] = 0.3,
):
    """
    Apply GPU model splitting to a Flux2 transformer.

    Distributes transformer blocks across all available GPUs. Supports three modes:

    1. User-specified splits: If gpu_split_double and gpu_split_single are provided,
       uses those exact distributions.

    2. Default deterministic splits: For 2, 4, or 8 GPUs, uses pre-defined even splits
       that balance memory well for B300 GPUs.

    3. Greedy parameter-based: Falls back to the legacy algorithm that assigns
       blocks until each GPU reaches its fair share of parameters.

    Args:
        transformer: The Flux2 model to split
        gpu_split_double: List of double blocks per GPU, e.g., [4, 4] for 2 GPUs
        gpu_split_single: List of single blocks per GPU, e.g., [24, 24] for 2 GPUs
        use_stream_transfers: If True, use CUDA stream-based transfers for
            overlapped compute/transfer (recommended for best throughput)
        sync_per_block: If True, synchronize device after each block's transfers.
            Default False for better pipelining when using stream transfers.
            Set True only if debugging transfer issues without stream manager.
        output_device: Device where final output should reside. If None, uses
            cuda:0 (the first GPU). This is important for loss computation and
            backward pass - set this to match where your loss is computed.
        other_module_params: Estimated parameter count for non-transformer modules
        other_module_param_count_scale: Multiplier for other_module_params

    Memory Distribution Example (2x B300 with FLUX.2, deterministic split):
        GPU 0: Embedders + 4 double + 24 single (~50GB transformer + TE cached)
        GPU 1: 4 double + 24 single (~50GB transformer)

    Memory Distribution Example (4x B300 with FLUX.2, deterministic split):
        GPU 0: Embedders + 2 double + 12 single (~25GB)
        GPU 1: 2 double + 12 single (~25GB)
        GPU 2: 2 double + 12 single (~25GB)
        GPU 3: 2 double + 12 single (~25GB)

    Notes on Synchronization:
        - With use_stream_transfers=True: The DeviceTransferManager handles all
          synchronization via CUDA events. sync_per_block has no effect.
        - With use_stream_transfers=False and sync_per_block=False (default):
          Tensors are transferred with non_blocking=True but no explicit sync.
          PyTorch's default stream ordering ensures correctness, but there's
          no overlap between compute and transfer.
        - With use_stream_transfers=False and sync_per_block=True:
          Each block waits for transfers to complete before compute. This is
          the safest but slowest option - use only for debugging.
    """
    gpu_ids = list(range(torch.cuda.device_count()))
    n_gpus = len(gpu_ids)

    if n_gpus < 2:
        print("[GPU Splitter] Warning: Only 1 GPU detected, GPU splitting will have no effect")
        return

    print(f"[GPU Splitter] Splitting FLUX.2 model across {n_gpus} GPUs")

    # Reset any previous split state
    for block in transformer.double_blocks:
        _reset_block(block)
    for block in transformer.single_blocks:
        _reset_block(block)

    # Determine split strategy
    use_deterministic = False
    double_split = None
    single_split = None

    # Priority 1: User-specified splits
    if gpu_split_double is not None and gpu_split_single is not None:
        if len(gpu_split_double) != n_gpus:
            raise ValueError(f"gpu_split_double has {len(gpu_split_double)} elements but {n_gpus} GPUs available")
        if len(gpu_split_single) != n_gpus:
            raise ValueError(f"gpu_split_single has {len(gpu_split_single)} elements but {n_gpus} GPUs available")
        double_split = gpu_split_double
        single_split = gpu_split_single
        use_deterministic = True
        print(f"[GPU Splitter] Using user-specified split configuration")

    # Priority 2: Default deterministic splits for common configurations
    elif n_gpus in DEFAULT_SPLITS:
        double_split = DEFAULT_SPLITS[n_gpus]["double"]
        single_split = DEFAULT_SPLITS[n_gpus]["single"]
        use_deterministic = True
        print(f"[GPU Splitter] Using default deterministic split for {n_gpus} GPUs")

    # Apply the appropriate split
    if use_deterministic:
        _apply_deterministic_split(
            transformer, gpu_ids, double_split, single_split,
            use_stream_transfers=use_stream_transfers,
            sync_per_block=sync_per_block,
            output_device=output_device,
        )
    else:
        _apply_greedy_split(
            transformer, gpu_ids, other_module_params, other_module_param_count_scale,
            use_stream_transfers=use_stream_transfers,
            sync_per_block=sync_per_block,
            output_device=output_device,
        )

    # Store list of devices for stream manager initialization
    transformer._split_devices = [torch.device(f"cuda:{gpu_id}") for gpu_id in gpu_ids]
    transformer._use_stream_transfers = use_stream_transfers
    transformer._sync_per_block = sync_per_block
    transformer._output_device = output_device if output_device is not None else torch.device(f"cuda:{gpu_ids[0]}")

    # Wrap the .to() method to respect GPU splitting
    transformer._pre_gpu_split_to = transformer.to
    transformer.to = partial(new_device_to_flux2, transformer)

    print("[GPU Splitter] FLUX.2 model splitting complete")
    if use_stream_transfers:
        print("[GPU Splitter] NOTE: Stream transfers enabled. Ensure DeviceTransferManager is")
        print("              instantiated in get_noise_prediction() for this to take effect.")
