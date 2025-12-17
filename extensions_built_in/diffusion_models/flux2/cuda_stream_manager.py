"""
CUDA Stream-Based Device Transfer Manager for FLUX.2 GPU Splitting

This module provides fine-grained synchronization for multi-GPU model parallelism,
enabling compute on GPU N to overlap with transfers to GPU N+1.

Key Features:
- Dedicated transfer stream per GPU (separate from compute)
- CUDA event-based synchronization (no full device syncs)
- Pre-staged modulation tensor caching with shape validation
- Packed modulation transfers for efficiency
- Automatic cleanup on context exit
- Configurable synchronization behavior
- Debug logging support (set FLUX_TRANSFER_DEBUG=1 to enable)

Synchronization Pattern:
    GPU N (compute stream):    [Compute Block] ---record(compute_event)--->
    GPU N (transfer stream):                   wait(compute_event)--[Transfer]-record(transfer_event)->
    GPU N+1 (compute stream):                                       wait(transfer_event)--[Compute]-->

Usage:
    with DeviceTransferManager(devices=[...]) as mgr:
        mgr.stage_modulation_tensors(mod_img, mod_txt, mod_single)
        for block in blocks:
            tensors = mgr.transfer_to_device(tensors, block._split_device)
            output = block.forward(*tensors)
"""

import os
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import torch
from torch import Tensor
import logging

logger = logging.getLogger(__name__)

# Global debug flag (can be set via FLUX_TRANSFER_DEBUG env var)
_transfer_debug_default = os.environ.get("FLUX_TRANSFER_DEBUG", "").lower() in ("1", "true", "yes")


@dataclass
class ModulationCacheKey:
    """
    Key for validating modulation tensor cache compatibility.

    This ensures cached tensors are only reused when shapes and properties match,
    which is critical for:
    - Bucketing (variable resolutions → different sequence lengths)
    - Changing batch sizes
    - Different LoRA configurations that might alter hidden dimensions
    - Device consistency
    """
    batch_size: int
    seq_len: int  # For modulation vectors this is typically 1
    hidden_dim: int
    dtype: torch.dtype
    device: torch.device  # Include device for cross-device validation

    @classmethod
    def from_tensor(cls, t: Tensor) -> "ModulationCacheKey":
        """Create cache key from a tensor."""
        return cls(
            batch_size=t.shape[0],
            seq_len=t.shape[1] if t.ndim > 2 else 1,
            hidden_dim=t.shape[-1],
            dtype=t.dtype,
            device=t.device,
        )

    def is_compatible(self, other: "ModulationCacheKey") -> bool:
        """
        Check if two cache keys are compatible.

        Returns False if any dimension, dtype, or device differs, which
        triggers cache invalidation and rebuild.
        """
        return (
            self.batch_size == other.batch_size
            and self.seq_len == other.seq_len
            and self.hidden_dim == other.hidden_dim
            and self.dtype == other.dtype
            and self.device == other.device
        )

    def describe_mismatch(self, other: "ModulationCacheKey") -> str:
        """Describe what's different between two keys (for debugging)."""
        mismatches = []
        if self.batch_size != other.batch_size:
            mismatches.append(f"batch_size: {self.batch_size} vs {other.batch_size}")
        if self.seq_len != other.seq_len:
            mismatches.append(f"seq_len: {self.seq_len} vs {other.seq_len}")
        if self.hidden_dim != other.hidden_dim:
            mismatches.append(f"hidden_dim: {self.hidden_dim} vs {other.hidden_dim}")
        if self.dtype != other.dtype:
            mismatches.append(f"dtype: {self.dtype} vs {other.dtype}")
        if self.device != other.device:
            mismatches.append(f"device: {self.device} vs {other.device}")
        return ", ".join(mismatches) if mismatches else "no mismatch"


@dataclass
class DeviceState:
    """Per-device state for transfer management."""
    device: torch.device
    transfer_stream: torch.cuda.Stream
    compute_event: torch.cuda.Event
    transfer_event: torch.cuda.Event
    # Cached modulation tensors for this device (packed as single tensors)
    mod_img_cache: Optional[Tensor] = None
    mod_txt_cache: Optional[Tensor] = None
    mod_single_cache: Optional[Tensor] = None
    # Cache keys for shape validation
    mod_img_key: Optional[ModulationCacheKey] = None
    mod_txt_key: Optional[ModulationCacheKey] = None
    mod_single_key: Optional[ModulationCacheKey] = None
    # Track if modulation tensors have been staged
    mods_staged: bool = False


def _pack_modulation_tensors(mod: Tuple, is_double: bool = True) -> Tensor:
    """
    Pack modulation tensors into a single contiguous tensor for efficient transfer.

    For double blocks: mod = ((shift1, scale1, gate1), (shift2, scale2, gate2))
                       -> packed shape [6, B, seq_len, hidden_dim]
    For single blocks: mod = (shift, scale, gate)
                       -> packed shape [3, B, seq_len, hidden_dim]

    Args:
        mod: Modulation tuple(s)
        is_double: True for double block modulation (6 tensors), False for single (3 tensors)

    Returns:
        Packed tensor [6, ...] or [3, ...]
    """
    if is_double:
        # mod = ((shift1, scale1, gate1), (shift2, scale2, gate2))
        mod1, mod2 = mod
        tensors = list(mod1) + list(mod2)  # 6 tensors total
    else:
        # mod = (shift, scale, gate)
        tensors = list(mod)  # 3 tensors

    return torch.stack(tensors, dim=0)


def _unpack_modulation_tensors(packed: Tensor, is_double: bool = True):
    """
    Unpack a single packed tensor back into modulation tuples.

    Args:
        packed: Packed tensor [6, ...] or [3, ...]
        is_double: True for double block modulation, False for single

    Returns:
        For double: ((shift1, scale1, gate1), (shift2, scale2, gate2))
        For single: (shift, scale, gate)
    """
    tensors = [packed[i] for i in range(packed.shape[0])]

    if is_double:
        return (tuple(tensors[:3]), tuple(tensors[3:6]))
    else:
        return tuple(tensors)


class DeviceTransferManager:
    """
    Manages CUDA stream-based transfers between GPUs for model parallelism.

    This class creates dedicated transfer streams per GPU and uses CUDA events
    for fine-grained synchronization, allowing compute on one GPU to overlap
    with data transfer to the next GPU.

    The key insight is that PyTorch's default stream handles compute, while
    a separate transfer stream handles D2D (device-to-device) copies. Events
    establish happens-before relationships between streams.
    """

    def __init__(
        self,
        devices: List[torch.device],
        enable_timing: bool = False,
        sync_on_exit: bool = False,
        output_device: Optional[torch.device] = None,
    ):
        """
        Initialize transfer manager for the given devices.

        Args:
            devices: List of torch.device objects representing GPUs to manage.
                     These should be in the order blocks will be executed.
            enable_timing: If True, create events with timing enabled (adds overhead).
            sync_on_exit: If True, synchronize all devices on context exit.
                         Default False for performance (caller should handle sync).
            output_device: Device where final output should reside. If None, uses
                          devices[0]. Important for backward pass compatibility.
        """
        self.devices = [
            d if isinstance(d, torch.device) else torch.device(d)
            for d in devices
        ]
        self.enable_timing = enable_timing
        self.sync_on_exit = sync_on_exit
        self.output_device = output_device if output_device is not None else (
            self.devices[0] if self.devices else None
        )
        self._device_states: Dict[torch.device, DeviceState] = {}
        self._initialized = False
        self._debug = _transfer_debug_default  # Can override via set_debug()

    def set_debug(self, enabled: bool = True):
        """Enable/disable debug logging."""
        self._debug = enabled

    def _log_debug(self, msg: str):
        """Log debug message if debug mode is enabled."""
        if self._debug:
            logger.info(f"[DeviceTransferManager] {msg}")

    def _ensure_initialized(self):
        """Lazily initialize CUDA resources on first use."""
        if self._initialized:
            return

        for device in self.devices:
            if device.type != "cuda":
                continue

            with torch.cuda.device(device):
                state = DeviceState(
                    device=device,
                    transfer_stream=torch.cuda.Stream(device=device),
                    compute_event=torch.cuda.Event(enable_timing=self.enable_timing),
                    transfer_event=torch.cuda.Event(enable_timing=self.enable_timing),
                )
                self._device_states[device] = state

        self._initialized = True
        self._log_debug(f"Initialized with {len(self._device_states)} devices")

    def __enter__(self):
        """
        Context manager entry - initializes resources and resets per-step state.

        This method is safe to call multiple times (manager reuse). CUDA streams
        and events are initialized once and reused. Per-step state (modulation
        caches, staging flags) are reset each entry.
        """
        self._ensure_initialized()
        # Reset per-step state to allow manager reuse across training steps
        self._reset_per_step_state()
        return self

    def _reset_per_step_state(self):
        """
        Reset per-step state for manager reuse.

        Called at the start of each forward pass to ensure clean state.
        Does NOT destroy CUDA streams/events (those are reusable).
        """
        for state in self._device_states.values():
            # Clear modulation caches (will be re-staged this step)
            state.mod_img_cache = None
            state.mod_txt_cache = None
            state.mod_single_cache = None
            # Note: We keep cache keys to enable shape validation
            # state.mod_img_key = None  # Keep for validation
            # state.mod_txt_key = None
            # state.mod_single_key = None
            state.mods_staged = False
        self._log_debug("Reset per-step state for manager reuse")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - joins streams and optionally synchronizes.

        IMPORTANT: For backward pass safety, we ensure the default stream on
        the output device waits for all transfer/compute events. This creates
        the happens-before relationship that autograd requires.

        sync_on_exit behavior:
        - False (default): Only joins the default stream on output_device to ensure
          autograd backward pass safety. Does NOT call global synchronize_all().
          This is the recommended setting for performance.
        - True: Additionally calls synchronize_all() for a full device barrier.
          This is heavier and typically not needed for correctness.
        """
        # Join default stream to transfer completion on output device
        # This is CRITICAL for backward pass safety - autograd uses default stream
        self._join_default_stream()

        if self.sync_on_exit:
            # Optional: full device barrier (not needed for correctness, only for debugging)
            self.synchronize_all()
            self._log_debug("sync_on_exit=True: performed full synchronize_all()")

        self._clear_mod_caches()
        return False

    def _join_default_stream(self):
        """
        Ensure the default stream on the output device sees all prior operations.

        This is critical for autograd safety - the backward pass uses the default
        stream and must see all completed forward operations.

        Implementation:
        - For each device with recorded events, make the output device's default
          stream wait on those events.
        - This creates a happens-before relationship WITHOUT a full device sync.
        - The result: when forward() returns, calling .backward() on the output
          tensor is safe because the default stream has visibility of all work.
        """
        if not self._initialized or self.output_device is None:
            return

        output_state = self._device_states.get(self.output_device)
        if output_state is None:
            self._log_debug(f"Warning: output_device {self.output_device} not in managed devices")
            return

        # Get the default stream on the output device (where autograd will run backward)
        default_stream = torch.cuda.current_stream(self.output_device)

        # Make the default stream wait on compute and transfer events from ALL devices
        # This ensures the entire forward pass is visible before backward starts
        for device, state in self._device_states.items():
            # The compute event marks when compute finished on this device
            # wait_event is safe to call even if the event was never recorded
            # (it will be a no-op in that case)
            default_stream.wait_event(state.compute_event)
            # The transfer event marks when transfers to this device completed
            default_stream.wait_event(state.transfer_event)

        self._log_debug(f"Joined default stream on {self.output_device} - backward pass safe")

    def _get_state(self, device: torch.device) -> Optional[DeviceState]:
        """Get state for a device, initializing if needed."""
        self._ensure_initialized()
        device = device if isinstance(device, torch.device) else torch.device(device)
        return self._device_states.get(device)

    def stage_modulation_tensors(
        self,
        mod_img: Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]],
        mod_txt: Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]],
        mod_single: Tuple[Tensor, ...],
        source_device: Optional[torch.device] = None,
    ):
        """
        Pre-stage modulation tensors to all GPUs with packed transfers.

        Modulation tensors are computed once on GPU 0 (from the time/guidance embeddings)
        and used identically by all blocks. Rather than transferring them with each
        block's inputs, we transfer them once at the start of forward().

        This implementation packs all modulation tensors into single contiguous
        tensors before transfer, reducing the overhead from many small copies.

        Args:
            mod_img: ((shift, scale, gate), (shift, scale, gate)) for image stream
            mod_txt: ((shift, scale, gate), (shift, scale, gate)) for text stream
            mod_single: (shift, scale, gate) for single blocks
            source_device: Device where tensors currently reside (default: GPU 0)
        """
        self._ensure_initialized()

        if source_device is None:
            # Infer from first tensor
            source_device = mod_img[0][0].device

        # Create cache keys for shape validation
        img_key = ModulationCacheKey.from_tensor(mod_img[0][0])
        txt_key = ModulationCacheKey.from_tensor(mod_txt[0][0])
        single_key = ModulationCacheKey.from_tensor(mod_single[0])

        # Pack tensors on source device for efficient transfer
        packed_img = _pack_modulation_tensors(mod_img, is_double=True)
        packed_txt = _pack_modulation_tensors(mod_txt, is_double=True)
        packed_single = _pack_modulation_tensors(mod_single, is_double=False)

        source_state = self._get_state(source_device)

        for device, state in self._device_states.items():
            # Check if cache is still valid
            cache_valid = (
                state.mods_staged
                and state.mod_img_key is not None
                and state.mod_img_key.is_compatible(img_key)
                and state.mod_txt_key is not None
                and state.mod_txt_key.is_compatible(txt_key)
                and state.mod_single_key is not None
                and state.mod_single_key.is_compatible(single_key)
            )

            if cache_valid and device == source_device:
                # Source device with valid cache - just update references
                state.mod_img_cache = packed_img
                state.mod_txt_cache = packed_txt
                state.mod_single_cache = packed_single
                self._log_debug(f"Reusing modulation cache on {device}")
                continue

            if device == source_device:
                # Source device - cache the packed originals
                state.mod_img_cache = packed_img
                state.mod_txt_cache = packed_txt
                state.mod_single_cache = packed_single
                state.mod_img_key = img_key
                state.mod_txt_key = txt_key
                state.mod_single_key = single_key
                state.mods_staged = True
                self._log_debug(f"Staged modulation on source device {device}")
                continue

            # Invalidate cache if shapes changed
            if not cache_valid and state.mod_img_key is not None:
                mismatch = state.mod_img_key.describe_mismatch(img_key)
                self._log_debug(f"Invalidating modulation cache on {device}: {mismatch}")

            # Transfer packed tensors to other devices on their transfer streams
            with torch.cuda.stream(state.transfer_stream):
                # Wait for source compute to complete
                if source_state is not None:
                    state.transfer_stream.wait_event(source_state.compute_event)

                # Transfer packed tensors (3 transfers instead of 12+)
                state.mod_img_cache = packed_img.to(device, non_blocking=True)
                state.mod_txt_cache = packed_txt.to(device, non_blocking=True)
                state.mod_single_cache = packed_single.to(device, non_blocking=True)

                # Update cache keys
                state.mod_img_key = img_key
                state.mod_txt_key = txt_key
                state.mod_single_key = single_key

                # Record that transfers are queued
                state.transfer_event.record()

            state.mods_staged = True
            self._log_debug(f"Queued packed modulation transfer to {device}")

    def get_staged_modulation_tensors(
        self,
        device: torch.device,
        block_type: str,  # "double" or "single"
    ) -> Union[
        Tuple[Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]], Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]],
        Tuple[Tensor, ...],
        None,
    ]:
        """
        Get pre-staged modulation tensors for a device, unpacking from cache.

        Args:
            device: Target device
            block_type: "double" for DoubleStreamBlock, "single" for SingleStreamBlock

        Returns:
            For "double": (mod_img, mod_txt) tuple of unpacked modulation tuples
            For "single": (shift, scale, gate) tuple
            None if not staged
        """
        state = self._get_state(device)
        if state is None or not state.mods_staged:
            return None  # Fallback - caller should use original tensors

        if block_type == "double":
            if state.mod_img_cache is None or state.mod_txt_cache is None:
                return None
            # Unpack from cached packed tensors
            mod_img = _unpack_modulation_tensors(state.mod_img_cache, is_double=True)
            mod_txt = _unpack_modulation_tensors(state.mod_txt_cache, is_double=True)
            return mod_img, mod_txt
        elif block_type == "single":
            if state.mod_single_cache is None:
                return None
            return _unpack_modulation_tensors(state.mod_single_cache, is_double=False)
        else:
            raise ValueError(f"Unknown block_type: {block_type}")

    def transfer_tensors(
        self,
        tensors: Tuple[Tensor, ...],
        target_device: torch.device,
        source_device: Optional[torch.device] = None,
        wait_for_source_compute: bool = True,
    ) -> Tuple[Tensor, ...]:
        """
        Transfer tensors to target device using the transfer stream.

        This queues the transfers on the target device's transfer stream
        and returns immediately. The caller must call wait_for_transfer()
        before using the returned tensors.

        Args:
            tensors: Tuple of tensors to transfer
            target_device: Device to transfer to
            source_device: Device tensors are on (inferred if None)
            wait_for_source_compute: If True, wait for source device's
                compute to complete before starting transfer

        Returns:
            Tuple of tensors on target device (may not be ready yet!)
        """
        target_device = (
            target_device if isinstance(target_device, torch.device)
            else torch.device(target_device)
        )

        # Infer source device from first non-None tensor
        if source_device is None:
            for t in tensors:
                if t is not None:
                    source_device = t.device
                    break

        # If already on target, return as-is
        if source_device == target_device:
            return tensors

        target_state = self._get_state(target_device)
        source_state = self._get_state(source_device) if source_device else None

        if target_state is None:
            # Fallback for non-CUDA or unmanaged devices
            return tuple(
                t.to(target_device, non_blocking=True) if t is not None else None
                for t in tensors
            )

        # Queue transfers on target's transfer stream
        with torch.cuda.stream(target_state.transfer_stream):
            # Optionally wait for source compute to complete
            if wait_for_source_compute and source_state is not None:
                target_state.transfer_stream.wait_event(source_state.compute_event)

            transferred = tuple(
                t.to(target_device, non_blocking=True) if t is not None else None
                for t in tensors
            )

            # Record that transfer is queued
            target_state.transfer_event.record()

        return transferred

    def transfer_nested_tuples(
        self,
        nested: Tuple,
        target_device: torch.device,
        source_device: Optional[torch.device] = None,
    ) -> Tuple:
        """
        Transfer nested tuple of tensors (like modulation tuples).

        Handles structures like ((t1, t2, t3), (t4, t5, t6)) for double blocks.
        """
        target_device = (
            target_device if isinstance(target_device, torch.device)
            else torch.device(target_device)
        )

        if source_device is None:
            # Find first tensor to infer source
            def find_device(obj):
                if isinstance(obj, Tensor):
                    return obj.device
                elif isinstance(obj, tuple):
                    for item in obj:
                        d = find_device(item)
                        if d is not None:
                            return d
                return None
            source_device = find_device(nested)

        if source_device == target_device:
            return nested

        target_state = self._get_state(target_device)
        source_state = self._get_state(source_device) if source_device else None

        def recursive_transfer(obj):
            if obj is None:
                return None
            if isinstance(obj, Tensor):
                return obj.to(target_device, non_blocking=True)
            elif isinstance(obj, tuple):
                return tuple(recursive_transfer(item) for item in obj)
            else:
                return obj

        if target_state is None:
            return recursive_transfer(nested)

        with torch.cuda.stream(target_state.transfer_stream):
            if source_state is not None:
                target_state.transfer_stream.wait_event(source_state.compute_event)

            result = recursive_transfer(nested)
            target_state.transfer_event.record()

        return result

    def wait_for_transfer(self, device: torch.device):
        """
        Wait for transfers to device to complete.

        This should be called on the compute stream before using transferred tensors.
        It inserts a dependency from the compute stream to the transfer stream.

        Args:
            device: Device whose transfers to wait for
        """
        state = self._get_state(device)
        if state is None:
            return

        # Make current (compute) stream wait for transfers
        torch.cuda.current_stream(device).wait_event(state.transfer_event)

    def mark_compute_complete(self, device: torch.device):
        """
        Mark that compute on this device is complete.

        This should be called after a block's forward() completes. It records
        an event that the next device's transfer stream can wait on.

        Args:
            device: Device that completed compute
        """
        state = self._get_state(device)
        if state is None:
            return

        # Record on compute (current) stream
        state.compute_event.record(torch.cuda.current_stream(device))

    def synchronize_all(self):
        """
        Synchronize all managed devices.

        This should be called at the end of forward() to ensure all
        operations are complete before returning.

        WARNING: This is a heavy operation that blocks until all work completes.
        Prefer using _join_default_stream() for autograd safety without full sync.
        """
        for device in self._device_states.keys():
            torch.cuda.synchronize(device)
        self._log_debug("Synchronized all devices")

    def synchronize_output_device(self):
        """
        Synchronize only the output device.

        Lighter weight than synchronize_all() but still ensures the output
        is ready for use.
        """
        if self.output_device is not None and self.output_device.type == "cuda":
            torch.cuda.synchronize(self.output_device)
            self._log_debug(f"Synchronized output device {self.output_device}")

    def _clear_mod_caches(self):
        """Clear modulation tensor caches."""
        for state in self._device_states.values():
            state.mod_img_cache = None
            state.mod_txt_cache = None
            state.mod_single_cache = None
            state.mod_img_key = None
            state.mod_txt_key = None
            state.mod_single_key = None
            state.mods_staged = False

    def get_output_device(self) -> Optional[torch.device]:
        """Get the configured output device."""
        return self.output_device


# Global instance for the current forward pass
_current_transfer_manager: Optional[DeviceTransferManager] = None


def get_transfer_manager() -> Optional[DeviceTransferManager]:
    """Get the current transfer manager (if any)."""
    return _current_transfer_manager


def set_transfer_manager(manager: Optional[DeviceTransferManager]):
    """Set the current transfer manager."""
    global _current_transfer_manager
    _current_transfer_manager = manager
