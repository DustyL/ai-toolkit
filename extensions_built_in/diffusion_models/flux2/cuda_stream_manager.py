"""
CUDA Stream-Based Device Transfer Manager for FLUX.2 GPU Splitting

This module provides fine-grained synchronization for multi-GPU model parallelism,
enabling compute on GPU N to overlap with transfers to GPU N+1.

Key Features:
- Dedicated transfer stream per GPU (separate from compute)
- CUDA event-based synchronization (no full device syncs)
- Pre-staged modulation tensor caching
- Automatic cleanup on context exit

Synchronization Pattern:
    GPU N (compute stream):    [Compute Block] ───record(compute_event)───►
    GPU N (transfer stream):                   wait(compute_event)──[Transfer]─record(transfer_event)─►
    GPU N+1 (compute stream):                                       wait(transfer_event)──[Compute]──►

Usage:
    with DeviceTransferManager(devices=[...]) as mgr:
        mgr.stage_modulation_tensors(mod_img, mod_txt, mod_single)
        for block in blocks:
            tensors = mgr.transfer_to_device(tensors, block._split_device)
            output = block.forward(*tensors)
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import torch
from torch import Tensor


@dataclass
class DeviceState:
    """Per-device state for transfer management."""
    device: torch.device
    transfer_stream: torch.cuda.Stream
    compute_event: torch.cuda.Event
    transfer_event: torch.cuda.Event
    # Cached modulation tensors for this device
    mod_img_cache: Optional[Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]] = None
    mod_txt_cache: Optional[Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]] = None
    mod_single_cache: Optional[Tuple[Tensor, ...]] = None
    # Track if modulation tensors have been staged
    mods_staged: bool = False


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
    ):
        """
        Initialize transfer manager for the given devices.

        Args:
            devices: List of torch.device objects representing GPUs to manage.
                     These should be in the order blocks will be executed.
            enable_timing: If True, create events with timing enabled (adds overhead).
        """
        self.devices = [
            d if isinstance(d, torch.device) else torch.device(d)
            for d in devices
        ]
        self.enable_timing = enable_timing
        self._device_states: Dict[torch.device, DeviceState] = {}
        self._initialized = False

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

    def __enter__(self):
        """Context manager entry - initializes resources."""
        self._ensure_initialized()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - synchronizes and cleans up."""
        self.synchronize_all()
        self._clear_mod_caches()
        return False

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
        Pre-stage modulation tensors to all GPUs.

        Modulation tensors are computed once on GPU 0 (from the time/guidance embeddings)
        and used identically by all blocks. Rather than transferring them with each
        block's inputs, we transfer them once at the start of forward().

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

        source_state = self._get_state(source_device)

        for device, state in self._device_states.items():
            if device == source_device:
                # Source device - just cache the originals
                state.mod_img_cache = mod_img
                state.mod_txt_cache = mod_txt
                state.mod_single_cache = mod_single
                state.mods_staged = True
                continue

            # Transfer to other devices on their transfer streams
            with torch.cuda.stream(state.transfer_stream):
                # Wait for source compute to complete
                if source_state is not None:
                    state.transfer_stream.wait_event(source_state.compute_event)

                # Transfer mod_img: ((t1, t2, t3), (t4, t5, t6))
                state.mod_img_cache = (
                    tuple(t.to(device, non_blocking=True) for t in mod_img[0]),
                    tuple(t.to(device, non_blocking=True) for t in mod_img[1]),
                )

                # Transfer mod_txt: ((t1, t2, t3), (t4, t5, t6))
                state.mod_txt_cache = (
                    tuple(t.to(device, non_blocking=True) for t in mod_txt[0]),
                    tuple(t.to(device, non_blocking=True) for t in mod_txt[1]),
                )

                # Transfer mod_single: (shift, scale, gate)
                state.mod_single_cache = tuple(
                    t.to(device, non_blocking=True) for t in mod_single
                )

                # Record that transfers are queued
                state.transfer_event.record()

            state.mods_staged = True

    def get_staged_modulation_tensors(
        self,
        device: torch.device,
        block_type: str,  # "double" or "single"
    ) -> Union[
        Tuple[Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]], Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]],
        Tuple[Tensor, ...],
    ]:
        """
        Get pre-staged modulation tensors for a device.

        Args:
            device: Target device
            block_type: "double" for DoubleStreamBlock, "single" for SingleStreamBlock

        Returns:
            For "double": (mod_img, mod_txt) tuple
            For "single": (shift, scale, gate) tuple
        """
        state = self._get_state(device)
        if state is None or not state.mods_staged:
            return None  # Fallback - caller should use original tensors

        if block_type == "double":
            return state.mod_img_cache, state.mod_txt_cache
        elif block_type == "single":
            return state.mod_single_cache
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
        """
        for device in self._device_states.keys():
            torch.cuda.synchronize(device)

    def _clear_mod_caches(self):
        """Clear modulation tensor caches."""
        for state in self._device_states.values():
            state.mod_img_cache = None
            state.mod_txt_cache = None
            state.mod_single_cache = None
            state.mods_staged = False


# Global instance for the current forward pass
_current_transfer_manager: Optional[DeviceTransferManager] = None


def get_transfer_manager() -> Optional[DeviceTransferManager]:
    """Get the current transfer manager (if any)."""
    return _current_transfer_manager


def set_transfer_manager(manager: Optional[DeviceTransferManager]):
    """Set the current transfer manager."""
    global _current_transfer_manager
    _current_transfer_manager = manager
