"""
Memory Manager for AI Toolkit Layer Offloading.

This module provides the MemoryManager class for managing CPU-GPU weight
offloading during training. The per-module state architecture ensures:

- No cross-layer contention (each module has dedicated streams/events)
- Proper cleanup when offloading is disabled or model is freed
- Clear error messages for unsupported configurations
"""

import os
import warnings
import torch
from typing import List, Optional
from .manager_modules import (
    LinearLayerMemoryManager,
    ConvLayerMemoryManager,
    clear_device_state,
    ModuleOffloadState,
    _is_quantized_tensor,
)
from toolkit.print import print_acc

LINEAR_MODULES = [
    "Linear",
    "LoRACompatibleLinear",
    "QLinear",
]
CONV_MODULES = [
    "Conv2d",
    "LoRACompatibleConv",
    "QConv2d",
]

UNMANAGED_MODULES = [
    "LayerNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "Embedding",
    "EmbeddingBag",
    "RNNBase",
    "LSTM",
    "GRU",
    "RNN",
    "Conv3d"
]

UNMANAGED_MODULES_INCLUDES = ["RotaryEmbedding", "Norm", "RotaryPosEmbed"]

# Maximum recommended streams per device to avoid driver overhead
MAX_RECOMMENDED_STREAMS_PER_DEVICE = 128


class MemoryManager:
    """
    Manages CPU-GPU weight offloading for large model training.

    The MemoryManager attaches to a module and manages its child Linear/Conv
    layers, keeping their weights on CPU and transferring them to GPU only
    when needed for forward/backward passes.

    Usage:
        # Attach to a model
        MemoryManager.attach(model, device=torch.device("cuda:0"))

        # Train normally - offloading happens automatically

        # When done, detach to clean up resources
        MemoryManager.detach(model)
    """

    # Track streams per device to warn about proliferation
    _streams_per_device: dict = {}

    def __init__(
        self,
        module: torch.nn.Module,
        process_device: torch.device = torch.device("cpu"),
    ):
        self.module: torch.nn.Module = module
        self.process_device: torch.device = process_device
        self.unmanaged_modules: List[torch.nn.Module] = []
        self._managed_layers: List[torch.nn.Module] = []  # Track managed layers for cleanup

    def memory_managed_to(self, *args, **kwargs):
        """Custom .to() method that handles memory-managed modules properly."""
        # first move all the unmanaged modules
        for module in self.unmanaged_modules:
            if isinstance(module, torch.nn.Parameter):
                # Parameter cannot move this way
                module.data = module.data.to(*args, **kwargs)
            else:
                module.to(*args, **kwargs)
        # check for a dtype argument
        dtype = None
        if "dtype" in kwargs:
            dtype = kwargs["dtype"]
        elif len(args) > 0:
            for i, arg in enumerate(args):
                if isinstance(arg, torch.dtype):
                    dtype = arg
                    break
        if dtype is not None:
            return self.module._mm_to(dtype=dtype)
        return self.module

    @classmethod
    def attach(
        cls,
        module: torch.nn.Module,
        device: torch.device,
        offload_percent: float = 1.0,
        ignore_modules: Optional[List[torch.nn.Module]] = None,
        deterministic: bool = True,
    ):
        """
        Attach memory management to a module.

        Args:
            module: The module to manage (typically a transformer or large model)
            device: The target CUDA device for computation
            offload_percent: Fraction of eligible layers to offload (0.0-1.0).
                            1.0 = offload all, 0.5 = offload first 50% of layers
            ignore_modules: List of modules to exclude from offloading
            deterministic: If True (default), use deterministic layer selection.
                          If False, use random selection (legacy behavior).
        """
        if ignore_modules is None:
            ignore_modules = []

        if hasattr(module, "_memory_manager"):
            # already attached
            return

        module._memory_manager = cls(module, device)
        debug_mm = os.environ.get("AITK_MM_DEBUG") == "1"
        if debug_mm and not hasattr(module, "_mm_debug_logged"):
            print_acc(f"[MM] Attaching MemoryManager to {module.__class__.__name__} with offload_percent={offload_percent} device={device}")
            module._mm_debug_logged = True

        # override the to method to handle memory management
        module._mm_to = module.to
        module.to = module._memory_manager.memory_managed_to

        # add ignore modules to unmanaged list
        for im in ignore_modules:
            module._memory_manager.unmanaged_modules.append(im)

        # First pass: collect all eligible modules and check for quantized training issues
        eligible_modules = []
        modules_processed = set(id(x) for x in ignore_modules)
        quantized_training_warnings = []

        for name, sub_module in module.named_modules():
            for child_name, child_module in sub_module.named_modules():
                if id(child_module) in modules_processed:
                    continue

                is_linear = child_module.__class__.__name__ in LINEAR_MODULES
                is_conv = child_module.__class__.__name__ in CONV_MODULES
                is_unmanaged = (
                    child_module.__class__.__name__ in UNMANAGED_MODULES or
                    any(inc in child_module.__class__.__name__ for inc in UNMANAGED_MODULES_INCLUDES)
                )

                if is_linear or is_conv:
                    # Check for quantized weights with requires_grad (unsupported)
                    if hasattr(child_module, 'weight') and child_module.weight is not None:
                        weight = child_module.weight
                        if _is_quantized_tensor(weight) and getattr(weight, 'requires_grad', False):
                            full_name = f"{name}.{child_name}" if child_name else name
                            quantized_training_warnings.append(full_name)

                    eligible_modules.append((child_module, 'linear' if is_linear else 'conv'))
                    modules_processed.add(id(child_module))
                elif is_unmanaged:
                    module._memory_manager.unmanaged_modules.append(child_module)
                    modules_processed.add(id(child_module))

        # Warn about quantized training issues early
        if quantized_training_warnings:
            warnings.warn(
                f"Layer offloading: Found {len(quantized_training_warnings)} quantized layers with "
                f"requires_grad=True. Gradients cannot be computed for quantized parameters. "
                f"Either freeze these layers or disable quantization for training. "
                f"First few: {quantized_training_warnings[:3]}",
                UserWarning
            )

        # Calculate how many layers to offload
        total_eligible = len(eligible_modules)
        num_to_offload = int(total_eligible * offload_percent)

        if debug_mm:
            print_acc(f"[MM] Found {total_eligible} eligible layers, will offload {num_to_offload}")

        # Deterministic selection: offload first N layers (consistent across runs)
        # This is better for reproducibility and cache efficiency
        if deterministic:
            layers_to_offload = set(range(num_to_offload))
        else:
            # Legacy random selection (not recommended)
            import random
            all_indices = list(range(total_eligible))
            random.shuffle(all_indices)
            layers_to_offload = set(all_indices[:num_to_offload])

        managed_count = 0
        skipped_count = 0

        # Second pass: attach memory managers to selected layers
        for idx, (child_module, module_type) in enumerate(eligible_modules):
            if idx in layers_to_offload:
                if module_type == 'linear':
                    LinearLayerMemoryManager.attach(child_module, module._memory_manager)
                else:
                    ConvLayerMemoryManager.attach(child_module, module._memory_manager)

                module._memory_manager._managed_layers.append(child_module)
                managed_count += 1

                # Attach to ARA as well
                if hasattr(child_module, "ara_lora_ref"):
                    ara = child_module.ara_lora_ref()
                    if not hasattr(ara, "_memory_manager"):
                        MemoryManager.attach(ara, device)
            else:
                module._memory_manager.unmanaged_modules.append(child_module)
                skipped_count += 1

        # Track and warn about stream proliferation
        device_key = str(device)
        if device_key not in cls._streams_per_device:
            cls._streams_per_device[device_key] = 0
        cls._streams_per_device[device_key] += managed_count * 2  # 2 streams per layer

        if cls._streams_per_device[device_key] > MAX_RECOMMENDED_STREAMS_PER_DEVICE:
            warnings.warn(
                f"Layer offloading: {cls._streams_per_device[device_key]} streams allocated on {device}. "
                f"This exceeds the recommended maximum of {MAX_RECOMMENDED_STREAMS_PER_DEVICE}. "
                f"Consider reducing offload_percent or using a stream pool for better performance.",
                RuntimeWarning
            )

        if debug_mm:
            print_acc(f"[MM] Attached to {managed_count} layers, skipped {skipped_count} layers")
            print_acc(f"[MM] Total streams on {device}: {cls._streams_per_device[device_key]}")

    @classmethod
    def detach(cls, module: torch.nn.Module):
        """
        Detach memory management from a module and clean up resources.

        This should be called when:
        - Training is complete
        - Switching from offloaded to non-offloaded mode
        - Before deleting a model to ensure proper cleanup

        Args:
            module: The module to detach memory management from
        """
        if not hasattr(module, "_memory_manager"):
            return

        manager = module._memory_manager
        debug_mm = os.environ.get("AITK_MM_DEBUG") == "1"

        if debug_mm:
            print_acc(f"[MM] Detaching MemoryManager from {module.__class__.__name__}")

        # Track streams being freed
        device_key = str(manager.process_device)
        streams_freed = 0

        # Detach all managed layers
        detached_count = 0
        for child_module in manager._managed_layers:
            if hasattr(child_module, "_layer_memory_manager"):
                child_module._layer_memory_manager.detach()
                detached_count += 1
                streams_freed += 2  # 2 streams per layer

        # Update stream tracking
        if device_key in cls._streams_per_device:
            cls._streams_per_device[device_key] = max(0, cls._streams_per_device[device_key] - streams_freed)

        # Restore original .to() method
        if hasattr(module, "_mm_to"):
            module.to = module._mm_to
            delattr(module, "_mm_to")

        # Remove debug flag
        if hasattr(module, "_mm_debug_logged"):
            delattr(module, "_mm_debug_logged")

        # Remove manager reference
        delattr(module, "_memory_manager")

        if debug_mm:
            print_acc(f"[MM] Detached {detached_count} layers, freed {streams_freed} streams")

        # Try to free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def is_attached(cls, module: torch.nn.Module) -> bool:
        """Check if memory management is attached to a module."""
        return hasattr(module, "_memory_manager")

    @classmethod
    def get_managed_layer_count(cls, module: torch.nn.Module) -> int:
        """Get the number of layers being managed for a module."""
        if not hasattr(module, "_memory_manager"):
            return 0
        return len(module._memory_manager._managed_layers)

    @classmethod
    def synchronize_all(cls, module: torch.nn.Module):
        """
        Synchronize all CUDA streams used by managed layers.

        Call this before optimizer.step() to ensure all gradient
        transfers from GPU to CPU have completed.

        Args:
            module: The module with memory management attached
        """
        if not hasattr(module, "_memory_manager"):
            return

        manager = module._memory_manager

        for child_module in manager._managed_layers:
            if hasattr(child_module, "_layer_memory_manager"):
                layer_manager = child_module._layer_memory_manager
                if hasattr(layer_manager, "_state") and layer_manager._state is not None:
                    state = layer_manager._state
                    if state.transfer_grad_stream is not None:
                        try:
                            state.transfer_grad_stream.synchronize()
                        except Exception:
                            pass

    @classmethod
    def clear_all_state(cls):
        """
        Clear all global device state.

        This clears any legacy global state that may have accumulated.
        Call this after training is complete to free all resources.
        """
        clear_device_state()
        cls._streams_per_device.clear()

    @classmethod
    def get_stream_count(cls, device: Optional[torch.device] = None) -> int:
        """
        Get the number of streams allocated for layer offloading.

        Args:
            device: Specific device to check, or None for total across all devices.

        Returns:
            Number of streams allocated.
        """
        if device is not None:
            return cls._streams_per_device.get(str(device), 0)
        return sum(cls._streams_per_device.values())

    @classmethod
    def log_status(cls, module: torch.nn.Module, module_name: str = "model"):
        """
        Log the current memory management status for a module.

        Call this at training startup to provide visibility into offloading config.
        Automatically warns if stream count exceeds recommended threshold.

        Args:
            module: The module to report status for
            module_name: Human-readable name for logging (e.g., "transformer", "unet")
        """
        if not hasattr(module, "_memory_manager"):
            return

        manager = module._memory_manager
        device = manager.process_device
        device_key = str(device)
        managed_count = len(manager._managed_layers)
        stream_count = cls._streams_per_device.get(device_key, 0)

        # Count total eligible layers for percentage calculation
        total_layers = managed_count + len([
            m for m in manager.unmanaged_modules
            if m.__class__.__name__ in LINEAR_MODULES + CONV_MODULES
        ])

        if total_layers > 0:
            offload_pct = (managed_count / total_layers) * 100
        else:
            offload_pct = 0

        print_acc(f"[MemoryManager] {module_name}: {managed_count} layers offloaded "
                  f"({offload_pct:.1f}%), {stream_count} streams on {device}")

        # Warn if approaching or exceeding threshold
        if stream_count > MAX_RECOMMENDED_STREAMS_PER_DEVICE:
            print_acc(f"[MemoryManager] WARNING: Stream count ({stream_count}) exceeds "
                      f"recommended max ({MAX_RECOMMENDED_STREAMS_PER_DEVICE}). "
                      f"Consider reducing offload_percent.")
        elif stream_count > MAX_RECOMMENDED_STREAMS_PER_DEVICE * 0.8:
            print_acc(f"[MemoryManager] Note: Stream count ({stream_count}) approaching "
                      f"recommended max ({MAX_RECOMMENDED_STREAMS_PER_DEVICE}).")
