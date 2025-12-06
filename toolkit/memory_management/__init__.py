"""
Memory Management Module for AI Toolkit Layer Offloading.

This module provides CPU-GPU weight offloading for training large models
like Flux.2-dev that don't fit entirely in GPU memory.

Key classes and functions:
- MemoryManager: Main class for attaching/detaching offloading to models
- clear_device_state: Clean up global state (legacy mode)
- ModuleOffloadState: Per-module state for streams/events/buffers
- LoHALayerMemoryManager: Specialized manager for LoHA (LyCORIS) modules

Usage:
    from toolkit.memory_management import MemoryManager

    # Attach to model
    MemoryManager.attach(model, device=torch.device("cuda:0"))

    # Train normally...

    # Clean up when done
    MemoryManager.detach(model)

LoHA Support:
    LoHA modules from LyCORIS are automatically detected and managed.
    They use nn.Parameter directly instead of nn.Linear, requiring
    specialized handling for their 4+ parameter structure.
"""

from .manager import MemoryManager
from .manager_modules import clear_device_state, ModuleOffloadState
from .loha_layer_manager import (
    LoHALayerMemoryManager,
    LoHAModuleOffloadState,
    _is_loha_module,
    _is_lokr_module,
)

__all__ = [
    "MemoryManager",
    "clear_device_state",
    "ModuleOffloadState",
    "LoHALayerMemoryManager",
    "LoHAModuleOffloadState",
    "_is_loha_module",
    "_is_lokr_module",
]
