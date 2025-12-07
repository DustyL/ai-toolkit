"""
Distributed training utilities for multi-GPU training with DDP/FSDP.

This module provides helper functions for initializing and managing
distributed training across multiple GPUs using PyTorch's distributed
package. Supports both DistributedDataParallel (DDP) and 
FullyShardedDataParallel (FSDP) training modes.

Usage:
    # Launch with: torchrun --nproc_per_node=4 train.py
    
    from toolkit.distributed import init_distributed, is_main, cleanup
    
    init_distributed()
    # ... training code ...
    if is_main():
        save_checkpoint()
    cleanup()
"""

import os
import torch
import torch.distributed as dist
from typing import Optional


def init_distributed(backend: str = "nccl") -> None:
    """
    Initialize distributed training environment.
    
    Args:
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)
    """
    if dist.is_initialized():
        return
    
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


def is_main() -> bool:
    """Return True if this is the main process (rank 0)."""
    return get_rank() == 0


def get_rank() -> int:
    """Get the global rank of this process."""
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    """Get the total number of processes."""
    return int(os.environ.get("WORLD_SIZE", 1))


def get_local_rank() -> int:
    """Get the local rank (GPU index on this node)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_initialized()


def cleanup() -> None:
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()
