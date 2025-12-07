import importlib
import os
from typing import Any, List, Mapping, Optional, Set, Type

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

__all__ = [
    "get_flux2_wrap_policy",
    "get_fsdp_mixed_precision",
    "save_fsdp_checkpoint",
    "load_fsdp_checkpoint",
    "validate_fsdp_config",
    "ShardingStrategy",
]


def get_flux2_wrap_policy() -> ModuleWrapPolicy:
    """
    Build an FSDP auto-wrap policy for FLUX.2 blocks.

    The FLUX modules are imported lazily so the import path only executes when
    FSDP wrapping is requested, preventing errors in environments without FLUX.
    """
    try:
        # FLUX2 blocks are in the extensions_built_in diffusion_models
        model_module = importlib.import_module("extensions_built_in.diffusion_models.flux2.src.model")
        double_stream_block = getattr(model_module, "DoubleStreamBlock")
        single_stream_block = getattr(model_module, "SingleStreamBlock")
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "FLUX.2 layers are not available; ensure the flux2 extension is present."
        ) from exc

    block_types: Set[Type[torch.nn.Module]] = {
        double_stream_block,
        single_stream_block,
    }
    return ModuleWrapPolicy(block_types)


def get_fsdp_mixed_precision(dtype: Optional[str] = "bf16") -> Optional[MixedPrecision]:
    """
    Create a MixedPrecision configuration for FSDP.

    Args:
        dtype: Precision mode ('bf16', 'fp16', 'fp32', or None).

    Returns:
        MixedPrecision for bf16/fp16, or None when running in full precision.
    """
    if dtype is None:
        return None

    dtype_normalized = dtype.lower()
    if dtype_normalized == "bf16":
        target_dtype = torch.bfloat16
    elif dtype_normalized == "fp16":
        target_dtype = torch.float16
    elif dtype_normalized == "fp32":
        return None
    else:
        raise ValueError(f"Unsupported mixed precision dtype '{dtype}'.")

    return MixedPrecision(
        param_dtype=target_dtype,
        reduce_dtype=target_dtype,
        buffer_dtype=target_dtype,
    )


def _rank_checkpoint_path(path: str, rank: int) -> str:
    base, ext = os.path.splitext(path)
    ext = ext or ".pt"
    return f"{base}.rank{rank}{ext}"


def _get_config_value(config: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def save_fsdp_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    path: str,
    rank: int,
    use_sharded: bool = True,
) -> str:
    """
    Save an FSDP checkpoint containing model and optimizer state.

    Args:
        model: FSDP-wrapped model to checkpoint.
        optimizer: Optimizer whose state should be saved.
        path: Base path for the checkpoint. Sharded saves append `.rank{rank}`.
        rank: Current process rank.
        use_sharded: Whether to save sharded (per-rank) or full (rank 0) state.

    Returns:
        The path that was written for this rank.
    """
    base_path = os.fspath(path)
    target_path = _rank_checkpoint_path(base_path, rank) if use_sharded else base_path
    os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)

    state_dict_type = (
        StateDictType.SHARDED_STATE_DICT
        if use_sharded
        else StateDictType.FULL_STATE_DICT
    )
    state_config = (
        ShardedStateDictConfig(offload_to_cpu=True, rank0_only=False)
        if use_sharded
        else FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    )
    optim_state_config = (
        ShardedOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
        if use_sharded
        else FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    )

    with FSDP.state_dict_type(
        model,
        state_dict_type,
        state_dict_config=state_config,
        optim_state_dict_config=optim_state_config,
    ):
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer)

    if not use_sharded and rank != 0:
        return target_path

    checkpoint = {
        "model": model_state,
        "optimizer": optim_state,
        "state_dict_type": "sharded" if use_sharded else "full",
    }
    torch.save(checkpoint, target_path)
    return target_path


def load_fsdp_checkpoint(
    model: FSDP,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
) -> StateDictType:
    """
    Load an FSDP checkpoint, handling both sharded and full state dicts.

    Args:
        model: FSDP-wrapped model to restore.
        optimizer: Optimizer to restore; pass None to skip optimizer loading.
        path: Base checkpoint path. Sharded checkpoints are expected to carry
            the `.rank{rank}` suffix added by `save_fsdp_checkpoint`.

    Returns:
        The StateDictType that was loaded.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    base_path = os.fspath(path)
    sharded_path = _rank_checkpoint_path(base_path, rank)
    checkpoint_path = sharded_path if os.path.exists(sharded_path) else base_path

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_tag = checkpoint.get("state_dict_type")
    if state_tag == "sharded" or checkpoint_path == sharded_path:
        state_dict_type = StateDictType.SHARDED_STATE_DICT
        state_config = ShardedStateDictConfig(offload_to_cpu=True, rank0_only=False)
        optim_state_config = ShardedOptimStateDictConfig(
            offload_to_cpu=True, rank0_only=False
        )
    else:
        state_dict_type = StateDictType.FULL_STATE_DICT
        state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_state_config = FullOptimStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )

    with FSDP.state_dict_type(
        model,
        state_dict_type,
        state_dict_config=state_config,
        optim_state_dict_config=optim_state_config,
    ):
        model.load_state_dict(checkpoint["model"])
        if optimizer is not None and "optimizer" in checkpoint:
            optim_state = FSDP.optim_state_dict_to_load(
                checkpoint["optimizer"], model, optimizer
            )
            optimizer.load_state_dict(optim_state)

    if dist.is_initialized():
        dist.barrier()

    return state_dict_type


def validate_fsdp_config(train_config: Mapping[str, Any]) -> List[str]:
    """
    Validate FSDP-related training configuration and return any errors found.

    Args:
        train_config: Mapping of configuration options for the training run.

    Returns:
        A list of error strings; empty when the configuration is valid.
    """
    errors: List[str] = []
    dist_mode = str(_get_config_value(train_config, "dist_mode", "")).lower()
    using_fsdp = dist_mode == "fsdp" or bool(_get_config_value(train_config, "use_fsdp"))

    if not using_fsdp:
        return errors

    if dist_mode == "ddp":
        errors.append("FSDP cannot be enabled while dist_mode='ddp'; use 'fsdp' instead.")
    if _get_config_value(train_config, "gpu_split"):
        errors.append("FSDP is incompatible with gpu_split (different parallelism model).")
    if _get_config_value(train_config, "layer_offloading"):
        errors.append("FSDP cannot be combined with layer_offloading; disable offloading.")
    if _get_config_value(train_config, "low_vram"):
        errors.append("FSDP conflicts with low_vram mode, which overrides .to() behavior.")

    return errors
