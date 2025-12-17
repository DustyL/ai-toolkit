"""
Unified Virtual Memory helpers for GH200 deployments.

These utilities port the SimpleTuner GH200 optimisations into ai-toolkit.
They embrace a best-effort mindset: when the runtime is not backed by a
UVM-enabled PyTorch build, callbacks simply no-op.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Mapping, Optional, Sequence

import torch

logger = logging.getLogger("GH200Helpers")

try:  # pragma: no cover - optional import
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
except Exception:  # pragma: no cover - defensive
    DataLoaderBatchDTO = None  # type: ignore

try:  # pragma: no cover - optional import
    from toolkit.prompt_utils import PromptEmbeds
except Exception:  # pragma: no cover - defensive
    PromptEmbeds = None  # type: ignore

_UVM_HINT_OVERRIDE: Optional[bool] = None


def gh200_uvm_enabled() -> bool:
    """
    Determine whether GH200-specific optimisations should run.

    Environment variable precedence (default-enabled for GH200 deployments):
        AITK_GH200_ENABLE_UVM_HINTS = {1, true, yes, on}
    """

    if _UVM_HINT_OVERRIDE is not None:
        return _UVM_HINT_OVERRIDE
    flag = os.environ.get("AITK_GH200_ENABLE_UVM_HINTS", "1")
    return flag.strip().lower() in {"1", "true", "yes", "on"}


def set_uvm_hint_override(value: Optional[bool]) -> None:
    """
    Allow callers to override environment-driven behaviour at runtime.

    Passing ``None`` clears the override.
    """

    global _UVM_HINT_OVERRIDE
    _UVM_HINT_OVERRIDE = value


def _element_span_bytes(tensor: torch.Tensor) -> int:
    try:
        storage = tensor.untyped_storage()
        nbytes = storage.nbytes()
        if nbytes:
            return int(nbytes)
    except Exception:
        pass

    try:
        return int(tensor.element_size() * tensor.numel())
    except Exception:
        return 0


def _current_device() -> int:
    try:
        return int(torch.cuda.current_device())
    except Exception:
        return 0


def _cudart():
    if not hasattr(torch.cuda, "cudart"):
        return None
    try:
        return torch.cuda.cudart()
    except Exception:
        return None


def _apply_preferred_location(tensor: torch.Tensor, *, prefer_cpu: bool) -> bool:
    if not gh200_uvm_enabled():
        return False
    if not torch.cuda.is_available() or not tensor.is_cuda:
        return False

    try:
        if hasattr(torch.cuda, "memory_advise"):
            location = "cpu" if prefer_cpu else _current_device()
            torch.cuda.memory_advise(tensor, "set_preferred_location", device=location)
            torch.cuda.memory_advise(tensor, "set_accessed_by", device=_current_device())
            return True
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("torch.cuda.memory_advise failed: %s", exc)

    cudart = _cudart()
    if cudart is None:
        return False

    size = _element_span_bytes(tensor)
    if size <= 0:
        return False

    try:
        ptr = tensor.data_ptr()
    except Exception:
        try:
            ptr = tensor.storage().data_ptr()
        except Exception:
            return False

    try:
        advise = cudart.cudaMemAdvise
        preferred_location = cudart.cudaMemAdviseSetPreferredLocation
        accessed_by = cudart.cudaMemAdviseSetAccessedBy
        cpu_id = getattr(cudart, "cudaCpuDeviceId", -1)

        target = cpu_id if prefer_cpu else _current_device()
        advise(ptr, size, preferred_location, target)
        advise(ptr, size, accessed_by, _current_device())
        return True
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("cudaMemAdvise failed: %s", exc)
        return False


def prefer_cpu_residency(tensor: torch.Tensor) -> bool:
    """Hint the runtime to keep this CUDA tensor resident in Grace DDR memory."""
    return _apply_preferred_location(tensor, prefer_cpu=True)


def prefer_gpu_residency(tensor: torch.Tensor) -> bool:
    """Hint the runtime to keep this CUDA tensor resident in Hopper HBM."""
    return _apply_preferred_location(tensor, prefer_cpu=False)


def prefetch_to_device(tensor: torch.Tensor, device: int | None = None) -> bool:
    """
    Attempt to prefetch a managed tensor to ``device`` asynchronously.
    Returns ``True`` on success, ``False`` otherwise.
    """

    if not gh200_uvm_enabled():
        return False
    if not torch.cuda.is_available() or not tensor.is_cuda:
        return False

    device = _current_device() if device is None else int(device)

    try:
        if hasattr(torch.cuda, "prefetch_to_device"):
            torch.cuda.prefetch_to_device(tensor, device)
            return True
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("torch.cuda.prefetch_to_device failed: %s", exc)

    cudart = _cudart()
    if cudart is None:
        return False

    size = _element_span_bytes(tensor)
    if size <= 0:
        return False

    try:
        ptr = tensor.data_ptr()
    except Exception:
        try:
            ptr = tensor.storage().data_ptr()
        except Exception:
            return False

    try:
        cudart.cudaMemPrefetchAsync(ptr, size, device)
        return True
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("cudaMemPrefetchAsync failed: %s", exc)
        return False


def _iter_tensors(obj) -> Iterable[torch.Tensor]:
    if obj is None:
        return
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, torch.nn.Parameter):
        yield obj.detach()
    elif PromptEmbeds is not None and isinstance(obj, PromptEmbeds):
        yield from _iter_tensors(obj.text_embeds)
        if obj.pooled_embeds is not None:
            yield from _iter_tensors(obj.pooled_embeds)
        if getattr(obj, "attention_mask", None) is not None:
            yield from _iter_tensors(obj.attention_mask)
    elif isinstance(obj, Mapping):
        for value in obj.values():
            yield from _iter_tensors(value)
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        for value in obj:
            yield from _iter_tensors(value)
    else:
        for attr in ("text_embeds", "pooled_embeds"):
            if hasattr(obj, attr):
                yield from _iter_tensors(getattr(obj, attr))


def iter_tensors(obj) -> Iterable[torch.Tensor]:
    """Public helper to iterate over all tensors contained within ``obj``."""
    yield from _iter_tensors(obj)


def optimize_batch_for_gh200(batch) -> None:
    """
    Apply heuristic UVM placement for common diffusion training tensors.

    Current policy:
      * Latent tensors -> prefer CPU residency (Grace RAM).
      * Text encoder outputs -> prefer GPU residency.
    """

    if not gh200_uvm_enabled():
        return

    def _apply_latent_policy(latent_like):
        for tensor in iter_tensors(latent_like):
            prefer_cpu_residency(tensor)

    def _apply_text_policy(text_like):
        for tensor in iter_tensors(text_like):
            prefer_gpu_residency(tensor)
            prefetch_to_device(tensor)

    if DataLoaderBatchDTO is not None and isinstance(batch, DataLoaderBatchDTO):
        _apply_latent_policy(getattr(batch, "latents", None))
        _apply_latent_policy(getattr(batch, "unconditional_latents", None))
        _apply_text_policy(getattr(batch, "prompt_embeds", None))
        return

    if isinstance(batch, Mapping):
        if "latents" in batch:
            _apply_latent_policy(batch.get("latents"))
        if "unconditional_latents" in batch:
            _apply_latent_policy(batch.get("unconditional_latents"))
        for key in (
            "prompt_embeds",
            "unconditional_prompt_embeds",
            "encoder_hidden_states",
            "text_embeddings",
            "pooled_prompt_embeds",
        ):
            if key in batch:
                _apply_text_policy(batch.get(key))
