"""
Attention backend selection and loading for Flux.2-dev.

Supports three backends:
- SDPA: PyTorch's scaled_dot_product_attention (always available)
- Flash: Flash Attention 2 (Triton-based, requires flash-attn package)
- CuTE: Flash Attention CuTE (CUTLASS DSL, Blackwell SM >= 10.0)

Key Design Decisions:
1. Per-device backend selection (for split_model_over_gpus)
2. Caching by (device_index, dtype, head_dim) - seqlen NOT cached
3. Runtime seqlen threshold (checked per-call, not at init)
4. AMP/autocast dtype guards (check actual tensor dtype, not model dtype)
5. Robust fallback chain with rank-0 logging
6. Strategy vs dispatch: init selects strategy, runtime dispatches based on inputs
"""

import logging
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Dict, Set

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AttentionBackendInfo:
    """Information about an attention backend's availability."""
    name: str
    available: bool
    reason: str  # Why unavailable, or version info if available


# =============================================================================
# Logging Utilities
# =============================================================================

def _log_rank0(log_fn: Callable, message: str):
    """Log only on rank 0 to avoid DDP spam."""
    try:
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except:
        pass
    log_fn(message)


# =============================================================================
# Failure Tracking for Backend Disabling
# =============================================================================
#
# Scope: Per-backend, per-process. NOT persisted across restarts.
# - _failure_counts: incremented on each runtime kernel failure
# - _disabled_backends: once failures >= threshold, backend is disabled for this run
# - Resets: Only on process restart (not configurable, not persistent)
# - Logging: Disable logs are rank-0 only to avoid DDP spam
#
# IMPORTANT: Once a backend is added to _disabled_backends, NO further attempts
# are made to use that backend for the remainder of the process. Calls go
# directly to SDPA without even trying the disabled backend.
#
_failure_counts: Dict[str, int] = {'cute': 0, 'flash': 0}
_MAX_FAILURES_BEFORE_DISABLE = 3  # After this many failures, permanently use SDPA for this run
_disabled_backends: Set[str] = set()  # Per-backend, per-process; cleared only on restart

# Module-level flags for throttled warnings (separate flags for each warning type)
_causal_warning_logged = False  # causal=True provided
_mixed_dtype_warning_logged = False  # q/k/v have different dtypes
_fp32_fallback_warning_logged = False  # fp32 tensors with CuTE/Flash backend
_device_mismatch_warned = False  # context.device != tensor.device


def _invalidate_selection_cache_for_backend(backend: str):
    """
    Invalidate selection cache entries that reference a disabled backend.

    Called when a backend is disabled after repeated failures. This ensures
    that future select_backend() calls don't return stale cached selections.
    """
    keys_to_remove = [k for k, v in _selection_cache.items() if v == backend]
    for key in keys_to_remove:
        del _selection_cache[key]


# =============================================================================
# Cache Structures
# =============================================================================

# Cache: availability info (computed once globally per device)
# Key: (device_type, device_index) to avoid CPU/GPU collisions
# NOTE: CPU has index=None which we normalize to 0, but device.type differentiates
_availability_cache: Dict[Tuple[str, int], Dict[str, AttentionBackendInfo]] = {}

# Cache: selected backend per (device_type, device_index, dtype, head_dim)
# IMPORTANT: seqlen is NOT part of cache key - it varies per batch and is checked at runtime
# NOTE: When a backend is disabled at runtime, stale cache entries are invalidated
_selection_cache: Dict[Tuple[str, int, torch.dtype, int], str] = {}

# Cache: attention functions per backend name
_function_cache: Dict[str, Callable] = {}


# =============================================================================
# Availability Checking
# =============================================================================

def get_backend_availability(device: torch.device) -> Dict[str, AttentionBackendInfo]:
    """
    Check which attention backends are available on the given device.
    Results are cached per (device_type, device_index) to avoid CPU/GPU collisions.
    """
    device_type = device.type
    device_idx = device.index if device.index is not None else 0
    cache_key = (device_type, device_idx)

    if cache_key in _availability_cache:
        return _availability_cache[cache_key]

    availability = {}

    # SDPA is always available (PyTorch built-in)
    availability['sdpa'] = AttentionBackendInfo(
        name='PyTorch SDPA',
        available=True,
        reason=f'PyTorch {torch.__version__}'
    )

    # Non-CUDA devices: only SDPA is available
    if device_type != 'cuda':
        availability['flash'] = AttentionBackendInfo(
            name='Flash Attention 2',
            available=False,
            reason=f'Requires CUDA device, got {device_type}'
        )
        availability['cute'] = AttentionBackendInfo(
            name='Flash Attention CuTE',
            available=False,
            reason=f'Requires CUDA device, got {device_type}'
        )
        _availability_cache[cache_key] = availability
        return availability

    # Check Flash Attention 2
    try:
        from flash_attn import flash_attn_func
        availability['flash'] = AttentionBackendInfo(
            name='Flash Attention 2',
            available=True,
            reason='flash-attn installed'
        )
    except ImportError as e:
        availability['flash'] = AttentionBackendInfo(
            name='Flash Attention 2',
            available=False,
            reason=f'ImportError: {e}'
        )

    # Check Flash Attention CuTE (requires Blackwell SM >= 10.0)
    try:
        sm_major, sm_minor = torch.cuda.get_device_capability(device)
        if sm_major < 10:
            availability['cute'] = AttentionBackendInfo(
                name='Flash Attention CuTE',
                available=False,
                reason=f'Requires SM >= 10.0, got SM {sm_major}.{sm_minor}'
            )
        else:
            try:
                from flash_attn.cute import flash_attn_func as cute_attn_func
                availability['cute'] = AttentionBackendInfo(
                    name='Flash Attention CuTE',
                    available=True,
                    reason=f'SM {sm_major}.{sm_minor} (Blackwell)'
                )
            except ImportError as e:
                availability['cute'] = AttentionBackendInfo(
                    name='Flash Attention CuTE',
                    available=False,
                    reason=f'ImportError: {e}'
                )
    except Exception as e:
        availability['cute'] = AttentionBackendInfo(
            name='Flash Attention CuTE',
            available=False,
            reason=f'Error checking SM: {e}'
        )

    _availability_cache[cache_key] = availability
    return availability


# =============================================================================
# Backend Selection
# =============================================================================

def select_backend(
    requested: str,
    device: torch.device,
    dtype: torch.dtype,
    head_dim: int,
) -> str:
    """
    Select the best available attention backend based on constraints.

    Args:
        requested: User-requested backend ('auto', 'sdpa', 'flash', 'cute')
        device: Target CUDA device
        dtype: Model dtype (bf16/fp16 for CuTE/Flash, fp32 falls back to SDPA)
        head_dim: Attention head dimension (must be multiple of 16 for CuTE)

    Returns:
        Selected backend name ('sdpa', 'flash', or 'cute')
    """
    device_type = device.type
    device_idx = device.index if device.index is not None else 0
    cache_key = (device_type, device_idx, dtype, head_dim)

    if cache_key in _selection_cache:
        return _selection_cache[cache_key]

    availability = get_backend_availability(device)

    # dtype constraint: CuTE/Flash require bf16 or fp16
    dtype_ok = dtype in (torch.bfloat16, torch.float16)

    # head_dim constraint: CuTE requires multiple of 16
    head_dim_ok = head_dim % 16 == 0

    selected = 'sdpa'  # Default fallback

    if requested == 'auto':
        # Priority: cute > flash > sdpa (if constraints met)
        if (availability['cute'].available and dtype_ok and head_dim_ok
                and 'cute' not in _disabled_backends):
            selected = 'cute'
        elif availability['flash'].available and dtype_ok and 'flash' not in _disabled_backends:
            selected = 'flash'
        else:
            selected = 'sdpa'

    elif requested == 'cute':
        if not availability['cute'].available:
            _log_rank0(logger.warning,
                f"[Attention] CuTE requested but unavailable: {availability['cute'].reason}. Using SDPA.")
            selected = 'sdpa'
        elif not dtype_ok:
            _log_rank0(logger.warning,
                f"[Attention] CuTE requires bf16/fp16, got {dtype}. Using SDPA.")
            selected = 'sdpa'
        elif not head_dim_ok:
            _log_rank0(logger.warning,
                f"[Attention] CuTE requires head_dim % 16 == 0, got {head_dim}. Using SDPA.")
            selected = 'sdpa'
        elif 'cute' in _disabled_backends:
            _log_rank0(logger.warning,
                "[Attention] CuTE disabled due to runtime failures. Using SDPA.")
            selected = 'sdpa'
        else:
            selected = 'cute'

    elif requested == 'flash':
        if not availability['flash'].available:
            _log_rank0(logger.warning,
                f"[Attention] Flash requested but unavailable: {availability['flash'].reason}. Using SDPA.")
            selected = 'sdpa'
        elif not dtype_ok:
            _log_rank0(logger.warning,
                f"[Attention] Flash requires bf16/fp16, got {dtype}. Using SDPA.")
            selected = 'sdpa'
        elif 'flash' in _disabled_backends:
            _log_rank0(logger.warning,
                "[Attention] Flash disabled due to runtime failures. Using SDPA.")
            selected = 'sdpa'
        else:
            selected = 'flash'

    elif requested == 'sdpa':
        selected = 'sdpa'

    else:
        _log_rank0(logger.warning,
            f"[Attention] Unknown backend '{requested}', using SDPA.")
        selected = 'sdpa'

    _selection_cache[cache_key] = selected
    return selected


# =============================================================================
# Attention Functions
# =============================================================================

def _get_sdpa_fn() -> Callable:
    """Get SDPA attention function (native layout, no transpose needed)."""
    def sdpa_attention(q: Tensor, k: Tensor, v: Tensor, causal: bool = False) -> Tensor:
        # SDPA uses native [B, H, L, D] layout
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return sdpa_attention


def _get_flash_fn() -> Callable:
    """Get Flash Attention 2 function with layout handling and runtime error guard."""
    from flash_attn import flash_attn_func

    # Capture SDPA fallback for runtime errors
    sdpa_fallback = _get_sdpa_fn()

    def flash_attention(q: Tensor, k: Tensor, v: Tensor, causal: bool = False) -> Tensor:
        global _failure_counts, _disabled_backends

        # If permanently disabled due to repeated failures, use SDPA directly
        if 'flash' in _disabled_backends:
            return sdpa_fallback(q, k, v, causal=causal)

        # Flash expects [B, L, H, D], we have [B, H, L, D]
        # MANDATORY: .contiguous() is required - stride bugs occur without it
        q_t = q.transpose(1, 2).contiguous()
        k_t = k.transpose(1, 2).contiguous()
        v_t = v.transpose(1, 2).contiguous()

        try:
            out = flash_attn_func(q_t, k_t, v_t, causal=causal)
        except Exception as e:
            # Runtime failure: fall back to SDPA
            _failure_counts['flash'] += 1
            if _failure_counts['flash'] >= _MAX_FAILURES_BEFORE_DISABLE:
                _disabled_backends.add('flash')
                _invalidate_selection_cache_for_backend('flash')
                _log_rank0(
                    logger.error,
                    f"[Attention] Flash failed {_failure_counts['flash']} times; "
                    f"permanently disabling for this run. Last error: {e}"
                )
            else:
                _log_rank0(
                    logger.warning,
                    f"[Attention] Flash kernel failed ({_failure_counts['flash']}/{_MAX_FAILURES_BEFORE_DISABLE}): "
                    f"{e}; falling back to SDPA for this call"
                )
            return sdpa_fallback(q, k, v, causal=causal)

        # Back to [B, H, L, D] - non-contiguous but safe (downstream rearrange handles it)
        return out.transpose(1, 2)

    return flash_attention


def _get_cute_fn() -> Callable:
    """Get Flash Attention CuTE function with layout handling and runtime error guard."""
    from flash_attn.cute import flash_attn_func as cute_attn_func

    # Capture SDPA fallback for runtime errors
    sdpa_fallback = _get_sdpa_fn()

    def cute_attention(q: Tensor, k: Tensor, v: Tensor, causal: bool = False) -> Tensor:
        global _failure_counts, _disabled_backends

        # If permanently disabled due to repeated failures, use SDPA directly
        if 'cute' in _disabled_backends:
            return sdpa_fallback(q, k, v, causal=causal)

        # CuTE expects [B, L, H, D], we have [B, H, L, D]
        # MANDATORY: .contiguous() is required - stride bugs occur without it
        q_t = q.transpose(1, 2).contiguous()
        k_t = k.transpose(1, 2).contiguous()
        v_t = v.transpose(1, 2).contiguous()

        try:
            out, _ = cute_attn_func(q_t, k_t, v_t, causal=causal)
        except Exception as e:
            # Runtime/JIT failure: fall back to SDPA
            _failure_counts['cute'] += 1
            if _failure_counts['cute'] >= _MAX_FAILURES_BEFORE_DISABLE:
                _disabled_backends.add('cute')
                _invalidate_selection_cache_for_backend('cute')
                _log_rank0(
                    logger.error,
                    f"[Attention] CuTE failed {_failure_counts['cute']} times; "
                    f"permanently disabling for this run. Last error: {e}"
                )
            else:
                _log_rank0(
                    logger.warning,
                    f"[Attention] CuTE kernel failed ({_failure_counts['cute']}/{_MAX_FAILURES_BEFORE_DISABLE}): "
                    f"{e}; falling back to SDPA for this call"
                )
            return sdpa_fallback(q, k, v, causal=causal)

        # Back to [B, H, L, D] - non-contiguous but safe (downstream rearrange handles it)
        return out.transpose(1, 2)

    return cute_attention


def get_attention_function(backend: str) -> Callable:
    """
    Get the attention function for a given backend.
    Functions are cached after first creation.
    """
    if backend in _function_cache:
        return _function_cache[backend]

    if backend == 'sdpa':
        fn = _get_sdpa_fn()
    elif backend == 'flash':
        fn = _get_flash_fn()
    elif backend == 'cute':
        fn = _get_cute_fn()
    else:
        raise ValueError(f"Unknown backend: {backend}")

    _function_cache[backend] = fn
    return fn


# =============================================================================
# Attention Context (Per-Block)
# =============================================================================

@dataclass
class AttentionContext:
    """
    Per-block attention context for Flux.2 transformer blocks.

    Encapsulates:
    - Selected backend (strategy chosen at init)
    - Backend function
    - Sequence length threshold for CuTE
    - Device and dtype info
    - Device checking mode (strict vs warn)
    """
    backend: str  # 'sdpa', 'flash', or 'cute'
    backend_fn: Callable
    cute_min_seqlen: int
    device: torch.device
    dtype: torch.dtype
    strict_device_check: bool = False  # If True, raise on device mismatch; else warn and proceed

    def get_effective_backend(self, seqlen: int) -> Tuple[str, Callable]:
        """
        Get the effective backend for the given sequence length.

        Applies cute_min_seqlen threshold at runtime.

        Boundary Behavior:
            - seqlen <  cute_min_seqlen -> use SDPA (overhead makes CuTE slower)
            - seqlen >= cute_min_seqlen -> use CuTE (speedup outweighs overhead)

        Note: seqlen is NOT part of the selection cache key; it varies per batch
        and is checked fresh on every attention call.
        """
        if self.backend == 'cute' and seqlen < self.cute_min_seqlen:
            # Below threshold: use SDPA for this call
            return 'sdpa', get_attention_function('sdpa')
        return self.backend, self.backend_fn

    def attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool = False,
    ) -> Tensor:
        """
        Compute attention with runtime guards.

        Handles (in order):
        1. Device consistency check -> error or warn based on strict_device_check
        2. Causal check -> force SDPA if causal=True
        3. Mixed dtype check -> force SDPA if q/k/v dtypes differ
        4. Sequence length threshold -> SDPA if seqlen < threshold
        5. AMP/fp32 dtype check -> SDPA if fp32

        fp32 Behavior:
            Fall back to SDPA (never cast to bf16). SDPA handles fp32 natively.
            A one-time warning is logged for visibility.

        Mixed Precision / Autocast:
            Selection uses actual tensor dtype (q.dtype), NOT model dtype.
            Under autocast, q/k/v will be bf16 even if model weights are fp32.

        Device Check Behavior:
            If strict_device_check=True (recommended for split_model_over_gpus),
            raises RuntimeError on device mismatch. Otherwise, logs warning and proceeds.

        Note on masks:
            Flux.2 does not use attention masks (it's not a causal LM).
            Mask support is not implemented; if needed, use SDPA directly.
        """
        global _causal_warning_logged, _mixed_dtype_warning_logged, _fp32_fallback_warning_logged, _device_mismatch_warned

        # 1. Device consistency check
        # Catches stale contexts after .to() moves or mis-splits
        if q.device != self.device:
            if self.strict_device_check:
                raise RuntimeError(
                    f"[Attention] Device mismatch: context.device={self.device}, "
                    f"q.device={q.device}. This indicates a stale context after .to() "
                    f"or a mis-split in split_model_over_gpus. Recreate contexts after "
                    f"device changes or fix the split configuration."
                )
            else:
                if not _device_mismatch_warned:
                    _log_rank0(
                        logger.warning,
                        f"[Attention] Device mismatch: context.device={self.device}, "
                        f"q.device={q.device}. This may indicate a stale context after .to(). "
                        f"WARNING: If devices have different architectures (e.g., Blackwell vs Ampere), "
                        f"CuTE/Flash kernels may fail or produce incorrect results. "
                        f"Set strict_device_check=True or reinitialize contexts after device changes."
                    )
                    _device_mismatch_warned = True
                # Proceed with tensor's device, but warn

        # Also verify k/v are on same device as q (basic sanity - always strict)
        if not (q.device == k.device == v.device):
            raise RuntimeError(
                f"[Attention] q/k/v must be on same device: "
                f"q={q.device}, k={k.device}, v={v.device}"
            )

        # 2. Causal safety check (CuTE/Flash support causal, but Flux.2 doesn't use it)
        if causal:
            if not _causal_warning_logged:
                _log_rank0(
                    logger.warning,
                    "[Attention] causal=True provided; using SDPA for causal attention"
                )
                _causal_warning_logged = True
            return get_attention_function('sdpa')(q, k, v, causal=causal)

        # 3. Mixed dtype check (q, k, v should all have same dtype)
        # Behavior: Force SDPA, log once (rank-0), do NOT attempt CuTE/Flash
        if not (q.dtype == k.dtype == v.dtype):
            if not _mixed_dtype_warning_logged:
                _log_rank0(
                    logger.warning,
                    f"[Attention] Mixed dtypes (q={q.dtype}, k={k.dtype}, v={v.dtype}); "
                    f"forcing SDPA (CuTE/Flash not attempted)"
                )
                _mixed_dtype_warning_logged = True
            return get_attention_function('sdpa')(q, k, v, causal=causal)

        # 4. Seqlen threshold check
        seqlen = q.shape[2]  # q is [B, H, L, D], L = concatenated txt+img tokens
        backend, fn = self.get_effective_backend(seqlen)

        # 5. AMP guard: if dtype is fp32 but we need bf16/fp16 for CuTE/Flash
        # Design decision: fall back to SDPA rather than casting to bf16
        if q.dtype == torch.float32 and backend in ('cute', 'flash'):
            if not _fp32_fallback_warning_logged:
                _log_rank0(
                    logger.info,
                    f"[Attention] fp32 tensors with {backend} backend (configured: {self.backend}); "
                    f"falling back to SDPA. Consider enabling autocast or setting dtype=bf16 for better performance."
                )
                _fp32_fallback_warning_logged = True
            return get_attention_function('sdpa')(q, k, v, causal=causal)

        return fn(q, k, v, causal=causal)


def create_attention_context(
    requested_backend: str,
    device: torch.device,
    dtype: torch.dtype,
    head_dim: int,
    cute_min_seqlen: int = 1024,
    strict_device_check: bool = False,
) -> AttentionContext:
    """
    Create an AttentionContext for a transformer block.

    Args:
        requested_backend: User-requested backend ('auto', 'sdpa', 'flash', 'cute')
        device: Target CUDA device for this block
        dtype: Model dtype (determines CuTE/Flash eligibility)
        head_dim: Attention head dimension (CuTE requires % 16 == 0)
        cute_min_seqlen: Minimum seqlen to use CuTE (default 1024)
        strict_device_check: If True, raise on context/tensor device mismatch;
                            recommended for split_model_over_gpus scenarios

    Returns:
        AttentionContext configured for the block
    """
    backend = select_backend(requested_backend, device, dtype, head_dim)
    backend_fn = get_attention_function(backend)

    return AttentionContext(
        backend=backend,
        backend_fn=backend_fn,
        cute_min_seqlen=cute_min_seqlen,
        device=device,
        dtype=dtype,
        strict_device_check=strict_device_check,
    )


# =============================================================================
# Initialization and Logging
# =============================================================================

_backend_logged: Set[Tuple[str, int, str]] = set()


def log_backend_selection(device: torch.device, backend: str, reason: str = ""):
    """Log backend selection once per (device_type, device_index, backend) tuple, rank-0 only."""
    device_type = device.type
    device_idx = device.index if device.index is not None else 0
    key = (device_type, device_idx, backend)

    if key in _backend_logged:
        return

    _backend_logged.add(key)

    availability = get_backend_availability(device)
    info = availability.get(backend, AttentionBackendInfo(backend, False, "unknown"))

    msg = f"[Attention] Device {device_idx}: Using {info.name}"
    if reason:
        msg += f" ({reason})"
    elif info.available:
        msg += f" ({info.reason})"

    _log_rank0(logger.info, msg)


def initialize_attention_for_flux2(
    transformer,
    requested_backend: str,
    dtype: torch.dtype,
    head_dim: int = 128,
    cute_min_seqlen: int = 1024,
    strict_device_check: bool = False,
):
    """
    Initialize attention contexts for all blocks in a Flux.2 transformer.

    CRITICAL TIMING: Must be called AFTER add_model_gpu_splitter_to_flux2()
    because we need _split_device to be set on blocks first.

    Rule: Splitting + context initialization should be considered FINAL.
    Calling .to() after that is UNSUPPORTED unless contexts are manually recreated.

    Args:
        transformer: Flux2 transformer model
        requested_backend: User-requested backend ('auto', 'sdpa', 'flash', 'cute')
        dtype: Model dtype (NOTE: CuTE/Flash require bf16/fp16; if dtype=fp32,
               backend selection falls back to SDPA even if autocast is used at runtime)
        head_dim: Attention head dimension (default 128 for Flux.2)
        cute_min_seqlen: Minimum seqlen to use CuTE
        strict_device_check: If True, raise on context/tensor device mismatch;
                            recommended for split_model_over_gpus scenarios
    """
    # Process double blocks
    for i, block in enumerate(transformer.double_blocks):
        device = getattr(block, '_split_device', None)
        if device is None:
            # Not split - use default device
            device = next(block.parameters()).device

        ctx = create_attention_context(
            requested_backend=requested_backend,
            device=device,
            dtype=dtype,
            head_dim=head_dim,
            cute_min_seqlen=cute_min_seqlen,
            strict_device_check=strict_device_check,
        )
        block._attention_context = ctx

        # Log once per unique (device, backend) combination
        log_backend_selection(device, ctx.backend)

    # Process single blocks
    for i, block in enumerate(transformer.single_blocks):
        device = getattr(block, '_split_device', None)
        if device is None:
            device = next(block.parameters()).device

        ctx = create_attention_context(
            requested_backend=requested_backend,
            device=device,
            dtype=dtype,
            head_dim=head_dim,
            cute_min_seqlen=cute_min_seqlen,
            strict_device_check=strict_device_check,
        )
        block._attention_context = ctx

        log_backend_selection(device, ctx.backend)


# =============================================================================
# Utility Functions
# =============================================================================

def clear_caches():
    """Clear all caches. Useful for testing."""
    global _causal_warning_logged, _mixed_dtype_warning_logged, _fp32_fallback_warning_logged, _device_mismatch_warned
    _availability_cache.clear()
    _selection_cache.clear()
    _function_cache.clear()
    _backend_logged.clear()
    _failure_counts['cute'] = 0
    _failure_counts['flash'] = 0
    _disabled_backends.clear()
    _causal_warning_logged = False
    _mixed_dtype_warning_logged = False
    _fp32_fallback_warning_logged = False
    _device_mismatch_warned = False
