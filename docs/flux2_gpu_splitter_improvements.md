# FLUX.2 GPU Splitter: Analysis & Improvement Plan

## Overview

The `flux2_gpu_splitter.py` script implements **naive model parallelism** for FLUX.2, distributing transformer blocks across multiple GPUs. This document analyzes the current implementation and proposes improvements for 2x and 4x B300 GPU configurations.

---

## Current Implementation Summary

### Architecture
- **FLUX.2 Transformer**: 8 DoubleStreamBlocks + 48 SingleStreamBlocks (~12B params)
- **Text Encoder**: Mistral-3.1-24B (~24B params, stays on GPU 0)
- **VAE**: ~0.5B params (stays on GPU 0)

### How It Works
1. Wraps each block's `forward()` to move inputs to the block's assigned device (`_split_device`)
2. Last single block routes output back to GPU 0 via `_split_output_device`
3. Overrides `Flux2.to()` to respect split devices and not coalesce params
4. Uses **greedy parameter-based distribution**: starts GPU 0 with estimated "other module" budget, assigns blocks until per-GPU param threshold is met

### What's Good
- Defensive reset of previous split state (`_reset_block`)
- Handles dtype-only `.to()` calls vs device moves; preserves per-block devices
- Handles CPU move by uniform device assignment with stashed original devices for restoration
- Warns and returns early if <2 GPUs detected

---

## Issues & Concerns

### Critical Issues

#### 1. `non_blocking=True` Race Condition
```python
def _move_tensor_to_device(tensor: Tensor, device: torch.device) -> Tensor:
    if tensor.device != device:
        return tensor.to(device, non_blocking=True)  # DANGEROUS
    return tensor
```

**Problem**: `non_blocking=True` returns before transfer completes. Computation may start on garbage data.

**Fix Options**:
- Add explicit synchronization after transfers
- Use blocking transfers for correctness
- Implement proper CUDA stream management with event synchronization

#### 2. Parameter-Based Distribution is Crude

**Problem**: The greedy heuristic assumes `params ≈ memory`. For B300 GPUs:
- Activation memory dominates when gradient checkpointing is off
- Earlier blocks see larger activations (full sequence before concatenation in single blocks)
- GPU 0 gets overloaded with TE + first blocks; GPU 1 underused

**Impact**: Memory imbalance leads to OOMs on GPU 0 while GPU 1 has headroom.

#### 3. Modulation Tensors Copied Repeatedly

In `Flux2.forward()`:
```python
double_block_mod_img = self.double_stream_modulation_img(vec)  # Computed once on GPU 0
for block in self.double_blocks:  # Blocks on different GPUs
    img, txt = block(..., double_block_mod_img, ...)  # Copied each time
```

**Impact**: 8+ redundant copies per forward pass = bandwidth waste + latency.

### Medium Priority Issues

#### 4. Text Encoder Stays on GPU 0
- The splitter only touches `transformer.double_blocks/single_blocks`
- Mistral TE (~24B) remains entirely on GPU 0
- For training with frozen TE, use `cache_text_embeddings: true` to avoid this

#### 5. `.to()` Override Risks
- Can interfere with `torch.compile` or other wrappers if applied after splitting
- Comment warns about discarding post-split wrappers on reset
- **Rule**: Compile AFTER splitting, avoid re-splitting

#### 6. `_split_output_device` Edge Cases
- Only set for last single block once during split
- Restoration logic in `.to()` only triggers with `has_device=True`
- dtype-only calls don't touch it (probably fine)

#### 7. No CUDA Stream Parallelism
All transfers on default stream = serialized execution:
```
GPU0 block → transfer → GPU1 block → transfer → ...
```

With separate streams per GPU, transfers and compute could overlap.

### Low Priority / Nice-to-Have

#### 8. Gradient Checkpointing Interaction
- If enabled, activations recomputed on block's device (good)
- But heuristic doesn't adjust for different activation loads
- Earlier blocks may need different treatment

#### 9. Layer Offloading Conflict
- Model splitting and `layer_offloading` can fight each other
- Offloading tries to move layers to CPU; splitting assigns GPU devices
- **Recommendation**: Disable offloading when using GPU split

---

## Recommended Block Distributions

### FLUX.2 Architecture
| Block Type | Count | Params/Block | Total |
|------------|-------|--------------|-------|
| DoubleStreamBlock | 8 | ~400M | ~3.2B |
| SingleStreamBlock | 48 | ~180M | ~8.6B |
| Other (embedders, final_layer) | - | - | ~0.2B |
| **Transformer Total** | 56 | - | **~12B** |

### 2x B300 (128GB HBM3 each)

**Even Split (Recommended)**:
| GPU | Double Blocks | Single Blocks | Est. VRAM |
|-----|--------------|---------------|-----------|
| 0 | 4 (0-3) | 24 (0-23) | ~50GB + TE cached |
| 1 | 4 (4-7) | 24 (24-47) | ~50GB |

**Note**: With `cache_text_embeddings: true`, TE never loads. With TE on GPU 0, expect ~96GB usage.

### 4x B300

**Even Split (Recommended)**:
| GPU | Double Blocks | Single Blocks | Est. VRAM |
|-----|--------------|---------------|-----------|
| 0 | 2 (0-1) | 12 (0-11) | ~25GB + embedders |
| 1 | 2 (2-3) | 12 (12-23) | ~25GB |
| 2 | 2 (4-5) | 12 (24-35) | ~25GB |
| 3 | 2 (6-7) | 12 (36-47) | ~25GB |

**Activation-Aware Alternative** (if OOM on early GPUs):
| GPU | Double Blocks | Single Blocks | Rationale |
|-----|--------------|---------------|-----------|
| 0 | 2 | 8 | Early blocks = larger activations |
| 1 | 2 | 12 | Medium |
| 2 | 2 | 14 | Medium-late |
| 3 | 2 | 14 | Late blocks = smaller activations |

---

## Implementation Plan

### Phase 1: Deterministic Block Split (High Priority)

Replace greedy parameter-based assignment with configurable deterministic splits:

```python
def add_model_gpu_splitter_to_flux2(
    transformer: "Flux2",
    gpu_ids: list[int] = None,
    double_split: list[int] = None,  # e.g., [4, 4] for 2 GPUs
    single_split: list[int] = None,  # e.g., [24, 24] for 2 GPUs
    # Legacy params for fallback
    other_module_params: Optional[int] = 25e9,
    other_module_param_count_scale: Optional[float] = 0.3
):
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))

    n_gpus = len(gpu_ids)

    # Use explicit splits if provided
    if double_split is not None and single_split is not None:
        assert len(double_split) == n_gpus, f"double_split must have {n_gpus} elements"
        assert len(single_split) == n_gpus, f"single_split must have {n_gpus} elements"
        assert sum(double_split) == 8, "double_split must sum to 8"
        assert sum(single_split) == 48, "single_split must sum to 48"
        use_deterministic = True
    else:
        # Default even splits for common configurations
        if n_gpus == 2:
            double_split = [4, 4]
            single_split = [24, 24]
            use_deterministic = True
        elif n_gpus == 4:
            double_split = [2, 2, 2, 2]
            single_split = [12, 12, 12, 12]
            use_deterministic = True
        else:
            use_deterministic = False  # Fallback to greedy

    if use_deterministic:
        _apply_deterministic_split(transformer, gpu_ids, double_split, single_split)
    else:
        _apply_greedy_split(transformer, gpu_ids, other_module_params, other_module_param_count_scale)
```

### Phase 2: Fix Synchronization (High Priority)

```python
def _move_tensor_to_device(tensor: Tensor, device: torch.device, sync: bool = True) -> Tensor:
    """Move tensor to device with optional synchronization."""
    if tensor.device != device:
        tensor = tensor.to(device, non_blocking=True)
        if sync:
            # Ensure transfer completes before computation
            torch.cuda.current_stream(device).synchronize()
        return tensor
    return tensor
```

Or with CUDA events for fine-grained control:
```python
class DeviceTransferManager:
    def __init__(self, devices: list[torch.device]):
        self.streams = {d: torch.cuda.Stream(device=d) for d in devices}
        self.events = {d: torch.cuda.Event() for d in devices}

    def transfer(self, tensor: Tensor, target_device: torch.device) -> Tensor:
        if tensor.device == target_device:
            return tensor

        src_stream = self.streams.get(tensor.device, torch.cuda.current_stream(tensor.device))
        dst_stream = self.streams[target_device]

        # Record event on source stream
        event = self.events[tensor.device]
        event.record(src_stream)

        # Wait for source computation to complete on destination stream
        dst_stream.wait_event(event)

        with torch.cuda.stream(dst_stream):
            return tensor.to(target_device, non_blocking=True)
```

### Phase 3: Config Integration (Medium Priority)

Add to `ModelConfig` in `config_modules.py`:
```python
# GPU splitting options
self.split_model_over_gpus: bool = kwargs.get('split_model_over_gpus', False)
self.gpu_split_double: list[int] = kwargs.get('gpu_split_double', None)  # e.g., [4, 4]
self.gpu_split_single: list[int] = kwargs.get('gpu_split_single', None)  # e.g., [24, 24]
```

Add validation:
```python
if self.split_model_over_gpus and self.layer_offloading:
    print("Warning: GPU splitting and layer_offloading may conflict. Consider disabling one.")
```

### Phase 4: Robust `.to()` Handling (Medium Priority)

```python
def new_device_to_flux2(self: "Flux2", *args, **kwargs):
    # Parse arguments more robustly
    device = None
    dtype = None

    # Handle kwargs first
    device = kwargs.pop('device', None)
    dtype = kwargs.pop('dtype', None)

    # Handle positional args
    remaining_args = []
    for arg in args:
        if isinstance(arg, torch.device) or (isinstance(arg, str) and ('cuda' in arg or 'cpu' in arg)):
            if device is None:
                device = torch.device(arg) if isinstance(arg, str) else arg
            else:
                remaining_args.append(arg)
        elif isinstance(arg, torch.dtype):
            if dtype is None:
                dtype = arg
            else:
                remaining_args.append(arg)
        else:
            remaining_args.append(arg)

    # ... rest of implementation
```

---

## Config Examples

### 2x B300 with Even Split
```yaml
model:
  name_or_path: "black-forest-labs/FLUX.2-dev"
  arch: flux2
  split_model_over_gpus: true
  gpu_split_double: [4, 4]
  gpu_split_single: [24, 24]
  layer_offloading: false  # Disable to avoid conflict
  quantize: false

train:
  cache_text_embeddings: true  # Keep TE off GPU
  batch_size: 2  # Can increase with split
```

### 4x B300 with Even Split
```yaml
model:
  name_or_path: "black-forest-labs/FLUX.2-dev"
  arch: flux2
  split_model_over_gpus: true
  gpu_split_double: [2, 2, 2, 2]
  gpu_split_single: [12, 12, 12, 12]
  layer_offloading: false
  quantize: false

train:
  cache_text_embeddings: true
  batch_size: 4  # 4x batch with 4 GPUs
```

### 4x B300 Activation-Aware (If OOM on Early GPUs)
```yaml
model:
  split_model_over_gpus: true
  gpu_split_double: [2, 2, 2, 2]
  gpu_split_single: [8, 12, 14, 14]  # Bias toward later GPUs
```

### With Stream-Based Transfers (Experimental)
```yaml
model:
  name_or_path: "black-forest-labs/FLUX.2-dev"
  arch: flux2
  split_model_over_gpus: true
  gpu_split_double: [4, 4]
  gpu_split_single: [24, 24]
  use_stream_transfers: true  # Enable overlapped compute/transfer
  layer_offloading: false
```

**Note**: `use_stream_transfers` enables CUDA stream-based transfers that can overlap
compute on one GPU with data transfer to the next GPU. This is experimental and may
provide throughput improvements on systems with high GPU interconnect bandwidth.

---

## Testing Checklist

- [ ] 2-GPU smoke test: 10 steps, verify no OOM, loss decreases
- [ ] 4-GPU smoke test: 10 steps, verify no OOM, loss decreases
- [ ] Memory balance: Check `nvidia-smi` for even distribution
- [ ] Gradient flow: Verify gradients non-zero on all GPUs
- [ ] `.to(dtype)` preserves device assignments
- [ ] `.to('cpu')` and back to CUDA works
- [ ] `torch.compile` after split works
- [ ] Sampling after training works (output routing correct)

---

## References

- FLUX.1 splitter: `toolkit/models/flux.py:add_model_gpu_splitter_to_flux()`
- FLUX.2 model: `extensions_built_in/diffusion_models/flux2/src/model.py`
- Config modules: `toolkit/config_modules.py`

---

## Changelog

| Date | Change |
|------|--------|
| 2024-12-05 | Initial analysis and improvement plan |
| 2024-12-05 | Implemented deterministic splits, config options, synchronization fix |
| 2024-12-05 | Added CUDA stream manager for overlapped compute/transfer (experimental) |
