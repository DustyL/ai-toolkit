# FSDP Implementation Plan for ai-toolkit

## Overview
Enable FullyShardedDataParallel (FSDP) training to allow FLUX.2 training without gradient checkpointing by sharding model parameters across GPUs.

## Supported Configurations

### Network Types (ALL SUPPORTED)
| Type | Class | FSDP Compatible |
|------|-------|-----------------|
| lora | LoConModule | ✅ |
| loha | LohaModule | ✅ |
| lokr | LokrModule | ✅ |
| locon | LoConModule | ✅ |
| dora | DoRAModule | ✅ |
| boft | ButterflyOFTModule | ✅ |
| diag-oft | DiagOFTModule | ✅ |

### Model Types
- FLUX.2 (primary target)
- FLUX.1-dev/schnell
- Future: SD3, SDXL (with appropriate block classes)

## Architecture

### FSDP Wrapping Strategy

```
┌─────────────────────────────────────────────────────┐
│                    TRAINING FLOW                     │
├─────────────────────────────────────────────────────┤
│  1. Load transformer on CPU (not GPU!)              │
│  2. Create LoRA/LoHA/LoKr network                   │
│  3. network.apply_to(transformer) - hook forwards   │
│  4. Freeze base params if adapter-only training     │
│  5. FSDP wrap transformer with auto-wrap policy     │
│  6. FSDP wrap network (top-level only, no recurse)  │
│  7. Create optimizer on FSDP-wrapped params         │
│  8. Training loop with FSDP state dict handling     │
└─────────────────────────────────────────────────────┘
```

### Auto-Wrap Policy for FLUX2
```python
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from extensions_built_in.diffusion_models.flux2.src.model import DoubleStreamBlock, SingleStreamBlock

wrap_policy = ModuleWrapPolicy({
    DoubleStreamBlock,   # 8 blocks
    SingleStreamBlock,   # 48 blocks
})
```

### Mixed Precision Configuration
```python
from torch.distributed.fsdp import MixedPrecision

mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)
```

## Implementation Files

### 1. New: `toolkit/fsdp_utils.py`
```python
# Functions:
- get_flux2_wrap_policy() -> ModuleWrapPolicy
- get_fsdp_mixed_precision(dtype: str) -> MixedPrecision
- save_fsdp_checkpoint(model, optimizer, path, rank)
- load_fsdp_checkpoint(model, optimizer, path)
- validate_fsdp_config(train_config, model_config)
```

### 2. Modify: `jobs/process/BaseSDTrainProcess.py`
- Add `_prepare_with_fsdp()` method
- Update `prepare_accelerator()` routing
- Modify checkpoint save/load for FSDP

### 3. Modify: `toolkit/distributed.py`
- Add FSDP-specific initialization if needed

## Configuration

### YAML Config Example
```yaml
train:
  distributed: true
  dist_backend: "nccl"
  dist_mode: "fsdp"
  
  # FSDP-specific options
  fsdp_auto_wrap: true
  fsdp_sharding: "full"        # full (ZeRO-3) or hybrid
  fsdp_mixed_precision: "bf16"
  fsdp_cpu_offload: false      # Optional: offload params to CPU
  fsdp_backward_prefetch: true
  
  # Standard training (works with any network type)
  batch_size: 2
  gradient_checkpointing: false  # Can disable with FSDP!
  
network:
  type: "loha"  # or lora, lokr, locon, dora, boft, diag-oft
  linear: 64
```

## Mutual Exclusions (Validation Required)

| Feature | Compatible with FSDP |
|---------|---------------------|
| GPU Splitter | ❌ NO - different parallelism model |
| Layer Offloading | ❌ NO - conflicts with FSDP memory management |
| low_vram mode | ❌ NO - overrides .to() behavior |
| gradient_checkpointing | ✅ YES - but not needed with FSDP |
| DDP | ❌ NO - use one or the other |

## Checkpoint Strategy

### During Training (Resume Support)
- Use `StateDictType.SHARDED_STATE_DICT`
- Each rank saves its shard
- Fast save/load, no gather overhead

### Final Export (User-Facing)
- LoRA/LoHA weights: Standard safetensors (unchanged)
- Full model (optional): `FULL_STATE_DICT` on rank 0

## Launch Command
```bash
# 4 GPUs on single node
torchrun --nproc_per_node=4 run.py config/train_flux2_loha_fsdp.yaml

# Multi-node (2 nodes × 4 GPUs each)
torchrun --nnodes=2 --nproc_per_node=4 --rdzv_backend=c10d \
    --rdzv_endpoint=master:29500 run.py config/train_flux2_loha_fsdp.yaml
```

## Memory Comparison (FLUX.2, batch=1, no grad ckpt)

| Mode | Per-GPU Memory | Effective Batch |
|------|----------------|-----------------|
| Single GPU | OOM | N/A |
| DDP 4×GPU | OOM per GPU | N/A |
| FSDP 4×GPU | ~20GB | 4 |
| FSDP 4×GPU + CPU offload | ~12GB | 4 |

## Implementation Phases

### Phase 1: Core FSDP Support
- [ ] Create `toolkit/fsdp_utils.py`
- [ ] Add `_prepare_with_fsdp()` to BaseSDTrainProcess
- [ ] Add config validation (mutual exclusions)

### Phase 2: Checkpoint Handling
- [ ] Implement FSDP state dict save/load
- [ ] Test resume from FSDP checkpoints
- [ ] Verify LoRA/LoHA export still works

### Phase 3: Testing & Optimization
- [ ] Single-GPU FSDP smoke test
- [ ] Multi-GPU training test
- [ ] Benchmark vs DDP memory usage
- [ ] Tune prefetch settings

## Notes

- FSDP requires PyTorch 2.0+ (2.2+ recommended)
- NCCL backend required for GPU training
- All network types (LoRA, LoHA, LoKr, etc.) work identically with FSDP
- Optimizer must be created AFTER FSDP wrapping
