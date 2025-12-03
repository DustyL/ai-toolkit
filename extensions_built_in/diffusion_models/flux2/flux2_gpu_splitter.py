"""
GPU Model Splitter for FLUX.2

Distributes FLUX.2 transformer blocks across multiple GPUs for memory-efficient
training of large models without gradient checkpointing.

This implementation mirrors the approach used for FLUX.1 in toolkit/models/flux.py
but adapted for the FLUX.2 architecture which has different block structures.

FLUX.2 Architecture:
- 8 double blocks (DoubleStreamBlock): Process image and text streams separately
- 48 single blocks (SingleStreamBlock): Process concatenated img+txt stream
- Transformer: ~12B parameters
- Full model with Mistral TE: ~32B parameters total
"""

from functools import partial
from typing import Optional, TYPE_CHECKING
import torch
from torch import Tensor

if TYPE_CHECKING:
    from .src.model import Flux2


def _move_tensor_to_device(tensor: Tensor, device: torch.device) -> Tensor:
    """Move tensor to device if not already there. Uses non_blocking for efficiency."""
    if tensor.device != device:
        return tensor.to(device, non_blocking=True)
    return tensor


def _move_tuple_to_device(t: tuple, device: torch.device) -> tuple:
    """Move all tensors in a tuple to device. Handles None items gracefully."""
    return tuple(
        _move_tensor_to_device(item, device) if item is not None else None
        for item in t
    )


def split_gpu_double_block_forward(
    self,
    img: Tensor,
    txt: Tensor,
    pe: Tensor,
    pe_ctx: Tensor,
    mod_img: tuple,
    mod_txt: tuple,
) -> tuple[Tensor, Tensor]:
    """
    Wrapped forward for DoubleStreamBlock that handles device transfers.

    Moves all inputs to the block's assigned GPU before calling the original forward.
    """
    device = self._split_device

    # Move main tensors
    img = _move_tensor_to_device(img, device)
    txt = _move_tensor_to_device(txt, device)
    pe = _move_tensor_to_device(pe, device)
    pe_ctx = _move_tensor_to_device(pe_ctx, device)

    # Move modulation tuples - each is ((shift, scale, gate), (shift, scale, gate))
    # mod_img and mod_txt are tuples of 2 tuples, each containing 3 tensors
    mod_img = tuple(_move_tuple_to_device(m, device) for m in mod_img)
    mod_txt = tuple(_move_tuple_to_device(m, device) for m in mod_txt)

    return self._pre_gpu_split_forward(img, txt, pe, pe_ctx, mod_img, mod_txt)


def split_gpu_single_block_forward(
    self,
    x: Tensor,
    pe: Tensor,
    mod: tuple,
) -> Tensor:
    """
    Wrapped forward for SingleStreamBlock that handles device transfers.

    Moves all inputs to the block's assigned GPU before calling the original forward.
    If this is the last block, routes output back to GPU 0.
    """
    device = self._split_device

    # Move main tensors
    x = _move_tensor_to_device(x, device)
    pe = _move_tensor_to_device(pe, device)

    # Move modulation tuple - (shift, scale, gate)
    mod = _move_tuple_to_device(mod, device)

    x_out = self._pre_gpu_split_forward(x, pe, mod)

    # Route output to specified device (used for last block to return to GPU 0)
    if hasattr(self, "_split_output_device"):
        return x_out.to(self._split_output_device)

    return x_out


def new_device_to_flux2(self: "Flux2", *args, **kwargs):
    """
    Custom .to() method that respects GPU splitting.

    When a model is split across GPUs, calling .to(device) should:
    - Move global layers (embeddings, modulation, final_layer) to target device
    - Move transformer blocks to their assigned split devices

    This prevents .to() from accidentally moving all parameters to one device.

    Supports:
    - .to(device) - move to device, blocks go to split devices
    - .to(dtype) - change dtype only, preserve device placement
    - .to(device, dtype) - both
    - .to(device=..., dtype=...) - kwargs form
    - .to("cpu") - moves ALL layers to CPU, clears split device assignments

    Note: Moving to CPU effectively disables GPU splitting until the model
    is moved back to a CUDA device.
    """
    # Extract device from args or kwargs
    device_in_kwargs = 'device' in kwargs
    device_in_args = any(isinstance(arg, (str, torch.device)) for arg in args)
    has_device = device_in_kwargs or device_in_args

    device = None
    # Remove device from kwargs if present
    if device_in_kwargs:
        device = kwargs.pop('device')

    # Extract device from args if present
    if device_in_args:
        args = list(args)
        for idx, arg in enumerate(args):
            if isinstance(arg, (str, torch.device)):
                device = arg
                del args[idx]
                break

    # Normalize device to torch.device for comparison
    if device is not None:
        device = torch.device(device)

    # Check if moving to CPU - this requires special handling
    # When moving to CPU, we move ALL layers uniformly (no split)
    moving_to_cpu = has_device and device is not None and device.type == "cpu"

    # Helper to call .to() with or without device argument
    def to_with_optional_device(module, target_device):
        """Call module.to() with device only if device was specified in original call."""
        if has_device and target_device is not None:
            return module.to(target_device, *args, **kwargs)
        else:
            # dtype-only call: just pass remaining args/kwargs
            return module.to(*args, **kwargs)

    # Move global input layers to target device (GPU 0) or just change dtype
    self.pe_embedder = to_with_optional_device(self.pe_embedder, device)
    self.img_in = to_with_optional_device(self.img_in, device)
    self.time_in = to_with_optional_device(self.time_in, device)
    self.guidance_in = to_with_optional_device(self.guidance_in, device)
    self.txt_in = to_with_optional_device(self.txt_in, device)

    # Move modulation layers to target device (they're lightweight and used globally)
    self.double_stream_modulation_img = to_with_optional_device(self.double_stream_modulation_img, device)
    self.double_stream_modulation_txt = to_with_optional_device(self.double_stream_modulation_txt, device)
    self.single_stream_modulation = to_with_optional_device(self.single_stream_modulation, device)

    # Move transformer blocks
    # If moving to CPU: move all blocks to CPU uniformly and update _split_device
    # If moving to CUDA: move blocks to their assigned split devices
    # If dtype-only: just change dtype, preserve device placement
    for i, block in enumerate(self.double_blocks):
        if moving_to_cpu:
            # CPU mode: all blocks go to CPU
            # Store original split device for potential restoration, then update to CPU
            if not hasattr(block, "_original_split_device"):
                block._original_split_device = block._split_device
            block._split_device = device  # Update so forward wrapper uses CPU
            block_device = device
        elif has_device:
            # CUDA mode: blocks go to their split devices
            # Restore original split device if we're coming back from CPU
            if hasattr(block, "_original_split_device"):
                block._split_device = block._original_split_device
            block_device = block._split_device
        else:
            # dtype-only: no device change
            block_device = None
        self.double_blocks[i] = to_with_optional_device(block, block_device)

    for i, block in enumerate(self.single_blocks):
        if moving_to_cpu:
            if not hasattr(block, "_original_split_device"):
                block._original_split_device = block._split_device
            block._split_device = device
            block_device = device
        elif has_device:
            if hasattr(block, "_original_split_device"):
                block._split_device = block._original_split_device
            block_device = block._split_device
        else:
            block_device = None
        self.single_blocks[i] = to_with_optional_device(block, block_device)

    # Move output layer to target device
    self.final_layer = to_with_optional_device(self.final_layer, device)

    # Update _split_output_device to match where final_layer now lives
    # This ensures the last single block routes output to the correct device
    if has_device and device is not None:
        last_block = self.single_blocks[-1]
        if moving_to_cpu:
            # Store original output device for restoration
            if not hasattr(last_block, "_original_split_output_device"):
                last_block._original_split_output_device = last_block._split_output_device
            last_block._split_output_device = device
        else:
            # Restore original output device when moving back to CUDA
            if hasattr(last_block, "_original_split_output_device"):
                last_block._split_output_device = last_block._original_split_output_device
            else:
                last_block._split_output_device = device

    return self


def add_model_gpu_splitter_to_flux2(
    transformer: "Flux2",
    # Estimated params for text encoder (~24B), VAE (~0.5B), and misc
    other_module_params: Optional[int] = 25e9,
    # Scale down non-trainable params in distribution calculation
    other_module_param_count_scale: Optional[float] = 0.3,
    # Split strategy: "contiguous" (default) or "greedy"
    split_strategy: Optional[str] = "contiguous"
):
    """
    Apply GPU model splitting to a Flux2 transformer.

    Distributes transformer blocks across available GPUs. Double blocks (8 total) are
    always pinned to GPU 0 to avoid mid-forward transfers. Only single blocks (48 total)
    are distributed across GPUs.

    Split Strategies:
        - "contiguous" (default): Finds optimal split point within single blocks to
          create consecutive ranges per GPU. Guarantees exactly 2 cross-GPU transfers:
          one at the split point, one returning to GPU 0 for final_layer.

        - "greedy": Assigns each single block to the GPU with lowest current param total.
          Can cause interleaved assignments with many cross-GPU transfers (~20-30 per
          forward pass). Only use for edge cases where one GPU is much slower or has
          different memory constraints.

    Args:
        transformer: The Flux2 model to split
        other_module_params: Estimated parameter count for non-transformer modules
                           (text encoder, VAE, etc.) that share GPU 0
        other_module_param_count_scale: Multiplier for other_module_params. Applied
                                       only to GPU 0's starting load when computing
                                       the split point. Use 0.0 to ignore TE/VAE weight.
        split_strategy: "contiguous" (default) or "greedy"

    Memory Distribution Example (2x B200 with FLUX.2, contiguous, scale=0.0):
        GPU 0: Mistral TE + VAE + embedders + 8 double + ~20 single blocks
        GPU 1: ~28 single blocks
        Transfers: 1 (at single split point) + 1 (back to GPU 0 for final_layer) = 2 total

    With gradient_checkpointing=false, activations are distributed across GPUs too.
    """
    gpu_id_list = [i for i in range(torch.cuda.device_count())]

    if len(gpu_id_list) < 2:
        print("Warning: Only 1 GPU detected, GPU splitting will have no effect")
        return

    # Validate split_strategy
    if split_strategy not in ("contiguous", "greedy"):
        print(f"[GPU Splitter] Warning: Unknown split_strategy '{split_strategy}', using 'contiguous' instead")
        split_strategy = "contiguous"

    print(f"[GPU Splitter] Splitting FLUX.2 model across {len(gpu_id_list)} GPUs (strategy: {split_strategy})")

    # Contiguous strategy only supports 2 GPUs; fall back to greedy for >2
    if len(gpu_id_list) > 2 and split_strategy == "contiguous":
        print("[GPU Splitter] WARNING: Contiguous strategy only supports 2 GPUs. "
              f"Falling back to 'greedy' for {len(gpu_id_list)} GPUs.")
        print("[GPU Splitter] HINT: For 3+ GPUs, consider FSDP/DeepSpeed data parallelism for better scaling. "
              "Greedy model splitting is a stopgap with high transfer overhead.")
        split_strategy = "greedy"

    # Scale other module params (applied to GPU 0's starting load)
    gpu0_starting_load = other_module_params * other_module_param_count_scale

    # Track per-GPU param totals (start with TE/VAE estimate on GPU 0)
    gpu_param_totals = {i: 0 for i in gpu_id_list}
    gpu_param_totals[0] = gpu0_starting_load

    # Track block distribution for logging
    block_distribution = {i: {"double": 0, "single": 0} for i in gpu_id_list}

    # Clear any stale state from previous splits (defensive, in case re-splitting)
    # Must restore original forward before clearing to avoid infinite recursion
    # Note: This discards any post-split wrappers (e.g., torch.compile) applied after
    # the initial split. Re-splitting should be done before applying such wrappers.
    def _reset_block(block):
        orig_forward = getattr(block, "_pre_gpu_split_forward", None)
        if orig_forward is not None:
            block.forward = orig_forward  # restore original first
        for attr in ("_original_split_device", "_original_split_output_device",
                     "_pre_gpu_split_forward", "_split_device", "_split_output_device"):
            if hasattr(block, attr):
                delattr(block, attr)

    for block in transformer.double_blocks:
        _reset_block(block)
    for block in transformer.single_blocks:
        _reset_block(block)

    # Pre-calculate block params
    double_params_list = [sum(p.numel() for p in b.parameters()) for b in transformer.double_blocks]
    single_params_list = [sum(p.numel() for p in b.parameters()) for b in transformer.single_blocks]

    total_double_params = sum(double_params_list)
    total_single_params = sum(single_params_list)
    total_block_params = total_double_params + total_single_params

    print(f"[GPU Splitter] Total block params: {total_block_params/1e9:.2f}B "
          f"({total_double_params/1e9:.2f}B double + {total_single_params/1e9:.2f}B single)")
    print(f"[GPU Splitter] GPU0 extra load (TE/VAE estimate): {gpu0_starting_load/1e9:.2f}B "
          f"(only GPU0 carries this, affects split point calculation)")

    # ============ ALWAYS PIN DOUBLE BLOCKS TO GPU 0 ============
    # Double blocks (8 total) process both img and txt streams and come first in the forward pass.
    # Pinning them to GPU0 guarantees exactly one transfer point (at the double→single boundary
    # or within singles), avoiding mid-double transfers that would add extra communication.
    gpu0_device = torch.device("cuda:0")

    for i, block in enumerate(transformer.double_blocks):
        block._pre_gpu_split_forward = block.forward
        block.forward = partial(split_gpu_double_block_forward, block)
        block._split_device = gpu0_device

        block_distribution[0]["double"] += 1
        gpu_param_totals[0] += double_params_list[i]

    if split_strategy == "contiguous":
        # ============ CONTIGUOUS STRATEGY ============
        # Find optimal split point within single blocks only.
        # Guarantees exactly 1 transfer (at split) + 1 return (to final_layer on GPU0) = 2 total.

        # Handle edge case: no single blocks (unlikely but defensive)
        if not single_params_list:
            print("[GPU Splitter] WARNING: No single blocks found, all work on GPU0")
            split_idx = 0
            best_diff = 0.0
        else:
            # GPU 0 already has: TE/VAE + all doubles
            # Find the split point that minimizes |params_gpu0 - params_gpu1|
            gpu0_base = gpu0_starting_load + total_double_params

            # Compute cumulative sums for singles
            single_cumsum = []
            running = 0
            for params in single_params_list:
                running += params
                single_cumsum.append(running)

            # Find split index that minimizes imbalance: O(N) scan over N+1 possible split points
            # split_idx=k means singles[0:k] on GPU0, singles[k:] on GPU1
            best_split_idx = 0
            best_diff = float('inf')

            for k in range(len(single_params_list) + 1):
                if k == 0:
                    gpu0_total = gpu0_base
                    gpu1_total = total_single_params
                else:
                    gpu0_total = gpu0_base + single_cumsum[k - 1]
                    gpu1_total = total_single_params - single_cumsum[k - 1]

                diff = abs(gpu0_total - gpu1_total)
                if diff < best_diff:
                    best_diff = diff
                    best_split_idx = k

            split_idx = best_split_idx

        # Assign single blocks contiguously
        for i, block in enumerate(transformer.single_blocks):
            if i < split_idx:
                current_gpu_idx = 0
            else:
                # For >2 GPUs, could extend to find multiple split points
                # For now, all remaining go to GPU 1
                current_gpu_idx = min(len(gpu_id_list) - 1, 1)

            device = torch.device(f"cuda:{current_gpu_idx}")

            block._pre_gpu_split_forward = block.forward
            block.forward = partial(split_gpu_single_block_forward, block)
            block._split_device = device

            block_distribution[current_gpu_idx]["single"] += 1
            gpu_param_totals[current_gpu_idx] += single_params_list[i]

        # Log split details with percentage
        num_singles = len(transformer.single_blocks)
        total_distributed = total_block_params + gpu0_starting_load
        imbalance_pct = (best_diff / total_distributed * 100) if total_distributed > 0 else 0.0

        if split_idx == 0:
            print(f"[GPU Splitter] Contiguous split: all {num_singles} singles on GPU1 "
                  f"(imbalance: {best_diff/1e9:.2f}B, {imbalance_pct:.1f}%)")
        elif split_idx == num_singles:
            print(f"[GPU Splitter] Contiguous split: all {num_singles} singles on GPU0 "
                  f"(imbalance: {best_diff/1e9:.2f}B, {imbalance_pct:.1f}%)")
        else:
            print(f"[GPU Splitter] Contiguous split: singles 0-{split_idx-1} on GPU0, "
                  f"{split_idx}-{num_singles-1} on GPU1 (imbalance: {best_diff/1e9:.2f}B, {imbalance_pct:.1f}%)")

    else:
        # ============ GREEDY STRATEGY ============
        # Assigns each single block to GPU with lowest current param total.
        # WARNING: Can cause interleaved assignments with many cross-GPU transfers.
        # Only use for edge cases (asymmetric GPUs, memory constraints).

        print("[GPU Splitter] WARNING: Greedy strategy may cause 20-30+ cross-GPU transfers per forward pass")

        for i, block in enumerate(transformer.single_blocks):
            # Pick GPU with lowest current param total
            current_gpu_idx = min(gpu_param_totals, key=gpu_param_totals.get)
            device = torch.device(f"cuda:{current_gpu_idx}")

            block._pre_gpu_split_forward = block.forward
            block.forward = partial(split_gpu_single_block_forward, block)
            block._split_device = device

            block_distribution[current_gpu_idx]["single"] += 1
            gpu_param_totals[current_gpu_idx] += single_params_list[i]

    # Force last single block to route output back to GPU 0
    # This is where final_layer expects the output
    transformer.single_blocks[-1]._split_output_device = torch.device("cuda:0")

    # Wrap the .to() method to respect GPU splitting
    transformer._pre_gpu_split_to = transformer.to
    transformer.to = partial(new_device_to_flux2, transformer)

    # Log distribution
    print("[GPU Splitter] Block distribution:")
    for gpu_id, counts in block_distribution.items():
        params_b = gpu_param_totals[gpu_id] / 1e9
        print(f"  GPU {gpu_id}: {counts['double']} double + {counts['single']} single blocks ({params_b:.2f}B params)")

    print("[GPU Splitter] FLUX.2 model splitting complete")
