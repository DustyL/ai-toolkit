"""
Integration tests for LoHA layer offloading with Flux2-style training.

These tests simulate a real training scenario with:
1. A network structure similar to Flux2 with LoHA adapters
2. Forward/backward passes with loss computation
3. Optimizer steps with offloaded gradients
4. Multiple training iterations

Note: These tests require CUDA and may take longer to run.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class MockTransformerBlock(nn.Module):
    """Simplified transformer block for testing."""

    def __init__(self, dim: int = 256, heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.Linear(dim, dim * 3)  # QKV
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.dim = dim
        self.heads = heads

    def forward(self, x):
        # Simple attention-like computation
        h = self.norm1(x)
        qkv = self.attn(h)
        q, k, v = qkv.chunk(3, dim=-1)
        attn = F.scaled_dot_product_attention(
            q.reshape(-1, self.heads, q.size(-1) // self.heads).unsqueeze(0),
            k.reshape(-1, self.heads, k.size(-1) // self.heads).unsqueeze(0),
            v.reshape(-1, self.heads, v.size(-1) // self.heads).unsqueeze(0),
        ).squeeze(0).reshape(x.shape)
        x = x + self.proj(attn)
        x = x + self.mlp(self.norm2(x))
        return x


class MockLoHAAdapter(nn.Module):
    """
    LoHA adapter that wraps a Linear layer.

    Mimics the LyCORIS LohaModule structure for testing.
    """

    def __init__(self, original: nn.Linear, lora_dim: int = 8):
        super().__init__()

        in_dim = original.in_features
        out_dim = original.out_features

        # Store reference to original
        self.org_module = original
        self.org_weight = original.weight

        # LoHA parameters
        self.hada_w1_a = nn.Parameter(torch.randn(out_dim, lora_dim) * 0.1)
        self.hada_w1_b = nn.Parameter(torch.randn(lora_dim, in_dim))
        self.hada_w2_a = nn.Parameter(torch.randn(out_dim, lora_dim) * 0.1)
        self.hada_w2_b = nn.Parameter(torch.randn(lora_dim, in_dim))

        # Scale
        self.scale = 1.0
        self.scalar = nn.Parameter(torch.tensor(1.0))
        self.multiplier = 1.0
        self.lora_dim = lora_dim
        self.shape = (out_dim, in_dim)
        self.rank_dropout = 0.0
        self.rank_dropout_scale = False
        self.module_dropout = 0.0
        self.bypass_mode = None
        self.wd = False
        self.tucker = False
        self.hada_t1 = None
        self.hada_t2 = None

    def _current_weight(self):
        return self.org_weight

    def org_forward(self, x):
        # Call the original module's forward, which handles its own offloading if managed
        return self.org_module(x)

    def op(self, x, weight, bias=None, **kwargs):
        return F.linear(x, weight, bias)

    @property
    def kw_dict(self):
        return {}

    def get_weight(self, shape=None):
        scale = torch.tensor(self.scale, dtype=self.hada_w1_b.dtype, device=self.hada_w1_b.device)
        weight = (self.hada_w1_a @ self.hada_w1_b) * (self.hada_w2_a @ self.hada_w2_b) * scale
        if shape is not None:
            weight = weight.reshape(shape)
        return weight

    def forward(self, x):
        base = self.org_forward(x)
        diff_weight = self.get_weight(self.shape) * self.scalar
        delta = self.op(x, diff_weight * self.multiplier)
        return base + delta


class MockFlux2WithLoHA(nn.Module):
    """
    Simplified Flux2-like model with LoHA adapters for testing.

    Structure:
    - Input projection
    - Multiple transformer blocks (some with LoHA)
    - Output projection
    """

    def __init__(
        self,
        dim: int = 256,
        num_blocks: int = 4,
        num_loha_blocks: int = 2,
        lora_dim: int = 8,
    ):
        super().__init__()

        self.input_proj = nn.Linear(64, dim)

        # Create transformer blocks
        self.blocks = nn.ModuleList()
        self.loha_adapters = nn.ModuleList()

        for i in range(num_blocks):
            block = MockTransformerBlock(dim)
            self.blocks.append(block)

            # Add LoHA to first num_loha_blocks
            if i < num_loha_blocks:
                # Wrap the attention projection
                loha = MockLoHAAdapter(block.proj, lora_dim)
                self.loha_adapters.append(loha)
                # Replace forward to use LoHA
                block._loha_proj = loha
            else:
                self.loha_adapters.append(None)

        self.output_proj = nn.Linear(dim, 64)

    def forward(self, x):
        x = self.input_proj(x)

        for i, block in enumerate(self.blocks):
            if self.loha_adapters[i] is not None:
                # Use LoHA for projection
                h = block.norm1(x)
                qkv = block.attn(h)
                q, k, v = qkv.chunk(3, dim=-1)
                attn = F.scaled_dot_product_attention(
                    q.reshape(-1, block.heads, q.size(-1) // block.heads).unsqueeze(0),
                    k.reshape(-1, block.heads, k.size(-1) // block.heads).unsqueeze(0),
                    v.reshape(-1, block.heads, v.size(-1) // block.heads).unsqueeze(0),
                ).squeeze(0).reshape(x.shape)
                # Use LoHA adapter instead of regular proj
                x = x + self.loha_adapters[i](attn)
                x = x + block.mlp(block.norm2(x))
            else:
                x = block(x)

        x = self.output_proj(x)
        return x


class TestFlux2LoHAIntegration:
    """Integration tests simulating Flux2 training with LoHA."""

    def test_training_loop_with_offloading(self):
        """Run a complete training loop with LoHA offloading."""
        from toolkit.memory_management import MemoryManager

        # Create model
        model = MockFlux2WithLoHA(dim=128, num_blocks=4, num_loha_blocks=2, lora_dim=4)
        device = torch.device("cuda:0")

        # Freeze base model, train LoHA
        for param in model.parameters():
            param.requires_grad = False
        for loha in model.loha_adapters:
            if loha is not None:
                for param in loha.parameters():
                    param.requires_grad = True

        # Move to GPU first, then attach (MemoryManager offloads LoHA params to CPU)
        model = model.to(device)
        MemoryManager.attach(model, device, offload_percent=1.0)

        # Create optimizer (CPU parameters work with standard optimizers)
        loha_params = []
        for loha in model.loha_adapters:
            if loha is not None:
                loha_params.extend(loha.parameters())
        optimizer = torch.optim.AdamW(loha_params, lr=1e-4)

        # Training loop
        num_steps = 5
        losses = []

        for step in range(num_steps):
            optimizer.zero_grad()

            # Simulate batch
            x = torch.randn(2, 16, 64, device=device)
            target = torch.randn(2, 16, 64, device=device)

            # Forward
            output = model(x)

            # Loss
            loss = F.mse_loss(output, target)
            losses.append(loss.item())

            # Backward
            loss.backward()

            # Synchronize before optimizer step
            MemoryManager.synchronize_all(model)

            # Optimizer step
            optimizer.step()

        # Verify training happened
        assert len(losses) == num_steps
        assert all(l > 0 for l in losses), "All losses should be positive"

        # Cleanup
        MemoryManager.detach(model)

    def test_gradient_accumulation(self):
        """Test gradient accumulation works with offloading."""
        from toolkit.memory_management import MemoryManager

        model = MockFlux2WithLoHA(dim=64, num_blocks=2, num_loha_blocks=1, lora_dim=4)
        device = torch.device("cuda:0")

        # Setup training
        for param in model.parameters():
            param.requires_grad = False
        for loha in model.loha_adapters:
            if loha is not None:
                for param in loha.parameters():
                    param.requires_grad = True

        # Move to GPU first, then attach (MemoryManager offloads LoHA params to CPU)
        model = model.to(device)
        MemoryManager.attach(model, device, offload_percent=1.0)

        loha_params = [p for loha in model.loha_adapters if loha for p in loha.parameters()]
        optimizer = torch.optim.SGD(loha_params, lr=0.01)

        # Gradient accumulation
        accumulation_steps = 3
        optimizer.zero_grad()

        for acc_step in range(accumulation_steps):
            x = torch.randn(1, 8, 64, device=device)
            target = torch.randn(1, 8, 64, device=device)

            output = model(x)
            loss = F.mse_loss(output, target) / accumulation_steps
            loss.backward()

        # Synchronize and step
        MemoryManager.synchronize_all(model)
        optimizer.step()

        # Verify gradients were accumulated
        for loha in model.loha_adapters:
            if loha is not None:
                assert loha.hada_w1_a.grad is not None

        MemoryManager.detach(model)

    def test_mixed_precision_training(self):
        """Test LoHA offloading works with autocast."""
        from toolkit.memory_management import MemoryManager

        model = MockFlux2WithLoHA(dim=64, num_blocks=2, num_loha_blocks=1, lora_dim=4)
        device = torch.device("cuda:0")

        for param in model.parameters():
            param.requires_grad = False
        for loha in model.loha_adapters:
            if loha is not None:
                for param in loha.parameters():
                    param.requires_grad = True

        # Move to GPU first, then attach (MemoryManager offloads LoHA params to CPU)
        model = model.to(device)
        MemoryManager.attach(model, device, offload_percent=1.0)

        loha_params = [p for loha in model.loha_adapters if loha for p in loha.parameters()]
        optimizer = torch.optim.AdamW(loha_params, lr=1e-4)
        scaler = torch.amp.GradScaler()

        # Training with autocast
        for _ in range(3):
            optimizer.zero_grad()

            x = torch.randn(2, 8, 64, device=device)
            target = torch.randn(2, 8, 64, device=device)

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(x)
                loss = F.mse_loss(output, target)

            scaler.scale(loss).backward()
            MemoryManager.synchronize_all(model)
            scaler.step(optimizer)
            scaler.update()

        MemoryManager.detach(model)

    def test_checkpoint_compatibility(self):
        """Test that model state can be saved and loaded with offloading."""
        from toolkit.memory_management import MemoryManager
        import tempfile
        import os

        model = MockFlux2WithLoHA(dim=64, num_blocks=2, num_loha_blocks=1, lora_dim=4)
        device = torch.device("cuda:0")

        # Setup
        for loha in model.loha_adapters:
            if loha is not None:
                for param in loha.parameters():
                    param.requires_grad = True

        # Move to GPU first, then attach (MemoryManager offloads LoHA params to CPU)
        model = model.to(device)
        MemoryManager.attach(model, device, offload_percent=1.0)

        # Do some training
        x = torch.randn(2, 8, 64, device=device)
        output = model(x)
        loss = output.sum()
        loss.backward()
        MemoryManager.synchronize_all(model)

        # Save LoHA state
        loha_state = {}
        for i, loha in enumerate(model.loha_adapters):
            if loha is not None:
                loha_state[f"loha_{i}"] = {
                    k: v.cpu() for k, v in loha.state_dict().items()
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "loha_state.pt")
            torch.save(loha_state, path)

            # Load back
            loaded_state = torch.load(path, weights_only=True)
            assert "loha_0" in loaded_state

        MemoryManager.detach(model)


class TestMemoryBehavior:
    """Tests for memory management behavior."""

    def test_parameters_remain_on_cpu(self):
        """LoHA parameters should stay on CPU during training."""
        from toolkit.memory_management import MemoryManager

        model = MockFlux2WithLoHA(dim=64, num_blocks=2, num_loha_blocks=1, lora_dim=4)
        device = torch.device("cuda:0")

        for loha in model.loha_adapters:
            if loha is not None:
                for param in loha.parameters():
                    param.requires_grad = True

        # Move to GPU first, then attach (MemoryManager offloads LoHA params to CPU)
        model = model.to(device)
        MemoryManager.attach(model, device, offload_percent=1.0)

        # Parameters should be on CPU
        for loha in model.loha_adapters:
            if loha is not None:
                assert loha.hada_w1_a.device.type == "cpu"
                assert loha.hada_w1_b.device.type == "cpu"

        # Do forward/backward
        x = torch.randn(2, 8, 64, device=device)
        output = model(x)
        loss = output.sum()
        loss.backward()
        MemoryManager.synchronize_all(model)

        # Parameters should still be on CPU
        for loha in model.loha_adapters:
            if loha is not None:
                assert loha.hada_w1_a.device.type == "cpu"
                assert loha.hada_w1_a.grad.device.type == "cpu"

        MemoryManager.detach(model)

    def test_stream_cleanup_after_training(self):
        """CUDA streams should be cleaned up after detach."""
        from toolkit.memory_management import MemoryManager

        MemoryManager.clear_all_state()

        model = MockFlux2WithLoHA(dim=64, num_blocks=2, num_loha_blocks=2, lora_dim=4)
        device = torch.device("cuda:0")

        MemoryManager.attach(model, device, offload_percent=1.0)

        initial_streams = MemoryManager.get_stream_count(device)
        assert initial_streams > 0

        MemoryManager.detach(model)

        final_streams = MemoryManager.get_stream_count(device)
        assert final_streams == 0, f"Streams not cleaned up: {final_streams}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
