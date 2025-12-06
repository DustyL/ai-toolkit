"""
Unit tests for LoHA layer offloading memory management.

Tests:
1. LoHA module detection
2. Per-module state isolation for LoHA
3. Forward/backward pass correctness
4. Gradient computation accuracy vs non-offloaded LoHA
5. Detach cleanup for LoHA
6. Tucker decomposition support
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class MockLoHAModule(nn.Module):
    """
    Mock LoHA module that mimics LyCORIS LohaModule structure.

    This is a simplified version for testing purposes.
    Computes: ΔW = (w1_a @ w1_b) ⊙ (w2_a @ w2_b) * scale * scalar
    """

    def __init__(
        self,
        in_dim: int = 64,
        out_dim: int = 128,
        lora_dim: int = 8,
        use_tucker: bool = False,
        use_scalar: bool = True,
    ):
        super().__init__()

        # Core LoHA parameters
        self.hada_w1_a = nn.Parameter(torch.randn(out_dim, lora_dim) * 0.1)
        self.hada_w1_b = nn.Parameter(torch.randn(lora_dim, in_dim))
        self.hada_w2_a = nn.Parameter(torch.randn(out_dim, lora_dim) * 0.1)
        self.hada_w2_b = nn.Parameter(torch.randn(lora_dim, in_dim))

        # Tucker tensors (optional)
        self.tucker = use_tucker
        if use_tucker:
            self.hada_t1 = nn.Parameter(torch.randn(lora_dim, lora_dim) * 0.1)
            self.hada_t2 = nn.Parameter(torch.randn(lora_dim, lora_dim) * 0.1)
        else:
            self.hada_t1 = None
            self.hada_t2 = None

        # Scale and scalar
        self.scale = lora_dim / lora_dim  # alpha / rank
        self.scalar = nn.Parameter(torch.tensor(1.0)) if use_scalar else torch.tensor(1.0)

        # Module metadata
        self.lora_dim = lora_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.shape = (out_dim, in_dim)
        self.multiplier = 1.0
        self.rank_dropout = 0.0
        self.rank_dropout_scale = False
        self.module_dropout = 0.0
        self.bypass_mode = None
        self.wd = False  # weight decompose

        # Original forward (simulated Linear layer)
        self.org_weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)

    def _current_weight(self):
        return self.org_weight

    def org_forward(self, x):
        """Simulated original forward (Linear layer)."""
        return F.linear(x, self.org_weight)

    def op(self, x, weight, bias=None, **kwargs):
        """Operation function (Linear)."""
        return F.linear(x, weight, bias)

    @property
    def kw_dict(self):
        return {}

    def get_weight(self, shape=None):
        """Compute LoHA diff weight."""
        scale = torch.tensor(self.scale, dtype=self.hada_w1_b.dtype, device=self.hada_w1_b.device)

        if self.tucker and self.hada_t1 is not None:
            # Tucker decomposition
            # Simplified for 2D case
            w1 = self.hada_w1_a @ self.hada_t1 @ self.hada_w1_b
            w2 = self.hada_w2_a @ self.hada_t2 @ self.hada_w2_b
            weight = w1 * w2 * scale
        else:
            # Standard LoHA: (w1_a @ w1_b) ⊙ (w2_a @ w2_b)
            weight = (self.hada_w1_a @ self.hada_w1_b) * (self.hada_w2_a @ self.hada_w2_b) * scale

        if shape is not None:
            weight = weight.reshape(shape)

        return weight

    def forward(self, x):
        """Standard LoHA forward."""
        base = self.org_forward(x)
        base_weight = self._current_weight()
        diff_weight = self.get_weight(self.shape) * self.scalar

        new_weight = base_weight + diff_weight * self.multiplier
        delta_weight = new_weight - base_weight
        delta = self.op(x, delta_weight)

        return base + delta


class MockLoHANetwork(nn.Module):
    """Network containing multiple LoHA modules for testing."""

    def __init__(self, num_loha_layers: int = 3):
        super().__init__()

        self.loha_layers = nn.ModuleList([
            MockLoHAModule(64 if i == 0 else 128, 128, lora_dim=8)
            for i in range(num_loha_layers)
        ])

    def forward(self, x):
        for layer in self.loha_layers:
            x = layer(x)
            x = F.relu(x)
        return x


class TestLoHAModuleDetection:
    """Test that LoHA modules are correctly detected."""

    def test_detect_loha_module(self):
        """Should detect LoHA module by parameter names."""
        from toolkit.memory_management.loha_layer_manager import _is_loha_module

        loha = MockLoHAModule()
        assert _is_loha_module(loha), "Should detect LoHA module"

    def test_not_detect_linear(self):
        """Should not detect Linear as LoHA."""
        from toolkit.memory_management.loha_layer_manager import _is_loha_module

        linear = nn.Linear(64, 128)
        assert not _is_loha_module(linear), "Should not detect Linear as LoHA"

    def test_memory_manager_finds_loha(self):
        """MemoryManager should find LoHA modules in network."""
        from toolkit.memory_management import MemoryManager

        network = MockLoHANetwork(num_loha_layers=3)
        device = torch.device("cuda:0")

        MemoryManager.attach(network, device, offload_percent=1.0)

        # Should have 3 managed LoHA layers
        count = MemoryManager.get_managed_layer_count(network)
        assert count == 3, f"Expected 3 managed LoHA layers, got {count}"

        # Each LoHA layer should have a layer manager
        for layer in network.loha_layers:
            assert hasattr(layer, "_layer_memory_manager"), \
                "LoHA layer should have memory manager attached"

        MemoryManager.detach(network)


class TestLoHAPerModuleState:
    """Test per-module state isolation for LoHA."""

    def test_separate_state_objects(self):
        """Each LoHA module should have its own state."""
        from toolkit.memory_management import MemoryManager
        from toolkit.memory_management.loha_layer_manager import LoHAModuleOffloadState

        network = MockLoHANetwork(num_loha_layers=3)
        device = torch.device("cuda:0")

        MemoryManager.attach(network, device, offload_percent=1.0)

        # Collect states
        states = []
        for layer in network.loha_layers:
            if hasattr(layer, "_layer_memory_manager"):
                lmm = layer._layer_memory_manager
                if hasattr(lmm, "_state") and lmm._state is not None:
                    states.append(lmm._state)

        # Should have 3 separate states
        assert len(states) == 3, f"Expected 3 states, got {len(states)}"

        # All states should be different objects
        state_ids = [id(s) for s in states]
        assert len(set(state_ids)) == 3, "States should be different objects"

        # Each state should be LoHAModuleOffloadState
        for state in states:
            assert isinstance(state, LoHAModuleOffloadState), \
                "State should be LoHAModuleOffloadState"

        MemoryManager.detach(network)

    def test_loha_state_has_extra_buffers(self):
        """LoHA state should have buffers for 4 core parameters."""
        from toolkit.memory_management import MemoryManager
        from toolkit.memory_management.loha_layer_manager import LoHAModuleOffloadState

        loha = MockLoHAModule()
        network = nn.ModuleList([loha])
        device = torch.device("cuda:0")

        # Wrap in a container
        class Container(nn.Module):
            def __init__(self, modules):
                super().__init__()
                self.layers = modules

        container = Container(network)
        MemoryManager.attach(container, device, offload_percent=1.0)

        # Get state
        state = loha._layer_memory_manager._state

        # Check for LoHA-specific buffer attributes
        assert hasattr(state, 'w1_a_buffer'), "State should have w1_a_buffer"
        assert hasattr(state, 'w1_b_buffer'), "State should have w1_b_buffer"
        assert hasattr(state, 'w2_a_buffer'), "State should have w2_a_buffer"
        assert hasattr(state, 'w2_b_buffer'), "State should have w2_b_buffer"
        assert hasattr(state, 'grad_w1_a_buffer'), "State should have grad_w1_a_buffer"
        assert hasattr(state, 'grad_w1_b_buffer'), "State should have grad_w1_b_buffer"

        MemoryManager.detach(container)


class TestLoHAForwardBackward:
    """Test forward and backward pass correctness."""

    def test_forward_pass_works(self):
        """Forward pass should work with offloading enabled."""
        from toolkit.memory_management import MemoryManager

        network = MockLoHANetwork(num_loha_layers=2)
        device = torch.device("cuda:0")

        # Move to GPU first, then attach (MemoryManager offloads LoHA params to CPU)
        network = network.to(device)
        MemoryManager.attach(network, device, offload_percent=1.0)

        # Create input on GPU
        x = torch.randn(2, 64, device=device)

        # Forward should work
        y = network(x)

        assert y.shape == (2, 128), f"Expected shape (2, 128), got {y.shape}"
        assert y.device == device, "Output should be on same device as input"

        MemoryManager.detach(network)

    def test_backward_pass_computes_gradients(self):
        """Backward pass should compute gradients for LoHA parameters."""
        from toolkit.memory_management import MemoryManager

        network = MockLoHANetwork(num_loha_layers=2)
        device = torch.device("cuda:0")

        # Enable gradients
        for param in network.parameters():
            param.requires_grad = True

        # Move to GPU first, then attach (MemoryManager offloads LoHA params to CPU)
        network = network.to(device)
        MemoryManager.attach(network, device, offload_percent=1.0)

        # Forward + backward
        x = torch.randn(2, 64, device=device, requires_grad=True)
        y = network(x)
        loss = y.sum()
        loss.backward()

        # Synchronize
        MemoryManager.synchronize_all(network)

        # Check gradients exist for LoHA parameters
        for layer in network.loha_layers:
            assert layer.hada_w1_a.grad is not None, "w1_a should have gradient"
            assert layer.hada_w1_b.grad is not None, "w1_b should have gradient"
            assert layer.hada_w2_a.grad is not None, "w2_a should have gradient"
            assert layer.hada_w2_b.grad is not None, "w2_b should have gradient"

        MemoryManager.detach(network)


class TestLoHAGradientAccuracy:
    """Test gradient computation accuracy vs non-offloaded LoHA."""

    def test_gradient_matches_non_offloaded(self):
        """Offloaded gradients should match non-offloaded computation."""
        from toolkit.memory_management import MemoryManager

        # Create two identical LoHA modules
        torch.manual_seed(42)
        loha_offloaded = MockLoHAModule(in_dim=32, out_dim=64, lora_dim=4)
        torch.manual_seed(42)
        loha_reference = MockLoHAModule(in_dim=32, out_dim=64, lora_dim=4)

        device = torch.device("cuda:0")

        # Enable gradients on both
        for param in loha_offloaded.parameters():
            param.requires_grad = True
        for param in loha_reference.parameters():
            param.requires_grad = True

        # Move reference to GPU (no offloading)
        loha_reference = loha_reference.to(device)

        # Wrap offloaded in container and attach
        class Container(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.layer = module
            def forward(self, x):
                return self.layer(x)

        # Move to GPU first, then attach (MemoryManager offloads LoHA params to CPU)
        container = Container(loha_offloaded)
        container = container.to(device)
        MemoryManager.attach(container, device, offload_percent=1.0)

        # Same input for both
        torch.manual_seed(123)
        x_offloaded = torch.randn(4, 32, device=device, requires_grad=True)
        torch.manual_seed(123)
        x_reference = torch.randn(4, 32, device=device, requires_grad=True)

        # Forward + backward on offloaded
        y_offloaded = container(x_offloaded)
        loss_offloaded = y_offloaded.sum()
        loss_offloaded.backward()
        MemoryManager.synchronize_all(container)

        # Forward + backward on reference
        y_reference = loha_reference(x_reference)
        loss_reference = y_reference.sum()
        loss_reference.backward()

        # Compare outputs (should be very close)
        torch.testing.assert_close(
            y_offloaded, y_reference,
            rtol=1e-4, atol=1e-4,
            msg="Forward outputs should match"
        )

        # Compare gradients for w1_a (most complex gradient)
        grad_offloaded = loha_offloaded.hada_w1_a.grad
        grad_reference = loha_reference.hada_w1_a.grad

        # Move offloaded grad to GPU for comparison
        if grad_offloaded.device.type == "cpu":
            grad_offloaded = grad_offloaded.to(device)

        torch.testing.assert_close(
            grad_offloaded, grad_reference,
            rtol=1e-3, atol=1e-3,
            msg="w1_a gradients should match"
        )

        MemoryManager.detach(container)

    def test_scalar_gradient_matches_non_offloaded(self):
        """Scalar gradient should be computed correctly when use_scalar=True."""
        from toolkit.memory_management import MemoryManager

        # Create LoHA with trainable scalar
        torch.manual_seed(42)
        loha_offloaded = MockLoHAModule(in_dim=32, out_dim=64, lora_dim=4, use_scalar=True)
        torch.manual_seed(42)
        loha_reference = MockLoHAModule(in_dim=32, out_dim=64, lora_dim=4, use_scalar=True)

        device = torch.device("cuda:0")

        # Enable gradients on both
        for param in loha_offloaded.parameters():
            param.requires_grad = True
        for param in loha_reference.parameters():
            param.requires_grad = True

        # Move reference to GPU (no offloading)
        loha_reference = loha_reference.to(device)

        # Wrap offloaded in container and attach
        class Container(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.layer = module
            def forward(self, x):
                return self.layer(x)

        # Move to GPU first, then attach (MemoryManager offloads LoHA params to CPU)
        container = Container(loha_offloaded)
        container = container.to(device)
        MemoryManager.attach(container, device, offload_percent=1.0)

        # Same input for both
        torch.manual_seed(123)
        x_offloaded = torch.randn(4, 32, device=device, requires_grad=True)
        torch.manual_seed(123)
        x_reference = torch.randn(4, 32, device=device, requires_grad=True)

        # Forward + backward on offloaded
        y_offloaded = container(x_offloaded)
        loss_offloaded = y_offloaded.sum()
        loss_offloaded.backward()
        MemoryManager.synchronize_all(container)

        # Forward + backward on reference
        y_reference = loha_reference(x_reference)
        loss_reference = y_reference.sum()
        loss_reference.backward()

        # Scalar gradient should exist on offloaded
        assert loha_offloaded.scalar.grad is not None, "Scalar should have gradient when use_scalar=True"

        # Compare scalar gradients
        grad_scalar_offloaded = loha_offloaded.scalar.grad
        grad_scalar_reference = loha_reference.scalar.grad

        # Move offloaded grad to GPU for comparison
        if grad_scalar_offloaded.device.type == "cpu":
            grad_scalar_offloaded = grad_scalar_offloaded.to(device)

        torch.testing.assert_close(
            grad_scalar_offloaded, grad_scalar_reference,
            rtol=1e-3, atol=1e-3,
            msg="Scalar gradients should match"
        )

        MemoryManager.detach(container)


class TestLoHADetachCleanup:
    """Test that detach properly cleans up LoHA resources."""

    def test_detach_removes_markers(self):
        """Detach should remove all memory management markers."""
        from toolkit.memory_management import MemoryManager

        network = MockLoHANetwork(num_loha_layers=2)
        device = torch.device("cuda:0")

        # Wrap in container
        class Container(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net
            def forward(self, x):
                return self.net(x)

        container = Container(network)
        MemoryManager.attach(container, device, offload_percent=1.0)

        # Verify markers exist
        for layer in network.loha_layers:
            assert hasattr(layer, "_layer_memory_manager")
            assert hasattr(layer, "_memory_management_device")

        MemoryManager.detach(container)

        # Verify markers removed
        for layer in network.loha_layers:
            assert not hasattr(layer, "_layer_memory_manager")
            assert not hasattr(layer, "_memory_management_device")

    def test_detach_clears_state(self):
        """Detach should destroy per-module state objects."""
        from toolkit.memory_management import MemoryManager

        loha = MockLoHAModule()

        class Container(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.layer = module

        container = Container(loha)
        device = torch.device("cuda:0")

        MemoryManager.attach(container, device, offload_percent=1.0)

        # Get state before detach
        state = loha._layer_memory_manager._state

        MemoryManager.detach(container)

        # State should be destroyed
        assert state._destroyed, "State should be marked as destroyed"
        assert state.transfer_stream is None, "Stream should be cleared"
        assert state.w1_a_buffer is None, "LoHA buffer should be cleared"


class TestLoHAParametersOnCPU:
    """Test that LoHA parameters are moved to CPU after attach."""

    def test_parameters_on_cpu_after_attach(self):
        """LoHA parameters should be on CPU with pinned memory after attach."""
        from toolkit.memory_management import MemoryManager

        loha = MockLoHAModule()

        class Container(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.layer = module

        container = Container(loha)
        device = torch.device("cuda:0")

        MemoryManager.attach(container, device, offload_percent=1.0)

        # Parameters should be on CPU
        assert loha.hada_w1_a.device.type == "cpu", "w1_a should be on CPU"
        assert loha.hada_w1_b.device.type == "cpu", "w1_b should be on CPU"
        assert loha.hada_w2_a.device.type == "cpu", "w2_a should be on CPU"
        assert loha.hada_w2_b.device.type == "cpu", "w2_b should be on CPU"

        # Parameters should be marked as memory managed
        assert hasattr(loha.hada_w1_a, "_is_memory_managed")
        assert loha.hada_w1_a._is_memory_managed

        MemoryManager.detach(container)


class TestLoHAOffloadPercent:
    """Test partial offloading with offload_percent."""

    def test_partial_offload(self):
        """Should offload only specified percentage of LoHA layers."""
        from toolkit.memory_management import MemoryManager

        network = MockLoHANetwork(num_loha_layers=4)
        device = torch.device("cuda:0")

        # Offload 50% (2 of 4 layers)
        MemoryManager.attach(network, device, offload_percent=0.5, deterministic=True)

        # Should have 2 managed layers
        count = MemoryManager.get_managed_layer_count(network)
        assert count == 2, f"Expected 2 managed layers with 50% offload, got {count}"

        # First 2 should be managed (deterministic)
        assert hasattr(network.loha_layers[0], "_layer_memory_manager")
        assert hasattr(network.loha_layers[1], "_layer_memory_manager")
        assert not hasattr(network.loha_layers[2], "_layer_memory_manager")
        assert not hasattr(network.loha_layers[3], "_layer_memory_manager")

        MemoryManager.detach(network)


class TestMixedNetwork:
    """Test network with both Linear and LoHA layers."""

    def test_mixed_network(self):
        """Should handle networks with both Linear and LoHA layers."""
        from toolkit.memory_management import MemoryManager

        class MixedNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(64, 128)
                self.loha = MockLoHAModule(128, 128, lora_dim=8)
                self.linear2 = nn.Linear(128, 64)

            def forward(self, x):
                x = F.relu(self.linear1(x))
                x = F.relu(self.loha(x))
                x = self.linear2(x)
                return x

        network = MixedNetwork()
        device = torch.device("cuda:0")

        # Move to GPU first, then attach (MemoryManager offloads managed params to CPU)
        network = network.to(device)
        MemoryManager.attach(network, device, offload_percent=1.0)

        # Should manage all 3 layers (2 Linear + 1 LoHA)
        count = MemoryManager.get_managed_layer_count(network)
        assert count == 3, f"Expected 3 managed layers, got {count}"

        # All should have layer managers
        assert hasattr(network.linear1, "_layer_memory_manager")
        assert hasattr(network.loha, "_layer_memory_manager")
        assert hasattr(network.linear2, "_layer_memory_manager")

        # Forward should work
        x = torch.randn(2, 64, device=device)
        y = network(x)
        assert y.shape == (2, 64)

        MemoryManager.detach(network)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
