"""
Unit tests for layer offloading memory management.

Tests:
1. Per-module state isolation
2. Detach clears streams/events and removes markers
3. Deterministic layer selection
4. Quantized training early warning
5. Stream count tracking
"""

import pytest
import torch
import torch.nn as nn
from typing import Optional

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        return x


class TestPerModuleStateIsolation:
    """Test that each managed layer has its own state."""

    def test_separate_state_objects(self):
        """Two Linear layers should not share state objects."""
        from toolkit.memory_management import MemoryManager

        model = SimpleModel()
        device = torch.device("cuda:0")

        MemoryManager.attach(model, device, offload_percent=1.0)

        # Get state objects from each layer
        states = []
        for name, module in model.named_modules():
            if hasattr(module, "_layer_memory_manager"):
                lmm = module._layer_memory_manager
                if hasattr(lmm, "_state") and lmm._state is not None:
                    states.append(lmm._state)

        # Should have 3 separate state objects (one per Linear layer)
        assert len(states) == 3, f"Expected 3 states, got {len(states)}"

        # All states should be different objects
        state_ids = [id(s) for s in states]
        assert len(set(state_ids)) == 3, "States should be different objects"

        # Each state should have its own streams
        streams = [s.transfer_stream for s in states]
        stream_ids = [id(s) for s in streams]
        assert len(set(stream_ids)) == 3, "Streams should be different objects"

        # Cleanup
        MemoryManager.detach(model)

    def test_events_not_shared(self):
        """Events should not be shared between layers."""
        from toolkit.memory_management import MemoryManager

        model = SimpleModel()
        device = torch.device("cuda:0")

        MemoryManager.attach(model, device, offload_percent=1.0)

        # Collect all transfer_forward_done events
        events = []
        for name, module in model.named_modules():
            if hasattr(module, "_layer_memory_manager"):
                lmm = module._layer_memory_manager
                if hasattr(lmm, "_state") and lmm._state is not None:
                    events.append(lmm._state.transfer_forward_done)

        # All events should be different objects
        event_ids = [id(e) for e in events]
        assert len(set(event_ids)) == len(events), "Events should not be shared"

        # Cleanup
        MemoryManager.detach(model)


class TestDetachCleanup:
    """Test that detach properly cleans up resources."""

    def test_detach_removes_markers(self):
        """Detach should remove all memory management markers."""
        from toolkit.memory_management import MemoryManager

        model = SimpleModel()
        device = torch.device("cuda:0")

        MemoryManager.attach(model, device, offload_percent=1.0)

        # Verify markers exist
        assert hasattr(model, "_memory_manager")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                assert hasattr(module, "_layer_memory_manager")
                assert hasattr(module, "_memory_management_device")

        # Detach
        MemoryManager.detach(model)

        # Verify markers are removed
        assert not hasattr(model, "_memory_manager")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                assert not hasattr(module, "_layer_memory_manager")
                assert not hasattr(module, "_memory_management_device")

    def test_detach_clears_state(self):
        """Detach should destroy per-module state objects."""
        from toolkit.memory_management import MemoryManager

        model = SimpleModel()
        device = torch.device("cuda:0")

        MemoryManager.attach(model, device, offload_percent=1.0)

        # Collect state objects before detach
        states = []
        for name, module in model.named_modules():
            if hasattr(module, "_layer_memory_manager"):
                lmm = module._layer_memory_manager
                if hasattr(lmm, "_state"):
                    states.append(lmm._state)

        # Detach
        MemoryManager.detach(model)

        # States should be marked as destroyed
        for state in states:
            assert state._destroyed, "State should be marked as destroyed"
            assert state.transfer_stream is None, "Stream should be cleared"

    def test_is_attached_after_detach(self):
        """is_attached should return False after detach."""
        from toolkit.memory_management import MemoryManager

        model = SimpleModel()
        device = torch.device("cuda:0")

        MemoryManager.attach(model, device)
        assert MemoryManager.is_attached(model)

        MemoryManager.detach(model)
        assert not MemoryManager.is_attached(model)


class TestDeterministicSelection:
    """Test deterministic layer selection."""

    def test_deterministic_is_reproducible(self):
        """Deterministic selection should give same layers each time."""
        from toolkit.memory_management import MemoryManager

        model1 = SimpleModel()
        model2 = SimpleModel()
        device = torch.device("cuda:0")

        # Attach with 66% offload (2 of 3 layers)
        MemoryManager.attach(model1, device, offload_percent=0.66, deterministic=True)
        MemoryManager.attach(model2, device, offload_percent=0.66, deterministic=True)

        # Get managed layer names
        managed1 = []
        managed2 = []
        for name, module in model1.named_modules():
            if hasattr(module, "_layer_memory_manager"):
                managed1.append(name)
        for name, module in model2.named_modules():
            if hasattr(module, "_layer_memory_manager"):
                managed2.append(name)

        assert managed1 == managed2, "Deterministic selection should be reproducible"

        # Cleanup
        MemoryManager.detach(model1)
        MemoryManager.detach(model2)

    def test_deterministic_offloads_first_n(self):
        """Deterministic selection should offload the first N layers."""
        from toolkit.memory_management import MemoryManager

        model = SimpleModel()
        device = torch.device("cuda:0")

        # Attach with 34% offload (1 of 3 layers) - use 0.34 since int(3*0.33)=0
        MemoryManager.attach(model, device, offload_percent=0.34, deterministic=True)

        # Should have exactly 1 managed layer (int(3 * 0.34) = 1)
        count = MemoryManager.get_managed_layer_count(model)
        assert count == 1, f"Expected 1 managed layer, got {count}"

        # The first layer (linear1) should be managed
        assert hasattr(model.linear1, "_layer_memory_manager")
        assert not hasattr(model.linear2, "_layer_memory_manager")
        assert not hasattr(model.linear3, "_layer_memory_manager")

        # Cleanup
        MemoryManager.detach(model)


class TestStreamCountTracking:
    """Test stream count tracking for proliferation warnings."""

    def test_stream_count_increases_on_attach(self):
        """Stream count should increase when layers are attached."""
        from toolkit.memory_management import MemoryManager

        # Clear any existing state
        MemoryManager.clear_all_state()

        model = SimpleModel()
        device = torch.device("cuda:0")

        initial_count = MemoryManager.get_stream_count(device)
        assert initial_count == 0

        MemoryManager.attach(model, device, offload_percent=1.0)

        # 3 layers × 2 streams per layer = 6 streams
        count = MemoryManager.get_stream_count(device)
        assert count == 6, f"Expected 6 streams, got {count}"

        # Cleanup
        MemoryManager.detach(model)

    def test_stream_count_decreases_on_detach(self):
        """Stream count should decrease when layers are detached."""
        from toolkit.memory_management import MemoryManager

        # Clear any existing state
        MemoryManager.clear_all_state()

        model = SimpleModel()
        device = torch.device("cuda:0")

        MemoryManager.attach(model, device, offload_percent=1.0)
        assert MemoryManager.get_stream_count(device) == 6

        MemoryManager.detach(model)
        assert MemoryManager.get_stream_count(device) == 0


class TestForwardBackward:
    """Test that forward/backward passes work correctly."""

    def test_forward_pass(self):
        """Forward pass should work with offloading enabled."""
        from toolkit.memory_management import MemoryManager

        model = SimpleModel()
        device = torch.device("cuda:0")

        MemoryManager.attach(model, device, offload_percent=1.0)

        # Create input on GPU
        x = torch.randn(2, 64, device=device)

        # Forward should work
        y = model(x)

        assert y.shape == (2, 32)
        assert y.device == device

        # Cleanup
        MemoryManager.detach(model)

    def test_backward_pass(self):
        """Backward pass should compute gradients correctly."""
        from toolkit.memory_management import MemoryManager

        model = SimpleModel()
        device = torch.device("cuda:0")

        # Enable gradients on weights
        for param in model.parameters():
            param.requires_grad = True

        MemoryManager.attach(model, device, offload_percent=1.0)

        # Create input on GPU
        x = torch.randn(2, 64, device=device, requires_grad=True)

        # Forward + backward
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Synchronize to ensure gradients are transferred
        MemoryManager.synchronize_all(model)

        # Check that gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            # Gradients should be on CPU (offloaded)
            assert param.grad.device.type == "cpu", f"Gradient for {name} should be on CPU"

        # Cleanup
        MemoryManager.detach(model)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
