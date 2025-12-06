"""
Regression/diagnostic tests for Flux2 split-model multi-GPU plumbing.

These tests are lightweight, CUDA-only (single GPU is sufficient), and rely on
monkeypatching to avoid loading the real Flux2 weights. They cover:
- sync_per_block behavior (single + double blocks)
- streamed fallback behavior
- modulation cache safety and debug helpers
- DeviceTransferManager default-stream join, reuse, and sync_on_exit
- Split metadata safety across .to() round-trips and topology changes
- output_device enforcement during model loading
- Deterministic GPU splits for common configurations (2, 4, 8 GPUs)
- Debug logging for both splitter and transfer manager
"""

import types
import pytest
import torch
import logging

from extensions_built_in.diffusion_models.flux2 import flux2_gpu_splitter as splitter
from extensions_built_in.diffusion_models.flux2 import flux2_model
from extensions_built_in.diffusion_models.flux2.cuda_stream_manager import (
    DeviceTransferManager,
    ModulationCacheKey,
    logger as transfer_logger,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_splitter_debug(monkeypatch):
    """Keep splitter debug flag from leaking across tests."""
    monkeypatch.setattr(splitter, "_gpu_splitter_debug", False, raising=False)
    monkeypatch.setattr(splitter, "_debug_state_logged", False, raising=False)
    yield


def _build_dummy_transformer(num_double=2, num_single=2, robust_split_device=False):
    """Build a minimal dummy transformer for splitter tests.

    Args:
        num_double: Number of double blocks
        num_single: Number of single blocks
        robust_split_device: If True, use a DummyBlock that handles _split_device
            deletion gracefully (returns default instead of raising AttributeError).
            This works around a production code bug where _reset_block deletes
            _split_device but new_device_to_flux2 then tries to access it.
    """
    class DummyBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)
            self._split_device = torch.device("cuda:0")

        def forward(self, *a, **k):
            return a[0] if a else self.lin(torch.randn(1, 2, device=self._split_device))

    class RobustDummyBlock(torch.nn.Module):
        """DummyBlock that handles _split_device deletion gracefully.

        The production code has a bug where _reset_block deletes _split_device,
        but new_device_to_flux2 then tries to read it. This class uses __getattr__
        to return a default device when _split_device is missing.
        """
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)
            self._internal_split_device = torch.device("cuda:0")
            self._split_device = torch.device("cuda:0")

        def __getattr__(self, name):
            if name == "_split_device":
                # Return default if deleted
                return self.__dict__.get("_internal_split_device", torch.device("cuda:0"))
            return super().__getattr__(name)

        def __delattr__(self, name):
            if name == "_split_device":
                # Allow deletion but keep internal fallback
                if "_split_device" in self.__dict__:
                    del self.__dict__["_split_device"]
            else:
                super().__delattr__(name)

        def forward(self, *a, **k):
            dev = getattr(self, "_split_device", torch.device("cuda:0"))
            return a[0] if a else self.lin(torch.randn(1, 2, device=dev))

    BlockClass = RobustDummyBlock if robust_split_device else DummyBlock

    class DummyTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.double_blocks = [BlockClass() for _ in range(num_double)]
            self.single_blocks = [BlockClass() for _ in range(num_single)]
            self.pe_embedder = torch.nn.Linear(2, 2)
            self.img_in = torch.nn.Linear(2, 2)
            self.time_in = torch.nn.Linear(2, 2)
            self.guidance_in = torch.nn.Linear(2, 2)
            self.txt_in = torch.nn.Linear(2, 2)
            self.double_stream_modulation_img = torch.nn.Identity()
            self.double_stream_modulation_txt = torch.nn.Identity()
            self.single_stream_modulation = torch.nn.Identity()
            self.final_layer = torch.nn.Linear(2, 2)

    return DummyTransformer()


def _build_flux2_transformer(robust_split_device=False):
    """Build a dummy transformer matching FLUX.2 architecture (8 double, 48 single)."""
    return _build_dummy_transformer(num_double=8, num_single=48, robust_split_device=robust_split_device)


def _stub_flux2_dependencies(monkeypatch):
    """Patch heavy deps used by Flux2Model.load_model."""

    class DummyFlux(torch.nn.Module):
        def __init__(self, *a, **k):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.randn(1))

        def load_state_dict(self, *a, **k):
            return

        def to(self, *a, **k):
            return self

    class DummyAutoProcessor:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return None

    class DummyMistral(torch.nn.Module):
        def __init__(self, *a, **k):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.randn(1))

        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

    class DummyAutoEncoder(torch.nn.Module):
        def __init__(self, *a, **k):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.randn(1))

        def load_state_dict(self, *a, **k):
            return

        def to(self, *a, **k):
            return self

    class DummyAEParams:
        pass

    monkeypatch.setattr(flux2_model, "Flux2", DummyFlux)
    monkeypatch.setattr(flux2_model, "load_file", lambda *a, **k: {})
    monkeypatch.setattr(flux2_model, "AutoProcessor", DummyAutoProcessor)
    monkeypatch.setattr(flux2_model, "Mistral3ForConditionalGeneration", DummyMistral)
    monkeypatch.setattr(flux2_model, "AutoEncoderParams", DummyAEParams)
    monkeypatch.setattr(flux2_model, "AutoEncoder", DummyAutoEncoder)
    monkeypatch.setattr(
        flux2_model, "huggingface_hub", types.SimpleNamespace(hf_hub_download=lambda **kw: "/dev/null")
    )
    monkeypatch.setattr(flux2_model, "patch_dequantization_on_save", lambda *a, **k: None)
    monkeypatch.setattr(flux2_model, "quantize_model", lambda *a, **k: None)
    monkeypatch.setattr(flux2_model, "freeze", lambda *a, **k: None)
    monkeypatch.setattr(flux2_model, "quantize", lambda *a, **k: None)
    monkeypatch.setattr(flux2_model, "get_qtype", lambda *a, **k: None)
    monkeypatch.setattr(flux2_model, "flush", lambda *a, **k: None)
    monkeypatch.setattr(flux2_model, "MemoryManager", types.SimpleNamespace(attach=lambda *a, **k: None))


# ---------------------------------------------------------------------------
# Single Block Tests
# ---------------------------------------------------------------------------

def test_blocking_sync_flag_controls_synchronize(monkeypatch):
    """sync_per_block controls whether torch.cuda.synchronize is called (single block)."""
    calls = []

    def fake_sync(device=None):
        calls.append(device)

    monkeypatch.setattr(torch.cuda, "synchronize", fake_sync)

    class DummyBlock:
        def __init__(self):
            self._split_device = torch.device("cuda:0")

        def _pre_gpu_split_forward(self, x, pe, mod):
            return x

    blk = DummyBlock()
    x = torch.randn(1, device="cuda")
    pe = torch.randn(1, device="cuda")
    mod = (x, x, x)

    calls.clear()
    splitter.split_gpu_single_block_forward(blk, x, pe, mod, sync_per_block=False)
    assert calls == []

    calls.clear()
    splitter.split_gpu_single_block_forward(blk, x, pe, mod, sync_per_block=True)
    assert len(calls) == 1


def test_double_block_sync_flag_controls_synchronize(monkeypatch):
    """sync_per_block controls blocking sync for double block wrapper."""
    calls = []

    def fake_sync(device=None):
        calls.append(device)

    monkeypatch.setattr(torch.cuda, "synchronize", fake_sync)

    class DummyDouble:
        def __init__(self):
            self._split_device = torch.device("cuda:0")

        def _pre_gpu_split_forward(self, img, txt, pe, pe_ctx, mod_img, mod_txt):
            return img, txt

    blk = DummyDouble()
    t = torch.randn(1, device="cuda")
    mod_img = ((t, t, t), (t, t, t))
    mod_txt = ((t, t, t), (t, t, t))

    calls.clear()
    splitter.split_gpu_double_block_forward(blk, t, t, t, t, mod_img, mod_txt, sync_per_block=False)
    assert calls == []

    calls.clear()
    splitter.split_gpu_double_block_forward(blk, t, t, t, t, mod_img, mod_txt, sync_per_block=True)
    assert len(calls) == 1


def test_stream_fallback_respects_sync_per_block(monkeypatch):
    """Streamed single-block wrapper falls back and honors _sync_per_block."""
    calls = []

    def fake_sync(device=None):
        calls.append(device)

    monkeypatch.setattr(torch.cuda, "synchronize", fake_sync)
    monkeypatch.setattr(splitter, "get_transfer_manager", lambda: None)

    class DummyBlock:
        def __init__(self):
            self._split_device = torch.device("cuda:0")
            self._sync_per_block = True

        def _pre_gpu_split_forward(self, x, pe, mod):
            return x

    blk = DummyBlock()
    x = torch.randn(1, device="cuda")
    pe = torch.randn(1, device="cuda")
    mod = (x, x, x)

    calls.clear()
    splitter.split_gpu_single_block_forward_streamed(blk, x, pe, mod)
    assert len(calls) == 1  # synced once via fallback


def test_split_output_device_routing():
    """Single block should route output to _split_output_device when present."""
    class DummyBlock:
        def __init__(self, output_dev):
            self._split_device = torch.device("cuda:0")
            self._split_output_device = output_dev

        def _pre_gpu_split_forward(self, x, pe, mod):
            return x

    x = torch.randn(1, device="cuda")
    pe = torch.randn(1, device="cuda")
    mod = (x, x, x)

    blk_cpu = DummyBlock(torch.device("cpu"))
    out_cpu = splitter.split_gpu_single_block_forward(blk_cpu, x, pe, mod, sync_per_block=False)
    assert out_cpu.device.type == "cpu"

    class NoOutputAttr(DummyBlock):
        def __init__(self):
            self._split_device = torch.device("cuda:0")

    blk_default = NoOutputAttr()
    out_default = splitter.split_gpu_single_block_forward(blk_default, x, pe, mod, sync_per_block=False)
    assert out_default.device == torch.device("cuda:0")


def test_split_output_device_missing_attr_fallback():
    """Single block without _split_output_device should keep output on split device."""
    class DummyBlock:
        def __init__(self):
            self._split_device = torch.device("cuda:0")

        def _pre_gpu_split_forward(self, x, pe, mod):
            return x

    x = torch.randn(1, device="cuda")
    pe = torch.randn(1, device="cuda")
    mod = (x, x, x)
    blk = DummyBlock()
    out = splitter.split_gpu_single_block_forward(blk, x, pe, mod, sync_per_block=False)
    assert out.device == torch.device("cuda:0")


# ---------------------------------------------------------------------------
# DeviceTransferManager Tests
# ---------------------------------------------------------------------------

def test_modulation_cache_shape_invalidation():
    """DeviceTransferManager should update cache when shapes change."""
    dev = torch.device("cuda:0")
    mgr = DeviceTransferManager(devices=[dev], enable_timing=False, sync_on_exit=False, output_device=dev)
    with mgr:
        t1 = torch.randn(1, 1, 8, device=dev)
        mod_img = ((t1, t1, t1), (t1, t1, t1))
        mod_txt = ((t1, t1, t1), (t1, t1, t1))
        mod_single = (t1, t1, t1)
        mgr.stage_modulation_tensors(mod_img, mod_txt, mod_single, source_device=dev)
        key1 = mgr._device_states[dev].mod_img_key

    with mgr:
        t2 = torch.randn(2, 1, 8, device=dev)
        mod_img2 = ((t2, t2, t2), (t2, t2, t2))
        mod_txt2 = ((t2, t2, t2), (t2, t2, t2))
        mod_single2 = (t2, t2, t2)
        mgr.stage_modulation_tensors(mod_img2, mod_txt2, mod_single2, source_device=dev)
        key2 = mgr._device_states[dev].mod_img_key

    assert key1 != key2, f"modulation cache key did not change: {key1} vs {key2}"
    assert key2.batch_size == 2


def test_modulation_cache_key_describe_mismatch():
    """describe_mismatch should report differing fields."""
    dev = torch.device("cuda:0")
    k1 = ModulationCacheKey(batch_size=1, seq_len=1, hidden_dim=8, dtype=torch.float32, device=dev)
    k2 = ModulationCacheKey(batch_size=2, seq_len=1, hidden_dim=8, dtype=torch.float32, device=dev)
    msg = k1.describe_mismatch(k2)
    assert "batch_size" in msg and "1 vs 2" in msg


def test_modulation_cache_debug_mismatch_message():
    """describe_mismatch should list all differing fields."""
    dev = torch.device("cuda:0")
    k1 = ModulationCacheKey(1, 1, 8, torch.float32, dev)
    k2 = ModulationCacheKey(1, 2, 16, torch.float16, dev)
    msg = k1.describe_mismatch(k2)
    assert "seq_len" in msg and "hidden_dim" in msg and "dtype" in msg


def test_default_stream_join_backward():
    """Manager exit should leave backward safe on default stream."""
    dev = torch.device("cuda:0")
    mgr = DeviceTransferManager(devices=[dev], enable_timing=False, sync_on_exit=False, output_device=dev)
    x = torch.randn(4, device=dev, requires_grad=True)
    with mgr:
        y = (x * 2).sum()
        state = mgr._device_states[dev]
        state.compute_event.record(torch.cuda.current_stream(dev))
        state.transfer_event.record(torch.cuda.current_stream(dev))
    y.backward()
    assert x.grad is not None


def test_default_stream_join_called(monkeypatch):
    """_join_default_stream should be invoked on context exit."""
    dev = torch.device("cuda:0")
    mgr = DeviceTransferManager(devices=[dev], enable_timing=False, sync_on_exit=False, output_device=dev)
    called = {"count": 0}
    original = mgr._join_default_stream

    def wrapped():
        called["count"] += 1
        return original()

    monkeypatch.setattr(mgr, "_join_default_stream", wrapped)
    with mgr:
        pass
    assert called["count"] == 1


def test_transfer_manager_reuse_resets_state():
    """Manager reuse should reset per-step state and allow re-staging."""
    dev = torch.device("cuda:0")
    mgr = DeviceTransferManager(devices=[dev], enable_timing=False, sync_on_exit=False, output_device=dev)

    with mgr:
        t = torch.randn(1, 1, 4, device=dev)
        mod_img = ((t, t, t), (t, t, t))
        mod_txt = ((t, t, t), (t, t, t))
        mod_single = (t, t, t)
        mgr.stage_modulation_tensors(mod_img, mod_txt, mod_single, source_device=dev)
        state = mgr._device_states[dev]
        assert state.mods_staged is True
        assert state.mod_img_cache is not None
    state = mgr._device_states[dev]
    assert state.mod_img_cache is None
    assert state.mods_staged is False

    with mgr:
        t2 = torch.randn(2, 1, 4, device=dev)
        mod_img2 = ((t2, t2, t2), (t2, t2, t2))
        mod_txt2 = ((t2, t2, t2), (t2, t2, t2))
        mod_single2 = (t2, t2, t2)
        mgr.stage_modulation_tensors(mod_img2, mod_txt2, mod_single2, source_device=dev)
        state2 = mgr._device_states[dev]
        assert state2.mods_staged is True
        assert state2.mod_img_cache is not None
        assert state2.mod_img_key.batch_size == 2


def test_transfer_manager_sync_on_exit_triggers_synchronize_all(monkeypatch):
    """sync_on_exit=True should call synchronize_all() on context exit."""
    dev = torch.device("cuda:0")
    mgr = DeviceTransferManager(devices=[dev], enable_timing=False, sync_on_exit=True, output_device=dev)

    sync_all_calls = {"count": 0}
    original_sync_all = mgr.synchronize_all

    def tracked_sync_all():
        sync_all_calls["count"] += 1
        return original_sync_all()

    monkeypatch.setattr(mgr, "synchronize_all", tracked_sync_all)

    with mgr:
        # Minimal work inside context
        pass

    assert sync_all_calls["count"] == 1, \
        f"synchronize_all should be called exactly once with sync_on_exit=True, got {sync_all_calls['count']}"


def test_transfer_manager_sync_on_exit_false_skips_synchronize_all(monkeypatch):
    """sync_on_exit=False should NOT call synchronize_all() on context exit."""
    dev = torch.device("cuda:0")
    mgr = DeviceTransferManager(devices=[dev], enable_timing=False, sync_on_exit=False, output_device=dev)

    sync_all_calls = {"count": 0}
    original_sync_all = mgr.synchronize_all

    def tracked_sync_all():
        sync_all_calls["count"] += 1
        return original_sync_all()

    monkeypatch.setattr(mgr, "synchronize_all", tracked_sync_all)

    with mgr:
        pass

    assert sync_all_calls["count"] == 0, \
        f"synchronize_all should NOT be called with sync_on_exit=False, got {sync_all_calls['count']}"


def test_default_stream_join_multi_device():
    """
    _join_default_stream should handle multiple device states without error.

    We reuse cuda:0 as a second device entry to test the iteration logic
    without requiring multiple physical GPUs.
    """
    dev0 = torch.device("cuda:0")
    # Create a second "virtual" device reference (same physical GPU, different dict key)
    # We manually add a second device state to test the multi-device join path
    mgr = DeviceTransferManager(devices=[dev0], enable_timing=False, sync_on_exit=False, output_device=dev0)

    x = torch.randn(4, device=dev0, requires_grad=True)

    with mgr:
        y = (x * 2).sum()
        # Record events on device state
        state = mgr._device_states[dev0]
        state.compute_event.record(torch.cuda.current_stream(dev0))
        state.transfer_event.record(torch.cuda.current_stream(dev0))

        # Manually add a second device state to simulate multi-device scenario
        # This tests that _join_default_stream iterates over all states
        from extensions_built_in.diffusion_models.flux2.cuda_stream_manager import DeviceState
        with torch.cuda.device(dev0):
            extra_state = DeviceState(
                device=dev0,
                transfer_stream=torch.cuda.Stream(device=dev0),
                compute_event=torch.cuda.Event(),
                transfer_event=torch.cuda.Event(),
            )
        extra_state.compute_event.record(torch.cuda.current_stream(dev0))
        extra_state.transfer_event.record(torch.cuda.current_stream(dev0))
        # Use a synthetic key to add extra state
        mgr._device_states[torch.device("cuda:99")] = extra_state

    # Context exit calls _join_default_stream - should not raise
    # Backward should still work
    y.backward()
    assert x.grad is not None, "backward pass should succeed after multi-device join"


# ---------------------------------------------------------------------------
# DeviceTransferManager Debug Logging Tests
# ---------------------------------------------------------------------------

def test_transfer_manager_debug_logging_enabled(monkeypatch, caplog):
    """set_debug(True) should enable _log_debug to actually log messages."""
    dev = torch.device("cuda:0")
    mgr = DeviceTransferManager(devices=[dev], enable_timing=False, sync_on_exit=False, output_device=dev)

    # Enable debug mode
    mgr.set_debug(True)

    # Capture logging at INFO level
    with caplog.at_level(logging.INFO, logger=transfer_logger.name):
        # Call _log_debug directly
        mgr._log_debug("test debug message")

    # Check that the message was logged
    assert any("test debug message" in record.message for record in caplog.records), \
        "Debug message should appear in logs when debug is enabled"


def test_transfer_manager_debug_logging_disabled(monkeypatch, caplog):
    """set_debug(False) should silence _log_debug messages."""
    dev = torch.device("cuda:0")
    mgr = DeviceTransferManager(devices=[dev], enable_timing=False, sync_on_exit=False, output_device=dev)

    # Ensure debug mode is disabled
    mgr.set_debug(False)

    # Capture logging at INFO level
    with caplog.at_level(logging.INFO, logger=transfer_logger.name):
        # Call _log_debug directly
        mgr._log_debug("this should not appear")

    # Check that the message was NOT logged
    assert not any("this should not appear" in record.message for record in caplog.records), \
        "Debug message should NOT appear in logs when debug is disabled"


def test_transfer_manager_debug_logs_on_initialization(caplog):
    """Debug mode should log initialization message when enabled."""
    dev = torch.device("cuda:0")
    mgr = DeviceTransferManager(devices=[dev], enable_timing=False, sync_on_exit=False, output_device=dev)
    mgr.set_debug(True)

    with caplog.at_level(logging.INFO, logger=transfer_logger.name):
        # Force initialization by entering context
        with mgr:
            pass

    # Should have logged initialization
    assert any("Initialized" in record.message for record in caplog.records), \
        "Debug mode should log initialization message"


# ---------------------------------------------------------------------------
# Topology / .to() Safety Tests
# ---------------------------------------------------------------------------

def test_to_topology_safety_single_gpu(monkeypatch):
    """new_device_to_flux2 should survive CPU round-trip on single GPU."""
    transformer = _build_dummy_transformer()
    splitter.add_model_gpu_splitter_to_flux2(
        transformer,
        gpu_split_double=[2],
        gpu_split_single=[2],
        use_stream_transfers=False,
        sync_per_block=False,
        output_device=torch.device("cuda:0"),
    )
    transformer = splitter.new_device_to_flux2(transformer, torch.device("cpu"))
    transformer = splitter.new_device_to_flux2(transformer, torch.device("cuda:0"))
    assert all(getattr(b, "_split_device", None) == torch.device("cuda:0") for b in transformer.double_blocks)
    assert all(getattr(b, "_split_device", None) == torch.device("cuda:0") for b in transformer.single_blocks)


def test_to_topology_failure_clears_split_metadata(monkeypatch):
    """
    Regression test: Topology shrink (2->1 GPU) should clear stale split
    metadata and leave model in a usable single-device state.

    This test verifies the fix for the historical bug where new_device_to_flux2
    would crash after _reset_block deleted _split_device. The fix adds a
    moving_to_cuda branch to distinguish between "restoring valid split" and
    "split metadata was cleared".

    Asserts:
    - After GPU count shrinks, _split_devices should be cleared
    - Model should be safely usable on the remaining GPU
    - No stale device references pointing to non-existent GPUs
    """
    # Build a transformer with FLUX.2 block counts so splits match
    # Uses normal DummyBlock (not RobustDummyBlock) to catch regressions
    transformer = _build_flux2_transformer(robust_split_device=False)

    # Start with 2 GPUs available
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    splitter.add_model_gpu_splitter_to_flux2(
        transformer,
        gpu_split_double=[4, 4],
        gpu_split_single=[24, 24],
        use_stream_transfers=False,
        sync_per_block=False,
        output_device=torch.device("cuda:0"),
    )

    # Verify initial split was applied
    assert hasattr(transformer, "_split_devices"), "Transformer should have _split_devices after splitting"
    assert len(transformer._split_devices) == 2, "Should have 2 split devices"

    # Move to CPU
    transformer = splitter.new_device_to_flux2(transformer, torch.device("cpu"))

    # Now only 1 GPU available (simulating topology change)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    # Move back to CUDA - this triggers the topology validation
    transformer = splitter.new_device_to_flux2(transformer, torch.device("cuda:0"))

    # High-level assertions: split metadata should be cleared
    assert not hasattr(transformer, "_split_devices"), \
        "_split_devices should be cleared when GPU count shrinks below required"

    # All blocks should now point to cuda:0 (the only available GPU)
    # Note: blocks may have _split_device cleared by _reset_block, or set to cuda:0
    for i, blk in enumerate(transformer.double_blocks):
        dev = getattr(blk, "_split_device", torch.device("cuda:0"))
        # Either no _split_device or it's cuda:0
        assert dev.type == "cpu" or dev == torch.device("cuda:0"), \
            f"double_blocks[{i}]._split_device should be cpu or cuda:0, got {dev}"

    for i, blk in enumerate(transformer.single_blocks):
        dev = getattr(blk, "_split_device", torch.device("cuda:0"))
        assert dev.type == "cpu" or dev == torch.device("cuda:0"), \
            f"single_blocks[{i}]._split_device should be cpu or cuda:0, got {dev}"


def test_to_topology_recovery_allows_forward(monkeypatch):
    """
    Regression test: After topology failure recovery, model should be usable.

    This verifies that after GPU count shrinks and split metadata is cleared,
    the model layers can still perform forward passes on the remaining GPU.

    Uses normal DummyBlock (not RobustDummyBlock) to catch regressions in
    the new_device_to_flux2 fix.
    """
    transformer = _build_flux2_transformer(robust_split_device=False)

    # Start with 2 GPUs
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    splitter.add_model_gpu_splitter_to_flux2(
        transformer,
        gpu_split_double=[4, 4],
        gpu_split_single=[24, 24],
        use_stream_transfers=False,
        sync_per_block=False,
        output_device=torch.device("cuda:0"),
    )

    # Move to CPU
    transformer = splitter.new_device_to_flux2(transformer, torch.device("cpu"))

    # Shrink to 1 GPU
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    # Move back to CUDA
    transformer = splitter.new_device_to_flux2(transformer, torch.device("cuda:0"))

    # Try to move global layers to cuda:0 - should work
    transformer.pe_embedder = transformer.pe_embedder.to("cuda:0")
    transformer.final_layer = transformer.final_layer.to("cuda:0")

    # Basic forward on global layers should work
    x = torch.randn(1, 2, device="cuda:0")
    out = transformer.final_layer(x)
    assert out.device == torch.device("cuda:0"), "forward should work after recovery"


def test_to_topology_shrink_2_to_1_gpu_recovers_cleanly(monkeypatch):
    """
    Comprehensive regression test: moving from 2 GPUs to 1 GPU should:
    - Clear _split_devices
    - Not crash when calling .to("cuda:0")
    - Leave the model in a usable state for trivial forward-like operations.

    This is the canonical regression test for the historical bug where
    new_device_to_flux2 would crash with AttributeError after _reset_block
    deleted _split_device. The fix distinguishes moving_to_cuda (valid split
    restoration) from plain has_device (split was cleared, use normal .to()).

    Uses normal DummyBlock to ensure this test fails if the bug reappears.
    """
    transformer = _build_flux2_transformer(robust_split_device=False)

    # Start with 2 GPUs.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    splitter.add_model_gpu_splitter_to_flux2(
        transformer,
        gpu_split_double=[4, 4],
        gpu_split_single=[24, 24],
        use_stream_transfers=False,
        sync_per_block=False,
        output_device=torch.device("cuda:0"),
    )

    # Verify split was applied
    assert hasattr(transformer, "_split_devices")
    assert len(transformer._split_devices) == 2

    # Move to CPU (stash original split devices).
    transformer = splitter.new_device_to_flux2(transformer, torch.device("cpu"))

    # Now only 1 GPU available.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    # Move back to CUDA: should clear split metadata and NOT crash.
    # This is the line that would raise AttributeError before the fix.
    transformer = splitter.new_device_to_flux2(transformer, torch.device("cuda:0"))

    # Split metadata should be cleared or reset to a safe state.
    assert not hasattr(transformer, "_split_devices"), \
        "_split_devices should be cleared after topology shrink"

    # A trivial forward-like operation should succeed on cuda:0.
    # Test on a single block to verify blocks are in a usable state.
    block = transformer.single_blocks[0]
    x = torch.randn(1, 2, device=torch.device("cuda:0"))
    out = block(x)
    assert out.shape[-1] == 2, "block forward should return expected shape"
    assert out.device == torch.device("cuda:0"), "output should be on cuda:0"

    # Also verify final_layer works
    transformer.final_layer = transformer.final_layer.to("cuda:0")
    out2 = transformer.final_layer(x)
    assert out2.device == torch.device("cuda:0"), "final_layer forward should work"


# ---------------------------------------------------------------------------
# Deterministic GPU Split Tests
# ---------------------------------------------------------------------------

def test_gpu_splitter_respects_default_splits_for_4_gpus(monkeypatch):
    """
    When 4 GPUs are available and no user splits are specified, the splitter
    should use DEFAULT_SPLITS[4] which evenly distributes blocks.

    FLUX.2 has 8 double blocks and 48 single blocks.
    DEFAULT_SPLITS[4] = {
        "double": [2, 2, 2, 2],  # 2 blocks per GPU
        "single": [12, 12, 12, 12],  # 12 blocks per GPU
    }
    """
    # Build transformer with FLUX.2 block counts
    transformer = _build_flux2_transformer()

    # Mock 4 GPUs available
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    # Apply splitter with NO user-specified splits - should use defaults
    splitter.add_model_gpu_splitter_to_flux2(
        transformer,
        gpu_split_double=None,  # Use defaults
        gpu_split_single=None,  # Use defaults
        use_stream_transfers=False,
        sync_per_block=False,
        output_device=torch.device("cuda:0"),
    )

    # Count blocks per device
    double_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    single_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for blk in transformer.double_blocks:
        dev = getattr(blk, "_split_device", None)
        assert dev is not None, "Block should have _split_device"
        gpu_id = dev.index
        double_counts[gpu_id] += 1

    for blk in transformer.single_blocks:
        dev = getattr(blk, "_split_device", None)
        assert dev is not None, "Block should have _split_device"
        gpu_id = dev.index
        single_counts[gpu_id] += 1

    # Verify distribution matches DEFAULT_SPLITS[4]
    expected_double = splitter.DEFAULT_SPLITS[4]["double"]
    expected_single = splitter.DEFAULT_SPLITS[4]["single"]

    for gpu_id in range(4):
        assert double_counts[gpu_id] == expected_double[gpu_id], \
            f"GPU {gpu_id} should have {expected_double[gpu_id]} double blocks, got {double_counts[gpu_id]}"
        assert single_counts[gpu_id] == expected_single[gpu_id], \
            f"GPU {gpu_id} should have {expected_single[gpu_id]} single blocks, got {single_counts[gpu_id]}"


def test_gpu_splitter_respects_default_splits_for_2_gpus(monkeypatch):
    """
    When 2 GPUs are available and no user splits are specified, the splitter
    should use DEFAULT_SPLITS[2].

    DEFAULT_SPLITS[2] = {
        "double": [4, 4],  # 4 blocks per GPU
        "single": [24, 24],  # 24 blocks per GPU
    }
    """
    transformer = _build_flux2_transformer()

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    splitter.add_model_gpu_splitter_to_flux2(
        transformer,
        gpu_split_double=None,
        gpu_split_single=None,
        use_stream_transfers=False,
        sync_per_block=False,
        output_device=torch.device("cuda:0"),
    )

    double_counts = {0: 0, 1: 0}
    single_counts = {0: 0, 1: 0}

    for blk in transformer.double_blocks:
        dev = getattr(blk, "_split_device", None)
        assert dev is not None
        double_counts[dev.index] += 1

    for blk in transformer.single_blocks:
        dev = getattr(blk, "_split_device", None)
        assert dev is not None
        single_counts[dev.index] += 1

    expected_double = splitter.DEFAULT_SPLITS[2]["double"]
    expected_single = splitter.DEFAULT_SPLITS[2]["single"]

    for gpu_id in range(2):
        assert double_counts[gpu_id] == expected_double[gpu_id], \
            f"GPU {gpu_id} double blocks mismatch: expected {expected_double[gpu_id]}, got {double_counts[gpu_id]}"
        assert single_counts[gpu_id] == expected_single[gpu_id], \
            f"GPU {gpu_id} single blocks mismatch: expected {expected_single[gpu_id]}, got {single_counts[gpu_id]}"


def test_gpu_splitter_user_splits_override_defaults(monkeypatch):
    """User-specified splits should take precedence over DEFAULT_SPLITS."""
    transformer = _build_flux2_transformer()

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    # Custom asymmetric split
    custom_double = [3, 5]  # Sum = 8
    custom_single = [20, 28]  # Sum = 48

    splitter.add_model_gpu_splitter_to_flux2(
        transformer,
        gpu_split_double=custom_double,
        gpu_split_single=custom_single,
        use_stream_transfers=False,
        sync_per_block=False,
        output_device=torch.device("cuda:0"),
    )

    double_counts = {0: 0, 1: 0}
    single_counts = {0: 0, 1: 0}

    for blk in transformer.double_blocks:
        double_counts[blk._split_device.index] += 1

    for blk in transformer.single_blocks:
        single_counts[blk._split_device.index] += 1

    assert double_counts[0] == 3 and double_counts[1] == 5, \
        f"Custom double split not applied: got {double_counts}"
    assert single_counts[0] == 20 and single_counts[1] == 28, \
        f"Custom single split not applied: got {single_counts}"


# ---------------------------------------------------------------------------
# Splitter Debug Flag Tests
# ---------------------------------------------------------------------------

def test_splitter_debug_flag_toggle(monkeypatch, caplog):
    """set_gpu_splitter_debug should enable/disable logging."""
    # Enable debug
    splitter.set_gpu_splitter_debug(True)
    assert splitter._gpu_splitter_debug is True

    # Disable debug
    splitter.set_gpu_splitter_debug(False)
    assert splitter._gpu_splitter_debug is False


def test_splitter_log_debug_respects_flag(monkeypatch, caplog):
    """_log_debug should only log when _gpu_splitter_debug is True."""
    splitter._gpu_splitter_debug = False
    splitter._debug_state_logged = False

    with caplog.at_level(logging.INFO, logger=splitter.logger.name):
        splitter._log_debug("should not appear")

    assert not any("should not appear" in r.message for r in caplog.records)

    caplog.clear()
    splitter._gpu_splitter_debug = True
    splitter._debug_state_logged = False

    with caplog.at_level(logging.INFO, logger=splitter.logger.name):
        splitter._log_debug("should appear")

    assert any("should appear" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Flux2Model Tests
# ---------------------------------------------------------------------------

def test_output_device_enforced(monkeypatch):
    """
    When output_device is set in config, load_model should enforce device_torch.
    Heavy dependencies are stubbed.
    """
    captured = {}

    def fake_add_model_gpu_splitter_to_flux2(model, **kwargs):
        captured["output_device"] = kwargs.get("output_device")

    monkeypatch.setattr(flux2_model, "add_model_gpu_splitter_to_flux2", fake_add_model_gpu_splitter_to_flux2)
    _stub_flux2_dependencies(monkeypatch)

    cfg = flux2_model.ModelConfig(
        name_or_path="/dev/null",
        split_model_over_gpus=True,
        gpu_split_double=[4, 4],
        gpu_split_single=[24, 24],
        use_stream_transfers=False,
        output_device="cuda:3",
        arch="flux2",
    )
    model = flux2_model.Flux2Model(device=torch.device("cuda:0"), model_config=cfg, dtype="bf16")
    model.load_model()
    assert captured["output_device"] == torch.device("cuda:0")


def test_incompatible_split_options_warns(monkeypatch, capsys):
    """Splitting with incompatible flags should emit warnings (stdout)."""
    captured = {}

    def fake_add_model_gpu_splitter_to_flux2(model, **kwargs):
        captured["output_device"] = kwargs.get("output_device")

    monkeypatch.setattr(flux2_model, "add_model_gpu_splitter_to_flux2", fake_add_model_gpu_splitter_to_flux2)
    _stub_flux2_dependencies(monkeypatch)

    cfg = flux2_model.ModelConfig(
        name_or_path="/dev/null",
        split_model_over_gpus=True,
        gpu_split_double=[4, 4],
        gpu_split_single=[24, 24],
        use_stream_transfers=False,
        output_device="cuda:0",
        arch="flux2",
        low_vram=True,
        layer_offloading=True,
        quantize=True,
    )
    model = flux2_model.Flux2Model(device=torch.device("cuda:0"), model_config=cfg, dtype="bf16")
    model.load_model()
    out = capsys.readouterr().out
    assert "low_vram is not compatible" in out
    assert "layer_offloading is not compatible" in out
    assert "quantize is not compatible" in out
    assert model.model_config.low_vram is False
    assert model.model_config.layer_offloading is False
    assert model.model_config.quantize is False


def test_invalid_split_sum_raises_valueerror(monkeypatch):
    """Split sums that don't match block counts should raise ValueError."""
    transformer = _build_flux2_transformer()  # 8 double, 48 single

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    # Double split sums to 7, not 8
    with pytest.raises(ValueError, match="gpu_split_double sums to 7"):
        splitter.add_model_gpu_splitter_to_flux2(
            transformer,
            gpu_split_double=[3, 4],  # Sum = 7, should be 8
            gpu_split_single=[24, 24],  # Sum = 48, correct
            use_stream_transfers=False,
            sync_per_block=False,
            output_device=torch.device("cuda:0"),
        )

    # Single split sums to 47, not 48
    transformer2 = _build_flux2_transformer()
    with pytest.raises(ValueError, match="gpu_split_single sums to 47"):
        splitter.add_model_gpu_splitter_to_flux2(
            transformer2,
            gpu_split_double=[4, 4],  # Sum = 8, correct
            gpu_split_single=[23, 24],  # Sum = 47, should be 48
            use_stream_transfers=False,
            sync_per_block=False,
            output_device=torch.device("cuda:0"),
        )


def test_invalid_split_length_raises_valueerror(monkeypatch):
    """Split arrays with wrong length (mismatched GPU count) should raise ValueError."""
    transformer = _build_flux2_transformer()

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    # Double split has 1 element but 2 GPUs available
    with pytest.raises(ValueError, match="gpu_split_double has 1 elements but 2 GPUs"):
        splitter.add_model_gpu_splitter_to_flux2(
            transformer,
            gpu_split_double=[8],  # Wrong length: 1 instead of 2
            gpu_split_single=[24, 24],
            use_stream_transfers=False,
            sync_per_block=False,
            output_device=torch.device("cuda:0"),
        )

    # Single split has 3 elements but 2 GPUs available
    transformer2 = _build_flux2_transformer()
    with pytest.raises(ValueError, match="gpu_split_single has 3 elements but 2 GPUs"):
        splitter.add_model_gpu_splitter_to_flux2(
            transformer2,
            gpu_split_double=[4, 4],
            gpu_split_single=[16, 16, 16],  # Wrong length: 3 instead of 2
            use_stream_transfers=False,
            sync_per_block=False,
            output_device=torch.device("cuda:0"),
        )


def test_deterministic_split_sets_output_device_on_last_block(monkeypatch):
    """Deterministic split should set _split_output_device on the last single block."""
    transformer = _build_flux2_transformer()

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    output_dev = torch.device("cuda:0")
    splitter.add_model_gpu_splitter_to_flux2(
        transformer,
        gpu_split_double=[4, 4],
        gpu_split_single=[24, 24],
        use_stream_transfers=False,
        sync_per_block=False,
        output_device=output_dev,
    )

    # The last single block should have _split_output_device set
    last_block = transformer.single_blocks[-1]
    assert hasattr(last_block, "_split_output_device"), \
        "Last single block should have _split_output_device attribute"
    assert last_block._split_output_device == output_dev, \
        f"_split_output_device should be {output_dev}, got {last_block._split_output_device}"


def test_robust_dummy_block_survives_split_device_deletion():
    """
    RobustDummyBlock is a test helper that gracefully handles _split_device deletion.

    This test documents its purpose: it provides a fallback when _split_device is
    deleted (e.g., by _reset_block), returning a default device instead of raising
    AttributeError. This was used historically to work around a production bug that
    has since been fixed. The helper is retained for potential future resilience
    testing scenarios.
    """
    transformer = _build_dummy_transformer(num_double=2, num_single=2, robust_split_device=True)
    block = transformer.double_blocks[0]

    # Initially has _split_device
    assert block._split_device == torch.device("cuda:0")

    # Delete it (simulating what _reset_block does)
    del block._split_device

    # RobustDummyBlock should return a default instead of raising AttributeError
    # This is the key difference from regular DummyBlock
    assert block._split_device == torch.device("cuda:0"), \
        "RobustDummyBlock should return default device after _split_device deletion"

    # Forward should still work
    x = torch.randn(1, 2, device="cuda:0")
    out = block(x)
    assert out.device == torch.device("cuda:0")
