import json

import torch
import pytest
from safetensors.torch import save_file
from safetensors import safe_open


class DummyAlphaScheduler:
    def __init__(self):
        self.enabled = True
        self.current_phase_idx = 0
        self.total_steps = 0
        self.phases = [type("P", (), {"name": "phase0"})()]
        self.transition_history = []
        self._state = {
            "enabled": True,
            "current_phase_idx": 0,
            "total_steps": 10,
            "transition_history": [],
        }

    def state_dict(self):
        return self._state

    def load_state_dict(self, state):
        self._state = state
        self.current_phase_idx = state.get("current_phase_idx", 0)
        self.total_steps = state.get("total_steps", 0)


def test_alpha_scheduler_metadata_fallback(tmp_path):
    # Arrange: save a safetensors file with scheduler metadata only (no sidecar json)
    ckpt_path = tmp_path / "dummy.safetensors"
    scheduler_state = DummyAlphaScheduler().state_dict()
    save_file({"weight": torch.zeros(1)}, str(ckpt_path), metadata={"alpha_scheduler_state_json": json.dumps(scheduler_state)})

    # Act: read metadata and apply to a fresh scheduler
    with safe_open(str(ckpt_path), framework="pt", device="cpu") as f:
        meta = f.metadata()
    sched_json = meta.get("alpha_scheduler_state_json")
    parsed_state = json.loads(sched_json)
    sched = DummyAlphaScheduler()
    sched.load_state_dict(parsed_state)

    # Assert
    assert sched.total_steps == 10
    assert sched.current_phase_idx == 0
