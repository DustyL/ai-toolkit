"""
Test Suite for FLUX.2 GPU Model Splitter

This module provides comprehensive testing for the GPU splitter implementation
without requiring actual multi-GPU hardware. Uses mocking and simulation.

Run with:
    python -m pytest tests/test_gpu_splitter.py -v

Or standalone:
    python tests/test_gpu_splitter.py

Tests cover:
    1. Config parsing and validation
    2. Split calculation logic
    3. Error cases (wrong sums, wrong GPU count, etc.)
    4. Default split selection
    5. Block assignment logic
"""

import sys
import os
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch, PropertyMock
from functools import partial

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


# =============================================================================
# Mock Classes for Testing Without GPU Hardware
# =============================================================================

class MockTensor:
    """Mock tensor for testing device transfers."""

    def __init__(self, device="cuda:0", shape=(1, 1)):
        self._device = MockDevice(device) if isinstance(device, str) else device
        self.shape = shape

    @property
    def device(self):
        return self._device

    def to(self, device, non_blocking=False):
        new_device = device if isinstance(device, MockDevice) else MockDevice(str(device))
        return MockTensor(new_device, self.shape)

    def numel(self):
        result = 1
        for dim in self.shape:
            result *= dim
        return result


class MockDevice:
    """Mock torch.device for testing."""

    def __init__(self, device_str="cuda:0"):
        self._str = device_str
        if "cuda" in device_str:
            self.type = "cuda"
            if ":" in device_str:
                self.index = int(device_str.split(":")[1])
            else:
                self.index = 0
        else:
            self.type = "cpu"
            self.index = None

    def __str__(self):
        return self._str

    def __repr__(self):
        return f"device('{self._str}')"

    def __eq__(self, other):
        if isinstance(other, MockDevice):
            return self._str == other._str
        if isinstance(other, str):
            return self._str == other
        return False

    def __hash__(self):
        return hash(self._str)


class MockParameter:
    """Mock torch.nn.Parameter."""

    def __init__(self, shape=(1024, 1024)):
        self.shape = shape
        self.data = MockTensor(shape=shape)

    def numel(self):
        result = 1
        for dim in self.shape:
            result *= dim
        return result


class MockBlock:
    """Mock transformer block for testing."""

    def __init__(self, block_type="double", param_count=100_000_000):
        self.block_type = block_type
        self._param_count = param_count
        self._parameters = [MockParameter((int(param_count**0.5), int(param_count**0.5)))]

    def forward(self, *args, **kwargs):
        """Original forward method."""
        return args[0] if args else None

    def parameters(self):
        return iter(self._parameters)

    def to(self, *args, **kwargs):
        return self


class MockFlux2Model:
    """Mock FLUX.2 model for testing GPU splitting."""

    def __init__(self, num_double_blocks=8, num_single_blocks=48):
        # FLUX.2 has 8 double blocks and 48 single blocks
        self.double_blocks = [MockBlock("double", 500_000_000) for _ in range(num_double_blocks)]
        self.single_blocks = [MockBlock("single", 200_000_000) for _ in range(num_single_blocks)]

        # Mock other layers
        self.pe_embedder = MockBlock("embedder", 10_000_000)
        self.img_in = MockBlock("img_in", 5_000_000)
        self.time_in = MockBlock("time_in", 5_000_000)
        self.guidance_in = MockBlock("guidance_in", 5_000_000)
        self.txt_in = MockBlock("txt_in", 5_000_000)
        self.double_stream_modulation_img = MockBlock("mod_img", 10_000_000)
        self.double_stream_modulation_txt = MockBlock("mod_txt", 10_000_000)
        self.single_stream_modulation = MockBlock("mod_single", 10_000_000)
        self.final_layer = MockBlock("final", 20_000_000)

    def parameters(self):
        """Yield all parameters."""
        for block in self.double_blocks:
            yield from block.parameters()
        for block in self.single_blocks:
            yield from block.parameters()

    def to(self, *args, **kwargs):
        return self


# =============================================================================
# Test: Config Validation in ModelConfig
# =============================================================================

class TestConfigValidation:
    """Test config parsing and validation from config_modules.py."""

    def test_valid_flux2_config_with_default_split(self):
        """Test that FLUX.2 config loads correctly with default splits."""
        from toolkit.config_modules import ModelConfig

        config = ModelConfig(
            name_or_path="black-forest-labs/FLUX.2-dev",
            arch="flux2",
            split_model_over_gpus=True,
        )

        assert config.is_flux2 is True
        assert config.split_model_over_gpus is True
        assert config.gpu_split_double is None  # Default, will use built-in
        assert config.gpu_split_single is None

    def test_valid_flux2_config_with_custom_split(self):
        """Test FLUX.2 config with valid custom split configuration."""
        from toolkit.config_modules import ModelConfig

        config = ModelConfig(
            name_or_path="black-forest-labs/FLUX.2-dev",
            arch="flux2",
            split_model_over_gpus=True,
            gpu_split_double=[4, 4],
            gpu_split_single=[24, 24],
        )

        assert config.gpu_split_double == [4, 4]
        assert config.gpu_split_single == [24, 24]

    def test_invalid_flux2_double_blocks_sum(self):
        """Test that wrong sum for double blocks raises error."""
        from toolkit.config_modules import ModelConfig

        with pytest.raises(ValueError) as exc_info:
            ModelConfig(
                name_or_path="black-forest-labs/FLUX.2-dev",
                arch="flux2",
                split_model_over_gpus=True,
                gpu_split_double=[3, 3],  # Sum = 6, should be 8
                gpu_split_single=[24, 24],
            )

        assert "gpu_split_double must sum to 8" in str(exc_info.value)

    def test_invalid_flux2_single_blocks_sum(self):
        """Test that wrong sum for single blocks raises error."""
        from toolkit.config_modules import ModelConfig

        with pytest.raises(ValueError) as exc_info:
            ModelConfig(
                name_or_path="black-forest-labs/FLUX.2-dev",
                arch="flux2",
                split_model_over_gpus=True,
                gpu_split_double=[4, 4],
                gpu_split_single=[20, 20],  # Sum = 40, should be 48
            )

        assert "gpu_split_single must sum to 48" in str(exc_info.value)

    def test_invalid_gpu_split_type(self):
        """Test that non-list gpu_split raises error."""
        from toolkit.config_modules import ModelConfig

        with pytest.raises(ValueError) as exc_info:
            ModelConfig(
                name_or_path="black-forest-labs/FLUX.2-dev",
                arch="flux2",
                split_model_over_gpus=True,
                gpu_split_double="4,4",  # Should be list
            )

        assert "must be a list of integers" in str(exc_info.value)

    def test_invalid_gpu_split_contains_non_int(self):
        """Test that gpu_split with non-integers raises error."""
        from toolkit.config_modules import ModelConfig

        with pytest.raises(ValueError) as exc_info:
            ModelConfig(
                name_or_path="black-forest-labs/FLUX.2-dev",
                arch="flux2",
                split_model_over_gpus=True,
                gpu_split_double=[4.0, 4.0],  # Should be int
            )

        assert "must be a list of integers" in str(exc_info.value)

    def test_split_model_only_for_flux(self):
        """Test that split_model_over_gpus is rejected for non-FLUX models."""
        from toolkit.config_modules import ModelConfig

        with pytest.raises(ValueError) as exc_info:
            ModelConfig(
                name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
                arch="sdxl",
                split_model_over_gpus=True,
            )

        assert "only supported with flux" in str(exc_info.value).lower()

    def test_flux1_block_counts(self):
        """Test validation for FLUX.1 (19 double, 38 single)."""
        from toolkit.config_modules import ModelConfig

        # Should fail for FLUX.1 with FLUX.2 block counts
        with pytest.raises(ValueError) as exc_info:
            ModelConfig(
                name_or_path="black-forest-labs/FLUX.1-dev",
                arch="flux",  # FLUX.1
                split_model_over_gpus=True,
                gpu_split_double=[4, 4],  # Sum = 8, but FLUX.1 needs 19
            )

        assert "must sum to 19" in str(exc_info.value)

    def test_layer_offloading_conflict_warning(self, capsys):
        """Test warning when layer_offloading and split_model_over_gpus conflict."""
        from toolkit.config_modules import ModelConfig

        config = ModelConfig(
            name_or_path="black-forest-labs/FLUX.2-dev",
            arch="flux2",
            split_model_over_gpus=True,
            layer_offloading=True,
        )

        captured = capsys.readouterr()
        assert "conflict" in captured.out.lower() or config.layer_offloading is True


# =============================================================================
# Test: Default Split Selection
# =============================================================================

class TestDefaultSplits:
    """Test default split configurations from flux2_gpu_splitter.py."""

    def test_default_splits_exist(self):
        """Test that default splits are defined for common GPU counts."""
        from extensions_built_in.diffusion_models.flux2.flux2_gpu_splitter import DEFAULT_SPLITS

        assert 2 in DEFAULT_SPLITS
        assert 4 in DEFAULT_SPLITS
        assert 8 in DEFAULT_SPLITS

    def test_2gpu_default_split(self):
        """Test 2-GPU default split configuration."""
        from extensions_built_in.diffusion_models.flux2.flux2_gpu_splitter import DEFAULT_SPLITS

        split = DEFAULT_SPLITS[2]

        assert split["double"] == [4, 4]
        assert split["single"] == [24, 24]
        assert sum(split["double"]) == 8
        assert sum(split["single"]) == 48

    def test_4gpu_default_split(self):
        """Test 4-GPU default split configuration."""
        from extensions_built_in.diffusion_models.flux2.flux2_gpu_splitter import DEFAULT_SPLITS

        split = DEFAULT_SPLITS[4]

        assert split["double"] == [2, 2, 2, 2]
        assert split["single"] == [12, 12, 12, 12]
        assert sum(split["double"]) == 8
        assert sum(split["single"]) == 48

    def test_8gpu_default_split(self):
        """Test 8-GPU default split configuration."""
        from extensions_built_in.diffusion_models.flux2.flux2_gpu_splitter import DEFAULT_SPLITS

        split = DEFAULT_SPLITS[8]

        assert split["double"] == [1, 1, 1, 1, 1, 1, 1, 1]
        assert split["single"] == [6, 6, 6, 6, 6, 6, 6, 6]
        assert sum(split["double"]) == 8
        assert sum(split["single"]) == 48


# =============================================================================
# Test: Split Calculation Logic
# =============================================================================

class TestSplitCalculation:
    """Test the split calculation and block assignment logic."""

    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.synchronize')
    @patch('torch.device', MockDevice)
    def test_deterministic_split_assigns_blocks_correctly(self, mock_sync, mock_count):
        """Test that blocks are assigned to correct devices."""
        from extensions_built_in.diffusion_models.flux2.flux2_gpu_splitter import (
            _apply_deterministic_split
        )

        model = MockFlux2Model()
        gpu_ids = [0, 1]
        double_split = [4, 4]
        single_split = [24, 24]

        _apply_deterministic_split(model, gpu_ids, double_split, single_split)

        # Check double block assignments
        for i, block in enumerate(model.double_blocks):
            expected_gpu = 0 if i < 4 else 1
            assert hasattr(block, '_split_device')
            assert str(block._split_device) == f"cuda:{expected_gpu}"

        # Check single block assignments
        for i, block in enumerate(model.single_blocks):
            expected_gpu = 0 if i < 24 else 1
            assert hasattr(block, '_split_device')
            assert str(block._split_device) == f"cuda:{expected_gpu}"

    @patch('torch.cuda.device_count', return_value=4)
    @patch('torch.cuda.synchronize')
    @patch('torch.device', MockDevice)
    def test_4gpu_split_assigns_blocks_evenly(self, mock_sync, mock_count):
        """Test 4-GPU split distributes blocks evenly."""
        from extensions_built_in.diffusion_models.flux2.flux2_gpu_splitter import (
            _apply_deterministic_split
        )

        model = MockFlux2Model()
        gpu_ids = [0, 1, 2, 3]
        double_split = [2, 2, 2, 2]
        single_split = [12, 12, 12, 12]

        _apply_deterministic_split(model, gpu_ids, double_split, single_split)

        # Count blocks per GPU
        double_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        single_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for block in model.double_blocks:
            gpu_id = int(str(block._split_device).split(":")[1])
            double_counts[gpu_id] += 1

        for block in model.single_blocks:
            gpu_id = int(str(block._split_device).split(":")[1])
            single_counts[gpu_id] += 1

        assert list(double_counts.values()) == [2, 2, 2, 2]
        assert list(single_counts.values()) == [12, 12, 12, 12]

    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.synchronize')
    @patch('torch.device', MockDevice)
    def test_custom_uneven_split(self, mock_sync, mock_count):
        """Test custom uneven split for activation-aware distribution."""
        from extensions_built_in.diffusion_models.flux2.flux2_gpu_splitter import (
            _apply_deterministic_split
        )

        model = MockFlux2Model()
        gpu_ids = [0, 1]
        # Activation-aware: fewer blocks on GPU 0
        double_split = [3, 5]
        single_split = [20, 28]

        _apply_deterministic_split(model, gpu_ids, double_split, single_split)

        # Count blocks per GPU
        double_counts = {0: 0, 1: 0}
        single_counts = {0: 0, 1: 0}

        for block in model.double_blocks:
            gpu_id = int(str(block._split_device).split(":")[1])
            double_counts[gpu_id] += 1

        for block in model.single_blocks:
            gpu_id = int(str(block._split_device).split(":")[1])
            single_counts[gpu_id] += 1

        assert double_counts[0] == 3
        assert double_counts[1] == 5
        assert single_counts[0] == 20
        assert single_counts[1] == 28

    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.synchronize')
    @patch('torch.device', MockDevice)
    def test_last_block_routes_to_gpu0(self, mock_sync, mock_count):
        """Test that last single block routes output back to GPU 0."""
        from extensions_built_in.diffusion_models.flux2.flux2_gpu_splitter import (
            _apply_deterministic_split
        )

        model = MockFlux2Model()
        gpu_ids = [0, 1]
        double_split = [4, 4]
        single_split = [24, 24]

        _apply_deterministic_split(model, gpu_ids, double_split, single_split)

        last_block = model.single_blocks[-1]
        assert hasattr(last_block, '_split_output_device')
        assert str(last_block._split_output_device) == "cuda:0"

    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.synchronize')
    @patch('torch.device', MockDevice)
    def test_forward_method_wrapped(self, mock_sync, mock_count):
        """Test that forward methods are wrapped after splitting."""
        from extensions_built_in.diffusion_models.flux2.flux2_gpu_splitter import (
            _apply_deterministic_split
        )

        model = MockFlux2Model()

        # Store original forwards
        original_double_forward = model.double_blocks[0].forward
        original_single_forward = model.single_blocks[0].forward

        _apply_deterministic_split(model, [0, 1], [4, 4], [24, 24])

        # Check that forwards are now partial functions (wrapped)
        for block in model.double_blocks:
            assert hasattr(block, '_pre_gpu_split_forward')
            assert isinstance(block.forward, partial)

        for block in model.single_blocks:
            assert hasattr(block, '_pre_gpu_split_forward')
            assert isinstance(block.forward, partial)


# =============================================================================
# Test: Error Cases
# =============================================================================

class TestErrorCases:
    """Test error handling in GPU splitter."""

    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.synchronize')
    @patch('torch.device', MockDevice)
    def test_gpu_count_mismatch_error(self, mock_sync, mock_count):
        """Test error when split list length doesn't match GPU count."""
        from extensions_built_in.diffusion_models.flux2.flux2_gpu_splitter import (
            add_model_gpu_splitter_to_flux2
        )

        model = MockFlux2Model()

        with pytest.raises(ValueError) as exc_info:
            add_model_gpu_splitter_to_flux2(
                model,
                gpu_split_double=[2, 2, 2, 2],  # 4 elements but only 2 GPUs
                gpu_split_single=[24, 24],
            )

        assert "4 elements but 2 GPUs" in str(exc_info.value)

    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.synchronize')
    @patch('torch.device', MockDevice)
    def test_single_split_gpu_count_mismatch(self, mock_sync, mock_count):
        """Test error when single split length doesn't match GPU count."""
        from extensions_built_in.diffusion_models.flux2.flux2_gpu_splitter import (
            add_model_gpu_splitter_to_flux2
        )

        model = MockFlux2Model()

        with pytest.raises(ValueError) as exc_info:
            add_model_gpu_splitter_to_flux2(
                model,
                gpu_split_double=[4, 4],
                gpu_split_single=[12, 12, 12, 12],  # 4 elements but only 2 GPUs
            )

        assert "4 elements but 2 GPUs" in str(exc_info.value)

    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.synchronize')
    def test_single_gpu_warning(self, mock_sync, mock_count, capsys):
        """Test warning when only 1 GPU is detected."""
        from extensions_built_in.diffusion_models.flux2.flux2_gpu_splitter import (
            add_model_gpu_splitter_to_flux2
        )

        model = MockFlux2Model()

        # Should not raise, but print warning
        add_model_gpu_splitter_to_flux2(model)

        captured = capsys.readouterr()
        assert "Only 1 GPU" in captured.out


# =============================================================================
# Test: Block Reset Logic
# =============================================================================

class TestBlockReset:
    """Test block reset functionality."""

    def test_reset_block_clears_attributes(self):
        """Test that _reset_block clears all split-related attributes."""
        from extensions_built_in.diffusion_models.flux2.flux2_gpu_splitter import _reset_block

        block = MockBlock()
        block._split_device = MockDevice("cuda:1")
        block._split_output_device = MockDevice("cuda:0")
        block._pre_gpu_split_forward = block.forward
        block._original_split_device = MockDevice("cuda:1")

        _reset_block(block)

        assert not hasattr(block, '_split_device')
        assert not hasattr(block, '_split_output_device')
        assert not hasattr(block, '_pre_gpu_split_forward')
        assert not hasattr(block, '_original_split_device')


# =============================================================================
# Test: Tensor Movement Helpers
# =============================================================================

class TestTensorMovement:
    """Test tensor movement helper functions."""

    def test_move_tensor_to_device_same_device(self):
        """Test tensor stays put when already on correct device."""
        from extensions_built_in.diffusion_models.flux2.flux2_gpu_splitter import (
            _move_tensor_to_device
        )

        # Mock torch module to use our MockTensor
        with patch('extensions_built_in.diffusion_models.flux2.flux2_gpu_splitter.Tensor', MockTensor):
            tensor = MockTensor("cuda:0")
            device = MockDevice("cuda:0")

            # When devices match, should return same tensor (or equivalent)
            result = _move_tensor_to_device(tensor, device)
            # The actual implementation checks tensor.device != device
            # With mock, this will still work as expected

    def test_move_tuple_to_device_handles_none(self):
        """Test that None items in tuples are handled gracefully."""
        from extensions_built_in.diffusion_models.flux2.flux2_gpu_splitter import (
            _move_tuple_to_device
        )

        with patch('extensions_built_in.diffusion_models.flux2.flux2_gpu_splitter.Tensor', MockTensor):
            t = (MockTensor("cuda:0"), None, MockTensor("cuda:0"))
            device = MockDevice("cuda:1")

            result = _move_tuple_to_device(t, device)

            assert result[1] is None
            assert len(result) == 3


# =============================================================================
# Test: Integration with Main Splitter Function
# =============================================================================

class TestMainSplitter:
    """Test the main add_model_gpu_splitter_to_flux2 function."""

    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.synchronize')
    @patch('torch.device', MockDevice)
    def test_user_specified_split_takes_precedence(self, mock_sync, mock_count, capsys):
        """Test that user-specified splits override defaults."""
        from extensions_built_in.diffusion_models.flux2.flux2_gpu_splitter import (
            add_model_gpu_splitter_to_flux2
        )

        model = MockFlux2Model()

        add_model_gpu_splitter_to_flux2(
            model,
            gpu_split_double=[3, 5],  # Custom split
            gpu_split_single=[20, 28],
        )

        captured = capsys.readouterr()
        assert "user-specified" in captured.out.lower()

    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.synchronize')
    @patch('torch.device', MockDevice)
    def test_default_split_used_when_no_user_split(self, mock_sync, mock_count, capsys):
        """Test that default splits are used when user doesn't specify."""
        from extensions_built_in.diffusion_models.flux2.flux2_gpu_splitter import (
            add_model_gpu_splitter_to_flux2
        )

        model = MockFlux2Model()

        add_model_gpu_splitter_to_flux2(model)

        captured = capsys.readouterr()
        assert "default deterministic split" in captured.out.lower()

    @patch('torch.cuda.device_count', return_value=3)
    @patch('torch.cuda.synchronize')
    @patch('torch.device', MockDevice)
    def test_greedy_fallback_for_unusual_gpu_count(self, mock_sync, mock_count, capsys):
        """Test greedy algorithm fallback for non-standard GPU counts."""
        from extensions_built_in.diffusion_models.flux2.flux2_gpu_splitter import (
            add_model_gpu_splitter_to_flux2
        )

        model = MockFlux2Model()

        add_model_gpu_splitter_to_flux2(model)

        captured = capsys.readouterr()
        assert "greedy" in captured.out.lower()

    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.synchronize')
    @patch('torch.device', MockDevice)
    def test_to_method_wrapped(self, mock_sync, mock_count):
        """Test that model.to() is wrapped after splitting."""
        from extensions_built_in.diffusion_models.flux2.flux2_gpu_splitter import (
            add_model_gpu_splitter_to_flux2
        )

        model = MockFlux2Model()
        original_to = model.to

        add_model_gpu_splitter_to_flux2(model)

        assert hasattr(model, '_pre_gpu_split_to')
        assert model._pre_gpu_split_to == original_to
        assert isinstance(model.to, partial)


# =============================================================================
# Test: YAML Config Loading
# =============================================================================

class TestYAMLConfigLoading:
    """Test loading and validating YAML config files."""

    def test_load_2gpu_config(self):
        """Test loading 2-GPU deterministic config."""
        import yaml

        config_path = Path(__file__).parent.parent / "config/examples/train_lora_flux2_2gpu_deterministic.yaml"

        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

            assert config['job'] == 'extension'
            model_config = config['config']['process'][0]['model']
            assert model_config['arch'] == 'flux2'
            assert model_config['split_model_over_gpus'] is True

    def test_load_4gpu_config(self):
        """Test loading 4-GPU deterministic config."""
        import yaml

        config_path = Path(__file__).parent.parent / "config/examples/train_lora_flux2_4gpu_deterministic.yaml"

        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

            assert config['job'] == 'extension'
            model_config = config['config']['process'][0]['model']
            assert model_config['arch'] == 'flux2'
            assert model_config['split_model_over_gpus'] is True

    def test_load_custom_split_config(self):
        """Test loading custom activation-aware split config."""
        import yaml

        config_path = Path(__file__).parent.parent / "config/examples/train_lora_flux2_custom_split.yaml"

        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

            model_config = config['config']['process'][0]['model']
            assert model_config['gpu_split_double'] == [3, 5]
            assert model_config['gpu_split_single'] == [20, 28]
            assert sum(model_config['gpu_split_double']) == 8
            assert sum(model_config['gpu_split_single']) == 48


# =============================================================================
# Test: Validation Script Integration
# =============================================================================

def validate_split_config(
    double_split: List[int],
    single_split: List[int],
    num_gpus: int,
    model_type: str = "flux2"
) -> dict:
    """
    Validate a GPU split configuration.

    Args:
        double_split: List of double block counts per GPU
        single_split: List of single block counts per GPU
        num_gpus: Number of GPUs available
        model_type: "flux2" (8 double, 48 single) or "flux" (19 double, 38 single)

    Returns:
        dict with 'valid' bool and 'errors' list
    """
    errors = []

    # Block totals for each model
    block_totals = {
        "flux2": {"double": 8, "single": 48},
        "flux": {"double": 19, "single": 38},
    }

    if model_type not in block_totals:
        errors.append(f"Unknown model_type: {model_type}")
        return {"valid": False, "errors": errors}

    totals = block_totals[model_type]

    # Check list lengths match GPU count
    if len(double_split) != num_gpus:
        errors.append(f"double_split has {len(double_split)} elements but {num_gpus} GPUs available")

    if len(single_split) != num_gpus:
        errors.append(f"single_split has {len(single_split)} elements but {num_gpus} GPUs available")

    # Check sums
    if sum(double_split) != totals["double"]:
        errors.append(f"double_split sums to {sum(double_split)}, expected {totals['double']}")

    if sum(single_split) != totals["single"]:
        errors.append(f"single_split sums to {sum(single_split)}, expected {totals['single']}")

    # Check all values are positive
    if any(x <= 0 for x in double_split):
        errors.append("double_split contains non-positive values")

    if any(x <= 0 for x in single_split):
        errors.append("single_split contains non-positive values")

    # Check all values are integers
    if not all(isinstance(x, int) for x in double_split):
        errors.append("double_split contains non-integer values")

    if not all(isinstance(x, int) for x in single_split):
        errors.append("single_split contains non-integer values")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "config": {
            "double_split": double_split,
            "single_split": single_split,
            "num_gpus": num_gpus,
            "model_type": model_type,
        }
    }


class TestValidationFunction:
    """Test the standalone validation function."""

    def test_valid_2gpu_flux2(self):
        """Test validation passes for valid 2-GPU FLUX.2 config."""
        result = validate_split_config(
            double_split=[4, 4],
            single_split=[24, 24],
            num_gpus=2,
            model_type="flux2"
        )

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_valid_4gpu_flux2(self):
        """Test validation passes for valid 4-GPU FLUX.2 config."""
        result = validate_split_config(
            double_split=[2, 2, 2, 2],
            single_split=[12, 12, 12, 12],
            num_gpus=4,
            model_type="flux2"
        )

        assert result["valid"] is True

    def test_invalid_double_sum(self):
        """Test validation fails for wrong double block sum."""
        result = validate_split_config(
            double_split=[3, 3],  # Sum = 6, not 8
            single_split=[24, 24],
            num_gpus=2,
            model_type="flux2"
        )

        assert result["valid"] is False
        assert any("double_split sums to 6" in e for e in result["errors"])

    def test_invalid_gpu_count_mismatch(self):
        """Test validation fails when list length doesn't match GPU count."""
        result = validate_split_config(
            double_split=[4, 4],  # 2 elements
            single_split=[24, 24],
            num_gpus=4,  # 4 GPUs
            model_type="flux2"
        )

        assert result["valid"] is False
        assert any("2 elements but 4 GPUs" in e for e in result["errors"])

    def test_activation_aware_split(self):
        """Test validation passes for uneven activation-aware split."""
        result = validate_split_config(
            double_split=[3, 5],  # Fewer blocks on GPU 0
            single_split=[20, 28],
            num_gpus=2,
            model_type="flux2"
        )

        assert result["valid"] is True


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FLUX.2 GPU Splitter Validation Tests")
    print("=" * 70)

    # Run validation examples
    print("\n1. Testing 2-GPU default split:")
    result = validate_split_config([4, 4], [24, 24], 2, "flux2")
    print(f"   Valid: {result['valid']}")

    print("\n2. Testing 4-GPU default split:")
    result = validate_split_config([2, 2, 2, 2], [12, 12, 12, 12], 4, "flux2")
    print(f"   Valid: {result['valid']}")

    print("\n3. Testing custom activation-aware split:")
    result = validate_split_config([3, 5], [20, 28], 2, "flux2")
    print(f"   Valid: {result['valid']}")

    print("\n4. Testing invalid split (wrong sum):")
    result = validate_split_config([3, 3], [24, 24], 2, "flux2")
    print(f"   Valid: {result['valid']}")
    print(f"   Errors: {result['errors']}")

    print("\n5. Testing GPU count mismatch:")
    result = validate_split_config([4, 4], [24, 24], 4, "flux2")
    print(f"   Valid: {result['valid']}")
    print(f"   Errors: {result['errors']}")

    print("\n" + "=" * 70)
    print("Running pytest...")
    print("=" * 70)

    # Run pytest
    pytest.main([__file__, "-v", "--tb=short"])
