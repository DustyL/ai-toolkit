#!/usr/bin/env python3
"""
AI-Toolkit LyCORIS Advanced Features Patch Script

This script patches the current ai-toolkit installation with advanced LyCORIS features
from the user's fork, including:
- LoHA (Hadamard product) network type
- Enhanced LoKr with DoRA integration
- Advanced NetworkConfig parameters (weight_decompose, use_scalar, rs_lora, etc.)
- BOFT and Diag-OFT support (requires lycoris-lora package)

Usage:
    python patch_lycoris_advanced.py [--dry-run] [--backup]

Options:
    --dry-run   Show what would be changed without making changes
    --backup    Create backup of modified files before patching

Author: Claude Code (ported from user's ai-toolkit fork)
"""

import os
import sys
import shutil
import argparse
from datetime import datetime
from pathlib import Path

# Paths
AI_TOOLKIT_ROOT = Path("/root/ai-toolkit")
FORK_ROOT = Path("/root/ai-toolkit-fork-review")
BACKUP_DIR = AI_TOOLKIT_ROOT / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Files to copy directly
FILES_TO_COPY = [
    ("toolkit/models/loha.py", "toolkit/models/loha.py"),
]

def log(msg, level="INFO"):
    """Print log message with level prefix."""
    colors = {
        "INFO": "\033[94m",
        "OK": "\033[92m",
        "WARN": "\033[93m",
        "ERR": "\033[91m",
        "END": "\033[0m"
    }
    print(f"{colors.get(level, '')}{level}: {msg}{colors['END']}")


def backup_file(filepath, backup_dir):
    """Create backup of a file."""
    if not filepath.exists():
        return
    rel_path = filepath.relative_to(AI_TOOLKIT_ROOT)
    backup_path = backup_dir / rel_path
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(filepath, backup_path)
    log(f"Backed up: {rel_path}", "OK")


def copy_new_files(dry_run=False, backup_dir=None):
    """Copy new files from fork to ai-toolkit."""
    log("=" * 60)
    log("STEP 1: Copying new files")
    log("=" * 60)

    for src_rel, dst_rel in FILES_TO_COPY:
        src = FORK_ROOT / src_rel
        dst = AI_TOOLKIT_ROOT / dst_rel

        if not src.exists():
            log(f"Source file not found: {src}", "ERR")
            continue

        if dry_run:
            log(f"Would copy: {src_rel} -> {dst_rel}", "INFO")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if backup_dir and dst.exists():
                backup_file(dst, backup_dir)
            shutil.copy2(src, dst)
            log(f"Copied: {src_rel} -> {dst_rel}", "OK")


def patch_config_modules(dry_run=False, backup_dir=None):
    """Patch config_modules.py with advanced parameters."""
    log("=" * 60)
    log("STEP 2: Patching config_modules.py")
    log("=" * 60)

    filepath = AI_TOOLKIT_ROOT / "toolkit/config_modules.py"
    if not filepath.exists():
        log(f"File not found: {filepath}", "ERR")
        return False

    if backup_dir:
        backup_file(filepath, backup_dir)

    content = filepath.read_text()
    original_content = content

    # Patch 1: Update NetworkType literal
    old_network_type = "NetworkType = Literal['lora', 'locon', 'lorm', 'lokr']"
    new_network_type = "NetworkType = Literal['lora', 'locon', 'lorm', 'lokr', 'loha', 'dora', 'boft', 'diag-oft']"

    if old_network_type in content:
        content = content.replace(old_network_type, new_network_type)
        log("Updated NetworkType literal to include loha, dora, boft, diag-oft", "OK")
    elif new_network_type in content:
        log("NetworkType literal already patched", "INFO")
    else:
        log("Could not find NetworkType literal to patch", "WARN")

    # Patch 2: Add advanced LyCORIS parameters after lokr_factor
    advanced_params = '''
        # Advanced LyCORIS parameters
        # DoRA-style weight decomposition (works with lokr, loha)
        self.weight_decompose = kwargs.get('weight_decompose', False)
        # Direction of weight decomposition (True = output, False = input)
        self.wd_on_out = kwargs.get('wd_on_out', True)
        # Trainable scalar for weight difference
        self.use_scalar = kwargs.get('use_scalar', False)
        # Tucker decomposition for conv layers (loha, lokr)
        self.use_tucker = kwargs.get('use_tucker', False)
        # Rank-stabilized LoRA scaling (scale by sqrt(rank) instead of rank)
        self.rs_lora = kwargs.get('rs_lora', False)
        # Decompose both matrices in LoKr
        self.decompose_both = kwargs.get('decompose_both', False)
        # Scale rank dropout by mean
        self.rank_dropout_scale = kwargs.get('rank_dropout_scale', False)
        # Bypass mode for quantized models (Y = WX + ΔWX instead of Y = (W+ΔW)X)
        self.bypass_mode = kwargs.get('bypass_mode', None)
        # Unbalanced factorization for LoKr
        self.unbalanced_factorization = kwargs.get('unbalanced_factorization', False)
        # OFT constraint (regularization strength)
        self.constraint = kwargs.get('constraint', 0.0)
        # OFT rescaled (learnable rescaling)
        self.rescaled = kwargs.get('rescaled', False)
'''

    # Find insertion point (after lokr_factor)
    marker = "self.lokr_factor = kwargs.get('lokr_factor', -1)"
    if marker in content:
        if "self.weight_decompose" not in content:
            content = content.replace(
                marker,
                marker + advanced_params
            )
            log("Added advanced LyCORIS parameters to NetworkConfig", "OK")
        else:
            log("Advanced LyCORIS parameters already present", "INFO")
    else:
        log("Could not find lokr_factor marker for parameter insertion", "WARN")

    if dry_run:
        if content != original_content:
            log("Would modify config_modules.py", "INFO")
    else:
        if content != original_content:
            filepath.write_text(content)
            log("Saved config_modules.py", "OK")

    return True


def patch_lora_special(dry_run=False, backup_dir=None):
    """Patch lora_special.py with LoHA and enhanced module handling."""
    log("=" * 60)
    log("STEP 3: Patching lora_special.py")
    log("=" * 60)

    filepath = AI_TOOLKIT_ROOT / "toolkit/lora_special.py"
    if not filepath.exists():
        log(f"File not found: {filepath}", "ERR")
        return False

    if backup_dir:
        backup_file(filepath, backup_dir)

    content = filepath.read_text()
    original_content = content

    # Patch 1: Add LohaModule import
    old_import = "from toolkit.models.lokr import LokrModule"
    new_import = """from toolkit.models.lokr import LokrModule
from toolkit.models.loha import LohaModule"""

    if "from toolkit.models.loha import LohaModule" not in content:
        content = content.replace(old_import, new_import)
        log("Added LohaModule import", "OK")
    else:
        log("LohaModule import already present", "INFO")

    # Patch 2: Add LyCORIS BOFT/DiagOFT imports
    lycoris_imports = '''
# Import LyCORIS modules for advanced algorithms (BOFT, Diag-OFT)
try:
    from lycoris.modules.boft import ButterflyOFTModule
    from lycoris.modules.diag_oft import DiagOFTModule
    LYCORIS_AVAILABLE = True
except ImportError:
    LYCORIS_AVAILABLE = False
    ButterflyOFTModule = None
    DiagOFTModule = None
'''

    if "LYCORIS_AVAILABLE" not in content:
        # Insert after TYPE_CHECKING block
        marker = "if TYPE_CHECKING:"
        marker_end = "from toolkit.stable_diffusion_model import StableDiffusion"
        if marker in content and marker_end in content:
            idx = content.find(marker_end)
            if idx != -1:
                end_idx = content.find("\n", idx) + 1
                content = content[:end_idx] + lycoris_imports + content[end_idx:]
                log("Added LyCORIS BOFT/DiagOFT imports", "OK")
    else:
        log("LyCORIS imports already present", "INFO")

    # Patch 3: Add module class handling for loha, boft, diag-oft
    old_module_handling = '''        elif self.network_type.lower() == "lokr":
            self.module_class = LokrModule
            module_class = LokrModule
        self.network_config: NetworkConfig = kwargs.get("network_config", None)'''

    new_module_handling = '''        elif self.network_type.lower() == "lokr":
            self.module_class = LokrModule
            module_class = LokrModule
        elif self.network_type.lower() == "loha":
            self.module_class = LohaModule
            module_class = LohaModule
        elif self.network_type.lower() == "boft":
            if not LYCORIS_AVAILABLE or ButterflyOFTModule is None:
                raise ImportError("LyCORIS is required for BOFT. Install with: pip install lycoris-lora")
            self.module_class = ButterflyOFTModule
            module_class = ButterflyOFTModule
        elif self.network_type.lower() == "diag-oft":
            if not LYCORIS_AVAILABLE or DiagOFTModule is None:
                raise ImportError("LyCORIS is required for Diag-OFT. Install with: pip install lycoris-lora")
            self.module_class = DiagOFTModule
            module_class = DiagOFTModule
        self.network_config: NetworkConfig = kwargs.get("network_config", None)'''

    if 'elif self.network_type.lower() == "loha":' not in content:
        content = content.replace(old_module_handling, new_module_handling)
        log("Added module class handling for loha, boft, diag-oft", "OK")
    else:
        log("Module class handling already patched", "INFO")

    # Patch 4: Update peft_format condition to exclude loha
    old_peft = '''            if self.network_type.lower() != "lokr":
                self.peft_format = True'''
    new_peft = '''            if self.network_type.lower() not in ("lokr", "loha"):
                self.peft_format = True'''

    if old_peft in content:
        content = content.replace(old_peft, new_peft)
        log("Updated peft_format condition to exclude loha", "OK")
    elif 'not in ("lokr", "loha")' in content:
        log("peft_format condition already patched", "INFO")

    # Patch 5: Add module_kwargs for lokr advanced params
    old_lokr_kwargs = '''                            if self.network_type.lower() == "lokr":
                                module_kwargs["factor"] = self.network_config.lokr_factor'''

    new_lokr_kwargs = '''                            if self.network_type.lower() == "lokr":
                                module_kwargs["factor"] = self.network_config.lokr_factor
                                module_kwargs["decompose_both"] = self.network_config.decompose_both
                                module_kwargs["weight_decompose"] = self.network_config.weight_decompose
                                module_kwargs["wd_on_out"] = self.network_config.wd_on_out
                                module_kwargs["use_scalar"] = self.network_config.use_scalar
                                module_kwargs["rs_lora"] = self.network_config.rs_lora
                                module_kwargs["unbalanced_factorization"] = self.network_config.unbalanced_factorization
                                if self.network_config.use_tucker:
                                    module_kwargs["use_tucker"] = True

                            elif self.network_type.lower() == "loha":
                                # LoHa-specific parameters
                                module_kwargs["use_tucker"] = self.network_config.use_tucker
                                module_kwargs["use_scalar"] = self.network_config.use_scalar
                                module_kwargs["weight_decompose"] = self.network_config.weight_decompose
                                module_kwargs["wd_on_out"] = self.network_config.wd_on_out
                                module_kwargs["rs_lora"] = self.network_config.rs_lora
                                module_kwargs["rank_dropout_scale"] = self.network_config.rank_dropout_scale
                                if self.network_config.bypass_mode is not None:
                                    module_kwargs["bypass_mode"] = self.network_config.bypass_mode

                            elif self.network_type.lower() in ["boft", "diag-oft"]:
                                # OFT-specific parameters
                                module_kwargs["constraint"] = self.network_config.constraint
                                module_kwargs["rescaled"] = self.network_config.rescaled'''

    if 'module_kwargs["decompose_both"]' not in content:
        content = content.replace(old_lokr_kwargs, new_lokr_kwargs)
        log("Added advanced module_kwargs for lokr, loha, boft, diag-oft", "OK")
    else:
        log("Advanced module_kwargs already present", "INFO")

    # Patch 6: Add loha to lora_shape_dict handling
    old_shape_dict = '''                            if self.network_type.lower() == "lokr":
                                try:
                                    lora_shape_dict[lora_name] = [list(lora.lokr_w1.weight.shape), list(lora.lokr_w2.weight.shape)]
                                except:
                                    pass
                            else:'''

    new_shape_dict = '''                            if self.network_type.lower() == "lokr":
                                try:
                                    lora_shape_dict[lora_name] = [list(lora.lokr_w1.weight.shape), list(lora.lokr_w2.weight.shape)]
                                except:
                                    pass
                            elif self.network_type.lower() == "loha":
                                try:
                                    lora_shape_dict[lora_name] = [list(lora.hada_w1_a.shape), list(lora.hada_w1_b.shape)]
                                except:
                                    pass
                            else:'''

    if 'elif self.network_type.lower() == "loha":\n                                try:\n                                    lora_shape_dict' not in content:
        content = content.replace(old_shape_dict, new_shape_dict)
        log("Added loha shape dict handling", "OK")
    else:
        log("LoHa shape dict handling already present", "INFO")

    if dry_run:
        if content != original_content:
            log("Would modify lora_special.py", "INFO")
    else:
        if content != original_content:
            filepath.write_text(content)
            log("Saved lora_special.py", "OK")

    return True


def patch_network_mixins(dry_run=False, backup_dir=None):
    """Patch network_mixins.py with LohaModule handling."""
    log("=" * 60)
    log("STEP 4: Patching network_mixins.py")
    log("=" * 60)

    filepath = AI_TOOLKIT_ROOT / "toolkit/network_mixins.py"
    if not filepath.exists():
        log(f"File not found: {filepath}", "ERR")
        return False

    if backup_dir:
        backup_file(filepath, backup_dir)

    content = filepath.read_text()
    original_content = content

    # Patch 1: Add LohaModule to Module type union
    old_module_union = "Module = Union['LoConSpecialModule', 'LoRAModule', 'DoRAModule']"
    new_module_union = "Module = Union['LoConSpecialModule', 'LoRAModule', 'DoRAModule', 'LohaModule']"

    if old_module_union in content:
        content = content.replace(old_module_union, new_module_union)
        log("Added LohaModule to Module type union", "OK")
    elif "LohaModule" in content and "Module = Union" in content:
        log("LohaModule already in Module union", "INFO")

    # Patch 2: Add LohaModule import in TYPE_CHECKING
    if "from toolkit.models.loha import LohaModule" not in content:
        old_dora_import = "from toolkit.models.DoRA import DoRAModule"
        new_imports = """from toolkit.models.DoRA import DoRAModule
    from toolkit.models.loha import LohaModule"""
        content = content.replace(old_dora_import, new_imports)
        log("Added LohaModule import to TYPE_CHECKING block", "OK")
    else:
        log("LohaModule import already present", "INFO")

    # Patch 3: Add LohaModule forward handling (uses its own _call_forward)
    old_lokr_forward = '''        if self.__class__.__name__ == "LokrModule":
            return self._call_forward(x)'''

    new_forward_handling = '''        if self.__class__.__name__ == "LokrModule":
            return self._call_forward(x)

        if self.__class__.__name__ == "LohaModule":
            return self._call_forward(x)'''

    if 'if self.__class__.__name__ == "LohaModule":' not in content:
        content = content.replace(old_lokr_forward, new_forward_handling)
        log("Added LohaModule forward handling", "OK")
    else:
        log("LohaModule forward handling already present", "INFO")

    if dry_run:
        if content != original_content:
            log("Would modify network_mixins.py", "INFO")
    else:
        if content != original_content:
            filepath.write_text(content)
            log("Saved network_mixins.py", "OK")

    return True


def verify_patch():
    """Verify the patch was applied correctly."""
    log("=" * 60)
    log("STEP 5: Verifying patch")
    log("=" * 60)

    errors = []

    # Check loha.py exists
    loha_path = AI_TOOLKIT_ROOT / "toolkit/models/loha.py"
    if loha_path.exists():
        log("loha.py exists", "OK")
    else:
        errors.append("loha.py not found")
        log("loha.py not found", "ERR")

    # Check config_modules.py
    config_path = AI_TOOLKIT_ROOT / "toolkit/config_modules.py"
    if config_path.exists():
        content = config_path.read_text()
        if "'loha'" in content:
            log("NetworkType includes 'loha'", "OK")
        else:
            errors.append("NetworkType missing 'loha'")
        if "self.weight_decompose" in content:
            log("Advanced parameters present in NetworkConfig", "OK")
        else:
            errors.append("Advanced parameters missing")

    # Check lora_special.py
    lora_path = AI_TOOLKIT_ROOT / "toolkit/lora_special.py"
    if lora_path.exists():
        content = lora_path.read_text()
        if "from toolkit.models.loha import LohaModule" in content:
            log("LohaModule import present", "OK")
        else:
            errors.append("LohaModule import missing")
        if 'elif self.network_type.lower() == "loha":' in content:
            log("LoHA module class handling present", "OK")
        else:
            errors.append("LoHA module handling missing")

    # Check network_mixins.py
    mixins_path = AI_TOOLKIT_ROOT / "toolkit/network_mixins.py"
    if mixins_path.exists():
        content = mixins_path.read_text()
        if "LohaModule" in content:
            log("LohaModule referenced in network_mixins.py", "OK")
        else:
            errors.append("LohaModule not in network_mixins.py")

    if errors:
        log(f"Verification found {len(errors)} issue(s)", "WARN")
        for err in errors:
            log(f"  - {err}", "ERR")
        return False
    else:
        log("All verifications passed!", "OK")
        return True


def print_usage_examples():
    """Print usage examples for the new features."""
    print("""
================================================================================
                         PATCH APPLIED SUCCESSFULLY!
================================================================================

NEW NETWORK TYPES AVAILABLE:
  - loha    : LoHa (Hadamard product) - good for faces/personas
  - dora    : DoRA (Weight-Decomposed LoRA) - already existed
  - boft    : Butterfly OFT (requires: pip install lycoris-lora)
  - diag-oft: Diagonal OFT (requires: pip install lycoris-lora)

EXAMPLE CONFIG - LoHa with DoRA-style weight decomposition:
--------------------------------------------------------------------------------
network:
  type: loha
  linear: 128
  linear_alpha: 64
  dropout: 0.0
  # LoHa-specific options:
  use_tucker: false          # Tucker decomposition for conv layers
  use_scalar: true           # Trainable scalar for weight diff
  weight_decompose: true     # DoRA-style magnitude/direction separation
  wd_on_out: true            # Decomposition direction (True=output dim)
  rs_lora: true              # Rank-stabilized scaling
  rank_dropout_scale: false  # Scale dropout by mean
  network_kwargs:
    rank_dropout: 0.02
--------------------------------------------------------------------------------

EXAMPLE CONFIG - Enhanced LoKr with all options:
--------------------------------------------------------------------------------
network:
  type: lokr
  linear: 128
  linear_alpha: 64
  lokr_factor: -1            # Auto-optimal factorization
  decompose_both: true       # Decompose both matrices
  weight_decompose: true     # DoRA-style decomposition
  wd_on_out: true
  use_scalar: true
  rs_lora: true
  use_tucker: false
  unbalanced_factorization: false
  network_kwargs:
    rank_dropout: 0.02
--------------------------------------------------------------------------------

For FLUX.2 persona training, recommended configs are in:
  /root/ai-toolkit/config/flux2-dlay-*.yaml
""")


def main():
    parser = argparse.ArgumentParser(
        description="Patch ai-toolkit with advanced LyCORIS features from fork"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of modified files before patching"
    )
    args = parser.parse_args()

    log("=" * 60)
    log("AI-Toolkit LyCORIS Advanced Features Patch")
    log("=" * 60)

    if args.dry_run:
        log("DRY RUN MODE - No changes will be made", "WARN")

    backup_dir = None
    if args.backup and not args.dry_run:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup_dir = BACKUP_DIR
        log(f"Backup directory: {BACKUP_DIR}", "INFO")

    # Check paths exist
    if not AI_TOOLKIT_ROOT.exists():
        log(f"AI-Toolkit not found at {AI_TOOLKIT_ROOT}", "ERR")
        return 1

    if not FORK_ROOT.exists():
        log(f"Fork not found at {FORK_ROOT}", "ERR")
        return 1

    # Run patches
    copy_new_files(args.dry_run, backup_dir)
    patch_config_modules(args.dry_run, backup_dir)
    patch_lora_special(args.dry_run, backup_dir)
    patch_network_mixins(args.dry_run, backup_dir)

    if not args.dry_run:
        if verify_patch():
            print_usage_examples()
            return 0
        else:
            log("Patch completed with warnings - please review", "WARN")
            return 1
    else:
        log("Dry run complete - run without --dry-run to apply changes", "INFO")
        return 0


if __name__ == "__main__":
    sys.exit(main())
