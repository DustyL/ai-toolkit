import math
import os
from typing import Optional, Union, List, Type

import torch
from lycoris.kohya import LycorisNetwork, LoConModule
from lycoris.modules.glora import GLoRAModule
from torch import nn
from transformers import CLIPTextModel
from torch.nn import functional as F
from toolkit.network_mixins import ToolkitNetworkMixin, ToolkitModuleMixin, ExtractableModuleMixin

# diffusers specific stuff
LINEAR_MODULES = [
    'Linear',
    'LoRACompatibleLinear'
]
CONV_MODULES = [
    'Conv2d',
    'LoRACompatibleConv'
]

class LoConSpecialModule(ToolkitModuleMixin, LoConModule, ExtractableModuleMixin):
    def __init__(
            self,
            lora_name, org_module: nn.Module,
            multiplier=1.0,
            lora_dim=4, alpha=1,
            dropout=0., rank_dropout=0., module_dropout=0.,
            use_cp=False,
            network: 'LycorisSpecialNetwork' = None,
            use_bias=False,
            **kwargs,
    ):
        """ if alpha == 0 or None, alpha is rank (no scaling). """
        # call super of super
        ToolkitModuleMixin.__init__(self, network=network)
        torch.nn.Module.__init__(self)
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.cp = False

        # check if parent has bias. if not force use_bias to False
        if org_module.bias is None:
            use_bias = False

        self.scalar = nn.Parameter(torch.tensor(0.0))
        orig_module_name = org_module.__class__.__name__
        if orig_module_name in CONV_MODULES:
            self.isconv = True
            # For general LoCon
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            out_dim = org_module.out_channels
            self.down_op = F.conv2d
            self.up_op = F.conv2d
            if use_cp and k_size != (1, 1):
                self.lora_down = nn.Conv2d(in_dim, lora_dim, (1, 1), bias=False)
                self.lora_mid = nn.Conv2d(lora_dim, lora_dim, k_size, stride, padding, bias=False)
                self.cp = True
            else:
                self.lora_down = nn.Conv2d(in_dim, lora_dim, k_size, stride, padding, bias=False)
            self.lora_up = nn.Conv2d(lora_dim, out_dim, (1, 1), bias=use_bias)
        elif orig_module_name in LINEAR_MODULES:
            self.isconv = False
            self.down_op = F.linear
            self.up_op = F.linear
            if orig_module_name == 'GroupNorm':
                # RuntimeError: mat1 and mat2 shapes cannot be multiplied (56320x120 and 320x32)
                in_dim = org_module.num_channels
                out_dim = org_module.num_channels
            else:
                in_dim = org_module.in_features
                out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=use_bias)
        else:
            raise NotImplementedError
        self.shape = org_module.weight.shape

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer('alpha', torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.lora_up.weight)
        if self.cp:
            torch.nn.init.kaiming_uniform_(self.lora_mid.weight, a=math.sqrt(5))

        self.multiplier = multiplier
        self.org_module = [org_module]
        self.register_load_state_dict_post_hook(self.load_weight_hook)

    def load_weight_hook(self, *args, **kwargs):
        self.scalar = nn.Parameter(torch.ones_like(self.scalar))


class LycorisSpecialNetwork(ToolkitNetworkMixin, LycorisNetwork):
    UNET_TARGET_REPLACE_MODULE = [
        "Transformer2DModel",
        "ResnetBlock2D",
        "Downsample2D",
        "Upsample2D",
        # 'UNet2DConditionModel',
        # 'Conv2d',
        # 'Timesteps',
        # 'TimestepEmbedding',
        # 'Linear',
        # 'SiLU',
        # 'ModuleList',
        # 'DownBlock2D',
        # 'ResnetBlock2D',  # need
        # 'GroupNorm',
        # 'LoRACompatibleConv',
        # 'LoRACompatibleLinear',
        # 'Dropout',
        # 'CrossAttnDownBlock2D', # needed
        # 'Transformer2DModel',  # maybe not, has duplicates
        # 'BasicTransformerBlock', # duplicates
        # 'LayerNorm',
        # 'Attention',
        # 'FeedForward',
        # 'GEGLU',
        # 'UpBlock2D',
        # 'UNetMidBlock2DCrossAttn'
    ]
    UNET_TARGET_REPLACE_NAME = [
        "conv_in",
        "conv_out",
        "time_embedding.linear_1",
        "time_embedding.linear_2",
    ]
    def __init__(
            self,
            text_encoder: Union[List[CLIPTextModel], CLIPTextModel],
            unet,
            multiplier: float = 1.0,
            lora_dim: int = 4,
            alpha: float = 1,
            dropout: Optional[float] = None,
            rank_dropout: Optional[float] = None,
            module_dropout: Optional[float] = None,
            conv_lora_dim: Optional[int] = None,
            conv_alpha: Optional[float] = None,
            use_cp: Optional[bool] = False,
            network_module: Type[object] = LoConSpecialModule,
            train_unet: bool = True,
            train_text_encoder: bool = True,
            use_text_encoder_1: bool = True,
            use_text_encoder_2: bool = True,
            use_bias: bool = False,
            is_lorm: bool = False,
            **kwargs,
    ) -> None:
        # call ToolkitNetworkMixin super
        ToolkitNetworkMixin.__init__(
            self,
            train_text_encoder=train_text_encoder,
            train_unet=train_unet,
            is_lorm=is_lorm,
            **kwargs
        )
        # call the parent of the parent LycorisNetwork
        torch.nn.Module.__init__(self)

        # LyCORIS unique stuff
        if dropout is None:
            dropout = 0
        if rank_dropout is None:
            rank_dropout = 0
        if module_dropout is None:
            module_dropout = 0
        self.train_unet = train_unet
        self.train_text_encoder = train_text_encoder

        self.torch_multiplier = None
        # triggers a tensor update
        self.multiplier = multiplier
        self.lora_dim = lora_dim

        if not self.ENABLE_CONV or conv_lora_dim is None:
            conv_lora_dim = 0
            conv_alpha = 0

        self.conv_lora_dim = int(conv_lora_dim)
        if self.conv_lora_dim and self.conv_lora_dim != self.lora_dim:
            print('Apply different lora dim for conv layer')
            print(f'Conv Dim: {conv_lora_dim}, Linear Dim: {lora_dim}')
        elif self.conv_lora_dim == 0:
            print('Disable conv layer')

        self.alpha = alpha
        self.conv_alpha = float(conv_alpha)
        if self.conv_lora_dim and self.alpha != self.conv_alpha:
            print('Apply different alpha value for conv layer')
            print(f'Conv alpha: {conv_alpha}, Linear alpha: {alpha}')

        if 1 >= dropout >= 0:
            print(f'Use Dropout value: {dropout}')
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        # create module instances
        def create_modules(
                prefix,
                root_module: torch.nn.Module,
                target_replace_modules,
                target_replace_names=[]
        ) -> List[network_module]:
            print('Create LyCORIS Module')
            loras = []
            created_names = set()  # Track already created LoRA names to avoid duplicates
            skipped_duplicates = 0  # Count how many duplicates were skipped
            # remove this
            named_modules = root_module.named_modules()
            # add a few to tthe generator

            for name, module in named_modules:
                module_name = module.__class__.__name__
                if module_name in target_replace_modules:
                    if module_name in self.MODULE_ALGO_MAP:
                        algo = self.MODULE_ALGO_MAP[module_name]
                    else:
                        algo = network_module
                    for child_name, child_module in module.named_modules():
                        lora_name = prefix + '.' + name + '.' + child_name
                        lora_name = lora_name.replace('.', '_')
                        if lora_name.startswith('lora_unet_input_blocks_1_0_emb_layers_1'):
                            print(f"{lora_name}")

                        if child_module.__class__.__name__ in LINEAR_MODULES and lora_dim > 0:
                            # Skip if we've already created a LoRA for this name
                            if lora_name not in created_names:
                                lora = algo(
                                    lora_name, child_module, self.multiplier,
                                    self.lora_dim, self.alpha,
                                    self.dropout, self.rank_dropout, self.module_dropout,
                                    use_cp,
                                    network=self,
                                    parent=module,
                                    use_bias=use_bias,
                                    **kwargs
                                )
                                created_names.add(lora_name)
                            else:
                                lora = None
                                skipped_duplicates += 1
                        elif child_module.__class__.__name__ in CONV_MODULES:
                            # Skip if we've already created a LoRA for this name
                            if lora_name not in created_names:
                                k_size, *_ = child_module.kernel_size
                                if k_size == 1 and lora_dim > 0:
                                    lora = algo(
                                        lora_name, child_module, self.multiplier,
                                        self.lora_dim, self.alpha,
                                        self.dropout, self.rank_dropout, self.module_dropout,
                                        use_cp,
                                        network=self,
                                        parent=module,
                                        use_bias=use_bias,
                                        **kwargs
                                    )
                                    created_names.add(lora_name)
                                elif conv_lora_dim > 0:
                                    lora = algo(
                                        lora_name, child_module, self.multiplier,
                                        self.conv_lora_dim, self.conv_alpha,
                                        self.dropout, self.rank_dropout, self.module_dropout,
                                        use_cp,
                                        network=self,
                                        parent=module,
                                        use_bias=use_bias,
                                        **kwargs
                                    )
                                    created_names.add(lora_name)
                                else:
                                    continue
                            else:
                                lora = None
                                skipped_duplicates += 1
                        else:
                            continue
                        if lora is not None:
                            loras.append(lora)
                elif name in target_replace_names:
                    if name in self.NAME_ALGO_MAP:
                        algo = self.NAME_ALGO_MAP[name]
                    else:
                        algo = network_module
                    lora_name = prefix + '.' + name
                    lora_name = lora_name.replace('.', '_')
                    if module.__class__.__name__ == 'Linear' and lora_dim > 0:
                        lora = algo(
                            lora_name, module, self.multiplier,
                            self.lora_dim, self.alpha,
                            self.dropout, self.rank_dropout, self.module_dropout,
                            use_cp,
                            parent=module,
                            network=self,
                            use_bias=use_bias,
                            **kwargs
                        )
                    elif module.__class__.__name__ == 'Conv2d':
                        k_size, *_ = module.kernel_size
                        if k_size == 1 and lora_dim > 0:
                            lora = algo(
                                lora_name, module, self.multiplier,
                                self.lora_dim, self.alpha,
                                self.dropout, self.rank_dropout, self.module_dropout,
                                use_cp,
                                network=self,
                                parent=module,
                                use_bias=use_bias,
                                **kwargs
                            )
                        elif conv_lora_dim > 0:
                            lora = algo(
                                lora_name, module, self.multiplier,
                                self.conv_lora_dim, self.conv_alpha,
                                self.dropout, self.rank_dropout, self.module_dropout,
                                use_cp,
                                network=self,
                                parent=module,
                                use_bias=use_bias,
                                **kwargs
                            )
                        else:
                            continue
                    else:
                        continue
                    loras.append(lora)

            # Report if any duplicates were skipped
            if skipped_duplicates > 0:
                print(f"  Skipped {skipped_duplicates} duplicate LoRA module names")

            return loras

        if network_module == GLoRAModule:
            print('GLoRA enabled, only train transformer')
            # only train transformer (for GLoRA)
            LycorisSpecialNetwork.UNET_TARGET_REPLACE_MODULE = [
                "Transformer2DModel",
                "Attention",
            ]
            LycorisSpecialNetwork.UNET_TARGET_REPLACE_NAME = []

        if isinstance(text_encoder, list):
            text_encoders = text_encoder
            use_index = True
        else:
            text_encoders = [text_encoder]
            use_index = False

        self.text_encoder_loras = []
        if self.train_text_encoder:
            for i, te in enumerate(text_encoders):
                if not use_text_encoder_1 and i == 0:
                    continue
                if not use_text_encoder_2 and i == 1:
                    continue
                self.text_encoder_loras.extend(create_modules(
                    LycorisSpecialNetwork.LORA_PREFIX_TEXT_ENCODER + (f'{i + 1}' if use_index else ''),
                    te,
                    LycorisSpecialNetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE
                ))
        print(f"create LyCORIS for Text Encoder: {len(self.text_encoder_loras)} modules.")
        if self.train_unet:
            # Check if this is a Qwen model and adjust target modules accordingly
            unet_class_name = unet.__class__.__name__
            target_modules = LycorisSpecialNetwork.UNET_TARGET_REPLACE_MODULE

            if unet_class_name == "QwenImageTransformer2DModel":
                # For Qwen Image Edit models, target only the modules that directly contain
                # Linear/Conv2d layers to avoid duplicates from nested module matching
                target_modules = [
                    # Primary targets - these should contain Linear/Conv2d directly
                    "Attention",          # Contains to_q, to_k, to_v, to_out Linear layers
                    "FeedForward",       # Contains Linear layers for MLP
                    # Secondary targets if primary don't work
                    "MLP",
                    "FFN",
                    "GEGLU",
                    "GeGLU",
                ]
                print(f"Detected Qwen Image model, using targeted module search to avoid duplicates")

            self.unet_loras = create_modules(LycorisSpecialNetwork.LORA_PREFIX_UNET, unet,
                                             target_modules)

            # If no modules were found with the standard approach, try a fallback
            if len(self.unet_loras) == 0 and unet_class_name == "QwenImageTransformer2DModel":
                print("No modules found with standard targets, attempting automatic discovery...")

                # Collect all module class names that look trainable
                discovered_modules = set()
                for name, module in unet.named_modules():
                    module_class_name = module.__class__.__name__
                    # Look for modules that typically contain parameters
                    if any(keyword in module_class_name.lower() for keyword in
                           ["linear", "conv", "attention", "feedforward", "mlp", "ffn"]):
                        # Check if module has parameters (is trainable)
                        if any(p.requires_grad for p in module.parameters(recurse=False)):
                            discovered_modules.add(module_class_name)

                if discovered_modules:
                    print(f"Discovered potential target modules: {sorted(discovered_modules)}")
                    # Try with the discovered module types
                    self.unet_loras = create_modules(
                        LycorisSpecialNetwork.LORA_PREFIX_UNET,
                        unet,
                        list(discovered_modules)
                    )
                    print(f"Created {len(self.unet_loras)} LyCORIS modules using discovered targets")
        else:
            self.unet_loras = []
        print(f"create LyCORIS for U-Net: {len(self.unet_loras)} modules.")

        # Final warning if no modules were created
        if self.train_unet and len(self.unet_loras) == 0:
            print("\n" + "=" * 60)
            print("WARNING: No LyCORIS modules were created for U-Net!")
            print("This usually means the model architecture is incompatible.")
            print("Debugging tips:")
            print("1. Check that the model is loaded correctly")
            print("2. Verify the model architecture matches your config")
            print("=" * 60 + "\n")

        self.weights_sd = None

        # Check for duplicates - warn instead of failing
        names = set()
        duplicates = []
        for lora in self.text_encoder_loras + self.unet_loras:
            if lora.lora_name in names:
                duplicates.append(lora.lora_name)
            names.add(lora.lora_name)

        if duplicates:
            print(f"Warning: Found {len(duplicates)} duplicate LoRA names (handled gracefully):")
            for dup in duplicates[:5]:  # Show first 5 duplicates
                print(f"   - {dup}")
            if len(duplicates) > 5:
                print(f"   ... and {len(duplicates) - 5} more")
