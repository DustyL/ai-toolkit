import torch


def get_optimizer(
        params,
        optimizer_type='adam',
        learning_rate=1e-6,
        optimizer_params=None
):
    # Defensive import to avoid rare UnboundLocalError reports when torch is only imported at module scope
    import torch as _torch

    if optimizer_params is None:
        optimizer_params = {}
    lower_type = optimizer_type.lower()
    if lower_type.startswith("dadaptation"):
        # dadaptation optimizer does not use standard learning rate. 1 is the default value
        import dadaptation
        print("Using DAdaptAdam optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0
        if lower_type.endswith('lion'):
            optimizer = dadaptation.DAdaptLion(params, eps=1e-6, lr=use_lr, **optimizer_params)
        elif lower_type.endswith('adam'):
            optimizer = dadaptation.DAdaptLion(params, eps=1e-6, lr=use_lr, **optimizer_params)
        elif lower_type == 'dadaptation':
            # backwards compatibility
            optimizer = dadaptation.DAdaptAdam(params, eps=1e-6, lr=use_lr, **optimizer_params)
            # warn user that dadaptation is deprecated
            print("WARNING: Dadaptation optimizer type has been changed to DadaptationAdam. Please update your config.")
    elif lower_type.startswith("prodigy8bit"):
        from toolkit.optimizers.prodigy_8bit import Prodigy8bit
        print("Using Prodigy optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0

        print(f"Using lr {use_lr}")
        # let net be the neural network you want to train
        # you can choose weight decay value based on your problem, 0 by default
        optimizer = Prodigy8bit(params, lr=use_lr, eps=1e-6, **optimizer_params)
    elif lower_type.startswith("prodigy_plus") or lower_type.startswith("prodigyplus"):
        from prodigyplus import ProdigyPlusScheduleFree
        print("Using ProdigyPlusScheduleFree optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # Prodigy uses adaptive lr, so values of 0.1 to 1.0 are typical. default to 1.0
            use_lr = 1.0

        print(f"Using lr {use_lr}")

        # Set sensible defaults for split_groups (enables per-param-group d values)
        if 'split_groups' not in optimizer_params:
            optimizer_params['split_groups'] = True

        # Extract per-expert LR params for MoE support (convert to per-group lr)
        # These will be set by _prepare_moe_optimizer_params in lora_special.py
        high_noise_lr = optimizer_params.pop('high_noise_lr', None)
        low_noise_lr = optimizer_params.pop('low_noise_lr', None)
        # Also remove Automagic-specific params that don't apply to Prodigy
        for key in ['high_noise_lr_bump', 'low_noise_lr_bump',
                    'high_noise_min_lr', 'low_noise_min_lr',
                    'high_noise_max_lr', 'low_noise_max_lr']:
            optimizer_params.pop(key, None)

        optimizer = ProdigyPlusScheduleFree(params, lr=use_lr, **optimizer_params)

        # Add train/eval methods for Schedule-Free mode compatibility
        # These need to be called when switching between training and eval
        if hasattr(optimizer, 'train') and hasattr(optimizer, 'eval'):
            print("  Schedule-Free mode enabled - remember to call optimizer.train()/eval() when switching modes")
    elif lower_type.startswith("prodigy"):
        from prodigyopt import Prodigy

        print("Using Prodigy optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0

        print(f"Using lr {use_lr}")
        # let net be the neural network you want to train
        # you can choose weight decay value based on your problem, 0 by default
        optimizer = Prodigy(params, lr=use_lr, eps=1e-6, **optimizer_params)
    elif lower_type == "adam8":
        from toolkit.optimizers.adam8bit import Adam8bit

        optimizer = Adam8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
    elif lower_type == "adamw8":
        from toolkit.optimizers.adam8bit import Adam8bit

        optimizer = Adam8bit(params, lr=learning_rate, eps=1e-6, decouple=True, **optimizer_params)
    elif lower_type.endswith("8bit"):
        import bitsandbytes

        if lower_type == "adam8bit":
            return bitsandbytes.optim.Adam8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
        if lower_type == "ademamix8bit":
            return bitsandbytes.optim.AdEMAMix8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
        elif lower_type == "adamw8bit":
            return bitsandbytes.optim.AdamW8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
        elif lower_type == "lion8bit":
            return bitsandbytes.optim.Lion8bit(params, lr=learning_rate, **optimizer_params)
        else:
            raise ValueError(f'Unknown optimizer type {optimizer_type}')
    elif lower_type == 'adam':
        optimizer = _torch.optim.Adam(params, lr=float(learning_rate), eps=1e-6, **optimizer_params)
    elif lower_type == 'adamw':
        optimizer = _torch.optim.AdamW(params, lr=float(learning_rate), eps=1e-6, **optimizer_params)
    elif lower_type == 'lion':
        try:
            from lion_pytorch import Lion
            return Lion(params, lr=learning_rate, **optimizer_params)
        except ImportError:
            raise ImportError("Please install lion_pytorch to use Lion optimizer -> pip install lion-pytorch")
    elif lower_type == 'adagrad':
        optimizer = _torch.optim.Adagrad(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'adafactor':
        from toolkit.optimizers.adafactor import Adafactor
        if 'relative_step' not in optimizer_params:
            optimizer_params['relative_step'] = False
        if 'scale_parameter' not in optimizer_params:
            optimizer_params['scale_parameter'] = False
        if 'warmup_init' not in optimizer_params:
            optimizer_params['warmup_init'] = False
        optimizer = Adafactor(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'automagic':
        from toolkit.optimizers.automagic import Automagic
        optimizer = Automagic(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type in ('adamw_bf16', 'adamwbf16', 'adamw_bfloat16'):
        try:
            from adamw_bfloat16 import AdamW_BF16, LR
        except ImportError:
            raise ImportError(
                "Please install adamw_bfloat16 to use AdamW_BF16 optimizer -> "
                "pip install adamw_bfloat16 or pip install git+https://github.com/arogozhnikov/adamw_bfloat16"
            )

        # Allow dynamic parameter shapes to avoid recompilation warnings
        # AdamW_BF16 uses torch.compile internally, and LoRA has many different tensor shapes
        import torch._dynamo
        import torch._dynamo.config
        import adamw_bfloat16.torchcompiled as _tc
        torch._dynamo.config.force_parameter_static_shapes = False

        # Disable Dynamo compilation for the optimizer step to avoid shape-related recompiles entirely
        # This is more reliable than raising recompile_limit since LoRA can have 160+ unique shapes
        AdamW_BF16.step = torch._dynamo.disable()(AdamW_BF16.step)
        # Also disable Dynamo on the compiled inner step function to prevent step-counter guard recompiles
        _tc._make_step = torch._dynamo.disable()(_tc._make_step)

        # Allow pickle of LR when loading optimizer states with weights_only=True
        import torch.serialization
        torch.serialization.add_safe_globals([LR])

        print("Using AdamW_BF16 optimizer (memory-efficient bfloat16)")

        # Extract LR config if provided, otherwise use defaults
        lr_config = optimizer_params.pop('lr_config', {})
        preheat_steps = lr_config.get('preheat_steps', 3000)
        decay_power = lr_config.get('decay_power', -0.5)

        # Create the LR scheduler function
        lr_func = LR(
            lr=float(learning_rate),
            preheat_steps=preheat_steps,
            decay_power=decay_power
        )

        print(f"  LR config: base_lr={learning_rate}, preheat_steps={preheat_steps}, decay_power={decay_power}")

        # Default weight decay for AdamW
        if 'weight_decay' not in optimizer_params:
            optimizer_params['weight_decay'] = 0.01

        optimizer = AdamW_BF16(
            params,
            lr_function=lr_func,
            **optimizer_params
        )

        # Add dummy 'lr' field for compatibility with torch schedulers and logging
        # This won't affect the optimizer's built-in LR scheduling
        for pg in optimizer.param_groups:
            pg.setdefault('lr', float(learning_rate))
            pg.setdefault('initial_lr', float(learning_rate))

        # Add helper method to get current LR from built-in scheduler for logging
        def get_learning_rates(self):
            """Get current learning rate from built-in LR scheduler."""
            # Find the first param with state to get current step
            for pg in self.param_groups:
                for p in pg['params']:
                    if p in self.state and 'step' in self.state[p]:
                        step = self.state[p]['step']
                        return [pg['lr_function'](step)]
            # If no steps taken yet, return initial LR
            return [self.param_groups[0]['lr_function'](0.0)]

        # Bind the method to the optimizer instance
        import types
        optimizer.get_learning_rates = types.MethodType(get_learning_rates, optimizer)
    else:
        raise ValueError(f'Unknown optimizer type {optimizer_type}')
    return optimizer
