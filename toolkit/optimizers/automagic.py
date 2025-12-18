from typing import List
import torch
from toolkit.optimizers.optimizer_utils import Auto8bitTensor, copy_stochastic, stochastic_grad_accummulation
from optimum.quanto import QBytesTensor
import random
import math


class Automagic(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-6,  # lr is start lr
        min_lr=1e-7,
        max_lr=1e-3,
        lr_bump=1e-6,  # amount to bump the lr when adjusting
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        beta2=0.999,
        weight_decay=0.0,
        do_paramiter_swapping=False,
        paramiter_swapping_factor=0.1,
        high_noise_lr_bump=None,
        low_noise_lr_bump=None,
        high_noise_min_lr=None,
        low_noise_min_lr=None,
        high_noise_max_lr=None,
        low_noise_max_lr=None,
        **unused_kwargs,
    ):
        self.lr = lr
        if self.lr > 1e-3:
            print(f"Warning! Start lr is very high: {self.lr}. Forcing to 1e-6. this does not work like prodigy")
            self.lr = 1e-6
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_bump = lr_bump
        self.high_noise_defaults = {
            "lr_bump": high_noise_lr_bump,
            "min_lr": high_noise_min_lr,
            "max_lr": high_noise_max_lr,
        }
        self.low_noise_defaults = {
            "lr_bump": low_noise_lr_bump,
            "min_lr": low_noise_min_lr,
            "max_lr": low_noise_max_lr,
        }
        if unused_kwargs:
            ignored = ", ".join(unused_kwargs.keys())
            print(f"Automagic optimizer ignoring unsupported kwargs: {ignored}")

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "beta2": beta2,
            "weight_decay": weight_decay,
            "min_lr": min_lr,
            "max_lr": max_lr,
            "lr_bump": lr_bump,
        }
        super().__init__(params, defaults)

        self.base_lrs: List[float] = [
            lr for group in self.param_groups
        ]

        self.is_stochastic_rounding_accumulation = False

        # Ensure every param group has min/max/lr_bump keys so they can be overridden individually
        for group in self.param_groups:
            group.setdefault("min_lr", self.min_lr)
            group.setdefault("max_lr", self.max_lr)
            group.setdefault("lr_bump", self.lr_bump)

        # setup stochastic grad accum hooks
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad and param.dtype != torch.float32:
                    self.is_stochastic_rounding_accumulation = True
                    param.register_post_accumulate_grad_hook(
                        stochastic_grad_accummulation
                    )

        self.do_paramiter_swapping = do_paramiter_swapping
        self.paramiter_swapping_factor = paramiter_swapping_factor
        self._total_paramiter_size = 0
        # count total paramiters
        for group in self.param_groups:
            for param in group['params']:
                self._total_paramiter_size += torch.numel(param)
        # pretty print total paramiters with comma seperation
        print(f"Total training paramiters: {self._total_paramiter_size:,}")

        # Track gradient stability & recent losses for alpha scheduler exit criteria
        self._gradient_sign_agreements: List[float] = []
        self.recent_losses: List[float] = []

        # needs to be enabled to count paramiters
        if self.do_paramiter_swapping:
            self.enable_paramiter_swapping(self.paramiter_swapping_factor)

    def enable_paramiter_swapping(self, paramiter_swapping_factor=0.1):
        self.do_paramiter_swapping = True
        self.paramiter_swapping_factor = paramiter_swapping_factor
        # call it an initial time
        self.swap_paramiters()

    def swap_paramiters(self):
        all_params = []
        # deactivate all paramiters
        for group in self.param_groups:
            for param in group['params']:
                param.requires_grad_(False)
                # remove any grad
                param.grad = None
                all_params.append(param)
        # shuffle all paramiters
        random.shuffle(all_params)

        # keep activating paramiters until we are going to go over the target paramiters
        target_paramiters = int(
            self._total_paramiter_size * self.paramiter_swapping_factor)
        total_paramiters = 0
        for param in all_params:
            total_paramiters += torch.numel(param)
            if total_paramiters >= target_paramiters:
                break
            else:
                param.requires_grad_(True)

    @staticmethod
    def _get_lr(param_group, param_state):
        if 'avg_lr' in param_state:
            lr = param_state["avg_lr"]
        else:
            lr = 0.0
        return lr

    def _get_group_lr(self, group):
        group_lrs = []
        for p in group["params"]:
            group_lrs.append(self._get_lr(group, self.state[p]))
        # return avg
        if len(group_lrs) == 0:
            return self.lr
        return sum(group_lrs) / len(group_lrs)

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-
                    1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def step_hook(self):
        if not self.is_stochastic_rounding_accumulation:
            return
        # copy over stochastically rounded grads
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad and hasattr(param, "_accum_grad"):
                    param.grad = param._accum_grad
                    del param._accum_grad

    # automagic manages its own lr
    def get_learning_rates(self):

        lrs = [
            self._get_group_lr(group)
            for group in self.param_groups
        ]
        if len(lrs) == 0:
            lrs = self.base_lrs  # if called before stepping
        return lrs

    def get_avg_learning_rate(self):
        lrs = self.get_learning_rates()
        return sum(lrs) / len(lrs)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self.step_hook()
        loss = None
        if closure is not None:
            loss = closure()

        if loss is not None:
            loss_value = loss.item() if torch.is_tensor(loss) else float(loss)
            if math.isfinite(loss_value):
                self.recent_losses.append(loss_value)
                if len(self.recent_losses) > 1000:
                    self.recent_losses.pop(0)

        step_agreement_sum = 0.0
        step_agreement_weight = 0.0

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue

                grad = p.grad
                if grad.dtype != torch.float32:
                    grad = grad.to(torch.float32)
                if grad.is_sparse:
                    raise RuntimeError(
                        "Automagic does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                factored = len(grad_shape) >= 2
                # State Initialization
                if len(state) == 0:
                    self.initialize_state(p, group)
                else:
                    # Check if exp_avg_sq_row and exp_avg_sq_col exist for factored case
                    if factored:
                        if "exp_avg_sq_row" not in state or "exp_avg_sq_col" not in state:
                            state["exp_avg_sq_row"] = torch.zeros(p.shape[:-1]).to(grad)
                            state["exp_avg_sq_col"] = torch.zeros(p.shape[:-2] + p.shape[-1:]).to(grad)
                        else:
                            state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                            state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    # Check if exp_avg_sq exists for non-factored case
                    else:
                        if "exp_avg_sq" not in state:
                            state["exp_avg_sq"] = torch.zeros_like(grad)
                        else:
                            state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p

                if isinstance(p_data_fp32, QBytesTensor):
                    p_data_fp32 = p_data_fp32.dequantize()
                if p.dtype != torch.float32:
                    p_data_fp32 = p_data_fp32.clone().float()

                # Initialize step if it doesn't exist
                if "step" not in state:
                    state["step"] = 0
                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)

                # Use fixed beta2 from group instead of decay_rate calculation
                beta2 = group["beta2"]
                eps = group["eps"]
                if isinstance(eps, tuple) or isinstance(eps, list):
                    eps = eps[0]
                update = (grad**2) + eps
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2).add_(
                        update.mean(dim=-1), alpha=(1.0 - beta2))
                    exp_avg_sq_col.mul_(beta2).add_(
                        update.mean(dim=-2), alpha=(1.0 - beta2))

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(
                        exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2).add_(update, alpha=(1.0 - beta2))
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_(
                    (self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

                # Ensure state is properly initialized
                if 'last_polarity' not in state or 'lr_mask' not in state:
                    self.initialize_state(p, group)

                # Get signs of current last update and updates
                last_polarity = state['last_polarity']
                current_polarity = (update > 0).to(torch.bool)
                sign_agreement = torch.where(
                    last_polarity == current_polarity, 1, -1)
                state['last_polarity'] = current_polarity

                lr_mask = state['lr_mask'].to(torch.float32)
                group_lr_bump = group.get("lr_bump", self.lr_bump)
                group_min_lr = group.get("min_lr", self.min_lr)
                group_max_lr = group.get("max_lr", self.max_lr)

                # Update learning rate mask based on sign agreement
                new_lr = torch.where(
                    sign_agreement > 0,
                    lr_mask + group_lr_bump,  # Increase lr
                    lr_mask - group_lr_bump  # Decrease lr
                )

                # Clip learning rates to bounds
                new_lr = torch.clamp(
                    new_lr,
                    min=group_min_lr,
                    max=group_max_lr
                )

                # Apply the learning rate mask to the update
                update.mul_(new_lr)

                state['lr_mask'] = Auto8bitTensor(new_lr)
                state['avg_lr'] = torch.mean(new_lr)

                agreement_rate = (sign_agreement > 0).float().mean().item()
                weight = float(p.numel())
                step_agreement_sum += agreement_rate * weight
                step_agreement_weight += weight

                if group["weight_decay"] != 0:
                    # Apply weight decay with per-parameter learning rates
                    # Instead of using add_ with a tensor alpha (which isn't supported),
                    # we'll use element-wise multiplication to apply the weight decay
                    weight_decay_update = p_data_fp32 * (-group["weight_decay"]) * new_lr
                    p_data_fp32.add_(weight_decay_update)

                p_data_fp32.add_(-update)

                if p.dtype != torch.float32:
                    # apply stochastic rounding
                    copy_stochastic(p, p_data_fp32)

        if step_agreement_weight > 0:
            overall_agreement = step_agreement_sum / step_agreement_weight
            self._gradient_sign_agreements.append(overall_agreement)
            if len(self._gradient_sign_agreements) > 1000:
                self._gradient_sign_agreements.pop(0)

        return loss
    
    def initialize_state(self, p, group=None):
        state = self.state[p]
        state["step"] = 0

        # store the lr mask
        if 'lr_mask' not in state:
            base_lr = self.lr
            if group is not None:
                base_lr = group.get("lr", self.lr)
            state['lr_mask'] = Auto8bitTensor(torch.ones(
                p.shape).to(p.device, dtype=torch.float32) * base_lr
            )
        state['avg_lr'] = torch.mean(
            state['lr_mask'].to(torch.float32))
        if 'last_polarity' not in state:
            state['last_polarity'] = torch.zeros(
                p.shape, dtype=torch.bool, device=p.device)

        factored = len(p.shape) >= 2
        if factored:
            state["exp_avg_sq_row"] = torch.zeros(
                p.shape[:-1]).to(p)
            state["exp_avg_sq_col"] = torch.zeros(
                p.shape[:-2] + p.shape[-1:]).to(p)
        else:
            state["exp_avg_sq"] = torch.zeros_like(p)

        state["RMS"] = 0
    
    # override the state_dict to save the lr_mask
    def state_dict(self, *args, **kwargs):
        orig_state_dict = super().state_dict(*args, **kwargs)
        # convert the state to quantized tensor to scale and quantized
        new_sace_state = {}
        for p, state in orig_state_dict['state'].items():
            save_state = {k: v for k, v in state.items() if k != 'lr_mask'}

            # Check if lr_mask exists in the state before trying to access it
            if 'lr_mask' in state:
                save_state['lr_mask'] = state['lr_mask'].state_dict()

            new_sace_state[p] = save_state

        orig_state_dict['state'] = new_sace_state
        orig_state_dict['gradient_sign_agreements'] = list(self._gradient_sign_agreements)
        orig_state_dict['recent_losses'] = list(self.recent_losses)

        return orig_state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        grad_history = state_dict.pop('gradient_sign_agreements', None)
        loss_history = state_dict.pop('recent_losses', None)

        # Validate that the state_dict is from an Automagic optimizer
        is_valid_automagic_state = False

        # Check if state_dict has the expected structure
        if 'state' in state_dict and isinstance(state_dict['state'], dict):
            # Check if at least one state entry has an lr_mask, which is specific to Automagic
            for param_id, param_state in state_dict['state'].items():
                if isinstance(param_state, dict) and 'lr_mask' in param_state:
                    is_valid_automagic_state = True
                    break

        if not is_valid_automagic_state:
            return

        # First, call the parent class's load_state_dict to load the basic optimizer state
        # We'll handle the lr_mask separately
        state_dict_copy = {
            'state': {},
            'param_groups': state_dict['param_groups']
        }

        # Copy all state entries except lr_mask
        for param_id, param_state in state_dict['state'].items():
            state_dict_copy['state'][param_id] = {
                k: v for k, v in param_state.items() if k != 'lr_mask'
            }

        # Call parent class load_state_dict with the modified state dict
        super().load_state_dict(state_dict_copy)

        # Now handle the lr_mask separately
        # We need to map the saved parameters to the current parameters
        # This is tricky because the parameter IDs might be different

        # Get all current parameters that require gradients
        current_params = []
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    current_params.append((p, group))

        # If the number of parameters doesn't match, we can't reliably map them
        if len(current_params) != len(state_dict['param_groups'][0]['params']):
            print(f"WARNING: Number of parameters doesn't match between saved state ({len(state_dict['param_groups'][0]['params'])}) "
                  f"and current model ({len(current_params)}). Learning rate masks may not be correctly loaded.")

        # Map parameters by their position in the param_groups
        # This assumes the order of parameters is preserved between saving and loading
        saved_param_ids = list(state_dict['state'].keys())

        for i, (current_param, current_group) in enumerate(current_params):
            if i >= len(saved_param_ids):
                break

            saved_param_id = saved_param_ids[i]
            saved_state = state_dict['state'][saved_param_id]

            # Skip if this saved state doesn't have an lr_mask
            if 'lr_mask' not in saved_state:
                continue

            # Initialize the state for this parameter if it doesn't exist
            if current_param not in self.state:
                self.initialize_state(current_param, current_group)

            # Get the current state for this parameter
            current_state = self.state[current_param]

            # Load the lr_mask from the saved state
            saved_lr_mask = saved_state['lr_mask']

            # Reconstruct the Auto8bitTensor from its state dict
            try:
                # Make sure the shapes match
                if 'quantized' in saved_lr_mask and saved_lr_mask['quantized'].shape == current_param.shape:
                    current_state['lr_mask'] = Auto8bitTensor(saved_lr_mask)
                    current_state['avg_lr'] = torch.mean(current_state['lr_mask'].to(torch.float32))
                else:
                    print(f"WARNING: Shape mismatch for parameter {i}. "
                          f"Expected {current_param.shape}, got {saved_lr_mask['quantized'].shape if 'quantized' in saved_lr_mask else 'unknown'}. "
                          f"Initializing new lr_mask.")
                    # Initialize a new lr_mask
                    base_lr = current_group.get("lr", self.lr) if current_group is not None else self.lr
                    current_state['lr_mask'] = Auto8bitTensor(torch.ones(
                        current_param.shape).to(current_param.device, dtype=torch.float32) * base_lr
                    )
                    current_state['avg_lr'] = torch.mean(current_state['lr_mask'].to(torch.float32))
            except Exception as e:
                print(f"ERROR: Failed to load lr_mask for parameter {i}: {e}")
                # Initialize a new lr_mask
                base_lr = current_group.get("lr", self.lr) if current_group is not None else self.lr
                current_state['lr_mask'] = Auto8bitTensor(torch.ones(
                    current_param.shape).to(current_param.device, dtype=torch.float32) * base_lr
                )
                current_state['avg_lr'] = torch.mean(current_state['lr_mask'].to(torch.float32))

        if grad_history is not None:
            self._gradient_sign_agreements = list(grad_history)
        if loss_history is not None:
            self.recent_losses = list(loss_history)

    def get_gradient_sign_agreement_rate(self):
        """Return the rolling average of gradient sign agreement (0-1)."""
        if not self._gradient_sign_agreements:
            return 0.0

        recent = self._gradient_sign_agreements[-100:]
        return float(sum(recent) / len(recent))

    def get_recent_losses(self):
        """Get list of recent loss values for alpha scheduler."""
        return list(self.recent_losses)
