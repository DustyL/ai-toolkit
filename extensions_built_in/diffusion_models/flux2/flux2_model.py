import math
import os
from typing import TYPE_CHECKING, List, Optional

import huggingface_hub
import torch
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.memory_management.manager import MemoryManager
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.base_model import BaseModel
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.dequantize import patch_dequantization_on_save
from toolkit.accelerator import unwrap_model
from optimum.quanto import freeze, QTensor
from toolkit.util.quantize import quantize, get_qtype, quantize_model

from transformers import AutoProcessor, Mistral3ForConditionalGeneration
from .src.model import Flux2, Flux2Params
from .src.pipeline import Flux2Pipeline
from .src.autoencoder import AutoEncoder, AutoEncoderParams
from .flux2_gpu_splitter import add_model_gpu_splitter_to_flux2
from .src.attention_backends import initialize_attention_for_flux2
from safetensors.torch import load_file, save_file
from PIL import Image
import torch.nn.functional as F

# Import stream manager for multi-GPU transfers
from .cuda_stream_manager import (
    DeviceTransferManager,
    set_transfer_manager,
    get_transfer_manager,
)

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

from .src.sampling import (
    batched_prc_img,
    batched_prc_txt,
    encode_image_refs,
    scatter_ids,
)

scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True,
}

MISTRAL_PATH = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
FLUX2_VAE_FILENAME = "ae.safetensors"
FLUX2_TRANSFORMER_FILENAME = "flux2-dev.safetensors"

HF_TOKEN = os.getenv("HF_TOKEN", None)


class Flux2Model(BaseModel):
    arch = "flux2"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["Flux2"]
        # control images will come in as a list for encoding some things if true
        self.has_multiple_control_images = True
        # do not resize control images
        self.use_raw_control_images = True

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Flux2 model")
        # will be updated if we detect a existing checkpoint in training folder
        model_path = self.model_config.name_or_path
        transformer_path = model_path

        self.print_and_status_update("Loading transformer")
        with torch.device("meta"):
            transformer = Flux2(Flux2Params())

        # use local path if provided
        if os.path.exists(os.path.join(transformer_path, FLUX2_TRANSFORMER_FILENAME)):
            transformer_path = os.path.join(
                transformer_path, FLUX2_TRANSFORMER_FILENAME
            )

        if not os.path.exists(transformer_path):
            # assume it is from the hub
            transformer_path = huggingface_hub.hf_hub_download(
                repo_id=model_path,
                filename=FLUX2_TRANSFORMER_FILENAME,
                token=HF_TOKEN,
            )

        transformer_state_dict = load_file(transformer_path, device="cpu")

        # cast to dtype
        for key in transformer_state_dict:
            transformer_state_dict[key] = transformer_state_dict[key].to(dtype)

        transformer.load_state_dict(transformer_state_dict, assign=True)

        # Apply GPU model splitting if enabled (must be before any .to() calls)
        if self.model_config.split_model_over_gpus:
            # Warn about incompatible options
            if self.model_config.low_vram:
                print("Warning: low_vram is not compatible with split_model_over_gpus, disabling low_vram")
                self.model_config.low_vram = False
            if self.model_config.layer_offloading:
                print("Warning: layer_offloading is not compatible with split_model_over_gpus, disabling layer_offloading")
                self.model_config.layer_offloading = False
            if self.model_config.quantize:
                print("Warning: quantize is not compatible with split_model_over_gpus, disabling quantize")
                self.model_config.quantize = False

            self.print_and_status_update("Applying GPU model splitting")

            # POLICY: For training, final output MUST be on self.device_torch (where loss is computed)
            # User-configured output_device is ignored for training to prevent device mismatches
            user_output_device = getattr(self.model_config, 'output_device', None)
            if user_output_device is not None:
                user_dev = torch.device(user_output_device) if isinstance(user_output_device, str) else user_output_device
                if user_dev != self.device_torch:
                    print(f"[Flux2Model] WARNING: Ignoring output_device={user_output_device}. "
                          f"Training requires output on device_torch={self.device_torch} for loss computation.")
            # Always use device_torch for training path
            output_device = self.device_torch

            add_model_gpu_splitter_to_flux2(
                transformer,
                gpu_split_double=self.model_config.gpu_split_double,
                gpu_split_single=self.model_config.gpu_split_single,
                use_stream_transfers=self.model_config.use_stream_transfers,
                sync_per_block=getattr(self.model_config, 'sync_per_block', False),
                output_device=output_device,
                other_module_param_count_scale=self.model_config.split_model_other_module_param_count_scale
            )

        transformer.to(self.quantize_device, dtype=dtype)

        if self.model_config.quantize:
            # patch the state dict method
            patch_dequantization_on_save(transformer)
            self.print_and_status_update("Quantizing Transformer")
            quantize_model(self, transformer)
            flush()
        else:
            transformer.to(self.device_torch, dtype=dtype)
        flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_transformer_percent > 0
        ):
            MemoryManager.attach(
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
            )
            MemoryManager.log_status(transformer, "transformer")

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to("cpu")

        self.print_and_status_update("Loading Mistral")

        text_encoder: Mistral3ForConditionalGeneration = (
            Mistral3ForConditionalGeneration.from_pretrained(
                MISTRAL_PATH,
                torch_dtype=dtype,
            )
        )
        text_encoder.to(self.device_torch, dtype=dtype)

        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Mistral")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype))
            freeze(text_encoder)
            flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )
            MemoryManager.log_status(text_encoder, "text_encoder")

        tokenizer = AutoProcessor.from_pretrained(MISTRAL_PATH, fix_mistral_regex=True)

        self.print_and_status_update("Loading VAE")
        vae_path = self.model_config.vae_path

        if os.path.exists(os.path.join(model_path, FLUX2_VAE_FILENAME)):
            vae_path = os.path.join(model_path, FLUX2_VAE_FILENAME)

        if vae_path is None or not os.path.exists(vae_path):
            # assume it is from the hub
            vae_path = huggingface_hub.hf_hub_download(
                repo_id=model_path,
                filename=FLUX2_VAE_FILENAME,
                token=HF_TOKEN,
            )
        with torch.device("meta"):
            vae = AutoEncoder(AutoEncoderParams())

        vae_state_dict = load_file(vae_path, device="cpu")

        # cast to dtype
        for key in vae_state_dict:
            vae_state_dict[key] = vae_state_dict[key].to(dtype)

        vae.load_state_dict(vae_state_dict, assign=True)

        self.noise_scheduler = Flux2Model.get_train_scheduler()

        self.print_and_status_update("Making pipe")

        pipe: Flux2Pipeline = Flux2Pipeline(
            scheduler=self.noise_scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            transformer=None,
        )
        # for quantization, it works best to do these after making the pipe
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = [pipe.text_encoder]
        tokenizer = [pipe.tokenizer]

        flush()
        # just to make sure everything is on the right device and dtype
        text_encoder[0].to(self.device_torch)
        text_encoder[0].requires_grad_(False)
        text_encoder[0].eval()

        # CRITICAL: Skip global .to() when split_model_over_gpus is enabled
        # The GPU splitter has already placed blocks on their target devices via _split_device.
        # A global .to(device_torch) would collapse all blocks back to a single device,
        # undoing the split and causing attention contexts to be created on wrong devices.
        if not self.model_config.split_model_over_gpus:
            pipe.transformer = pipe.transformer.to(self.device_torch)
        flush()

        # Initialize attention backend contexts for all transformer blocks
        # TIMING: Must be AFTER GPU splitter (if enabled) so _split_device is set
        # This configures CuTE/Flash/SDPA per-block based on device capabilities
        # NOTE: For split_model_over_gpus, contexts use _split_device; otherwise parameter device
        # NOTE: strict_device_check=True for split mode catches stale contexts after .to() moves
        #
        # AUTOCAST HEURISTIC: If model dtype is fp32 but device supports bf16, use bf16
        # for backend selection. This allows CuTE/Flash to be selected when users train
        # with dtype=fp32 + autocast(bf16). Runtime guards still fall back to SDPA for
        # actual fp32 tensors (when autocast is disabled).
        backend_selection_dtype = dtype
        if dtype == torch.float32 and torch.cuda.is_bf16_supported():
            backend_selection_dtype = torch.bfloat16
            self.print_and_status_update(
                "Attention backend: using bf16 for selection (fp32 model + bf16-capable GPU). "
                "CuTE/Flash will be used under autocast; fp32 tensors fall back to SDPA."
            )

        initialize_attention_for_flux2(
            transformer=pipe.transformer,
            requested_backend=self.model_config.attention_backend,
            dtype=backend_selection_dtype,
            head_dim=128,  # Flux.2-dev uses 128 head_dim (6144 / 48 heads)
            cute_min_seqlen=self.model_config.cute_min_seqlen,
            strict_device_check=self.model_config.split_model_over_gpus,  # Fail-fast for split mode
        )

        # save it to the model class
        self.vae = vae
        self.text_encoder = text_encoder  # list of text encoders
        self.tokenizer = tokenizer  # list of tokenizers
        self.model = pipe.transformer
        self.pipeline = pipe
        self.print_and_status_update("Model Loaded")

    def get_generation_pipeline(self):
        scheduler = Flux2Model.get_train_scheduler()

        pipeline: Flux2Pipeline = Flux2Pipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer),
        )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: Flux2Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        gen_config.width = (
            gen_config.width // self.get_bucket_divisibility()
        ) * self.get_bucket_divisibility()
        gen_config.height = (
            gen_config.height // self.get_bucket_divisibility()
        ) * self.get_bucket_divisibility()

        control_img_list = []
        if gen_config.ctrl_img is not None:
            control_img = Image.open(gen_config.ctrl_img)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)
        elif gen_config.ctrl_img_1 is not None:
            control_img = Image.open(gen_config.ctrl_img_1)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)
        if gen_config.ctrl_img_2 is not None:
            control_img = Image.open(gen_config.ctrl_img_2)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)
        if gen_config.ctrl_img_3 is not None:
            control_img = Image.open(gen_config.ctrl_img_3)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)

        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            control_img_list=control_img_list,
            **extra,
        ).images[0]
        return img

    def _setup_transfer_manager(self) -> Optional[DeviceTransferManager]:
        """
        Set up or reuse the DeviceTransferManager for multi-GPU stream-based transfers.

        Returns:
            DeviceTransferManager if stream transfers are enabled and model is split,
            None otherwise.

        Note:
            For training, output_device is ALWAYS self.device_torch to ensure
            the final tensor is on the correct device for loss computation.

        Reuse:
            The manager is cached on the model instance and reused across training
            steps. CUDA streams/events are allocated once. Per-step state is reset
            in the manager's __enter__ method.
        """
        transformer = self.transformer

        # Check if model is split and stream transfers are enabled
        if not hasattr(transformer, '_use_stream_transfers'):
            return None
        if not transformer._use_stream_transfers:
            return None
        if not hasattr(transformer, '_split_devices'):
            return None

        # Reuse existing manager if available
        if hasattr(self, '_transfer_manager') and self._transfer_manager is not None:
            return self._transfer_manager

        # POLICY: Always use device_torch for training (loss computation device)
        # This matches the enforcement in load_model()
        output_device = self.device_torch

        # Create and configure the manager (will be reused across steps)
        mgr = DeviceTransferManager(
            devices=transformer._split_devices,
            enable_timing=False,  # Disable timing for production
            sync_on_exit=False,   # We handle sync via _join_default_stream
            output_device=output_device,
        )

        # Cache for reuse
        self._transfer_manager = mgr
        return mgr

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        guidance_embedding_scale: float,
        batch: "DataLoaderBatchDTO" = None,
        **kwargs,
    ):
        """
        Get noise prediction from the transformer.

        For multi-GPU setups with split_model_over_gpus=True and use_stream_transfers=True,
        this method sets up a DeviceTransferManager to handle efficient tensor transfers
        between GPUs with proper synchronization for the backward pass.

        The DeviceTransferManager:
        - Uses dedicated transfer streams per GPU (separate from compute)
        - Uses CUDA events for fine-grained synchronization (no full device syncs)
        - Pre-stages modulation tensors to all GPUs at once
        - Ensures the default stream on the output device sees all operations
          before returning, which is required for autograd backward pass safety
        """
        # Set up transfer manager for multi-GPU stream transfers
        transfer_mgr = self._setup_transfer_manager()

        with torch.no_grad():
            txt, txt_ids = batched_prc_txt(text_embeddings.text_embeds)
            packed_latents, img_ids = batched_prc_img(latent_model_input)

            # prepare image conditioning if any
            img_cond_seq: torch.Tensor | None = None
            img_cond_seq_ids: torch.Tensor | None = None

            # handle control images
            if batch.control_tensor_list is not None:
                batch_size, num_channels_latents, height, width = (
                    latent_model_input.shape
                )

                control_image_max_res = 1024 * 1024
                if self.model_config.model_kwargs.get("match_target_res", False):
                    # use the current target size to set the control image res
                    control_image_res = (
                        height
                        * self.pipeline.vae_scale_factor
                        * width
                        * self.pipeline.vae_scale_factor
                    )
                    control_image_max_res = control_image_res

                if len(batch.control_tensor_list) != batch_size:
                    raise ValueError(
                        "Control tensor list length does not match batch size"
                    )
                for control_tensor_list in batch.control_tensor_list:
                    # control tensor list is a list of tensors for this batch item
                    controls = []
                    # pack control
                    for control_img in control_tensor_list:
                        # control images are 0 - 1 scale, shape (1, ch, height, width)
                        control_img = control_img.to(
                            self.device_torch, dtype=self.torch_dtype
                        )
                        # if it is only 3 dim, add batch dim
                        if len(control_img.shape) == 3:
                            control_img = control_img.unsqueeze(0)

                        # resize to fit within max res while keeping aspect ratio
                        if self.model_config.model_kwargs.get(
                            "match_target_res", False
                        ):
                            ratio = control_img.shape[2] / control_img.shape[3]
                            c_width = math.sqrt(control_image_res * ratio)
                            c_height = c_width / ratio

                            c_width = round(c_width / 32) * 32
                            c_height = round(c_height / 32) * 32

                            control_img = F.interpolate(
                                control_img, size=(c_height, c_width), mode="bilinear"
                            )

                        # scale to -1 to 1
                        control_img = control_img * 2 - 1
                        controls.append(control_img)

                    img_cond_seq_item, img_cond_seq_ids_item = encode_image_refs(
                        self.vae, controls, limit_pixels=control_image_max_res
                    )
                    if img_cond_seq is None:
                        img_cond_seq = img_cond_seq_item
                        img_cond_seq_ids = img_cond_seq_ids_item
                    else:
                        img_cond_seq = torch.cat(
                            (img_cond_seq, img_cond_seq_item), dim=0
                        )
                        img_cond_seq_ids = torch.cat(
                            (img_cond_seq_ids, img_cond_seq_ids_item), dim=0
                        )

            img_input = packed_latents
            img_input_ids = img_ids

            if img_cond_seq is not None:
                assert img_cond_seq_ids is not None, (
                    "You need to provide either both or neither of the sequence conditioning"
                )
                img_input = torch.cat((img_input, img_cond_seq), dim=1)
                img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)

            guidance_vec = torch.full(
                (img_input.shape[0],),
                guidance_embedding_scale,
                device=img_input.device,
                dtype=img_input.dtype,
            )

            cast_dtype = self.model.dtype

        # Run transformer forward pass
        # If transfer manager is set up, use context manager for proper stream handling
        if transfer_mgr is not None:
            # Debug: Log that we're using stream-based path
            if transfer_mgr._debug:
                print(f"[Flux2Model] Using stream-based multi-GPU path (output_device={transfer_mgr.output_device})")
            with transfer_mgr:
                # Set global transfer manager for block forwards to access
                set_transfer_manager(transfer_mgr)

                try:
                    packed_noise_pred = self.transformer(
                        x=img_input.to(self.device_torch, cast_dtype),
                        x_ids=img_input_ids.to(self.device_torch),
                        timesteps=timestep.to(self.device_torch, cast_dtype) / 1000,
                        ctx=txt.to(self.device_torch, cast_dtype),
                        ctx_ids=txt_ids.to(self.device_torch),
                        guidance=guidance_vec.to(self.device_torch, cast_dtype),
                    )
                finally:
                    # Always clear the global transfer manager
                    set_transfer_manager(None)

            # The transfer_mgr context exit calls _join_default_stream()
            # which ensures the default stream sees all operations - required
            # for autograd backward pass safety
        else:
            # Standard path without stream-based transfers
            packed_noise_pred = self.transformer(
                x=img_input.to(self.device_torch, cast_dtype),
                x_ids=img_input_ids.to(self.device_torch),
                timesteps=timestep.to(self.device_torch, cast_dtype) / 1000,
                ctx=txt.to(self.device_torch, cast_dtype),
                ctx_ids=txt_ids.to(self.device_torch),
                guidance=guidance_vec.to(self.device_torch, cast_dtype),
            )

        if img_cond_seq is not None:
            packed_noise_pred = packed_noise_pred[:, : packed_latents.shape[1]]

        if isinstance(packed_noise_pred, QTensor):
            packed_noise_pred = packed_noise_pred.dequantize()

        noise_pred = torch.cat(scatter_ids(packed_noise_pred, img_ids)).squeeze(2)

        return noise_pred

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)

        prompt_embeds, prompt_embeds_mask = self.pipeline.encode_prompt(
            prompt, device=self.device_torch
        )
        pe = PromptEmbeds(prompt_embeds)
        return pe

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        if not output_path.endswith(".safetensors"):
            output_path = output_path + ".safetensors"
        # only save the unet
        transformer: Flux2 = unwrap_model(self.model)
        state_dict = transformer.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, QTensor):
                v = v.dequantize()
            save_dict[k] = v.clone().to("cpu", dtype=save_dtype)

        meta = get_meta_for_safetensors(meta, name="flux2")
        save_file(save_dict, output_path, metadata=meta)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_base_model_version(self):
        return "flux2"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["double_blocks", "single_blocks"]

    def convert_lora_weights_before_save(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("transformer.", "diffusion_model.")
            new_sd[new_key] = value
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        return new_sd

    def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        # Move to vae to device if on cpu
        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        # move to device and dtype
        image_list = [image.to(device, dtype=dtype) for image in image_list]
        images = torch.stack(image_list).to(device, dtype=dtype)

        latents = self.vae.encode(images)

        return latents
