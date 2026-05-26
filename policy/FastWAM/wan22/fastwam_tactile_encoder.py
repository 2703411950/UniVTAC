"""FastWAM with UniVTAC tactile encoder.

Instead of concatenating tactile images into the visual grid, this module:
1. Encodes each tactile image (tac_left/tac_right) through a pretrained ResNet18
2. Projects the 512-dim tactile features to text_dim
3. Appends tactile tokens to the DiT cross-attention context

This keeps visual and tactile processing separate, allowing the model to
attend to each modality independently.
"""

from typing import Any, Optional
import torch
import torch.nn as nn
from torchvision import models

from ..logging_config import get_logger
from .fastwam import FastWAM
from .helpers.loader import load_wan22_ti2v_5b_components
from .action_dit import ActionDiT
from .mot import MoT

logger = get_logger(__name__)

TACTILE_LATENT_DIM = 512  # UniVTAC ResNet18 output dim


class TactileEncoder(nn.Module):
    """ResNet18-based tactile encoder (UniVTAC architecture).

    Encodes a 224x224 tactile image into a 512-dim latent vector.
    Uses pretrained ImageNet ResNet18 backbone.
    """

    def __init__(
        self,
        latent_dim: int = TACTILE_LATENT_DIM,
        pretrained: bool = True,
        freeze: bool = False,
    ):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, latent_dim)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode tactile image [B, C, H, W] → [B, latent_dim]."""
        return self.backbone(x)


class FastWAMTactileEncoder(FastWAM):
    """FastWAM with separate tactile encoding via UniVTAC ResNet18 backbone.

    Extra args vs FastWAM:
        tactile_encoder_freeze: freeze the ResNet18 backbone (default False)
        tactile_encoder_ckpt: path to UniVTAC pretrained checkpoint (optional)
    """

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        tactile_encoder_freeze: bool = False,
        tactile_encoder_ckpt: Optional[str] = None,
    ):
        super().__init__(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )

        # Build tactile encoder: ResNet18 → 512-dim
        self.tactile_encoder = TactileEncoder(
            latent_dim=TACTILE_LATENT_DIM,
            pretrained=True,
            freeze=tactile_encoder_freeze,
        ).to(device=device, dtype=torch_dtype)

        # Load UniVTAC checkpoint if provided
        if tactile_encoder_ckpt is not None:
            logger.info(f"Loading tactile encoder checkpoint: {tactile_encoder_ckpt}")
            state = torch.load(tactile_encoder_ckpt, map_location="cpu", weights_only=True)
            # strip "backbone." prefix if present (from UniVTAC training)
            if any(k.startswith("backbone.") for k in state):
                state = {k[len("backbone."):]: v for k, v in state.items()}
            self.tactile_encoder.backbone.load_state_dict(state, strict=False)
            logger.info("Tactile encoder checkpoint loaded.")

        # Project tactile latents (2 × 512) to context token space
        self.tactile_proj = nn.Sequential(
            nn.Linear(TACTILE_LATENT_DIM, self.text_dim),
            nn.SiLU(),
        ).to(device=device, dtype=torch_dtype)

        self.tactile_enabled = True

    def encode_tactile(self, tactile_images: torch.Tensor) -> torch.Tensor:
        """Encode tactile images to context tokens.

        Args:
            tactile_images: [B, 2, 3, H, W] — two tactile images per sample

        Returns:
            [B, 2, text_dim] — one token per tactile sensor
        """
        B, num_tactile, C, H, W = tactile_images.shape
        # Merge batch and tactile dims for single forward pass, convert to model dtype
        flat = tactile_images.view(B * num_tactile, C, H, W).to(
            device=self.device, dtype=self.torch_dtype
        )
        latents = self.tactile_encoder(flat)            # [B*2, 512]
        latents = latents.view(B, num_tactile, -1)       # [B, 2, 512]
        tokens = self.tactile_proj(latents)               # [B, 2, text_dim]
        return tokens

    def _append_tactile_to_context(self, context, context_mask, tactile_tokens):
        """Append tactile tokens to context sequence.

        Args:
            context: [B, L, text_dim]
            context_mask: [B, L]
            tactile_tokens: [B, num_tactile, text_dim]

        Returns:
            context: [B, L+num_tactile, text_dim]
            context_mask: [B, L+num_tactile]
        """
        B = context.shape[0]
        num_tactile = tactile_tokens.shape[1]
        ones = torch.ones(B, num_tactile, dtype=context_mask.dtype, device=context_mask.device)
        context = torch.cat([context, tactile_tokens.to(context.dtype)], dim=1)
        context_mask = torch.cat([context_mask, ones], dim=1)
        return context, context_mask

    def build_inputs(self, sample, tiled: bool = False):
        """Override: encode tactile images and append to context after parent's build."""
        inputs = super().build_inputs(sample, tiled=tiled)

        # Append tactile tokens to context if tactile_images is in sample
        if "tactile_images" in sample and sample["tactile_images"] is not None:
            tactile_images = sample["tactile_images"]  # [B, 2, C, H, W] — no time dim
            if tactile_images.ndim != 5:
                raise ValueError(
                    f"Expected `tactile_images` with shape [B, 2, C, H, W], got {tuple(tactile_images.shape)}"
                )
            tactile_tokens = self.encode_tactile(tactile_images)  # [B, 2, text_dim]
            inputs["context"], inputs["context_mask"] = self._append_tactile_to_context(
                inputs["context"], inputs["context_mask"], tactile_tokens
            )

        return inputs

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        tactile_images: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        """Override to support tactile_images during inference."""
        if tactile_images is None:
            return super().infer_action(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                negative_prompt=negative_prompt,
                text_cfg_scale=text_cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
            )

        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        # --- Text/proprio context ---
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        # Append proprio
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(context, context_mask, proprio)

        # Encode tactile and append
        tactile_tokens = self.encode_tactile(tactile_images)
        context, context_mask = self._append_tactile_to_context(context, context_mask, tactile_tokens)

        # --- Latent initialization ---
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        # --- Tiled context handling ---
        if text_cfg_scale != 1.0:
            raise NotImplementedError("CFG not supported with tactile encoder yet.")

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],), dtype=first_frame_latents.dtype, device=self.device
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            latents_action = self.infer_action_scheduler.step(pred_action_posi, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    def load_checkpoint(self, ckpt_path, strict: bool = True):
        """Override to handle tactile encoder/projection weights."""
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.load_state_dict(state, strict=strict)
        logger.info(f"Loaded checkpoint: {ckpt_path}")

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        tactile_encoder_freeze: bool = False,
        tactile_encoder_ckpt: Optional[str] = None,
    ) -> "FastWAMTactileEncoder":
        """Load Wan2.2 components, build FastWAMTactileEncoder, return it."""
        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )
        video_expert = components.dit
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM construction.")
        action_dit_config = dict(action_dit_config or {})
        action_dit_config.setdefault("freq_dim", int(video_dit_config.get("freq_dim", 256)))
        action_dit_config.setdefault("text_dim", int(video_dit_config.get("text_dim", 4096)))
        if action_dit_config.get("freq_dim") != video_dit_config.get("freq_dim"):
            raise ValueError("ActionDiT `freq_dim` must match video expert `freq_dim`.")
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if action_dit_pretrained_path is None and not skip_dit_load_from_pretrain:
            raise ValueError(
                "`action_dit_pretrained_path` is required for FastWAM; pass a path or set `skip_dit_load_from_pretrain=True`."
            )

        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            tactile_encoder_freeze=tactile_encoder_freeze,
            tactile_encoder_ckpt=tactile_encoder_ckpt,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model
