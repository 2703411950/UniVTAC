"""Helpers for decoding / visualizing predicted future tactile trajectories."""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision.transforms.functional import to_pil_image


def _load_raw_checkpoint(ckpt_path: str) -> Dict[str, torch.Tensor]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Tactile checkpoint not found: {ckpt_path}")

    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")

    if isinstance(state, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    if isinstance(state, dict):
        return state
    raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")


def describe_decoder_keys(ckpt_path: str) -> None:
    state = _load_raw_checkpoint(ckpt_path)
    dec_keys = sorted(key for key in state if key.startswith("decoders."))
    print(f"[TactileDecoder] Found {len(dec_keys)} decoder keys in {ckpt_path}")
    for key in dec_keys[:40]:
        tensor = state[key]
        shape = tuple(tensor.shape) if hasattr(tensor, "shape") else type(tensor)
        print(f"  {key}: {shape}")
    if len(dec_keys) > 40:
        print(f"  ... ({len(dec_keys) - 40} more keys)")


class _MLPImageDecoder(nn.Module):
    def __init__(self, state: Dict[str, torch.Tensor]):
        super().__init__()
        weight_keys = sorted(
            (key, value)
            for key, value in state.items()
            if key.endswith(".weight") and value.ndim == 2
        )
        if not weight_keys:
            raise ValueError("No linear decoder weights found.")

        layers = []
        for idx, (key, weight) in enumerate(weight_keys):
            prefix = key[: -len(".weight")]
            bias_key = f"{prefix}.bias"
            linear = nn.Linear(weight.shape[1], weight.shape[0], bias=bias_key in state)
            linear.weight.data.copy_(weight)
            if bias_key in state:
                linear.bias.data.copy_(state[bias_key])
            layers.append(linear)
            if idx + 1 < len(weight_keys):
                layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*layers)
        out_features = weight_keys[-1][1].shape[0]
        side = int(round((out_features // 3) ** 0.5))
        self.image_shape = (3, side, side) if side * side * 3 == out_features else None

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        flat = self.net(latent)
        if self.image_shape is None:
            raise RuntimeError("Decoder output size is not a square RGB image.")
        return torch.sigmoid(flat.view(-1, *self.image_shape))


class TactileDecoderBank(nn.Module):
    def __init__(self, ckpt_path: str):
        super().__init__()
        full_state = _load_raw_checkpoint(ckpt_path)
        dec_state = {
            key[len("decoders."):]: value
            for key, value in full_state.items()
            if key.startswith("decoders.")
        }
        if not dec_state:
            raise ValueError(f"No decoders.* keys found in {ckpt_path}")

        sensor_prefixes = [
            sensor
            for sensor in ("left", "right", "0", "1")
            if any(key.startswith(f"{sensor}.") for key in dec_state)
        ]

        self.decoders = nn.ModuleDict()
        if sensor_prefixes:
            for sensor in sensor_prefixes:
                sub_state = {
                    key[len(f"{sensor}."):]: value
                    for key, value in dec_state.items()
                    if key.startswith(f"{sensor}.")
                }
                self.decoders[sensor] = _MLPImageDecoder(sub_state)
        else:
            self.decoders["shared"] = _MLPImageDecoder(dec_state)

    @classmethod
    def try_load(cls, ckpt_path: Optional[str]) -> Optional["TactileDecoderBank"]:
        if not ckpt_path or not os.path.exists(ckpt_path):
            return None
        try:
            return cls(ckpt_path)
        except Exception as exc:
            print(f"[TactileDecoder] Failed to load decoder from {ckpt_path}: {exc}")
            return None

    def decode_pair(
        self,
        left_latent: torch.Tensor,
        right_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if "left" in self.decoders and "right" in self.decoders:
            return self.decoders["left"](left_latent), self.decoders["right"](right_latent)
        if "0" in self.decoders and "1" in self.decoders:
            return self.decoders["0"](left_latent), self.decoders["1"](right_latent)
        shared = self.decoders["shared"]
        return shared(left_latent), shared(right_latent)


def invert_target_projection(target_proj: nn.Linear, pred_latent: torch.Tensor) -> torch.Tensor:
    weight = target_proj.weight.float()
    bias = target_proj.bias.float()
    pinv = torch.linalg.pinv(weight)
    return (pred_latent.float() - bias) @ pinv


def split_concat_latent(concat_latent: torch.Tensor, latent_dim: int = 512):
    return concat_latent[..., :latent_dim], concat_latent[..., latent_dim:]


def decode_pred_tactile_images(
    target_proj: nn.Linear,
    decoder_bank: TactileDecoderBank,
    pred_latent: torch.Tensor,
    latent_dim: int = 512,
):
    if pred_latent.ndim == 3:
        pred_latent = pred_latent[0]
    concat_latent = invert_target_projection(target_proj, pred_latent)
    left_latent, right_latent = split_concat_latent(concat_latent, latent_dim=latent_dim)
    return decoder_bank.decode_pair(left_latent, right_latent)


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().float()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    else:
        image = image.clamp(0, 1).numpy()
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return image


def plot_latent_metrics(result: dict, save_path):
    steps = np.arange(len(result["per_step_mse"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(steps, result["per_step_mse"], marker="o")
    axes[0].set_title("Tactile latent MSE")
    axes[0].set_xlabel("Future step")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(steps, result["per_step_cos"], marker="o", color="tab:orange")
    axes[1].set_title("Tactile latent cosine")
    axes[1].set_xlabel("Future step")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(result["caption"], fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_latent_heatmap(result: dict, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(result["gt_tactiles"].T, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("GT tactile latent")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(result["pred_tactiles"].T, aspect="auto", origin="lower", cmap="viridis")
    axes[1].set_title("Predicted tactile latent")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tactile_image_comparison(result: dict, timesteps: tuple[int, ...], save_path):
    has_pred = result["pred_left_imgs"] is not None
    num_rows = 4 if has_pred else 2
    fig, axes = plt.subplots(num_rows, len(timesteps), figsize=(2.6 * len(timesteps), 2.4 * num_rows))
    if len(timesteps) == 1:
        axes = np.expand_dims(axes, axis=1)

    row_titles = (
        ["GT tac_left", "GT tac_right", "Pred tac_left", "Pred tac_right"]
        if has_pred
        else ["GT tac_left", "GT tac_right"]
    )

    for col, step in enumerate(timesteps):
        axes[0, col].imshow(tensor_to_numpy_image(result["gt_left_imgs"][step]))
        axes[0, col].set_title(f"t={step}")
        axes[0, col].axis("off")
        axes[1, col].imshow(tensor_to_numpy_image(result["gt_right_imgs"][step]))
        axes[1, col].axis("off")
        if has_pred:
            axes[2, col].imshow(tensor_to_numpy_image(result["pred_left_imgs"][step]))
            axes[2, col].axis("off")
            axes[3, col].imshow(tensor_to_numpy_image(result["pred_right_imgs"][step]))
            axes[3, col].axis("off")

    for row_idx, title in enumerate(row_titles):
        axes[row_idx, 0].set_ylabel(title, fontsize=9)

    subtitle = "GT vs predicted tactile images" if has_pred else "GT tactile images (decoder unavailable)"
    fig.suptitle(f"{result['caption']}\n{subtitle}", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_contact_sheet(result: dict, timesteps: tuple[int, ...], save_path, tile_size=256):
    has_pred = result["pred_left_imgs"] is not None
    rows = 4 if has_pred else 2
    cols = len(timesteps)
    canvas = np.ones((rows * tile_size, cols * tile_size, 3), dtype=np.float32)

    for col, step in enumerate(timesteps):
        images = [
            tensor_to_numpy_image(result["gt_left_imgs"][step]),
            tensor_to_numpy_image(result["gt_right_imgs"][step]),
        ]
        if has_pred:
            images.extend([
                tensor_to_numpy_image(result["pred_left_imgs"][step]),
                tensor_to_numpy_image(result["pred_right_imgs"][step]),
            ])
        for row, image in enumerate(images):
            pil = to_pil_image(torch.from_numpy(image).permute(2, 0, 1))
            pil = pil.resize((tile_size, tile_size))
            image = np.asarray(pil).astype(np.float32) / 255.0
            canvas[row * tile_size:(row + 1) * tile_size, col * tile_size:(col + 1) * tile_size] = image

    plt.imsave(save_path, np.clip(canvas, 0, 1))
