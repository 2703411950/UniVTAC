"""Smoke test for marker-based tactile WLA deploy config (no Isaac Sim).

Config:
  policy/wla/deploy_tactile_marker_predict_tactile.yml

Example:
  cd /data1/cyy/UniVTAC
  python policy/wla/smoke_test_tactile_marker.py
  python policy/wla/smoke_test_tactile_marker.py --config policy/wla/deploy_tactile_marker_predict_tactile.yml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from policy.wla.deploy_policy import (
    _build_mantis_from_training_config,
    _ensure_mantis_import_path,
    _load_checkpoint_state_dict,
    _load_mantis_transforms,
    _load_training_config,
    _prefer_conda_site_packages,
    _resolve_training_config_path,
)

DEFAULT_CONFIG = Path(__file__).with_name("deploy_tactile_marker_predict_tactile.yml")


def _stack_marker_pair(left_marker, right_marker):
    """Stack left/right markers into [1, 2, 2, num_markers, 2]."""
    markers = torch.stack(
        [
            left_marker.detach().cpu().float(),
            right_marker.detach().cpu().float(),
        ],
        dim=0,
    )
    return markers.unsqueeze(0)


def build_dummy_observation(num_markers: int):
    marker = torch.zeros((2, num_markers, 2), dtype=torch.float32)
    return {
        "observation": {
            "head": {"rgb": torch.zeros((270, 480, 3), dtype=torch.uint8)},
        },
        "embodiment": {
            "joint": torch.tensor(
                [0.0, 0.48, 0.0, -2.2, 0.0, 2.7, 0.76, 0.0043, 0.0043],
                dtype=torch.float32,
            )
        },
        "tactile": {
            "left_tactile": {"marker": marker.clone()},
            "right_tactile": {"marker": marker.clone()},
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--return_tactile", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg.get("tactile_input_type") != "marker":
        raise ValueError("This smoke test requires tactile_input_type: marker in deploy config.")

    mantis_root = Path(cfg["mantis_root"])
    _ensure_mantis_import_path(mantis_root)
    _prefer_conda_site_packages()

    from models.mantis import Mantis, MantisConfig

    transforms = _load_mantis_transforms(mantis_root)
    normalize_and_pad = transforms.normalize_and_pad
    unnormalize_and_unpad = transforms.unnormalize_and_unpad
    make_transform = transforms._make_transform

    with open(cfg["norm_stats_path"], "r", encoding="utf-8") as f:
        norm_stats = json.load(f)[cfg.get("unnorm_key", "insert_hdmi_tactile_lerobot")]

    input_size = int(cfg["target_image_size"]) // int(cfg["vae_downsample_f"])
    config_path = _resolve_training_config_path(mantis_root, cfg["training_config"])
    model_config = _load_training_config(config_path)

    if model_config.get("tactile_input_type") != "marker":
        raise ValueError(f"Training config tactile_input_type must be marker, got {model_config.get('tactile_input_type')}")

    model = _build_mantis_from_training_config(model_config, input_size, Mantis, MantisConfig)
    checkpoint_path = Path(cfg["checkpoint_path"])
    print(f"[smoke] Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(_load_checkpoint_state_dict(checkpoint_path), strict=True)

    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    dtype_name = str(cfg.get("dtype", "bfloat16"))
    model_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype_name, torch.bfloat16)
    model.to(device=device, dtype=model_dtype)
    model.eval()

    if hasattr(model, "vae"):
        del model.vae
    if hasattr(model.model, "transformer"):
        del model.model.transformer
    if hasattr(model.model, "connector"):
        del model.model.connector

    num_markers = int(cfg.get("tactile_num_markers", model_config.get("tactile_num_markers", 1200)))
    primary_image_size = int(cfg.get("primary_image_size", 256))
    primary_transform = make_transform(primary_image_size)
    obs = build_dummy_observation(num_markers)

    head = obs["observation"]["head"]["rgb"]
    if head.shape[-1] == 3:
        head = head.permute(2, 0, 1).float() / 255.0
        head = head[[2, 1, 0], ...]
    input_images = [[primary_transform(head)]]

    state = obs["embodiment"]["joint"][: int(cfg.get("original_action_dim", 8))].float()
    state, _ = normalize_and_pad(state, norm_stats["observation.state"], int(cfg.get("max_state_dim", 8)))
    state = state.unsqueeze(0).to(device=device, dtype=model_dtype)

    tactile_markers = _stack_marker_pair(
        obs["tactile"]["left_tactile"]["marker"],
        obs["tactile"]["right_tactile"]["marker"],
    ).to(device=device, dtype=model_dtype)

    sample_kwargs = {
        "caption": cfg.get("instruction", "clean"),
        "input_images": input_images,
        "num_images_per_prompt": 1,
        "states": state,
        "tactile_markers": tactile_markers,
    }
    if args.return_tactile or model_config.get("predict_tactile", False):
        sample_kwargs["return_tactile"] = True

    with torch.inference_mode():
        if cfg.get("use_autocast", True) and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=model_dtype):
                model_out = model.sample_actions(**sample_kwargs)
        else:
            model_out = model.sample_actions(**sample_kwargs)

    pred_tactiles = None
    if sample_kwargs.get("return_tactile"):
        actions, pred_tactiles = model_out
    else:
        actions = model_out

    actions = torch.as_tensor(actions, dtype=torch.float32)
    actions = unnormalize_and_unpad(
        actions,
        norm_stats["action"],
        int(cfg.get("original_action_dim", 8)),
    )
    action_horizon = int(cfg.get("action_horizon", 32))
    if action_horizon > 0:
        actions = actions[:action_horizon]

    print(
        f"smoke_ok tactile_input=marker "
        f"num_actions={len(actions)} action_shape={tuple(actions[0].shape)}"
    )
    if pred_tactiles is not None:
        pred_tactiles = torch.as_tensor(pred_tactiles)
        print(
            f"pred_tactiles_shape={tuple(pred_tactiles.shape)} "
            f"pred_tactile_dim={pred_tactiles.shape[-1]}"
        )


if __name__ == "__main__":
    main()
