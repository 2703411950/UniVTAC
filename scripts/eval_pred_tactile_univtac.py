#!/usr/bin/env python3
"""Offline eval: predicted future tactile vs GT on LeRobot dataset.

Designed for UniVTAC + WLA deploy config:
  policy/wla/deploy_tactile_encoder_predict_tactile.yml

Example:
  cd /data1/cyy/UniVTAC
  bash scripts/run_insert_hdmi_tactile_pred_eval.sh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

UNIVTAC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UNIVTAC_ROOT))

from policy.wla.deploy_policy import (
    _build_mantis_from_training_config,
    _ensure_mantis_import_path,
    _load_checkpoint_state_dict,
    _load_training_config,
    _prefer_conda_site_packages,
    _resolve_training_config_path,
)
from policy.wla.tactile_pred_utils import (
    TactileDecoderBank,
    decode_pred_tactile_images,
    describe_decoder_keys,
    plot_latent_heatmap,
    plot_latent_metrics,
    plot_tactile_image_comparison,
    save_contact_sheet,
)

DEFAULT_DEPLOY_CONFIG = UNIVTAC_ROOT / "policy/wla/deploy_tactile_encoder_predict_tactile.yml"
DEFAULT_TIMESTEPS = (0, 7, 15, 23, 31)


def load_deploy_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_training_config(mantis_root: Path, training_config: str) -> dict:
    config_path = _resolve_training_config_path(mantis_root, training_config)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(deploy_cfg: dict):
    mantis_root = Path(deploy_cfg["mantis_root"])
    if not mantis_root.exists():
        raise FileNotFoundError(f"mantis_root does not exist: {mantis_root}")

    _ensure_mantis_import_path(mantis_root)
    _prefer_conda_site_packages()

    from models.mantis import MantisConfig
    from dataset import LeRobotEvalDataset, load_custom_lerobot_dataset

    config_path = _resolve_training_config_path(
        mantis_root,
        deploy_cfg["training_config"],
    )
    model_config = _load_training_config(config_path)
    raw_model_config = load_raw_training_config(mantis_root, deploy_cfg["training_config"])
    input_size = int(deploy_cfg["target_image_size"]) // int(deploy_cfg["vae_downsample_f"])

    model = _build_mantis_from_training_config(
        model_config,
        input_size,
        __import__("models.mantis", fromlist=["Mantis"]).Mantis,
        MantisConfig,
    )
    checkpoint_path = Path(deploy_cfg["checkpoint_path"])
    print(f"[Eval] Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(_load_checkpoint_state_dict(checkpoint_path), strict=True)

    device = deploy_cfg.get("device", "cuda:0")
    dtype_name = deploy_cfg.get("dtype", "bfloat16")
    model_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype_name, torch.bfloat16)

    model.to(device=device, dtype=model_dtype)
    model.eval()
    return model, model_config, raw_model_config, mantis_root, LeRobotEvalDataset, load_custom_lerobot_dataset


def build_eval_dataset(model_config: dict, mantis_root: Path, dataset_root_dir: str | None, LeRobotEvalDataset, load_custom_lerobot_dataset):
    model_args = SimpleNamespace(**model_config)
    norm_stats_path = model_config["norm_stats_path"]
    if not os.path.isabs(norm_stats_path):
        norm_stats_path = str(mantis_root / norm_stats_path)

    data_args = SimpleNamespace(
        dataset_root_dir=dataset_root_dir or model_config["dataset_root_dir"],
        norm_stats_path=norm_stats_path,
        primary_image_key=model_config.get("primary_image_key", ""),
        auxiliary_image_keys=model_config.get("auxiliary_image_keys", []),
        tactile_image_keys=model_config.get("tactile_image_keys", []),
        tactile_marker_keys=model_config.get("tactile_marker_keys", []),
        unnorm_key=model_config["unnorm_key"],
    )
    training_args = SimpleNamespace(data_seed=42)

    base_dataset, norm_stats, primary_image_key, auxiliary_image_key, tactile_image_key, tactile_marker_key = (
        load_custom_lerobot_dataset(data_args, model_args, training_args)
    )
    return LeRobotEvalDataset(
        base_dataset=base_dataset,
        primary_image_size=model_config["primary_image_size"],
        auxiliary_image_size=model_config["auxiliary_image_size"],
        primary_image_key=primary_image_key,
        auxiliary_image_key=auxiliary_image_key,
        norm_stats=norm_stats,
        model_args=model_args,
        instruction_dict=None,
        tactile_image_key=tactile_image_key,
        tactile_marker_key=tactile_marker_key,
    )


@torch.no_grad()
def run_single_sample(model, sample, decoder_bank, device):
    states = sample["states"].unsqueeze(0).to(device)
    tactile_images = sample["tactile_images"].unsqueeze(0).to(device)
    target_tactile_images = sample["target_tactile_images"].unsqueeze(0).to(device)

    pred_actions, pred_tactiles = model.sample_actions(
        caption=sample["caption"],
        input_images=[sample["input_images"]],
        states=states,
        tactile_images=tactile_images,
        return_tactile=True,
    )

    gt_tactiles = model.model.encode_target_tactile_trajectory(
        target_tactile_images=target_tactile_images.to(device=device)
    )
    gt_tactiles_np = gt_tactiles[0].float().cpu().numpy()
    pred_tactiles_np = np.asarray(pred_tactiles, dtype=np.float32)

    per_step_mse = np.mean((pred_tactiles_np - gt_tactiles_np) ** 2, axis=1)
    per_step_cos = []
    for step in range(pred_tactiles_np.shape[0]):
        pred_vec = pred_tactiles_np[step]
        gt_vec = gt_tactiles_np[step]
        denom = np.linalg.norm(pred_vec) * np.linalg.norm(gt_vec) + 1e-8
        per_step_cos.append(float(np.dot(pred_vec, gt_vec) / denom))

    pred_left_imgs = pred_right_imgs = None
    if decoder_bank is not None and model.model.tactile_target_proj is not None:
        pred_left_imgs, pred_right_imgs = decode_pred_tactile_images(
            target_proj=model.model.tactile_target_proj,
            decoder_bank=decoder_bank,
            pred_latent=torch.as_tensor(pred_tactiles_np, device=device),
            latent_dim=model.config.tactile_latent_dim,
        )
        pred_left_imgs = pred_left_imgs.float().cpu()
        pred_right_imgs = pred_right_imgs.float().cpu()

    return {
        "caption": sample["caption"],
        "pred_actions": pred_actions,
        "pred_tactiles": pred_tactiles_np,
        "gt_tactiles": gt_tactiles_np,
        "per_step_mse": per_step_mse,
        "per_step_cos": np.asarray(per_step_cos, dtype=np.float32),
        "gt_left_imgs": target_tactile_images[0, :, 0].float().cpu(),
        "gt_right_imgs": target_tactile_images[0, :, 1].float().cpu(),
        "pred_left_imgs": pred_left_imgs,
        "pred_right_imgs": pred_right_imgs,
        "action_mse": float(np.mean((pred_actions - sample["actions"].numpy()) ** 2)),
    }


def parse_indices(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_timesteps(text: str | None) -> tuple[int, ...]:
    if not text:
        return DEFAULT_TIMESTEPS
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def main():
    parser = argparse.ArgumentParser(description="Evaluate predicted tactile trajectories for insert_hdmi WLA model.")
    parser.add_argument(
        "--deploy_config",
        default=str(DEFAULT_DEPLOY_CONFIG),
        help="UniVTAC deploy yaml, default: policy/wla/deploy_tactile_encoder_predict_tactile.yml",
    )
    parser.add_argument("--output_dir", default=str(UNIVTAC_ROOT / "eval_result/wla/insert_HDMI/tactile_pred_vis"))
    parser.add_argument("--dataset_root_dir", default=None, help="Override dataset_root_dir in training config")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--sample_indices", default="0,50,100,200,500")
    parser.add_argument("--timesteps", default="0,7,15,23,31")
    parser.add_argument("--inspect_decoder", action="store_true")
    args = parser.parse_args()

    deploy_cfg = load_deploy_config(Path(args.deploy_config))
    if not deploy_cfg.get("use_tactile_encoder", False):
        raise ValueError("Deploy config must use tactile encoder.")

    model, model_config, raw_model_config, mantis_root, LeRobotEvalDataset, load_custom_lerobot_dataset = build_model(deploy_cfg)
    if not model_config.get("predict_tactile", False):
        raise ValueError("Training config must have predict_tactile: true")

    tactile_encoder_ckpt = raw_model_config.get("tactile_encoder_ckpt")
    if args.inspect_decoder and tactile_encoder_ckpt:
        describe_decoder_keys(tactile_encoder_ckpt)

    eval_dataset = build_eval_dataset(
        model_config,
        mantis_root,
        args.dataset_root_dir,
        LeRobotEvalDataset,
        load_custom_lerobot_dataset,
    )

    decoder_bank = TactileDecoderBank.try_load(tactile_encoder_ckpt)
    if decoder_bank is not None:
        decoder_bank = decoder_bank.to(deploy_cfg.get("device", "cuda:0"))
        decoder_bank.eval()
    else:
        print("[WARN] Tactile image decoder unavailable. Saving GT images + latent plots only.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timesteps = parse_timesteps(args.timesteps)
    chunk_size = int(model_config.get("chunk_size", 32))
    timesteps = tuple(step for step in timesteps if 0 <= step < chunk_size)

    sample_indices = parse_indices(args.sample_indices)
    if not sample_indices:
        sample_indices = list(range(min(args.num_samples, len(eval_dataset))))
    else:
        sample_indices = sample_indices[: args.num_samples]

    device = deploy_cfg.get("device", "cuda:0")
    summary = []
    for rank, dataset_idx in enumerate(sample_indices):
        print(f"[{rank + 1}/{len(sample_indices)}] sample {dataset_idx}")
        sample = eval_dataset[dataset_idx]
        result = run_single_sample(model, sample, decoder_bank, device)

        sample_dir = output_dir / f"sample_{dataset_idx:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        np.savez(
            sample_dir / "tactile_pred_vs_gt.npz",
            pred_tactiles=result["pred_tactiles"],
            gt_tactiles=result["gt_tactiles"],
            per_step_mse=result["per_step_mse"],
            per_step_cos=result["per_step_cos"],
            pred_actions=result["pred_actions"],
            caption=result["caption"],
        )
        plot_latent_metrics(result, sample_dir / "latent_metrics.png")
        plot_latent_heatmap(result, sample_dir / "latent_heatmap.png")
        plot_tactile_image_comparison(result, timesteps, sample_dir / "tactile_image_compare.png")
        save_contact_sheet(result, timesteps, sample_dir / "tactile_contact_sheet.png")

        summary.append(
            {
                "dataset_idx": dataset_idx,
                "caption": result["caption"],
                "mean_tactile_mse": float(np.mean(result["per_step_mse"])),
                "mean_tactile_cos": float(np.mean(result["per_step_cos"])),
                "action_mse": result["action_mse"],
                "has_pred_images": result["pred_left_imgs"] is not None,
            }
        )
        print(
            f"  tactile MSE={summary[-1]['mean_tactile_mse']:.5f}, "
            f"cos={summary[-1]['mean_tactile_cos']:.4f}, "
            f"action MSE={summary[-1]['action_mse']:.5f}"
        )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Done. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
