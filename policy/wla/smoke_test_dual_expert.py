"""Smoke test for dual-expert WLA deploy configs (no Isaac Sim).

Example:
  cd /data1/cyy/UniVTAC
  python policy/wla/smoke_test_dual_expert.py
  python policy/wla/smoke_test_dual_expert.py --config policy/wla/deploy_dual_expert_lift_bottle_predict_tactile.yml
  python policy/wla/smoke_test_dual_expert.py --return-tactile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from policy.wla.deploy_policy import Policy

DEFAULT_CONFIG = Path(__file__).with_name("deploy_dual_expert_insert_hdmi_predict_tactile.yml")


def build_dummy_observation(use_tactile_encoder: bool):
    obs = {
        "observation": {
            "head": {"rgb": torch.zeros((270, 480, 3), dtype=torch.uint8)},
        },
        "embodiment": {
            "joint": torch.tensor(
                [0.0, 0.48, 0.0, -2.2, 0.0, 2.7, 0.76, 0.0043, 0.0043],
                dtype=torch.float32,
            )
        },
    }
    if use_tactile_encoder:
        obs["tactile"] = {
            "left_tactile": {"rgb_marker": torch.zeros((240, 320, 3), dtype=torch.uint8)},
            "right_tactile": {"rgb_marker": torch.zeros((240, 320, 3), dtype=torch.uint8)},
        }
    return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--return-tactile", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = Path(cfg["checkpoint_path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Update checkpoint_path in the deploy config after training finishes."
        )

    cfg["debug_log"] = True
    if args.return_tactile:
        cfg["eval_predict_tactile"] = True

    policy = Policy(cfg)
    expert_type = getattr(policy.model.config, "expert_type", "joint")
    predict_tactile = getattr(policy.model.config, "predict_tactile", False)
    if expert_type != "dual":
        raise ValueError(f"Expected expert_type=dual, got {expert_type!r}")
    if not predict_tactile:
        raise ValueError("Dual-expert deploy config must set predict_tactile=true in training config.")

    obs = build_dummy_observation(use_tactile_encoder=cfg.get("use_tactile_encoder", False))

    class Task:
        instruction = cfg.get("instruction", "clean")

    if args.return_tactile:
        policy.eval_predict_tactile = True
        policy.temporal_agg = False

    actions, pred_tactiles = policy._sample_action_chunk(Task(), obs)
    print(
        f"smoke_ok config={cfg_path.name} "
        f"expert_type={expert_type} predict_tactile={predict_tactile} "
        f"num_actions={len(actions)} action_shape={tuple(actions[0].shape)} "
        f"history_obs={policy.use_history_obs} num_head_images={len(policy.encode_obs(obs)[0][0])}"
    )
    if pred_tactiles is not None:
        print(f"pred_tactiles_shape={tuple(pred_tactiles.shape)}")


if __name__ == "__main__":
    main()
