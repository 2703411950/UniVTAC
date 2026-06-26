import sys
import argparse
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from policy.wla.deploy_policy import Policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("deploy.yml")))
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg.update({"task_name": "insert_HDMI", "task_config": "demo"})
    cfg["debug_log"] = True

    policy = Policy(cfg)
    obs = {
        "observation": {
            "head": {"rgb": torch.zeros((270, 480, 3), dtype=torch.uint8)},
        },
        "embodiment": {
            "joint": torch.tensor([
                0.0, 0.48, 0.0, -2.2, 0.0, 2.7, 0.76, 0.0043, 0.0043
            ], dtype=torch.float32)
        },
    }
    if cfg.get("use_tactile_encoder", False) or cfg.get("use_tactile_images", True):
        obs["tactile"] = {
            "left_tactile": {"rgb_marker": torch.zeros((240, 320, 3), dtype=torch.uint8)},
            "right_tactile": {"rgb_marker": torch.zeros((240, 320, 3), dtype=torch.uint8)},
        }

    class Task:
        instruction = "clean"

    actions, _ = policy._sample_action_chunk(Task(), obs)
    print(f"smoke_ok num_actions={len(actions)} action_shape={tuple(actions[0].shape)}")


if __name__ == "__main__":
    main()
