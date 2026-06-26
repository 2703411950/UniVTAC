"""Regression tests for tactile prediction eval finalization."""

from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np
import torch

from policy.wla.deploy_policy import Policy


class _DummyInnerModel:
    tactile_target_proj = None

    def __init__(self):
        self.mode = None
        self.shape = None

    def encode_target_tactile_trajectory(
        self,
        target_tactile_images=None,
        target_tactile_markers=None,
    ):
        if target_tactile_markers is not None:
            target = target_tactile_markers
            self.mode = "marker"
        elif target_tactile_images is not None:
            target = target_tactile_images
            self.mode = "image"
        else:
            raise AssertionError("Expected image or marker target tactile input.")

        self.shape = tuple(target.shape)
        batch_size, horizon = target.shape[:2]
        return torch.zeros((batch_size, horizon, 64), device=target.device)


class _DummyModel:
    def __init__(self):
        self.model = _DummyInnerModel()
        self.config = SimpleNamespace(tactile_latent_dim=512)


def _make_policy(mode: str, save_root: Path):
    policy = object.__new__(Policy)
    policy.action_horizon = 4
    policy.device = torch.device("cpu")
    policy.tactile_input_type = mode
    policy.tactile_eval_timesteps = ()
    policy._tactile_decoder_bank = None
    policy._tactile_eval_active = True
    policy._tactile_chunk_start = 0
    policy._pred_tactiles = np.ones((4, 64), dtype=np.float32)
    policy._episode_idx = 0
    policy._tactile_eval_summary = []
    policy._current_task_save_root = save_root
    policy.model = _DummyModel()

    if mode == "marker":
        policy._gt_tac_left = [torch.full((2, 5, 2), float(i)) for i in range(4)]
        policy._gt_tac_right = [torch.full((2, 5, 2), float(i + 1)) for i in range(4)]
    else:
        policy._gt_tac_left = [torch.full((3, 8, 8), float(i)) for i in range(4)]
        policy._gt_tac_right = [torch.full((3, 8, 8), float(i + 1)) for i in range(4)]
    return policy


def test_finalize_uses_image_targets():
    with tempfile.TemporaryDirectory() as tmpdir:
        save_root = Path(tmpdir)
        policy = _make_policy("image", save_root)
        policy._finalize_tactile_eval_chunk(SimpleNamespace(save_root=save_root, instruction=""))

        assert policy.model.model.mode == "image"
        assert policy.model.model.shape == (1, 4, 2, 3, 8, 8)
        assert (save_root / "tactile_pred" / "summary.json").is_file()


def test_finalize_uses_marker_targets():
    with tempfile.TemporaryDirectory() as tmpdir:
        save_root = Path(tmpdir)
        policy = _make_policy("marker", save_root)
        policy._finalize_tactile_eval_chunk(SimpleNamespace(save_root=save_root, instruction=""))

        assert policy.model.model.mode == "marker"
        assert policy.model.model.shape == (1, 4, 2, 2, 5, 2)
        assert (save_root / "tactile_pred" / "summary.json").is_file()


if __name__ == "__main__":
    test_finalize_uses_image_targets()
    test_finalize_uses_marker_targets()
    print("tactile_eval_finalize_test: OK")
