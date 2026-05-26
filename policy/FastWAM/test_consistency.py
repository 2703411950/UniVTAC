"""
Consistency test: verify standalone preprocessing matches FastWAM training.

Run in univtac_fastwam_copy env from UniVTAC root:
    python policy/FastWAM/test_consistency.py
"""

import sys
import os
import json

import torch
import torchvision.transforms.functional as transforms_F

UNIVTAC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# UNIVTAC_ROOT is .../cyy/Issca_lab/UniVTAC
# FastWAM is .../cyy/FastWAM
FASTWAM_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(UNIVTAC_ROOT)),
    "FastWAM"
)
sys.path.insert(0, UNIVTAC_ROOT)

from policy.FastWAM.dataset_utils import (
    ResizeSmallestSideAspectPreserving,
    CenterCrop,
    Normalize,
)
from policy.FastWAM.normalizer import LinearNormalizer
from policy.FastWAM.action_state_merger import ConcatLeftAlign


def test_preprocessing_consistency():
    print("=" * 70)
    print("CONSISTENCY TEST: Standalone vs Training Preprocessing")
    print("=" * 70)

    sample_path = "/tmp/fastwam_test_sample.pt"
    print(f"\n[1] Loading raw sample...")
    sample = torch.load(sample_path, map_location="cpu")
    raw_images = sample["images"]
    raw_state = sample["state"]
    for k, v in raw_images.items():
        print(f"  {k}: {v.shape}, dtype={v.dtype}")
    print(f"  state: {raw_state.shape}")

    num_frames = 33
    action_video_freq_ratio = 4
    cam_keys = ["cam_high", "cam_wrist", "tac_left", "tac_right"]
    video_sample_indices = list(range(0, num_frames, action_video_freq_ratio))
    print(f"  Video sample indices: {video_sample_indices}")

    dataset_stats_path = os.path.join(
        FASTWAM_ROOT,
        "runs/insert_hdmi_tactile_uncond_2cam224_1e-4/"
        "2026-05-25_22-36-18/dataset_stats.json",
    )

    shape_meta = {
        "action": [{"key": "default", "raw_shape": 8, "shape": 8}],
        "state": [{"key": "default", "raw_shape": 8, "shape": 8}],
    }

    grid_resize = ResizeSmallestSideAspectPreserving(args={"img_w": 448, "img_h": 448})
    grid_crop = CenterCrop(args={"img_w": 448, "img_h": 448})
    grid_norm = Normalize(args={"mean": 0.5, "std": 0.5})

    # ============================================================
    # Training-style preprocessing (replicating FastWAM exactly)
    # ============================================================
    print("\n[2] Training-style preprocessing (FastWAM pipeline)...")

    fw_cameras = {}
    for key in cam_keys:
        imgs = raw_images[key].float() / 255.0  # [T, C, H, W], [0, 1]
        imgs = imgs[video_sample_indices]
        resized = []
        for t in range(imgs.shape[0]):
            frame = transforms_F.resize(
                imgs[t], size=[224, 224],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            resized.append(frame)
        fw_cameras[key] = torch.stack(resized, dim=0)
        print(f"  FW {key}: {fw_cameras[key].shape}")

    fw_grid_list = []
    for t in range(fw_cameras["cam_high"].shape[0]):
        row0 = torch.cat([fw_cameras["cam_high"][t], fw_cameras["cam_wrist"][t]], dim=-1)
        row1 = torch.cat([fw_cameras["tac_left"][t], fw_cameras["tac_right"][t]], dim=-1)
        fw_grid_list.append(torch.cat([row0, row1], dim=-2))
    fw_grid = torch.stack(fw_grid_list, dim=0)
    print(f"  FW grid: {fw_grid.shape}")

    fw_transformed = grid_resize(fw_grid)
    fw_transformed = grid_crop(fw_transformed)
    fw_transformed = grid_norm(fw_transformed)
    fw_video = fw_transformed.permute(1, 0, 2, 3)  # [3, T_video, 448, 448]
    print(f"  FW video: range=[{fw_video.min():.6f}, {fw_video.max():.6f}]")

    with open(dataset_stats_path, "r") as f:
        ds_stats = json.load(f)
    ds_stats = _json_to_tensor_stats(ds_stats)
    normalizer = LinearNormalizer(
        shape_meta=shape_meta, use_stepwise_action_norm=False,
        default_mode="min/max", exception_mode=None, stats=ds_stats,
    )
    fw_state_raw = raw_state[:num_frames].clone().float()
    state_batch = {"state": {"default": fw_state_raw}}
    fw_state_norm = normalizer.forward(state_batch)["state"]["default"].clamp(-5.0, 5.0)
    print(f"  FW state norm: range=[{fw_state_norm.min():.6f}, {fw_state_norm.max():.6f}]")

    # ============================================================
    # Standalone preprocessing (matching deploy_policy.py exactly)
    # ============================================================
    print("\n[3] Standalone preprocessing (deploy_policy.py style)...")

    def sa_process_camera(rgb_chw_uint8_t):
        """CHW uint8 -> HWC uint8 -> process like deploy_policy.encode_obs"""
        hwc = rgb_chw_uint8_t.permute(1, 2, 0)          # [H, W, C] uint8
        img = hwc.permute(2, 0, 1).float() / 255.0      # [3, H, W], [0, 1]
        img = transforms_F.resize(
            img, size=[224, 224],
            interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True,
        )
        return img

    sa_video_frames = []
    sa_state_frames = []
    for t in video_sample_indices:
        sa_high = sa_process_camera(raw_images["cam_high"][t])
        sa_wrist = sa_process_camera(raw_images["cam_wrist"][t])
        sa_left = sa_process_camera(raw_images["tac_left"][t])
        sa_right = sa_process_camera(raw_images["tac_right"][t])

        row0 = torch.cat([sa_high, sa_wrist], dim=-1)
        row1 = torch.cat([sa_left, sa_right], dim=-1)
        sa_grid = torch.cat([row0, row1], dim=-2)

        sa_grid = grid_resize(sa_grid)
        sa_grid = grid_crop(sa_grid)
        sa_grid = grid_norm(sa_grid)
        sa_video_frames.append(sa_grid)
        sa_state_frames.append(raw_state[t].clone().float())

    sa_video = torch.stack(sa_video_frames, dim=1)
    print(f"  SA video: range=[{sa_video.min():.6f}, {sa_video.max():.6f}]")

    sa_state = torch.stack(sa_state_frames, dim=0)
    sa_state_batch = {"state": {"default": sa_state.unsqueeze(0)}}
    sa_state_norm = normalizer.forward(sa_state_batch)["state"]["default"][0].clamp(-5.0, 5.0)
    print(f"  SA state norm: range=[{sa_state_norm.min():.6f}, {sa_state_norm.max():.6f}]")

    # ================================================================
    # Compare
    # ================================================================
    print("\n[4] Comparing...")
    tol = 1e-5
    all_pass = True

    for t_idx in range(fw_video.shape[1]):
        diff = (fw_video[:, t_idx, :, :] - sa_video[:, t_idx, :, :]).abs()
        max_d, mean_d = diff.max().item(), diff.mean().item()
        if max_d > tol:
            all_pass = False
            print(f"  FAIL: Frame {t_idx} (t={video_sample_indices[t_idx]}): "
                  f"max_diff={max_d:.10f}, mean_diff={mean_d:.10f}")
    vid_max = (fw_video - sa_video).abs().max().item()
    print(f"  Video overall max_diff: {vid_max:.10f}")
    if vid_max <= tol:
        print(f"  PASS: All {fw_video.shape[1]} video frames match")

    state_diff = (fw_state_norm[video_sample_indices] - sa_state_norm).abs().max().item()
    print(f"  State norm max_diff: {state_diff:.10f}")
    if state_diff <= tol:
        print(f"  PASS: State normalization matches")
    else:
        print(f"  FAIL: State mismatch")
        all_pass = False

    # ================================================================
    # Action denorm roundtrip
    # ================================================================
    print("\n[5] Action denormalization roundtrip...")
    merger = ConcatLeftAlign()
    merger.set_shape_meta(shape_meta)

    raw_action = sample["action"].float()[:32]  # [32, 8]
    dummy_state = {"default": torch.zeros(32, 8)}
    fwd_batch = {"action": {"default": raw_action}, "state": dummy_state}
    fwd_batch = normalizer.forward(fwd_batch)
    fwd_batch = merger.forward(fwd_batch)  # action: [32, 8], state: [32, 8]
    fwd_action = fwd_batch["action"]  # [32, 8]
    fwd_state = fwd_batch["state"]    # [32, 8]

    # Backward: merger expects 3D [B, T, D]
    bwd_batch = {
        "action": fwd_action.clone().unsqueeze(0),  # [1, 32, 8]
        "state": fwd_state.clone().unsqueeze(0),    # [1, 32, 8]
    }
    bwd_batch = merger.backward(bwd_batch)
    bwd_batch = normalizer.backward(bwd_batch)
    bwd_action = bwd_batch["action"]["default"]  # [1, 32, 8]

    rtt_diff = (raw_action.unsqueeze(0) - bwd_action).abs().max().item()
    rtt_mean = (raw_action.unsqueeze(0) - bwd_action).abs().mean().item()
    print(f"  Roundtrip max_diff: {rtt_diff:.10f}, mean_diff: {rtt_mean:.10f}")
    if rtt_diff <= tol:
        print(f"  PASS: Action denormalization roundtrip")
    else:
        print(f"  FAIL: Roundtrip mismatch")
        all_pass = False

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)
    return all_pass


def _json_to_tensor_stats(stats):
    def _is_num_list(obj):
        if not isinstance(obj, list) or not obj:
            return False
        if isinstance(obj[0], (int, float)):
            return True
        if isinstance(obj[0], list):
            return all(_is_num_list(x) for x in obj)
        return False

    def _convert(obj):
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if _is_num_list(obj):
            return torch.tensor(obj, dtype=torch.float32)
        return obj
    return _convert(stats)


if __name__ == "__main__":
    sys.exit(0 if test_preprocessing_consistency() else 1)
