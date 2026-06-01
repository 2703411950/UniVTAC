"""
FastWAM policy deployment for UniVTAC simulation.

Supports three camera modes (set via `cam_mode` in deploy yml):
  - "grid_2x2":      4 cameras in 2x2 grid → 448×448 video (base FastWAM)
  - "horizontal":    2 cameras horizontal concat → 224×448 video, no tactile (base FastWAM)
  - "tactile_encoder": 2 visual cameras horizontal → 224×448 video
                       + 2 tactile cameras separate → ResNet18 → context (FastWAMTactileEncoder)
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from torchvision.utils import save_image

_sys_path = str(Path(__file__).parent.parent)
if _sys_path not in sys.path:
    sys.path.append(_sys_path)

from _base_policy import BasePolicy

from .dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from .normalizer import LinearNormalizer
from .action_state_merger import ConcatLeftAlign

DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


class Policy(BasePolicy):
    def __init__(self, args: dict):
        super().__init__(args)

        # ---- paths ----
        checkpoint_path = Path(args["checkpoint_path"])
        dataset_stats_path = Path(args["dataset_stats_path"])
        model_id = str(args["model_id"])
        tokenizer_model_id = str(args["tokenizer_model_id"])
        action_dit_pretrained_path = str(args["action_dit_pretrained_path"])

        diffsynth_base = str(args["diffsynth_model_base_path"])
        if "DIFFSYNTH_MODEL_BASE_PATH" not in os.environ:
            os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = diffsynth_base

        tokenizer_max_len = int(args.get("tokenizer_max_len", 128))
        action_horizon = int(args.get("action_horizon", 32))
        self.num_inference_steps = int(args.get("num_inference_steps", 10))
        self.device_str = str(args.get("device", "cuda:0"))
        self.debug_log = bool(args.get("debug_log", False))
        self.debug_log_infer_batches = int(args.get("debug_log_infer_batches", 4))
        self.debug_log_exec_steps = int(args.get("debug_log_exec_steps", 128))
        self.debug_save_images = bool(args.get("debug_save_images", True))
        self.clip_state_to_train_range = bool(args.get("clip_state_to_train_range", False))
        self.instruction_override = args.get("instruction", None)
        self._debug_dir: Path | None = None
        self._debug_jsonl: Path | None = None
        self._episode_idx = -1
        self._infer_batch_idx = 0
        self._exec_step_idx = 0

        # ---- camera mode ----
        self.cam_mode = str(args.get("cam_mode", "grid_2x2"))
        valid_modes = ("grid_2x2", "horizontal", "tactile_encoder")
        if self.cam_mode not in valid_modes:
            raise ValueError(f"cam_mode must be one of {valid_modes}, got {self.cam_mode}")
        self.use_tactile = self.cam_mode in ("grid_2x2", "tactile_encoder")
        if self.cam_mode in ("grid_2x2", "tactile_encoder") and "tactile" not in str(checkpoint_path):
            print(
                "[FastWAM][WARN] cam_mode uses tactile input but checkpoint_path does not contain "
                f"'tactile': {checkpoint_path}"
            )
        if self.cam_mode == "horizontal" and "tactile" in str(checkpoint_path):
            print(
                "[FastWAM][WARN] horizontal cam_mode ignores tactile input but checkpoint_path looks tactile: "
                f"{checkpoint_path}"
            )

        # ---- video size ----
        vw, vh = args.get("video_size", [448, 448])
        self.video_w = int(vw)
        self.video_h = int(vh)

        # ---- build model ----
        video_dit_config = {
            "has_image_input": False,
            "patch_size": [1, 2, 2],
            "in_dim": 48,
            "hidden_dim": 3072,
            "ffn_dim": 14336,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 48,
            "num_heads": 24,
            "attn_head_dim": 128,
            "num_layers": 30,
            "eps": 1e-6,
            "seperated_timestep": True,
            "require_clip_embedding": False,
            "require_vae_embedding": False,
            "fuse_vae_embedding_in_latents": True,
            "use_gradient_checkpointing": False,
            "video_attention_mask_mode": "first_frame_causal",
            "action_conditioned": False,
            "action_dim": 8,
            "action_group_causal_mask_mode": "group_diagonal",
        }

        action_dit_config = {
            "action_dim": 8,
            "hidden_dim": 1024,
            "ffn_dim": 4096,
            "num_heads": 24,
            "attn_head_dim": 128,
            "num_layers": 30,
            "text_dim": 4096,
            "freq_dim": 256,
            "eps": 1e-6,
            "use_gradient_checkpointing": False,
        }

        common_kwargs = dict(
            device=self.device_str,
            torch_dtype=torch.bfloat16,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            load_text_encoder=True,
            proprio_dim=8,
            redirect_common_files=True,
            video_dit_config=video_dit_config,
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=False,
            mot_checkpoint_mixed_attn=False,
            video_train_shift=5.0,
            video_infer_shift=5.0,
            video_num_train_timesteps=1000,
            action_train_shift=5.0,
            action_infer_shift=5.0,
            action_num_train_timesteps=1000,
            loss_lambda_video=1.0,
            loss_lambda_action=1.0,
        )

        if self.cam_mode == "tactile_encoder":
            from .wan22.fastwam_tactile_encoder import FastWAMTactileEncoder
            self.model = FastWAMTactileEncoder.from_wan22_pretrained(
                model_id=model_id,
                tactile_encoder_freeze=False,
                tactile_encoder_ckpt=None,
                **common_kwargs,
            )
        else:
            from .wan22.fastwam import FastWAM
            self.model = FastWAM.from_wan22_pretrained(
                model_id=model_id,
                **common_kwargs,
            )

        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        if self.cam_mode == "tactile_encoder":
            # Checkpoint saved by base FastWAM.save_checkpoint: only mot + proprio_encoder
            # Tactile encoder was ImageNet-pretrained and frozen during training,
            # so we load mot/proprio separately and keep tactile encoder as initialized.
            self.model.mot.load_state_dict(ckpt["mot"], strict=False)
            if "proprio_encoder" in ckpt and self.model.proprio_encoder is not None:
                self.model.proprio_encoder.load_state_dict(ckpt["proprio_encoder"])
            print(f"  Loaded MoT + proprio_encoder from checkpoint (tactile encoder kept as init)")
        else:
            self.model.load_checkpoint(str(checkpoint_path))
        self.model.eval()

        # ---- load normalizer stats ----
        with open(dataset_stats_path, "r") as f:
            dataset_stats = json.load(f)
        dataset_stats = _json_to_tensor_stats(dataset_stats)
        self.original_dataset_stats = _clone_tensor_stats(dataset_stats)
        self._apply_state_stats_overrides(dataset_stats, args)
        self.dataset_stats = dataset_stats

        shape_meta = {
            "action": [{"key": "default", "raw_shape": 8, "shape": 8}],
            "state": [{"key": "default", "raw_shape": 8, "shape": 8}],
        }
        self.normalizer = LinearNormalizer(
            shape_meta=shape_meta,
            use_stepwise_action_norm=False,
            default_mode="min/max",
            exception_mode=None,
            stats=dataset_stats,
        )
        self.action_state_merger = ConcatLeftAlign()
        self.action_state_merger.set_shape_meta(shape_meta)

        # ---- image transforms ----
        self.grid_resize = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_w, "img_h": self.video_h}
        )
        self.grid_crop = CenterCrop(args={"img_w": self.video_w, "img_h": self.video_h})
        self.grid_normalize = Normalize(args={"mean": 0.5, "std": 0.5})

        # ---- action buffer ----
        self.action_horizon = action_horizon
        self._action_buffer: list = []

        print(f"FastWAM policy loaded. mode={self.cam_mode}, device={self.device_str}, "
              f"video={self.video_w}x{self.video_h}, action_horizon={self.action_horizon}, "
              f"num_inference_steps={self.num_inference_steps}")
        if self.debug_log:
            print("[FastWAM] Debug logging enabled. Logs will be created under task.save_root/FastWAM_debug.")

    def _apply_state_stats_overrides(self, dataset_stats: dict, args: dict) -> None:
        state_min_override = args.get("state_global_min_override", None)
        state_max_override = args.get("state_global_max_override", None)
        if state_min_override is None and state_max_override is None:
            return

        state_stats = dataset_stats["state"]["default"]
        for stat_name, override in (
            ("global_min", state_min_override),
            ("global_max", state_max_override),
        ):
            if override is None:
                continue
            stat = state_stats[stat_name].clone().float()
            if len(override) != stat.numel():
                raise ValueError(
                    f"{stat_name}_override length must be {stat.numel()}, got {len(override)}"
                )
            for idx, value in enumerate(override):
                if value is not None:
                    stat[idx] = float(value)
            state_stats[stat_name] = stat

    # ------------------------------------------------------------------
    #  Observation encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _process_camera(rgb_tensor: torch.Tensor) -> torch.Tensor:
        """HWC uint8 -> CHW float [0,1], resize to 224x224."""
        img = rgb_tensor.permute(2, 0, 1).float() / 255.0
        img = transforms_F.resize(
            img, size=[224, 224],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        return img  # [3, 224, 224], [0, 1]

    def _ensure_debug_dir(self, task=None) -> None:
        if not self.debug_log or self._debug_dir is not None:
            return
        if task is not None and hasattr(task, "save_root"):
            root = Path(task.save_root)
        else:
            root = Path("eval_result") / "FastWAM_debug_standalone"
        self._debug_dir = root / "FastWAM_debug"
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        self._debug_jsonl = self._debug_dir / "trace.jsonl"
        summary = {
            "event": "init",
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cam_mode": self.cam_mode,
            "video_size": [self.video_w, self.video_h],
            "action_horizon": self.action_horizon,
            "num_inference_steps": self.num_inference_steps,
            "state_stats": {
                "original_global_min": self._tensor_summary(
                    self.original_dataset_stats["state"]["default"]["global_min"]
                ),
                "original_global_max": self._tensor_summary(
                    self.original_dataset_stats["state"]["default"]["global_max"]
                ),
                "effective_global_min": self._tensor_summary(
                    self.dataset_stats["state"]["default"]["global_min"]
                ),
                "effective_global_max": self._tensor_summary(
                    self.dataset_stats["state"]["default"]["global_max"]
                ),
            },
            "training_expected": {
                "insert_hdmi_tactile_uncond_2cam224_1e-4": {
                    "camera_order": ["cam_high", "cam_wrist", "tac_left", "tac_right"],
                    "camera_resize": [224, 224],
                    "concat_multi_camera": "grid_2x2",
                    "final_video_size": [448, 448],
                    "image_norm": "Normalize(mean=0.5, std=0.5)",
                    "state_dim": 8,
                    "action_dim": 8,
                    "action_horizon": 32,
                }
            },
        }
        self._write_debug(summary)

    @staticmethod
    def _tensor_summary(x) -> dict:
        if isinstance(x, torch.Tensor):
            t = x.detach().float().cpu()
        else:
            t = torch.as_tensor(x).detach().float().cpu()
        if t.numel() == 0:
            return {"shape": list(t.shape), "numel": 0}
        return {
            "shape": list(t.shape),
            "dtype": str(getattr(x, "dtype", t.dtype)),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "mean": float(t.mean().item()),
            "std": float(t.std(unbiased=False).item()),
            "first": [float(v) for v in t.flatten()[:8].tolist()],
        }

    def _write_debug(self, payload: dict) -> None:
        if not self.debug_log or self._debug_jsonl is None:
            return
        with open(self._debug_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _save_debug_image(self, name: str, tensor: torch.Tensor, value_range: str) -> None:
        if not self.debug_log or not self.debug_save_images or self._debug_dir is None:
            return
        img = tensor.detach().float().cpu()
        if img.ndim == 4:
            img = img[0]
        if value_range == "minus1_1":
            img = (img + 1.0) / 2.0
        img = img.clamp(0.0, 1.0)
        save_image(img, self._debug_dir / name)

    def encode_obs(self, observation: dict, instruction: str = "insert hdmi") -> dict:
        """Process UniVTAC observation into model input format.

        Returns dict with keys depending on cam_mode:
          Always:  input_image, proprio, prompt
          tactile_encoder: also tactile_images [2, 3, 224, 224]
        """
        cam_high = self._process_camera(observation["observation"]["head"]["rgb"])
        cam_wrist = self._process_camera(observation["observation"]["wrist"]["rgb"])
        debug_payload = None
        if self.debug_log and self._infer_batch_idx < self.debug_log_infer_batches:
            debug_payload = {
                "event": "encode_obs",
                "episode": self._episode_idx,
                "infer_batch": self._infer_batch_idx,
                "instruction": instruction,
                "raw": {
                    "cam_high": self._tensor_summary(observation["observation"]["head"]["rgb"]),
                    "cam_wrist": self._tensor_summary(observation["observation"]["wrist"]["rgb"]),
                    "joint8": self._tensor_summary(observation["embodiment"]["joint"][:8]),
                },
                "processed_224": {
                    "cam_high": self._tensor_summary(cam_high),
                    "cam_wrist": self._tensor_summary(cam_wrist),
                },
            }

        if self.cam_mode == "grid_2x2":
            # 4 cameras in 2x2 grid
            tac_left = self._process_camera(observation["tactile"]["left_tactile"]["rgb_marker"])
            tac_right = self._process_camera(observation["tactile"]["right_tactile"]["rgb_marker"])
            row0 = torch.cat([cam_high, cam_wrist], dim=-1)   # [3, 224, 448]
            row1 = torch.cat([tac_left, tac_right], dim=-1)   # [3, 224, 448]
            grid = torch.cat([row0, row1], dim=-2)            # [3, 448, 448]

        elif self.cam_mode == "horizontal":
            # 2 visual cameras, horizontal concat
            grid = torch.cat([cam_high, cam_wrist], dim=-1)   # [3, 224, 448]

        elif self.cam_mode == "tactile_encoder":
            # 2 visual cameras horizontal, 2 tactile separate
            tac_left = self._process_camera(observation["tactile"]["left_tactile"]["rgb_marker"])
            tac_right = self._process_camera(observation["tactile"]["right_tactile"]["rgb_marker"])
            grid = torch.cat([cam_high, cam_wrist], dim=-1)   # [3, 224, 448]
        if debug_payload is not None and self.cam_mode in ("grid_2x2", "tactile_encoder"):
            debug_payload["raw"]["tac_left"] = self._tensor_summary(observation["tactile"]["left_tactile"]["rgb_marker"])
            debug_payload["raw"]["tac_right"] = self._tensor_summary(observation["tactile"]["right_tactile"]["rgb_marker"])
            debug_payload["processed_224"]["tac_left"] = self._tensor_summary(tac_left)
            debug_payload["processed_224"]["tac_right"] = self._tensor_summary(tac_right)

        # Final transforms
        grid_before_final = grid
        grid = self.grid_resize(grid)
        grid = self.grid_crop(grid)
        grid = self.grid_normalize(grid)

        input_image = grid.unsqueeze(1).unsqueeze(0)  # [1, 3, 1, H, W]
        input_image = input_image.squeeze(2)           # [1, 3, H, W]

        # State normalization
        joint = observation["embodiment"]["joint"][:8].float().cpu()
        joint_for_norm = joint
        if self.clip_state_to_train_range:
            state_stats = self.dataset_stats["state"]["default"]
            state_min = state_stats["global_min"].float().cpu()
            state_max = state_stats["global_max"].float().cpu()
            joint_for_norm = torch.minimum(torch.maximum(joint_for_norm, state_min), state_max)
        joint_norm = self.normalizer.normalizers["state"]["default"].forward(
            joint_for_norm.unsqueeze(0)
        ).clamp(-5.0, 5.0)

        prompt = DEFAULT_PROMPT.format(task=instruction)

        result = {
            "input_image": input_image,
            "proprio": joint_norm,
            "prompt": prompt,
        }
        if self.cam_mode == "tactile_encoder":
            result["tactile_images"] = torch.stack([tac_left, tac_right], dim=0)  # [2, 3, 224, 224]

        if debug_payload is not None:
            debug_payload["grid_before_final"] = self._tensor_summary(grid_before_final)
            debug_payload["input_image"] = self._tensor_summary(input_image)
            debug_payload["joint_for_norm"] = self._tensor_summary(joint_for_norm)
            debug_payload["joint_norm"] = self._tensor_summary(joint_norm)
            debug_payload["clip_state_to_train_range"] = self.clip_state_to_train_range
            debug_payload["prompt"] = prompt
            self._write_debug(debug_payload)
            prefix = f"ep_{self._episode_idx:03d}_infer_{self._infer_batch_idx:03d}"
            self._save_debug_image(f"{prefix}_cam_high_224.png", cam_high, "0_1")
            self._save_debug_image(f"{prefix}_cam_wrist_224.png", cam_wrist, "0_1")
            if self.cam_mode in ("grid_2x2", "tactile_encoder"):
                self._save_debug_image(f"{prefix}_tac_left_224.png", tac_left, "0_1")
                self._save_debug_image(f"{prefix}_tac_right_224.png", tac_right, "0_1")
            self._save_debug_image(f"{prefix}_grid_before_final.png", grid_before_final, "0_1")
            self._save_debug_image(f"{prefix}_input_image_norm.png", input_image, "minus1_1")

        return result

    # ------------------------------------------------------------------
    #  Action postprocessing
    # ------------------------------------------------------------------

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        T = action.shape[0]
        batch = {
            "action": action.unsqueeze(0),
            "state": torch.zeros(1, T, 8),
        }
        batch = self.action_state_merger.backward(batch)
        batch = self.normalizer.backward(batch)
        return batch["action"]["default"][0].cpu().numpy()

    # ------------------------------------------------------------------
    #  Main eval loop
    # ------------------------------------------------------------------

    def eval(self, task, observation):
        self._ensure_debug_dir(task)
        if not self._action_buffer:
            instruction = self.instruction_override or getattr(task, "instruction", "insert hdmi")
            obs = self.encode_obs(observation, instruction=instruction)

            infer_kwargs = dict(
                prompt=obs["prompt"],
                input_image=obs["input_image"],
                action_horizon=self.action_horizon,
                proprio=obs["proprio"],
                num_inference_steps=self.num_inference_steps,
                seed=None,
            )
            if self.cam_mode == "tactile_encoder":
                infer_kwargs["tactile_images"] = obs["tactile_images"].unsqueeze(0)  # [1, 2, 3, 224, 224]

            result = self.model.infer_action(**infer_kwargs)
            actions = self._denormalize_action(result["action"])
            self._action_buffer = [actions[i] for i in range(actions.shape[0])]
            if self.debug_log and self._infer_batch_idx < self.debug_log_infer_batches:
                self._write_debug({
                    "event": "infer_action",
                    "episode": self._episode_idx,
                    "infer_batch": self._infer_batch_idx,
                    "normalized_action": self._tensor_summary(result["action"]),
                    "denormalized_action": self._tensor_summary(actions),
                    "denormalized_action_delta_abs": self._tensor_summary(np.abs(np.diff(actions, axis=0))),
                })
            self._infer_batch_idx += 1

        action = self._action_buffer.pop(0)
        if self.debug_log and self._exec_step_idx < self.debug_log_exec_steps:
            current_joint = observation["embodiment"]["joint"][:8].detach().float().cpu().numpy()
            self._write_debug({
                "event": "execute_action",
                "episode": self._episode_idx,
                "exec_step": self._exec_step_idx,
                "buffer_remaining": len(self._action_buffer),
                "current_joint8": self._tensor_summary(current_joint),
                "action": self._tensor_summary(action),
                "action_minus_current_joint": self._tensor_summary(action - current_joint),
            })
        self._exec_step_idx += 1
        action_tensor = torch.from_numpy(action).to(task.device).float()
        task.take_action(action_tensor, action_type="qpos")

    def reset(self):
        self._action_buffer.clear()
        self._episode_idx += 1
        self._infer_batch_idx = 0
        self._exec_step_idx = 0

    def close(self):
        if hasattr(self.model, "to"):
            self.model.to("cpu")


# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------

def _json_to_tensor_stats(stats: dict) -> dict:
    def _is_numeric_list(obj):
        if not isinstance(obj, list) or not obj:
            return False
        if isinstance(obj[0], (int, float)):
            return True
        if isinstance(obj[0], list):
            return all(_is_numeric_list(x) for x in obj)
        return False

    def _convert(obj):
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if _is_numeric_list(obj):
            return torch.tensor(obj, dtype=torch.float32)
        return obj

    return _convert(stats)


def _clone_tensor_stats(stats: dict):
    if isinstance(stats, torch.Tensor):
        return stats.clone()
    if isinstance(stats, dict):
        return {k: _clone_tensor_stats(v) for k, v in stats.items()}
    if isinstance(stats, list):
        return [_clone_tensor_stats(v) for v in stats]
    return stats
