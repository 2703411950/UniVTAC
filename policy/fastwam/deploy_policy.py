import sys
from pathlib import Path
from collections import deque

# Make the fastwam package importable. The package lives at policy/fastwam/fastwam/,
# so we add its parent (policy/fastwam/) to sys.path.
_FASTWAM_ROOT = Path(__file__).resolve().parent
if str(_FASTWAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_FASTWAM_ROOT))

import os
import numpy as np
import torch
import torchvision.transforms.functional as TF

# Ensure DiffSynth finds locally-cached Wan base models instead of downloading
os.environ.setdefault(
    "DIFFSYNTH_MODEL_BASE_PATH",
    str(Path(__file__).resolve().parents[4] / "FastWAM" / "checkpoints"),
)

from policy._base_policy import BasePolicy

from fastwam.runtime import (
    create_fastwam,
    _mixed_precision_to_model_dtype,
)
from fastwam.datasets.lerobot.utils.normalizer import (
    LinearNormalizer,
    load_dataset_stats_from_json,
)

DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

_POLICY_DIR = Path(__file__).resolve().parent


def _resolve_path(path_str, base=_POLICY_DIR):
    """Resolve a path: absolute paths stay as-is, relative paths resolve from base.

    If the resolved path exists, return it. Otherwise return the original string
    (e.g. HuggingFace repo IDs pass through unchanged).
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    resolved = (base / p).resolve()
    if resolved.exists():
        return resolved
    return p


class Policy(BasePolicy):
    def __init__(self, args):
        super().__init__(args)

        # --- Resolve paths ---
        ckpt_path = _resolve_path(args.get("ckpt_path", ""))
        stats_path = _resolve_path(args.get("dataset_stats_path", ""))

        # --- Device & dtype ---
        mixed_precision = args.get("mixed_precision", "bf16")
        self._model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
        self._device = args.get("device", "cuda") if torch.cuda.is_available() else "cpu"

        # --- Build model ---
        model_variant = args.get("model_variant", "fastwam")
        model_id = str(_resolve_path(args.get("model_id", "Wan-AI/Wan2.2-TI2V-5B")))
        tokenizer_model_id = str(_resolve_path(args.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B")))
        model_kwargs = dict(
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=args.get("tokenizer_max_len", 128),
            load_text_encoder=args.get("load_text_encoder", True),
            proprio_dim=args.get("proprio_dim", 8),
            video_dit_config=args.get("video_dit_config", {}),
            action_dit_config=args.get("action_dit_config", {}),
            action_dit_pretrained_path=None,
            skip_dit_load_from_pretrain=True,
            video_scheduler=args.get("video_scheduler", {}),
            action_scheduler=args.get("action_scheduler", {}),
            loss=args.get("loss", {}),
            mot_checkpoint_mixed_attn=False,
            redirect_common_files=True,
            model_dtype=self._model_dtype,
            device=self._device,
        )

        if model_variant == "fastwam":
            self.model = create_fastwam(**model_kwargs)
        elif model_variant == "fastwam_joint":
            from fastwam.runtime import create_fastwam_joint
            self.model = create_fastwam_joint(**model_kwargs)
        elif model_variant == "fastwam_idm":
            from fastwam.runtime import create_fastwam_idm
            self.model = create_fastwam_idm(**model_kwargs)
        else:
            raise ValueError(f"Unknown model_variant: {model_variant}")

        # --- Load checkpoint ---
        if ckpt_path.exists():
            print(f"[FastWAM] Loading checkpoint: {ckpt_path}")
            self.model.load_checkpoint(str(ckpt_path))
        else:
            raise FileNotFoundError(f"[FastWAM] Checkpoint not found: {ckpt_path}")

        self.model = self.model.to(self._device).eval()

        # --- Load normalizer ---
        self._proprio_dim = args.get("proprio_dim", 8)
        self._action_dim = args.get("action_dim", 8)
        self._load_text_encoder = args.get("load_text_encoder", True)
        self._text_dim = (
            args.get("video_dit_config", {}).get("text_dim")
            or args.get("action_dit_config", {}).get("text_dim")
            or 4096
        )
        self._normalizer = None
        if stats_path.exists():
            stats = load_dataset_stats_from_json(str(stats_path))
            self._normalizer = LinearNormalizer(
                shape_meta={
                    "action": [{"key": "default"}],
                    "state": [{"key": "default"}],
                },
                use_stepwise_action_norm=False,
                default_mode="min/max",
                exception_mode=None,
                stats=stats,
            )
            print(f"[FastWAM] Loaded dataset stats: {stats_path}")
        else:
            print(f"[FastWAM] WARNING: dataset stats not found at {stats_path}, "
                  "action/proprio will not be normalized")

        # --- Inference settings ---
        self._action_horizon = args.get("action_horizon", 32)
        self._replan_steps = args.get("replan_steps", 8)
        self._num_inference_steps = args.get("num_inference_steps", 10)
        self._seed = args.get("seed", 0)
        self._text_cfg_scale = args.get("text_cfg_scale", 1.0)
        self._negative_prompt = args.get("negative_prompt", "")
        self._rand_device = args.get("rand_device", "cpu")
        self._tiled = args.get("tiled", False)
        self._action_type = args.get("action_type", "qpos")

        # --- Camera settings ---
        self._camera_names = args.get("camera_names", "head")
        if isinstance(self._camera_names, str):
            self._camera_names = [n.strip() for n in self._camera_names.split("+")]
        self._camera_width = args.get("camera_width", 224)
        self._camera_height = args.get("camera_height", 224)
        self._concat_mode = args.get("concat_multi_camera", "horizontal")

        # --- Action queue for chunked execution ---
        self._pending_actions = deque()

        print(f"[FastWAM] Initialized on {self._device} | "
              f"action_horizon={self._action_horizon} | "
              f"replan={self._replan_steps} | "
              f"cameras={self._camera_names} | "
              f"action_type={self._action_type}")

    # ------------------------------------------------------------------
    #  Observation encoding
    # ------------------------------------------------------------------

    def _resolve_dotted_path(self, data, path):
        """Navigate a nested dict via a dotted path string.

        E.g. _resolve_dotted_path(obs, "tactile.left_tactile.rgb")
        returns obs["tactile"]["left_tactile"]["rgb"].
        """
        parts = path.split(".")
        cur = data
        for p in parts:
            cur = cur[p]
        return cur

    def _preprocess_camera(self, rgb_tensor):
        """Single-camera image preprocessing, exactly matching training.

        Training pipeline (per camera, per frame):
          1. ToTensor          → [C, H, W] float [0, 1]
          2. Resize((224,224)) → [C, 224, 224] float [0, 1]

        Input:  torch.Tensor [H, W, 3] uint8  (0-255)  — raw UniVTAC camera
        Output: torch.Tensor [3, 224, 224] float [0, 1]
        """
        if rgb_tensor.dtype != torch.uint8:
            rgb_tensor = rgb_tensor.clamp(0, 255).to(torch.uint8)
        tensor = rgb_tensor.permute(2, 0, 1).float() / 255.0  # ToTensor: [0, 1] 
        tensor = TF.resize(tensor, (self._camera_height, self._camera_width),
                          antialias=True)
        return tensor

    def _get_camera_rgb(self, observation, cam_ref):
        """Resolve a camera reference to an RGB tensor.

        ``cam_ref`` can be:
          - Dotted path: "tactile.left_tactile.rgb"
          - Plain name:  "head" (backward-compatible)
        """
        if "." in cam_ref:
            return self._resolve_dotted_path(observation, cam_ref)
        obs_data = observation.get("observation", {})
        if cam_ref in obs_data:
            return obs_data[cam_ref]["rgb"]
        if cam_ref == "head" and "head_camera" in obs_data:
            return obs_data["head_camera"]["rgb"]
        if cam_ref == "wrist" and "wrist_camera" in obs_data:
            return obs_data["wrist_camera"]["rgb"]
        return None

    def encode_obs(self, observation):
        """Convert UniVTAC observation to FastWAM model input.

        Returns dict with keys:
          - image_tensor:  [1, 3, H', W'] float [-1, 1]  (H',W' depend on concat mode)
          - proprio_tensor: [1, 8] float (normalized)
        """
        # --- 1. Camera images ---
        cam_tensors = []
        for cam_ref in self._camera_names:
            rgb = self._get_camera_rgb(observation, cam_ref)
            if rgb is None:
                raise KeyError(
                    f"Camera '{cam_ref}' not found. "
                    f"obs keys: {list(observation.get('observation', {}).keys())}, "
                    f"tactile keys: {list(observation.get('tactile', {}).keys())}"
                )
            cam_tensors.append(self._preprocess_camera(rgb))

        # --- 2. Multi-camera concatenation (matches training) ---
        if self._concat_mode == "grid_2x2":
            if len(cam_tensors) != 4:
                raise ValueError(
                    f"grid_2x2 requires exactly 4 cameras, got {len(cam_tensors)}"
                )
            # Order: [cam_high, cam_wrist, tac_left, tac_right]
            # grid:  cam0  | cam1
            #        cam2  | cam3
            row0 = torch.cat([cam_tensors[0], cam_tensors[1]], dim=2)  # [3, H, 2W]
            row1 = torch.cat([cam_tensors[2], cam_tensors[3]], dim=2)  # [3, H, 2W]
            image = torch.cat([row0, row1], dim=1)                     # [3, 2H, 2W]
        elif self._concat_mode == "horizontal":
            image = torch.cat(cam_tensors, dim=2)
        elif self._concat_mode == "vertical":
            image = torch.cat(cam_tensors, dim=1)
        else:
            raise ValueError(
                f"Unknown concat_multi_camera: {self._concat_mode}"
            )

        # --- 3. Normalize to [-1, 1] (matching training Normalize(0.5,0.5)) ---
        image = (image - 0.5) / 0.5       # [0,1] → [-1,1]
        image = image.unsqueeze(0)         # [1, 3, H, W]

        # --- 4. Proprio ---
        embodiment = observation.get("embodiment", {})
        joint = embodiment.get("joint")
        if joint is not None:
            if isinstance(joint, np.ndarray):
                joint = torch.from_numpy(joint).float()
            raw_proprio = joint[:self._proprio_dim].float()
            if raw_proprio.shape[0] < self._proprio_dim:
                pad = torch.zeros(self._proprio_dim - raw_proprio.shape[0])
                raw_proprio = torch.cat([raw_proprio, pad])
        else:
            raw_proprio = torch.zeros(self._proprio_dim)

        proprio = self._normalize_proprio(raw_proprio)
        return {"image_tensor": image, "proprio_tensor": proprio}

    # ------------------------------------------------------------------
    #  Normalization helpers (matching training min/max → [-1,1])
    # ------------------------------------------------------------------

    def _normalize_proprio(self, raw_proprio):
        if self._normalizer is None:
            return raw_proprio.unsqueeze(0)
        proprio = raw_proprio.float().cpu().unsqueeze(0)  # [1, D]
        batch = {"state": {"default": proprio}}
        batch = self._normalizer.forward(batch)
        # Clamp to [-1, 1] — the theoretical range of min/max normalization.
        # Without this, a physical value slightly outside the training min/max
        # can map to extreme normalized values (e.g. finger -4.3) that the model
        # has never seen, causing action explosion.
        return batch["state"]["default"].clamp(-1.0, 1.0)

    def _denormalize_action(self, action):
        """Denormalize [1, T, D] → [T, D] in physical joint units."""
        if self._normalizer is None:
            return action.squeeze(0)
        device = action.device
        batch = {
            "action": {"default": action.float()},
            "state": {
                "default": torch.zeros(1, self._proprio_dim,
                                       dtype=torch.float32, device=device)
            },
        }
        batch = self._normalizer.backward(batch)
        return batch["action"]["default"].squeeze(0)     # [T, D]

    # ------------------------------------------------------------------
    #  Inference
    # ------------------------------------------------------------------

    def _infer_action_chunk(self, observation, instruction):
        """Run one FastWAM inference, return denormalized action [T, D] numpy."""
        import time
        t0 = time.perf_counter()
        encoded = self.encode_obs(observation)
        image_tensor = encoded["image_tensor"].to(
            device=self._device, dtype=self._model_dtype
        )
        proprio_tensor = encoded["proprio_tensor"].to(
            device=self._device, dtype=self._model_dtype
        )
        t1 = time.perf_counter()

        if self._load_text_encoder:
            prompt = DEFAULT_PROMPT.format(task=instruction)
            context = None
            context_mask = None
        else:
            prompt = None
            context = torch.zeros(1, 1, self._text_dim,
                                  device=self._device, dtype=self._model_dtype)
            context_mask = torch.ones(1, 1, dtype=torch.bool, device=self._device)

        infer_out = self.model.infer_action(
            prompt=prompt,
            input_image=image_tensor,
            action_horizon=self._action_horizon,
            proprio=proprio_tensor,
            context=context,
            context_mask=context_mask,
            negative_prompt=self._negative_prompt,
            text_cfg_scale=self._text_cfg_scale,
            num_inference_steps=self._num_inference_steps,
            sigma_shift=None,
            seed=self._seed,
            rand_device=self._rand_device,
            tiled=self._tiled,
        )

        action = infer_out["action"].unsqueeze(0)  # [1, T, D]
        # Clamp to [-1, 1] before denorm — prevents out-of-distribution values
        # from exploding through the normalizer (especially for finger dim).
        action = action.clamp(-1.0, 1.0)
        # Log raw model output (normalized, [-1, 1] range)
        raw_action_np = action.squeeze(0).cpu().float().numpy()
        action = self._denormalize_action(action)   # [T, D]
        denorm_action_np = action.cpu().float().numpy()

        # Print first 4 action steps (raw + denormalized)
        print(f"[FASTWAM-ACTION] step={getattr(self, '_infer_cnt', 0):04d} "
              f"raw_range=[{raw_action_np.min():+.4f},{raw_action_np.max():+.4f}] "
              f"denorm_range=[{denorm_action_np.min():+.4f},{denorm_action_np.max():+.4f}]",
              flush=True)
        for i in range(min(4, self._action_horizon)):
            raw_str = " ".join(f"{v:+7.4f}" for v in raw_action_np[i])
            denorm_str = " ".join(f"{v:+7.4f}" for v in denorm_action_np[i])
            print(f"  t={i:02d}  raw=[{raw_str}]  denorm=[{denorm_str}]", flush=True)

        if not hasattr(self, '_infer_cnt'):
            self._infer_cnt = 0
        self._infer_cnt += 1

        t2 = time.perf_counter()
        print(f"[FASTWAM-TIME] encode={t1-t0:.3f}s  infer={t2-t1:.3f}s  total={t2-t0:.3f}s", flush=True)
        return denorm_action_np

    # ------------------------------------------------------------------
    #  Policy interface
    # ------------------------------------------------------------------

    def eval(self, task, observation):
        if not self._pending_actions:
            instruction = getattr(task, "instruction", "do the task")
            action_chunk = self._infer_action_chunk(observation, instruction)
            n_exec = min(self._replan_steps, action_chunk.shape[0])
            for i in range(n_exec):
                self._pending_actions.append(action_chunk[i].copy())

        if not self._pending_actions:
            return

        action = self._pending_actions.popleft()
        action_tensor = torch.from_numpy(action).to(task.device).float()
        task.take_action(action_tensor, action_type=self._action_type)

    def reset(self):
        self._pending_actions.clear()
        if hasattr(self.model, "reset"):
            self.model.reset()

    def close(self):
        self._pending_actions.clear()
