"""
FastWAM policy deployment for UniVTAC simulation.

Loads a trained FastWAM (base) model and runs inference on Insert HDMI task
with 4 cameras (cam_high, cam_wrist, tac_left, tac_right) in 2x2 grid layout.

The preprocessing pipeline is identical to FastWAM training.
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F

# Ensure parent is on path for BasePolicy import
_sys_path = str(Path(__file__).parent.parent)
if _sys_path not in sys.path:
    sys.path.append(_sys_path)

from _base_policy import BasePolicy

# Local utility imports
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

        # Tell DiffSynth loader where to find local model files
        diffsynth_base = str(args["diffsynth_model_base_path"])
        if "DIFFSYNTH_MODEL_BASE_PATH" not in os.environ:
            os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = diffsynth_base

        tokenizer_max_len = int(args.get("tokenizer_max_len", 128))
        action_horizon = int(args.get("action_horizon", 32))
        self.num_inference_steps = int(args.get("num_inference_steps", 10))
        self.device_str = str(args.get("device", "cuda:0"))

        # ---- build model ----
        from .wan22.fastwam import FastWAM

        # Video DiT config (must match training)
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
            "action_dim": 8,  # 7 arm joints + 1 gripper
            "action_group_causal_mask_mode": "group_diagonal",
        }

        # Action DiT config (must match training)
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

        self.model = FastWAM.from_wan22_pretrained(
            device=self.device_str,
            torch_dtype=torch.bfloat16,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            load_text_encoder=True,  # need to encode prompts at deployment
            proprio_dim=8,
            redirect_common_files=True,
            video_dit_config=video_dit_config,
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=False,
            mot_checkpoint_mixed_attn=False,  # matches training config
            video_train_shift=5.0,
            video_infer_shift=5.0,
            video_num_train_timesteps=1000,
            action_train_shift=5.0,
            action_infer_shift=5.0,
            action_num_train_timesteps=1000,
            loss_lambda_video=1.0,
            loss_lambda_action=1.0,
        )

        # ---- load finetuned checkpoint ----
        print(f"Loading checkpoint: {checkpoint_path}")
        self.model.load_checkpoint(str(checkpoint_path))
        self.model.eval()

        # ---- load normalizer stats ----
        with open(dataset_stats_path, "r") as f:
            dataset_stats = json.load(f)
        # Convert numeric lists back to tensors
        dataset_stats = _json_to_tensor_stats(dataset_stats)

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

        # ---- image transforms (matching training exactly) ----
        # After grid assembly: ResizeSmallestSide → CenterCrop → Normalize(0.5, 0.5)
        self.grid_resize = ResizeSmallestSideAspectPreserving(
            args={"img_w": 448, "img_h": 448}
        )
        self.grid_crop = CenterCrop(args={"img_w": 448, "img_h": 448})
        self.grid_normalize = Normalize(args={"mean": 0.5, "std": 0.5})

        # ---- action buffer ----
        self.action_horizon = action_horizon
        self._action_buffer: list = []  # queue of (action_8d,) tuples
        self._step_counter: int = 0

        # ---- instruction cache ----
        self._last_instruction: str | None = None
        self._cached_prompt: str | None = None

        print(f"FastWAM policy loaded. Device: {self.device_str}, "
              f"action_horizon={self.action_horizon}, "
              f"num_inference_steps={self.num_inference_steps}")

    # ------------------------------------------------------------------
    #  Observation encoding  (must match training preprocessing exactly)
    # ------------------------------------------------------------------

    def encode_obs(self, observation: dict, instruction: str = "insert hdmi") -> dict:
        """
        Process UniVTAC observation into model input format.

        UniVTAC input:
          observation["observation"]["head"]["rgb"]       — [480, 270, 3] uint8 HWC
          observation["observation"]["wrist"]["rgb"]      — [480, 270, 3] uint8 HWC
          observation["tactile"]["left_tactile"]["rgb_marker"]  — [240, 320, 3] uint8 HWC
          observation["tactile"]["right_tactile"]["rgb_marker"] — [240, 320, 3] uint8 HWC
          observation["embodiment"]["joint"]              — [9] float (7 arm + 2 gripper)

        Returns:
          dict with keys:
            "input_image": [1, 3, 448, 448] float32 in [-1, 1]
            "proprio": [1, 8] float32 normalized
            "prompt": str
        """
        # --- 1. Process each camera: HWC uint8 → CHW float [0,1] → Resize(224,224) ---
        def process_camera(rgb_tensor: torch.Tensor) -> torch.Tensor:
            """HWC uint8 -> CHW float [0,1], resize to 224x224."""
            # HWC -> CHW
            img = rgb_tensor.permute(2, 0, 1).float() / 255.0  # [C, H, W], [0, 1]
            # Resize to 224x224
            img = transforms_F.resize(
                img, size=[224, 224],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            return img  # [3, 224, 224], [0, 1]

        cam_high = process_camera(observation["observation"]["head"]["rgb"])
        cam_wrist = process_camera(observation["observation"]["wrist"]["rgb"])
        tac_left = process_camera(
            observation["tactile"]["left_tactile"]["rgb_marker"]
        )
        tac_right = process_camera(
            observation["tactile"]["right_tactile"]["rgb_marker"]
        )

        # --- 2. Assemble 2x2 grid (exactly matching training) ---
        # Top row: cam_high (left) | cam_wrist (right)
        # Bottom row: tac_left (left) | tac_right (right)
        row0 = torch.cat([cam_high, cam_wrist], dim=-1)    # [3, 224, 448]
        row1 = torch.cat([tac_left, tac_right], dim=-1)    # [3, 224, 448]
        grid = torch.cat([row0, row1], dim=-2)             # [3, 448, 448]

        # --- 3. Final transforms: ResizeSmallestSide → CenterCrop → Normalize ---
        # These are identity for 448x448 input, but included for correctness
        grid = self.grid_resize(grid)
        grid = self.grid_crop(grid)
        grid = self.grid_normalize(grid)  # [3, 448, 448], range [-1, 1]

        # Add batch and time dims: [1, 3, 1, 448, 448] → squeeze time → [1, 3, 448, 448]
        input_image = grid.unsqueeze(1).unsqueeze(0)  # [1, 3, 1, 448, 448]
        input_image = input_image.squeeze(2)           # [1, 3, 448, 448]

        # --- 4. State normalization (on CPU; normalizer stats are on CPU) ---
        joint = observation["embodiment"]["joint"][:8].float().cpu()  # [8]
        joint_norm = self.normalizer.normalizers["state"]["default"].forward(
            joint.unsqueeze(0)
        )  # [1, 8], range [-1, 1]
        joint_norm = joint_norm.clamp(-5.0, 5.0)

        # --- 5. Text prompt ---
        prompt = DEFAULT_PROMPT.format(task=instruction)

        return {
            "input_image": input_image,  # [1, 3, 448, 448], [-1, 1]
            "proprio": joint_norm,       # [1, 8], normalized
            "prompt": prompt,            # str
        }

    # ------------------------------------------------------------------
    #  Action postprocessing
    # ------------------------------------------------------------------

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        """Denormalize model output back to raw joint positions.

        Args:
            action: [T, 8] normalized actions from model (range ~[-1, 1])

        Returns:
            np.ndarray [T, 8] raw joint positions
        """
        # merger.backward expects plain 3D tensors [B, T, D] (output of merger.forward)
        T = action.shape[0]
        batch = {
            "action": action.unsqueeze(0),             # [1, T, 8]
            "state": torch.zeros(1, T, 8),             # [1, T, 8] dummy
        }
        batch = self.action_state_merger.backward(batch)
        batch = self.normalizer.backward(batch)
        return batch["action"]["default"][0].cpu().numpy()  # [T, 8]

    # ------------------------------------------------------------------
    #  Main eval loop
    # ------------------------------------------------------------------

    def eval(self, task, observation):
        """Run one inference step.

        Predicts action_horizon actions when buffer is empty,
        then executes them one at a time.
        """
        # Replenish buffer if needed
        if not self._action_buffer:
            instruction = getattr(task, "instruction", "insert hdmi")
            obs = self.encode_obs(observation, instruction=instruction)

            result = self.model.infer_action(
                prompt=obs["prompt"],
                input_image=obs["input_image"],
                action_horizon=self.action_horizon,
                proprio=obs["proprio"],
                num_inference_steps=self.num_inference_steps,
                seed=None,  # stochastic
            )
            # result["action"]: [action_horizon, 8] normalized
            actions = self._denormalize_action(result["action"])

            # Enqueue all actions
            self._action_buffer = [actions[i] for i in range(actions.shape[0])]

        # Pop and execute next action
        action = self._action_buffer.pop(0)
        action_tensor = torch.from_numpy(action).to(task.device).float()
        task.take_action(action_tensor, action_type="qpos")

    # ------------------------------------------------------------------
    #  Reset
    # ------------------------------------------------------------------

    def reset(self):
        self._action_buffer.clear()
        self._step_counter = 0

    def close(self):
        if hasattr(self.model, "to"):
            self.model.to("cpu")


# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------

def _json_to_tensor_stats(stats: dict) -> dict:
    """Recursively convert numeric lists in JSON-loaded stats to torch tensors."""

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
