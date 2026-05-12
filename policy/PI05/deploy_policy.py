#!/usr/bin/env python
"""
PI05 deploy policy for UniVTAC eval.

Loads a PI05 checkpoint and exposes the BasePolicy interface expected by
UniVTAC's eval_policy.py / parallel_eval_policy.py.
"""

import sys
import json
import os
import logging
import numpy as np
from pathlib import Path

import torch
from torchvision import transforms

sys.path.append(str(Path(__file__).parent.parent))
from .._base_policy import BasePolicy

from .config import PI05Config
from .modeling import PI05Pytorch, pad_vector, resize_with_pad_torch


def load_safetensors_state_dict(ckpt_dir: str) -> dict:
    """Load model.safetensors from a directory (can be local path or HuggingFace format)."""
    ckpt_path = Path(ckpt_dir) / "model.safetensors"
    if ckpt_path.exists():
        from safetensors.torch import load_file
        return load_file(str(ckpt_path))

    from transformers.utils import cached_file
    resolved = cached_file(ckpt_dir, "model.safetensors")
    from safetensors.torch import load_file
    return load_file(resolved)


def _fix_state_dict_keys(state_dict: dict) -> dict:
    """Remap PI05 checkpoint keys to match local model keys."""
    fixed = {}
    for key, value in state_dict.items():
        new_key = key
        # mlp naming changes
        if new_key.startswith("action_time_mlp_in."):
            new_key = new_key.replace("action_time_mlp_in.", "time_mlp_in.")
        elif new_key.startswith("action_time_mlp_out."):
            new_key = new_key.replace("action_time_mlp_out.", "time_mlp_out.")
        # skip state_proj (pi0 only, not pi05)
        if new_key.startswith("state_proj."):
            continue
        # map paligemma lm_head → embed_tokens (weight tying)
        if new_key == "paligemma_with_expert.paligemma.lm_head.weight":
            fixed["paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"] = value.clone()
            continue
        # skip gemma_expert lm_head (embed_tokens is None, not needed for inference)
        if new_key == "paligemma_with_expert.gemma_expert.lm_head.weight":
            continue
        # vision tower: remove .vision_model infix (transformer version mismatch)
        new_key = new_key.replace(".vision_tower.vision_model.", ".vision_tower.")
        fixed[new_key] = value
    return fixed


def _remap_state_dict(state_dict: dict, prefix: str = "model.") -> dict:
    """Add model. prefix to keys that don't already have it."""
    remapped = {}
    for key, value in state_dict.items():
        if not key.startswith(prefix):
            remapped[f"{prefix}{key}"] = value
        else:
            remapped[key] = value
    return remapped


class Policy(BasePolicy):
    """PI05 policy for UniVTAC eval."""

    def __init__(self, args):
        super().__init__(args)

        self.config = PI05Config(
            dtype=args.get("dtype", "float32"),
            num_inference_steps=args.get("num_inference_steps", 10),
            n_action_steps=args.get("n_action_steps", 50),
            device=args.get("device", "cuda"),
            freeze_vision_encoder=True,
            train_expert_only=False,
        )

        self.task_name = args["task_name"]
        self.camera_names = args.get("camera_names", ["cam_high"])
        self.use_tactile = args.get("use_tactile", False)

        # Load the pretrained model
        ckpt_dir = args.get("pretrained_path", "")
        tokenizer_name = args.get("tokenizer_name", "google/paligemma-3b-pt-224")
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        print(f"[PI05] Building model (dtype={self.config.dtype}) ...")
        self.model = PI05Pytorch(self.config)
        self.model.eval()

        if ckpt_dir and Path(ckpt_dir).exists():
            print(f"[PI05] Loading checkpoint from {ckpt_dir}")
            state_dict = load_safetensors_state_dict(ckpt_dir)
            state_dict = _fix_state_dict_keys(state_dict)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[PI05] Missing keys: {len(missing)} (first 5: {missing[:5]})")
            if unexpected:
                print(f"[PI05] Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
            print("[PI05] Checkpoint loaded.")
        else:
            print(f"[PI05] WARNING: Checkpoint not found at {ckpt_dir}, using random weights.")

        self.model.to(self.device)

        # Load PaliGemma tokenizer
        print(f"[PI05] Loading tokenizer from {tokenizer_name}")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load normalization stats
        self.stats = None
        stats_path = args.get("norm_stats_path", None)
        if stats_path and Path(stats_path).exists():
            with open(stats_path, "r") as f:
                self.stats = json.load(f)
            print(f"[PI05] Loaded norm stats from {stats_path}")

        self._action_queue = list()

        # Get task instruction from instructions
        self.instruction = ""
        deploy_dir = Path(__file__).parent.parent
        instructions_file = args.get("instuction_file", self.task_name)
        if instructions_file:
            inst_path = deploy_dir.parent / "instructions" / f"{instructions_file}.json"
            if inst_path.exists():
                with open(inst_path) as f:
                    instructions_data = json.load(f)
                inst_type = args.get("instruction_type", "seen")
                instructions_list = instructions_data.get(inst_type, ["Empty"])
                import random
                self.instruction = random.choice(instructions_list)
        if not self.instruction:
            self.instruction = "empty task"

        self._t = 0
        print(f"[PI05] Ready. Task: {self.task_name}, Camera: {self.camera_names}")

    def encode_obs(self, observation):
        """Convert UniVTAC observation → PI05 batch dict."""
        batch_size = 1

        # ── 1. Camera images ──
        images, img_masks = [], []
        for cam_name in self.camera_names:
            img = observation["observation"][cam_name]["rgb"]
            # HWC 0-255 → CHW [0,1]
            if isinstance(img, torch.Tensor):
                img = img.clone()
            else:
                img = torch.from_numpy(np.array(img))
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)  # HWC → CHW
            img = img.float() / 255.0
            img = resize_with_pad_torch(img.unsqueeze(0).permute(0, 2, 3, 1), *self.config.image_resolution)
            img = img.permute(0, 3, 1, 2)  # BHWC → BCHW
            # Normalize [0,1] → [-1,1] (SigLIP)
            img = img * 2.0 - 1.0
            images.append(img.to(self.device))
            mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            img_masks.append(mask)

        # ── 2. Tactile (optional, as extra cameras) ──
        if self.use_tactile:
            for tactile_key in ["left_tactile", "right_tactile"]:
                if tactile_key in observation.get("tactile", {}):
                    tac = observation["tactile"][tactile_key]["rgb_marker"]
                    if isinstance(tac, torch.Tensor):
                        tac = tac.clone()
                    else:
                        tac = torch.from_numpy(np.array(tac))
                    if tac.ndim == 3 and tac.shape[-1] == 3:
                        tac = tac.permute(2, 0, 1)
                    tac = tac.float() / 255.0
                    tac = resize_with_pad_torch(tac.unsqueeze(0).permute(0, 2, 3, 1), *self.config.image_resolution)
                    tac = tac.permute(0, 3, 1, 2)
                    tac = tac * 2.0 - 1.0
                    images.append(tac.to(self.device))
                    mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
                    img_masks.append(mask)

        # ── 3. Joint state → discretize + format prompt ──
        qpos = observation["embodiment"]["joint"][:8]
        if isinstance(qpos, np.ndarray):
            qpos_np = qpos
        else:
            qpos_np = qpos.cpu().numpy()

        # Normalize to [-1, 1] using stats (or identity)
        if self.stats is not None:
            qpos_mean = np.array(self.stats.get("qpos_mean", [0] * 8))
            qpos_std = np.array(self.stats.get("qpos_std", [1] * 8))
            qpos_normalized = (qpos_np - qpos_mean) / np.maximum(qpos_std, 1e-8)
            qpos_clamped = np.clip(qpos_normalized, -1.0, 1.0)
        else:
            qpos_clamped = qpos_np  # assume already in reasonable range

        # Discretize to 256 bins
        discretized = np.digitize(qpos_clamped, np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized))

        # Format task prompt
        task_text = self.instruction.strip().replace("_", " ").replace("\n", " ")
        full_prompt = f"Task: {task_text}, State: {state_str};\nAction: "

        # Tokenize
        tokenizer_out = self.tokenizer(
            full_prompt, max_length=self.config.tokenizer_max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        tokens = tokenizer_out["input_ids"].to(self.device)
        masks = tokenizer_out["attention_mask"].to(self.device).bool()

        return {"images": images, "img_masks": img_masks, "tokens": tokens, "masks": masks}

    def eval(self, task, observation):
        """Run one PI05 inference step."""
        obs_batch = self.encode_obs(observation)

        # Queue-based action execution (like LeRobot's select_action)
        if len(self._action_queue) == 0:
            actions = self.model.sample_actions(
                images=obs_batch["images"],
                img_masks=obs_batch["img_masks"],
                tokens=obs_batch["tokens"],
                masks=obs_batch["masks"],
            )
            # Unpad: [1, chunk_size, 32] → [1, chunk_size, real_action_dim]
            actions = actions[:, :, :self.config.real_action_dim]
            actions = actions.squeeze(0)  # [chunk_size, real_action_dim]
            self._action_queue = list(actions.cpu().numpy())

        action_np = self._action_queue.pop(0)
        action = torch.from_numpy(action_np).to(task.device).float()

        # Pad to 8D if needed (UniVTAC expects 8D: 7 arm + 1 gripper)
        if action.shape[-1] < 8:
            action = torch.cat([action, torch.zeros(8 - action.shape[-1], device=task.device)])

        exec_succ, eval_succ = task.take_action(action, action_type="qpos")
        self._t += 1

    def reset(self):
        self._t = 0
        self._action_queue = list()
