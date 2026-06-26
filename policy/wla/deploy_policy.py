import json
import importlib
import importlib.abc
import importlib.util
import os
import site
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torchvision.transforms.functional import to_tensor

from .._base_policy import BasePolicy


DEFAULT_MANTIS_ROOT = (
    "/inspire/hdd/global_user/yangyi-253108120173/inspire_shared/mount/"
    "advanced-machine-learning-and-deep-learning-applications/cyy/"
    "Mantis_flow_depth_use_metaqury"
)

_AWS_SDK_PREFIXES = ("boto3", "botocore", "s3transfer", "jmespath")


class _CondaAwsSdkFinder(importlib.abc.MetaPathFinder):
    def __init__(self, package_roots: dict[str, Path]):
        self.package_roots = package_roots

    def find_spec(self, fullname, path=None, target=None):
        for prefix, root in self.package_roots.items():
            if fullname != prefix and not fullname.startswith(f"{prefix}."):
                continue

            relative = fullname[len(prefix) + 1:] if fullname != prefix else ""
            if not relative:
                init_path = root / "__init__.py"
                if not init_path.exists():
                    return None
                return importlib.util.spec_from_file_location(
                    fullname,
                    init_path,
                    submodule_search_locations=[str(root)],
                )

            module_path = root.joinpath(*relative.split("."))
            file_path = module_path.with_suffix(".py")
            if file_path.exists():
                return importlib.util.spec_from_file_location(fullname, file_path)

            package_init = module_path / "__init__.py"
            if package_init.exists():
                return importlib.util.spec_from_file_location(
                    fullname,
                    package_init,
                    submodule_search_locations=[str(module_path)],
                )
        return None


def _purge_aws_sdk_modules():
    for module_name in list(sys.modules):
        if any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in _AWS_SDK_PREFIXES
        ):
            del sys.modules[module_name]


def _get_conda_site_packages():
    candidates = [
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        *map(Path, site.getsitepackages()),
    ]
    seen = set()
    resolved = []
    for candidate in candidates:
        path = str(candidate.resolve())
        if path in seen or not candidate.is_dir():
            continue
        seen.add(path)
        resolved.append(path)
    return resolved


def _install_conda_aws_sdk_finder(conda_site_packages):
    package_roots = {}
    for prefix in _AWS_SDK_PREFIXES:
        root = next(
            (
                Path(p) / prefix
                for p in conda_site_packages
                if (Path(p) / prefix / "__init__.py").exists()
            ),
            None,
        )
        if root is not None:
            package_roots[prefix] = root

    if "botocore" not in package_roots:
        raise ImportError(
            "Isaac Sim ships an incompatible botocore via pip_prebundle, but no conda "
            "botocore was found. Install it in the active UniVTAC env:\n"
            "  python -m pip install botocore boto3"
        )

    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not isinstance(finder, _CondaAwsSdkFinder)
    ]
    sys.meta_path.insert(0, _CondaAwsSdkFinder(package_roots))
    return package_roots


def _prefer_conda_site_packages():
    """Keep IsaacSim pip prebundle from shadowing packages needed by Mantis."""
    had_prebundle = any("pip_prebundle" in path for path in sys.path)
    conda_site_packages = _get_conda_site_packages()

    sys.path[:] = [p for p in sys.path if p not in conda_site_packages]
    sys.path[:0] = conda_site_packages
    sys.path[:] = [p for p in sys.path if "pip_prebundle" not in p]

    if not had_prebundle:
        return

    _purge_aws_sdk_modules()
    _install_conda_aws_sdk_finder(conda_site_packages)
    httpchecksum = importlib.import_module("botocore.httpchecksum")

    if not hasattr(httpchecksum, "DEFAULT_CHECKSUM_ALGORITHM"):
        raise ImportError(
            f"Loaded incompatible botocore from {httpchecksum.__file__}. "
            "Reinstall with: python -m pip install --upgrade botocore boto3"
        )
    if "pip_prebundle" in (httpchecksum.__file__ or ""):
        raise ImportError(
            f"botocore still resolves to Isaac pip_prebundle: {httpchecksum.__file__}"
        )


def _resolve_training_config_path(mantis_root: Path, config_ref: str) -> Path:
    path = Path(config_ref)
    if path.is_file():
        return path
    candidate = mantis_root / "configs" / config_ref
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"training config not found: {config_ref} "
        f"(also checked {candidate})"
    )


def _load_training_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config.get("use_tactile_encoder"):
        # Weights are restored from model.pt; avoid requiring encoder.pth on deploy machine.
        config["tactile_encoder_ckpt"] = None
    for key in ("modules_to_freeze", "modules_to_unfreeze"):
        if key in config and config[key] is not None:
            config[key] = tuple(config[key])
    return config


def _build_mantis_from_training_config(model_config: dict, input_size: int, Mantis, MantisConfig):
    model_kwargs = dict(model_config)
    model_kwargs["input_size"] = input_size
    return Mantis(config=MantisConfig(**model_kwargs))


def _load_checkpoint_state_dict(checkpoint_path: Path):
    return torch.load(str(checkpoint_path), map_location="cpu")


def _load_mantis_transforms(mantis_root: Path):
    transforms_path = mantis_root / "utils" / "transforms.py"
    spec = importlib.util.spec_from_file_location("mantis_transforms", transforms_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load transforms from {transforms_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_mantis_import_path(mantis_root: Path):
    mantis_root_str = str(mantis_root.resolve())
    for module_name in list(sys.modules):
        if module_name == "utils" or module_name.startswith("utils."):
            module_file = getattr(sys.modules[module_name], "__file__", "") or ""
            if module_file and not module_file.startswith(mantis_root_str):
                del sys.modules[module_name]

    if mantis_root_str in sys.path:
        sys.path.remove(mantis_root_str)
    sys.path.insert(0, mantis_root_str)


class Policy(BasePolicy):
    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self.mantis_root = Path(args.get("mantis_root", DEFAULT_MANTIS_ROOT))
        if not self.mantis_root.exists():
            raise FileNotFoundError(f"mantis_root does not exist: {self.mantis_root}")
        mantis_root_str = str(self.mantis_root)
        if mantis_root_str not in sys.path:
            sys.path.insert(0, mantis_root_str)
        _prefer_conda_site_packages()
        _ensure_mantis_import_path(self.mantis_root)

        from models.mantis import Mantis, MantisConfig
        transforms = _load_mantis_transforms(self.mantis_root)
        _make_transform = transforms._make_transform
        normalize_and_pad = transforms.normalize_and_pad
        resize_with_pad = transforms.resize_with_pad
        unnormalize_and_unpad = transforms.unnormalize_and_unpad

        self._normalize_and_pad = normalize_and_pad
        self._unnormalize_and_unpad = unnormalize_and_unpad
        self._resize_with_pad = resize_with_pad
        self.primary_image_size = int(args.get("primary_image_size", 256))
        self.auxiliary_image_size = int(args.get("auxiliary_image_size", 256))
        self.primary_image_transform = _make_transform(self.primary_image_size)
        self.auxiliary_image_transform = _make_transform(self.auxiliary_image_size)

        self.max_state_dim = int(args.get("max_state_dim", 8))
        self.original_action_dim = int(args.get("original_action_dim", 8))
        self.action_horizon = int(args.get("action_horizon", 32))
        self.replan_every = int(args.get("replan_every", self.action_horizon))
        self.temporal_agg = bool(args.get("temporal_agg", False))
        self.temporal_agg_query_frequency = int(args.get("temporal_agg_query_frequency", 1))
        self.temporal_agg_k = float(args.get("temporal_agg_k", 0.01))
        self.device = torch.device(args.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
        self.use_autocast = bool(args.get("use_autocast", True)) and self.device.type == "cuda"
        self.convert_rgb_to_bgr = bool(args.get("convert_rgb_to_bgr", True))
        self.use_tactile_encoder = bool(args.get("use_tactile_encoder", False))
        self.use_tactile_images = bool(args.get("use_tactile_images", not self.use_tactile_encoder))
        if self.use_tactile_encoder and self.use_tactile_images:
            raise ValueError(
                "use_tactile_encoder and use_tactile_images are mutually exclusive; "
                "set use_tactile_images=false when deploying tactile-encoder models."
            )
        self.debug_log = bool(args.get("debug_log", False))
        self.debug_save_images = bool(args.get("debug_save_images", False))
        self.debug_save_every = int(args.get("debug_save_every", 50))
        self.instruction_override = args.get("instruction", None)
        self.eval_predict_tactile = bool(args.get("eval_predict_tactile", False))
        self.tactile_eval_timesteps = tuple(
            args.get("tactile_eval_timesteps", [0, 7, 15, 23, 31])
        )
        if self.eval_predict_tactile and self.temporal_agg:
            print("[WLA] eval_predict_tactile enabled: disabling temporal_agg for aligned chunk comparison.")
            self.temporal_agg = False

        with open(args["norm_stats_path"], "r", encoding="utf-8") as f:
            all_norm_stats = json.load(f)
        self.norm_stats = all_norm_stats[args.get("unnorm_key", "insert_hdmi_tactile_lerobot")]

        input_size = int(args.get("target_image_size", 512)) // int(args.get("vae_downsample_f", 32))
        dtype_name = str(args.get("dtype", "bfloat16"))
        self.model_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(dtype_name, torch.bfloat16)

        use_pt_weights_only = bool(args.get("use_pt_weights_only", False))
        checkpoint_path = args.get("checkpoint_path")
        model_id = args.get("model_id")
        training_config = args.get("training_config")

        if use_pt_weights_only:
            if not checkpoint_path:
                raise ValueError("use_pt_weights_only requires checkpoint_path.")
            if not training_config:
                raise ValueError("use_pt_weights_only requires training_config.")
            config_path = _resolve_training_config_path(self.mantis_root, training_config)
            print(f"[WLA] Building model architecture from {config_path}")
            model_config = _load_training_config(config_path)
            self.model = _build_mantis_from_training_config(
                model_config,
                input_size,
                Mantis,
                MantisConfig,
            )
            checkpoint_path = Path(checkpoint_path)
            print(f"[WLA] Loading weights from {checkpoint_path}")
            self.model.load_state_dict(
                _load_checkpoint_state_dict(checkpoint_path),
                strict=True,
            )
        elif model_id:
            print(f"[WLA] Loading model config/weights from {model_id}")
            self.model = Mantis.from_pretrained(
                model_id,
                input_size=input_size,
                torch_dtype=self.model_dtype,
                dtype=self.model_dtype,
            )
            if checkpoint_path:
                checkpoint_path = Path(checkpoint_path)
                print(f"[WLA] Overriding weights from {checkpoint_path}")
                self.model.load_state_dict(
                    _load_checkpoint_state_dict(checkpoint_path),
                    strict=True,
                )
        else:
            raise ValueError(
                "Provide model_id, or set use_pt_weights_only=true with "
                "checkpoint_path and training_config."
            )

        self.model.to(device=self.device, dtype=self.model_dtype)
        self.model.eval()

        self.tactile_input_type = str(
            getattr(self.model.config, "tactile_input_type", args.get("tactile_input_type", "image"))
        )
        self.use_history_obs = bool(getattr(self.model.config, "use_history_obs", False))
        self.history_obs_step = int(
            args.get(
                "history_obs_step",
                getattr(self.model.config, "history_obs_step", 8),
            )
        )
        self._head_image_history: list[torch.Tensor] = []

        # Action-only deployment does not need the image decoder branch.
        if hasattr(self.model, "vae"):
            del self.model.vae
        if hasattr(self.model.model, "transformer"):
            del self.model.model.transformer
        if hasattr(self.model.model, "connector"):
            del self.model.model.connector

        self._action_buffer = []
        self._all_time_actions = None
        self._episode_idx = -1
        self._infer_count = 0
        self._exec_count = 0
        self._tactile_decoder_bank = None
        self._tactile_eval_active = False
        self._tactile_chunk_start = 0
        self._pred_tactiles = None
        self._gt_tac_left = []
        self._gt_tac_right = []
        self._tactile_eval_summary = []
        self._current_task_save_root = None
        if self.eval_predict_tactile:
            if not getattr(self.model.config, "predict_tactile", False):
                raise ValueError("eval_predict_tactile requires predict_tactile=true in training config.")
            from policy.wla.tactile_pred_utils import TactileDecoderBank

            training_config_ref = args.get("training_config")
            if training_config_ref:
                raw_config_path = _resolve_training_config_path(self.mantis_root, training_config_ref)
                with open(raw_config_path, "r", encoding="utf-8") as f:
                    raw_training_cfg = yaml.safe_load(f)
                self._tactile_decoder_bank = TactileDecoderBank.try_load(
                    raw_training_cfg.get("tactile_encoder_ckpt")
                )
                if self._tactile_decoder_bank is not None:
                    self._tactile_decoder_bank.to(self.device).eval()
            print(
                f"[WLA] eval_predict_tactile enabled, "
                f"decoder={'yes' if self._tactile_decoder_bank else 'no'}"
            )
        print(
            f"[WLA] Ready on {self.device}, dtype={self.model_dtype}, "
            f"action_horizon={self.action_horizon}, replan_every={self.replan_every}, "
            f"temporal_agg={self.temporal_agg}, "
            f"use_tactile_encoder={self.use_tactile_encoder}, "
            f"use_tactile_images={self.use_tactile_images}, "
            f"tactile_input_type={self.tactile_input_type}, "
            f"use_history_obs={self.use_history_obs}, "
            f"history_obs_step={self.history_obs_step}, "
            f"eval_predict_tactile={self.eval_predict_tactile}"
        )

    @staticmethod
    def _image_to_tensor(img, convert_rgb_to_bgr=True):
        if isinstance(img, torch.Tensor):
            tensor = img.detach().cpu()
            if tensor.ndim != 3:
                raise ValueError(f"Expected RGB tensor with 3 dims, got shape {tuple(tensor.shape)}")
            if tensor.shape[0] != 3 and tensor.shape[-1] == 3:
                tensor = tensor.permute(2, 0, 1)
            tensor = tensor.float()
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
        elif isinstance(img, np.ndarray):
            tensor = to_tensor(img)
        else:
            tensor = to_tensor(img)

        if convert_rgb_to_bgr:
            tensor = tensor[[2, 1, 0], ...]
        return tensor

    def _tactile_to_tensor(self, img):
        tensor = self._image_to_tensor(img, convert_rgb_to_bgr=self.convert_rgb_to_bgr)
        return self._resize_with_pad(tensor, self.auxiliary_image_size)

    @staticmethod
    def _marker_to_tensor(marker):
        if isinstance(marker, torch.Tensor):
            tensor = marker.detach().cpu().float()
        else:
            tensor = torch.as_tensor(np.asarray(marker), dtype=torch.float32)

        if tensor.ndim == 2 and tensor.shape[-1] == 2:
            tensor = torch.stack([torch.zeros_like(tensor), tensor], dim=0)
        if tensor.ndim != 3 or tensor.shape[0] != 2 or tensor.shape[-1] != 2:
            raise ValueError(
                "Expected tactile marker tensor [2, num_markers, 2], "
                f"got shape {tuple(tensor.shape)}"
            )
        return tensor

    @staticmethod
    def _get_tactile_marker(tactile_obs):
        if "marker" in tactile_obs:
            return tactile_obs["marker"]
        if "marker_motion" in tactile_obs:
            return tactile_obs["marker_motion"]
        raise KeyError("Tactile marker observation must contain 'marker' or 'marker_motion'.")

    def _build_head_input_images(self, head_tensor: torch.Tensor) -> list[torch.Tensor]:
        current = self.primary_image_transform(head_tensor)
        if not self.use_history_obs:
            return [current]

        history_frame = self.auxiliary_image_transform(head_tensor)
        self._head_image_history.append(history_frame)
        if len(self._head_image_history) > self.history_obs_step:
            self._head_image_history = self._head_image_history[-self.history_obs_step :]

        history = (
            self._head_image_history[0]
            if len(self._head_image_history) >= self.history_obs_step
            else current
        )
        return [history, current]

    def encode_obs(self, observation):
        head = self._image_to_tensor(
            observation["observation"]["head"]["rgb"],
            convert_rgb_to_bgr=self.convert_rgb_to_bgr,
        )
        images = self._build_head_input_images(head)
        tactile_images = None
        tactile_markers = None
        if self.use_tactile_images:
            left_tac = self._image_to_tensor(
                observation["tactile"]["left_tactile"]["rgb_marker"],
                convert_rgb_to_bgr=self.convert_rgb_to_bgr,
            )
            right_tac = self._image_to_tensor(
                observation["tactile"]["right_tactile"]["rgb_marker"],
                convert_rgb_to_bgr=self.convert_rgb_to_bgr,
            )
            images.extend([
                self.auxiliary_image_transform(left_tac),
                self.auxiliary_image_transform(right_tac),
            ])
        elif self.use_tactile_encoder:
            if self.tactile_input_type == "marker":
                left_marker = self._marker_to_tensor(
                    self._get_tactile_marker(observation["tactile"]["left_tactile"])
                )
                right_marker = self._marker_to_tensor(
                    self._get_tactile_marker(observation["tactile"]["right_tactile"])
                )
                tactile_markers = torch.stack([left_marker, right_marker], dim=0).unsqueeze(0)
            else:
                left_tac = self._tactile_to_tensor(
                    observation["tactile"]["left_tactile"]["rgb_marker"]
                )
                right_tac = self._tactile_to_tensor(
                    observation["tactile"]["right_tactile"]["rgb_marker"]
                )
                tactile_images = torch.stack([left_tac, right_tac], dim=0).unsqueeze(0)

        input_images = [images]

        state = observation["embodiment"]["joint"][: self.original_action_dim].detach().cpu().float()
        state, _ = self._normalize_and_pad(
            state,
            self.norm_stats["observation.state"],
            self.max_state_dim,
        )
        state = state.unsqueeze(0)
        return input_images, state, tactile_images, tactile_markers

    def _extract_tactile_pair(self, observation):
        if self.tactile_input_type == "marker":
            left_marker = self._marker_to_tensor(
                self._get_tactile_marker(observation["tactile"]["left_tactile"])
            )
            right_marker = self._marker_to_tensor(
                self._get_tactile_marker(observation["tactile"]["right_tactile"])
            )
            return left_marker, right_marker

        left_tac = self._tactile_to_tensor(
            observation["tactile"]["left_tactile"]["rgb_marker"]
        )
        right_tac = self._tactile_to_tensor(
            observation["tactile"]["right_tactile"]["rgb_marker"]
        )
        return left_tac, right_tac

    def _start_tactile_eval_chunk(self, task, pred_tactiles):
        if not self.eval_predict_tactile or pred_tactiles is None:
            return
        self._tactile_eval_active = True
        self._tactile_chunk_start = self._exec_count
        self._pred_tactiles = np.asarray(pred_tactiles, dtype=np.float32)
        self._gt_tac_left = []
        self._gt_tac_right = []
        self._current_task_save_root = Path(task.save_root)

    def _record_tactile_gt(self, observation):
        if not self._tactile_eval_active:
            return
        idx = self._exec_count - self._tactile_chunk_start
        if idx < 0 or idx >= self.action_horizon:
            return
        left_tac, right_tac = self._extract_tactile_pair(observation)
        if len(self._gt_tac_left) <= idx:
            self._gt_tac_left.extend([None] * (idx + 1 - len(self._gt_tac_left)))
            self._gt_tac_right.extend([None] * (idx + 1 - len(self._gt_tac_right)))
        self._gt_tac_left[idx] = left_tac.detach().cpu()
        self._gt_tac_right[idx] = right_tac.detach().cpu()

    def _finalize_tactile_eval_chunk(self, task):
        if not self._tactile_eval_active or self._pred_tactiles is None:
            return

        self._tactile_eval_active = False
        horizon = min(self.action_horizon, int(len(self._pred_tactiles)))
        if horizon <= 0:
            print(f"[WLA] tactile eval: empty predicted tactile chunk, skip.")
            self._pred_tactiles = None
            self._gt_tac_left = []
            self._gt_tac_right = []
            return

        valid_indices = [
            i
            for i in range(min(len(self._gt_tac_left), horizon))
            if i < len(self._gt_tac_left) and self._gt_tac_left[i] is not None
        ]
        if len(valid_indices) < 2:
            print(
                f"[WLA] tactile eval: only {len(valid_indices)} GT frames collected "
                f"(chunk_start={self._tactile_chunk_start}), skip."
            )
            self._pred_tactiles = None
            self._gt_tac_left = []
            self._gt_tac_right = []
            return

        def _pick(index):
            if index < len(self._gt_tac_left) and self._gt_tac_left[index] is not None:
                return self._gt_tac_left[index], self._gt_tac_right[index]
            for j in range(index, -1, -1):
                if j < len(self._gt_tac_left) and self._gt_tac_left[j] is not None:
                    return self._gt_tac_left[j], self._gt_tac_right[j]
            raise RuntimeError(f"Missing tactile GT frame for index {index}")

        try:
            left_stack = torch.stack([_pick(i)[0] for i in range(horizon)], dim=0)
            right_stack = torch.stack([_pick(i)[1] for i in range(horizon)], dim=0)
        except RuntimeError as exc:
            print(f"[WLA] tactile eval: {exc}, skip.")
            self._pred_tactiles = None
            self._gt_tac_left = []
            self._gt_tac_right = []
            return

        target_tactile = torch.stack([left_stack, right_stack], dim=1).unsqueeze(0)
        target_kwargs = (
            {"target_tactile_markers": target_tactile.to(device=self.device)}
            if self.tactile_input_type == "marker"
            else {"target_tactile_images": target_tactile.to(device=self.device)}
        )

        with torch.inference_mode():
            encoded_gt_tactiles = self.model.model.encode_target_tactile_trajectory(
                **target_kwargs
            )

        if encoded_gt_tactiles is None:
            print(
                "[WLA] tactile eval: model did not return GT tactile latents "
                f"(tactile_input_type={self.tactile_input_type}), skip."
            )
            self._pred_tactiles = None
            self._gt_tac_left = []
            self._gt_tac_right = []
            return

        gt_tactiles = encoded_gt_tactiles[0].float().cpu().numpy()

        pred_tactiles = self._pred_tactiles[:horizon]
        per_step_mse = np.mean((pred_tactiles - gt_tactiles) ** 2, axis=1)
        per_step_cos = []
        for step in range(horizon):
            pred_vec = pred_tactiles[step]
            gt_vec = gt_tactiles[step]
            denom = np.linalg.norm(pred_vec) * np.linalg.norm(gt_vec) + 1e-8
            per_step_cos.append(float(np.dot(pred_vec, gt_vec) / denom))
        per_step_cos = np.asarray(per_step_cos, dtype=np.float32)

        pred_left_imgs = pred_right_imgs = None
        if (
            self.tactile_input_type == "image"
            and self._tactile_decoder_bank is not None
            and self.model.model.tactile_target_proj is not None
        ):
            from policy.wla.tactile_pred_utils import decode_pred_tactile_images

            pred_left_imgs, pred_right_imgs = decode_pred_tactile_images(
                target_proj=self.model.model.tactile_target_proj,
                decoder_bank=self._tactile_decoder_bank,
                pred_latent=torch.as_tensor(pred_tactiles, device=self.device),
                latent_dim=self.model.config.tactile_latent_dim,
            )
            pred_left_imgs = pred_left_imgs.float().cpu()
            pred_right_imgs = pred_right_imgs.float().cpu()

        result = {
            "caption": getattr(task, "instruction", ""),
            "pred_tactiles": pred_tactiles,
            "gt_tactiles": gt_tactiles,
            "per_step_mse": per_step_mse,
            "per_step_cos": per_step_cos,
            "gt_left_imgs": left_stack if self.tactile_input_type == "image" else None,
            "gt_right_imgs": right_stack if self.tactile_input_type == "image" else None,
            "pred_left_imgs": pred_left_imgs,
            "pred_right_imgs": pred_right_imgs,
            "tactile_input_type": self.tactile_input_type,
            "chunk_start_step": self._tactile_chunk_start,
            "episode_idx": self._episode_idx,
        }

        save_root = Path(getattr(task, "save_root", self._current_task_save_root or Path("wla_debug")))
        chunk_dir = (
            save_root
            / "tactile_pred"
            / f"episode_{self._episode_idx:03d}_step_{self._tactile_chunk_start:05d}"
        )
        chunk_dir.mkdir(parents=True, exist_ok=True)

        from policy.wla.tactile_pred_utils import (
            plot_latent_heatmap,
            plot_latent_metrics,
            plot_latent_temporal_diff,
            plot_tactile_image_comparison,
            save_contact_sheet,
        )

        timesteps = tuple(
            step for step in self.tactile_eval_timesteps if 0 <= step < horizon
        )
        np.savez(
            chunk_dir / "tactile_pred_vs_gt.npz",
            pred_tactiles=pred_tactiles,
            gt_tactiles=gt_tactiles,
            per_step_mse=per_step_mse,
            per_step_cos=per_step_cos,
            chunk_start_step=self._tactile_chunk_start,
        )
        plot_latent_metrics(result, chunk_dir / "latent_metrics.png")
        plot_latent_heatmap(result, chunk_dir / "latent_heatmap.png")
        plot_latent_temporal_diff(result, chunk_dir / "latent_temporal_diff.png")
        if timesteps and self.tactile_input_type == "image":
            plot_tactile_image_comparison(result, timesteps, chunk_dir / "tactile_image_compare.png")
            save_contact_sheet(result, timesteps, chunk_dir / "tactile_contact_sheet.png")

        summary_item = {
            "episode_idx": self._episode_idx,
            "chunk_start_step": int(self._tactile_chunk_start),
            "tactile_input_type": self.tactile_input_type,
            "mean_tactile_mse": float(np.mean(per_step_mse)),
            "mean_tactile_cos": float(np.mean(per_step_cos)),
            "has_pred_images": pred_left_imgs is not None,
            "save_dir": str(chunk_dir),
        }
        self._tactile_eval_summary.append(summary_item)
        summary_file = save_root / "tactile_pred" / "summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
        existing.append(summary_item)
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        print(
            f"[WLA] tactile eval saved to {chunk_dir} "
            f"(MSE={summary_item['mean_tactile_mse']:.5f}, cos={summary_item['mean_tactile_cos']:.4f})"
        )

        self._pred_tactiles = None
        self._gt_tac_left = []
        self._gt_tac_right = []

    def _sample_action_chunk(self, task, observation):
        input_images, states, tactile_images, tactile_markers = self.encode_obs(observation)
        instruction = self.instruction_override or task.instruction or "clean"

        model_dtype = next(self.model.model.policy_head.parameters()).dtype
        states = states.to(dtype=model_dtype, device=self.device)
        sample_kwargs = {
            "caption": instruction,
            "input_images": input_images,
            "num_images_per_prompt": 1,
            "states": states,
        }
        if tactile_images is not None:
            sample_kwargs["tactile_images"] = tactile_images.to(
                dtype=model_dtype,
                device=self.device,
            )
        if tactile_markers is not None:
            sample_kwargs["tactile_markers"] = tactile_markers.to(
                dtype=model_dtype,
                device=self.device,
            )

        return_tactile = (
            self.eval_predict_tactile
            and getattr(self.model.config, "predict_tactile", False)
        )
        if return_tactile:
            sample_kwargs["return_tactile"] = True

        with torch.inference_mode():
            if self.use_autocast:
                with torch.autocast(device_type="cuda", dtype=self.model_dtype):
                    model_out = self.model.sample_actions(**sample_kwargs)
            else:
                model_out = self.model.sample_actions(**sample_kwargs)

        pred_tactiles = None
        if return_tactile:
            actions, pred_tactiles = model_out
        else:
            actions = model_out

        actions = torch.as_tensor(actions, dtype=torch.float32)
        actions = self._unnormalize_and_unpad(
            actions,
            self.norm_stats["action"],
            self.original_action_dim,
        )
        if self.action_horizon > 0:
            actions = actions[: self.action_horizon]

        if self.debug_log:
            print(
                f"[WLA] infer={self._infer_count} instruction={instruction!r} "
                f"chunk_shape={tuple(actions.shape)} first_action={actions[0].tolist()}"
            )
        self._infer_count += 1
        return [a.detach().clone() for a in actions], pred_tactiles

    def _action_chunk_to_numpy(self, action_chunk):
        return torch.stack(action_chunk).cpu().numpy().astype(np.float32)

    def _ensure_temporal_agg_buffer(self, task):
        if self._all_time_actions is not None:
            return
        max_steps = int(getattr(task.cfg, "step_lim", 600)) + self.action_horizon + 1
        self._all_time_actions = np.full(
            (max_steps, max_steps, self.original_action_dim),
            np.nan,
            dtype=np.float32,
        )

    def _eval_temporal_agg(self, task, observation):
        self._ensure_temporal_agg_buffer(task)
        t = self._exec_count

        should_query = (t % self.temporal_agg_query_frequency == 0)
        if not should_query:
            current = self._all_time_actions[: t + 1, t]
            should_query = not np.any(np.all(np.isfinite(current), axis=1))

        if should_query:
            actions = self._action_chunk_to_numpy(
                self._sample_action_chunk(task, observation)[0]
            )
            end_t = min(t + actions.shape[0], self._all_time_actions.shape[1])
            self._all_time_actions[t, t:end_t] = actions[: end_t - t]

        actions_for_curr_step = self._all_time_actions[: t + 1, t]
        actions_populated = np.all(np.isfinite(actions_for_curr_step), axis=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]
        if actions_for_curr_step.size == 0:
            actions = self._action_chunk_to_numpy(
                self._sample_action_chunk(task, observation)[0]
            )
            end_t = min(t + actions.shape[0], self._all_time_actions.shape[1])
            self._all_time_actions[t, t:end_t] = actions[: end_t - t]
            actions_for_curr_step = self._all_time_actions[: t + 1, t]
            actions_for_curr_step = actions_for_curr_step[
                np.all(np.isfinite(actions_for_curr_step), axis=1)
            ]

        exp_weights = np.exp(
            -self.temporal_agg_k * np.arange(len(actions_for_curr_step), dtype=np.float32)
        )
        exp_weights = exp_weights / exp_weights.sum()
        action = (actions_for_curr_step * exp_weights[:, None]).sum(axis=0).astype(np.float32)

        if self.debug_log:
            print(
                f"[WLA] temporal_agg t={t} query={should_query} "
                f"num_preds={len(actions_for_curr_step)} action={action.tolist()}"
            )
        return action

    def eval(self, task, observation):
        if self.temporal_agg:
            action = torch.from_numpy(
                self._eval_temporal_agg(task, observation)
            ).to(task.device).float()
        else:
            need_replan = len(self._action_buffer) == 0 or (
                self.replan_every > 0
                and self._exec_count > 0
                and self._exec_count % self.replan_every == 0
            )
            if need_replan:
                if self._tactile_eval_active:
                    self._finalize_tactile_eval_chunk(task)
                chunk, pred_tactiles = self._sample_action_chunk(task, observation)
                self._action_buffer = chunk
                self._start_tactile_eval_chunk(task, pred_tactiles)

            if self.eval_predict_tactile:
                self._record_tactile_gt(observation)

            action = self._action_buffer.pop(0).to(task.device).float()
        if self.debug_save_images and self._exec_count % self.debug_save_every == 0:
            self.save(task.get_frame_shot(observation), task.take_action_cnt)
        task.take_action(action, action_type="qpos")
        self._exec_count += 1

    def reset(self):
        if self._tactile_eval_active and self._current_task_save_root is not None:
            self._finalize_tactile_eval_chunk(
                SimpleNamespace(
                    save_root=self._current_task_save_root,
                    instruction="",
                )
            )
        self._episode_idx += 1
        self._action_buffer = []
        self._all_time_actions = None
        self._head_image_history = []
        self._infer_count = 0
        self._exec_count = 0

    def close(self):
        if self._tactile_eval_active and self._current_task_save_root is not None:
            self._finalize_tactile_eval_chunk(
                SimpleNamespace(
                    save_root=self._current_task_save_root,
                    instruction="",
                )
            )
        if self.model is not None and hasattr(self.model, "close"):
            self.model.close()

    def save(self, img, t):
        from PIL import Image, ImageDraw, ImageFont

        out_dir = Path("wla_debug")
        out_dir.mkdir(parents=True, exist_ok=True)
        obs = Image.fromarray(img.cpu().numpy())
        draw = ImageDraw.Draw(obs)
        draw.text((obs.width - 100, obs.height - 60), f"{t:03d}", fill=(255, 0, 0), font=ImageFont.load_default())
        obs.save(out_dir / f"episode_{self._episode_idx:03d}_step_{t:04d}.png")
