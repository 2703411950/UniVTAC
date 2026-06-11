import json
import importlib.abc
import importlib.util
import os
import site
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from .._base_policy import BasePolicy


DEFAULT_MANTIS_ROOT = (
    "/inspire/hdd/global_user/yangyi-253108120173/inspire_shared/mount/"
    "advanced-machine-learning-and-deep-learning-applications/cyy/"
    "Mantis_flow_depth_use_metaqury"
)


class _FixedModuleFinder(importlib.abc.MetaPathFinder):
    def __init__(self, fullname: str, file_path: Path):
        self.fullname = fullname
        self.file_path = Path(file_path)

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.fullname:
            return None
        return importlib.util.spec_from_file_location(fullname, self.file_path)


def _prefer_conda_site_packages():
    """Keep IsaacSim pip prebundle from shadowing packages needed by Mantis."""
    site_packages = site.getsitepackages()
    sys.path[:] = [p for p in sys.path if p not in site_packages]
    sys.path[:0] = site_packages

    removed_prefixes = []
    cleaned_path = []
    for path in sys.path:
        if "pip_prebundle" in path:
            removed_prefixes.append(path)
            continue
        cleaned_path.append(path)
    sys.path[:] = cleaned_path

    if not removed_prefixes:
        return

    for module_name, module in list(sys.modules.items()):
        if not (
            module_name == "boto3"
            or module_name.startswith("boto3.")
            or module_name == "botocore"
            or module_name.startswith("botocore.")
            or module_name == "s3transfer"
            or module_name.startswith("s3transfer.")
        ):
            continue
        module_file = getattr(module, "__file__", "") or ""
        if "pip_prebundle" in module_file or any(module_file.startswith(prefix) for prefix in removed_prefixes):
            del sys.modules[module_name]

    import botocore
    conda_botocore_paths = [
        str(Path(p) / "botocore")
        for p in site_packages
        if (Path(p) / "botocore" / "__init__.py").exists()
    ]
    if conda_botocore_paths:
        botocore.__path__ = conda_botocore_paths

    if not conda_botocore_paths:
        import botocore.httpchecksum as httpchecksum
    else:
        httpchecksum_path = Path(conda_botocore_paths[0]) / "httpchecksum.py"
        sys.meta_path[:] = [
            finder for finder in sys.meta_path
            if not (
                isinstance(finder, _FixedModuleFinder)
                and finder.fullname == "botocore.httpchecksum"
            )
        ]
        sys.meta_path.insert(0, _FixedModuleFinder("botocore.httpchecksum", httpchecksum_path))
        spec = importlib.util.spec_from_file_location("botocore.httpchecksum", httpchecksum_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load botocore.httpchecksum from {httpchecksum_path}")
        httpchecksum = importlib.util.module_from_spec(spec)
        sys.modules["botocore.httpchecksum"] = httpchecksum
        setattr(botocore, "httpchecksum", httpchecksum)
        spec.loader.exec_module(httpchecksum)
    if not hasattr(httpchecksum, "DEFAULT_CHECKSUM_ALGORITHM"):
        raise ImportError(f"Loaded incompatible botocore from {httpchecksum.__file__}")


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

        from models.mantis import Mantis
        from utils.transforms import _make_transform, normalize_and_pad, unnormalize_and_unpad

        self._normalize_and_pad = normalize_and_pad
        self._unnormalize_and_unpad = unnormalize_and_unpad
        self.primary_image_transform = _make_transform(int(args.get("primary_image_size", 256)))
        self.auxiliary_image_transform = _make_transform(int(args.get("auxiliary_image_size", 256)))

        self.max_state_dim = int(args.get("max_state_dim", 8))
        self.original_action_dim = int(args.get("original_action_dim", 8))
        self.action_horizon = int(args.get("action_horizon", 32))
        self.replan_every = int(args.get("replan_every", self.action_horizon))
        self.device = torch.device(args.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
        self.use_autocast = bool(args.get("use_autocast", True)) and self.device.type == "cuda"
        self.convert_rgb_to_bgr = bool(args.get("convert_rgb_to_bgr", True))
        self.use_tactile_images = bool(args.get("use_tactile_images", True))
        self.debug_log = bool(args.get("debug_log", False))
        self.debug_save_images = bool(args.get("debug_save_images", False))
        self.debug_save_every = int(args.get("debug_save_every", 50))
        self.instruction_override = args.get("instruction", None)

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

        print(f"[WLA] Loading model config/weights from {args['model_id']}")
        self.model = Mantis.from_pretrained(
            args["model_id"],
            input_size=input_size,
            torch_dtype=self.model_dtype,
            dtype=self.model_dtype,
        )

        checkpoint_path = Path(args["checkpoint_path"])
        print(f"[WLA] Loading checkpoint state_dict from {checkpoint_path}")
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(device=self.device, dtype=self.model_dtype)
        self.model.eval()

        # Action-only deployment does not need the image decoder branch.
        if hasattr(self.model, "vae"):
            del self.model.vae
        if hasattr(self.model.model, "transformer"):
            del self.model.model.transformer
        if hasattr(self.model.model, "connector"):
            del self.model.model.connector

        self._action_buffer = []
        self._episode_idx = -1
        self._infer_count = 0
        self._exec_count = 0
        print(
            f"[WLA] Ready on {self.device}, dtype={self.model_dtype}, "
            f"action_horizon={self.action_horizon}, replan_every={self.replan_every}, "
            f"use_tactile_images={self.use_tactile_images}"
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

    def encode_obs(self, observation):
        head = self._image_to_tensor(
            observation["observation"]["head"]["rgb"],
            convert_rgb_to_bgr=self.convert_rgb_to_bgr,
        )
        images = [self.primary_image_transform(head)]
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

        input_images = [images]

        state = observation["embodiment"]["joint"][: self.original_action_dim].detach().cpu().float()
        state, _ = self._normalize_and_pad(
            state,
            self.norm_stats["observation.state"],
            self.max_state_dim,
        )
        state = state.unsqueeze(0)
        return input_images, state

    def _sample_action_chunk(self, task, observation):
        input_images, states = self.encode_obs(observation)
        instruction = self.instruction_override or task.instruction or "clean"

        model_dtype = next(self.model.model.policy_head.parameters()).dtype
        states = states.to(dtype=model_dtype, device=self.device)

        with torch.inference_mode():
            if self.use_autocast:
                with torch.autocast(device_type="cuda", dtype=self.model_dtype):
                    actions = self.model.sample_actions(
                        caption=instruction,
                        input_images=input_images,
                        num_images_per_prompt=1,
                        states=states,
                    )
            else:
                actions = self.model.sample_actions(
                    caption=instruction,
                    input_images=input_images,
                    num_images_per_prompt=1,
                    states=states,
                )

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
        return [a.detach().clone() for a in actions]

    def eval(self, task, observation):
        if len(self._action_buffer) == 0 or (
            self.replan_every > 0 and self._exec_count > 0 and self._exec_count % self.replan_every == 0
        ):
            self._action_buffer = self._sample_action_chunk(task, observation)

        action = self._action_buffer.pop(0).to(task.device).float()
        if self.debug_save_images and self._exec_count % self.debug_save_every == 0:
            self.save(task.get_frame_shot(observation), task.take_action_cnt)
        task.take_action(action, action_type="qpos")
        self._exec_count += 1

    def reset(self):
        self._episode_idx += 1
        self._action_buffer = []
        self._infer_count = 0
        self._exec_count = 0

    def save(self, img, t):
        from PIL import Image, ImageDraw, ImageFont

        out_dir = Path("wla_debug")
        out_dir.mkdir(parents=True, exist_ok=True)
        obs = Image.fromarray(img.cpu().numpy())
        draw = ImageDraw.Draw(obs)
        draw.text((obs.width - 100, obs.height - 60), f"{t:03d}", fill=(255, 0, 0), font=ImageFont.load_default())
        obs.save(out_dir / f"episode_{self._episode_idx:03d}_step_{t:04d}.png")
