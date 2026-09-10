"""Experiment configuration: ONE yaml file fully defines ONE model's pipeline.

Sections: seed / data (dataset_dir + collection knobs) / model (size + arch
overrides) / train (SFT hyperparameters + output_dir) / eval (ckpt + output_dir)
/ sim / expert / task / grpo. Unknown keys are rejected so typos fail loudly.
CLI flags only override scalars for smoke runs; --out-dir redirects the three
output roots into one temporary workspace.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

import torch

# Ball color -> RGB (uint8)
COLOR_RGB = {
    "red": (222, 70, 70),
    "green": (70, 200, 100),
    "blue": (80, 130, 235),
}

# Bin colors (rim / interior)
BIN_RIM = (240, 160, 70)
BIN_FILL = (52, 45, 40)

# yaml `model:` keys (besides `size`) that override the preset architecture
ARCH_KEYS = ("ch", "extra_blocks", "d_model", "n_layer", "d_ff",
             "act_hidden", "ground_hidden", "use_aux_heads")

# Which config fields each yaml section may set (yaml key -> dataclass field)
_SECTIONS = {
    "sim": {k: k for k in (
        "link_lengths", "world_extent", "obs_res", "ball_radius", "n_balls",
        "bin_radius", "bin_r_range", "min_ball_dist", "min_ball_bin_dist",
        "action_max", "grasp_thresh", "place_overlap_frac", "max_steps",
        "target_r_range", "init_q1_range", "init_q2_range")},
    "expert": {k: k for k in (
        "expert_gain", "expert_fine_gain", "expert_fine_radius",
        "expert_noise", "expert_grasp_dist", "expert_release_dist")},
    "task": {k: k for k in ("proprio_dim", "action_dim", "lang_len")},
    "train": {"output_dir": "train_dir", "epochs": "epochs",
              "batch_size": "batch_size", "lr": "lr",
              "weight_decay": "weight_decay", "balanced_losses": "balanced_losses",
              "aux_target_weight": "aux_target_weight",
              "aux_bin_weight": "aux_bin_weight",
              "action_grip_weight": "action_grip_weight"},
    "eval": {"output_dir": "eval_dir", "gif_fps": "gif_fps",
             "frame_res": "frame_res", "ckpt": "eval_ckpt", "gifs": "eval_gifs"},
    "grpo": {"iterations": "grpo_iterations", "group_size": "grpo_group_size",
             "lr": "grpo_lr", "episodes": "grpo_episodes",
             "ckpt": "grpo_ckpt", "test_set": "grpo_test_set",
             "eval_every": "grpo_eval_every", "gifs": "grpo_gifs",
             "kl_coef": "grpo_kl_coef", "early_stop": "grpo_early_stop",
             "aux_weight": "grpo_aux_weight",
             "steps_per_domain": "grpo_steps_per_domain",
             "output_dir": "grpo_train_dir"},
}
# `data:` yaml keys -> dataclass fields
_DATA_KEYS = {"dataset_dir": "dataset_dir", "episodes_collect": "episodes_collect",
              "val_frac": "val_frac", "eval_episodes": "eval_episodes"}
_ROOT_KEYS = ("seed",)
_ALL_SECTIONS = ("model", "data") + tuple(_SECTIONS)


@dataclass
class Config:
    # ---------- Simulation ----------
    seed: int = 42
    link_lengths: tuple = (0.55, 0.55)   # (l1, l2)
    world_extent: float = 1.25           # world coords span [-E, E]^2
    obs_res: int = 64                    # model observation image resolution
    ball_radius: float = 0.075
    n_balls: int = 3                     # red / green / blue
    bin_radius: float = 0.20             # ball must lie fully inside the rim
    bin_r_range: tuple = (0.50, 0.95)    # bin center sampling radius range
    min_ball_dist: float = 0.40          # min distance between balls
    min_ball_bin_dist: float = 0.35      # min distance ball-to-bin center
    action_max: float = 0.15             # per-step joint delta clamp (rad)
    grasp_thresh: float = 0.09           # snap-on distance when closing gripper
    place_overlap_frac: float = 1.0      # 1.0 = the ball must lie fully inside the bin rim
    max_steps: int = 90
    target_r_range: tuple = (0.45, 1.0)  # ball sampling radius range
    init_q1_range: tuple = (-2.5, 2.5)
    init_q2_range: tuple = (-2.6, -0.5)  # elbow-down initial configurations

    # ---------- Expert ----------
    expert_gain: float = 0.55            # proportional IK gain (cruise)
    expert_fine_gain: float = 0.30       # slower gain near the goal
    expert_fine_radius: float = 0.18    # switch to fine gain within this distance
    expert_noise: float = 0.002          # demo joint-action noise (rad)
    expert_grasp_dist: float = 0.11      # close gripper within this distance of the ball
    expert_release_dist: float = 0.05   # release with the ball near the bin center

    # ---------- Language ----------
    lang_len: int = 10                   # instruction token sequence length

    # ---------- Model (yaml `model:` section) ----------
    model_size: str = "3m"               # preset key (3m/8m) or free experiment label
    model_arch: dict = field(default_factory=dict)   # arch overrides (ARCH_KEYS)
    balanced_losses: bool = True         # False: disable class-balanced losses (ablation)

    # ---------- Task I/O dims ----------
    proprio_dim: int = 6                 # q1, q2, ee_x, ee_y, grip, holding
    action_dim: int = 3                  # dq1, dq2, grip
    aux_target_weight: float = 1.0       # commanded-ball offset loss weight
    aux_bin_weight: float = 1.0          # bin offset loss weight
    action_grip_weight: float = 0.3      # grip MSE weight (after class balancing)

    # ---------- Training (behavior cloning) ----------
    episodes_collect: int = 10000        # demo episodes (~250k samples)
    val_frac: float = 0.30               # validation split by episode (avoids adjacent-frame leakage)
    batch_size: int = 256
    epochs: int = 15
    lr: float = 2.0e-4
    weight_decay: float = 1.0e-5

    # ---------- GRPO RL (stage 2; separate stage with its own dataset & artifacts) ----------
    grpo_iterations: int = 0             # 0 = skip GRPO (SFT-only)
    grpo_group_size: int = 8             # rollout members per iteration (same scenario)
    grpo_lr: float = 2.5e-5              # smaller than the SFT lr (fine-tuning)
    grpo_episodes: int = 3000            # ID RL training scenarios (fresh, disjoint seeds)
    ood_test_episodes: int = 300         # OOD test scenarios (disjoint from all training sets)
    grpo_ckpt: str | None = None         # SFT checkpoint to fine-tune (required; --ckpt overrides)
    grpo_test_set: str | None = None     # test set for GRPO evals (default: data.dataset_dir/test.npz)
    grpo_eval_every: int = 25            # light test-eval every N RL iterations (0 = off)
    grpo_gifs: int = 100                 # replay GIFs randomly saved by the post-GRPO eval
    grpo_kl_coef: float = 2.0            # anchor-to-SFT loss coefficient (stability)
    grpo_early_stop: int = 10            # stop when mean reward is stuck this many iterations
    grpo_aux_weight: float = 0.5         # aux-consistency loss weight in GRPO (0 = off)
    grpo_steps_per_domain: int = 20      # gradient steps per iteration (one ID scenario each)
    grpo_train_dir: str | None = None    # per-size GRPO artifacts root (yaml grpo.output_dir)

    # ---------- Inference / evaluation ----------
    eval_episodes: int = 500
    eval_ckpt: str | None = None         # checkpoint to evaluate (required; --ckpt overrides)
    eval_gifs: int = 30                  # random replay GIFs saved by the evaluation
    gif_fps: int = 12
    frame_res: int = 256                # visualization frame resolution

    # ---------- Output locations (all specified by the yaml) ----------
    dataset_dir: str = "dataset"         # yaml data.dataset_dir: train.npz / val.npz / test.npz
    train_dir: str | None = None         # yaml train.output_dir: training artifacts
    eval_dir: str | None = None          # yaml eval.output_dir: evaluation artifacts

    # ---------- YAML loading ----------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load one experiment from a yaml file (strict key validation)."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"config yaml must be a mapping: {path}")
        cfg = cls()
        raw = dict(raw)

        for k in _ROOT_KEYS:                       # root scalars
            if k in raw:
                setattr(cfg, k, raw.pop(k))

        model = raw.pop("model", None) or {}       # model section -> size + arch
        if not isinstance(model, dict):
            raise ValueError("`model:` section must be a mapping")
        bad = set(model) - {"size"} - set(ARCH_KEYS)
        if bad:
            raise ValueError(f"unknown `model:` keys {sorted(bad)}; "
                             f"valid: size, {', '.join(ARCH_KEYS)}")
        cfg.model_size = str(model.pop("size", cfg.model_size))
        cfg.model_arch = {k: model[k] for k in ARCH_KEYS if k in model}

        data = raw.pop("data", None) or {}         # data section (dir + knobs)
        if not isinstance(data, dict):
            raise ValueError("`data:` section must be a mapping")
        bad = set(data) - set(_DATA_KEYS)
        if bad:
            raise ValueError(f"unknown `data:` keys {sorted(bad)}; "
                             f"valid: {', '.join(_DATA_KEYS)}")
        for yk, fk in _DATA_KEYS.items():
            if yk in data:
                setattr(cfg, fk, data[yk])

        for sec, keys in _SECTIONS.items():        # remaining sections
            vals = raw.pop(sec, None) or {}
            if not isinstance(vals, dict):
                raise ValueError(f"`{sec}:` section must be a mapping")
            bad = set(vals) - set(keys)
            if bad:
                raise ValueError(f"unknown `{sec}:` keys {sorted(bad)}; "
                                 f"valid: {', '.join(keys)}")
            for yk, fk in keys.items():
                if yk in vals:
                    setattr(cfg, fk, vals[yk])

        if raw:                                    # anything left is a typo
            raise ValueError(f"unknown config keys/sections {sorted(raw)}; "
                             f"valid sections: {', '.join(_ALL_SECTIONS)} "
                             f"(root: {', '.join(_ROOT_KEYS)})")
        return cfg

    # ---------- Derived paths ----------
    @property
    def train_npz(self) -> str:
        """Training set (written by the dataset generation stage)."""
        return str(Path(self.dataset_dir) / "train.npz")

    @property
    def val_npz(self) -> str:
        """Validation set (held-out episodes, split by episode)."""
        return str(Path(self.dataset_dir) / "val.npz")

    @property
    def test_npz(self) -> str:
        """Fixed closed-loop test set: one initial scenario per episode."""
        return str(Path(self.dataset_dir) / "test.npz")

    @property
    def grpo_npz(self) -> str:
        """RL training scenarios (stored inside dataset/, third seed space)."""
        return str(Path(self.dataset_dir) / "grpo.npz")

    @property
    def grpo_ckpt_dir(self) -> str:
        """HF-format GRPO checkpoint directory (final RL model)."""
        suffix = "_noaux" if not self.model_arch.get("use_aux_heads", True) else ""
        return str(Path(self.grpo_train_dir) / f"pocketvla-{self.model_size}{suffix}-grpo")

    @property
    def grpo_log_path(self) -> str:
        return str(Path(self.grpo_train_dir) / "grpo_log.json")

    @property
    def grpo_curve_path(self) -> str:
        return str(Path(self.grpo_train_dir) / "grpo_curve.png")

    @property
    def demos_dir(self) -> str:
        """Expert demo replay GIFs from the dataset generation stage."""
        return str(Path(self.dataset_dir) / "demos")

    @property
    def hf_dir(self) -> str:
        """HF-format model directory: <train.output_dir>/pocketvla-<size>[_noaux]."""
        suffix = "_noaux" if not self.model_arch.get("use_aux_heads", True) else ""
        return str(Path(self.train_dir) / f"pocketvla-{self.model_size}{suffix}")

    @property
    def train_log_path(self) -> str:
        return str(Path(self.train_dir) / "train_log.json")

    @property
    def loss_curve_path(self) -> str:
        return str(Path(self.train_dir) / "loss_curve.png")

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        arch = ", ".join(f"{k}={self.model_arch[k]}" for k in ARCH_KEYS
                        if k in self.model_arch) or "preset defaults"
        return (f"model={self.model_size} ({arch})  dataset_dir={self.dataset_dir}  "
                f"train_dir={self.train_dir}  eval_dir={self.eval_dir}  "
                f"seed={self.seed}  lr={self.lr:.1e}  epochs={self.epochs}")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if missing; return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_device() -> torch.device:
    """Compute device; the accelerator is exposed through the CUDA compatibility layer."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_banner(device: torch.device) -> str:
    if device.type == "cuda":
        return f"Accelerator: {torch.cuda.get_device_name(0)} (via torch.cuda)"
    return "Accelerator: CPU"
