"""OOD scenario generator: standard scene plus one unseen-color distractor
(ball or square). Used by `collect` (OOD test set) and pvla.eval_ood.

Seed spaces (offsets vs cfg.seed): demos=seed, test=+20,000, RL=+40,000, OOD=+80,000.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .lang.tokenizer import COLORS
from .sim.arm2d import Arm2DPickPlace

OOD_TEST_SEED_OFFSET = 80_000     # OOD test scenarios
DISTRACTOR_COLORS = ("yellow", "purple", "orange", "cyan", "pink",
                     "brown", "white", "gray")


def gen_ood_scenarios(cfg: Config, n: int, seed_offset: int, seed: int) -> dict:
    """Build n OOD scenarios: standard 3 balls + bin, plus exactly ONE decoy
    object (unseen-color ball or square, 50/50) spaced from everything else.
    The only difference from the in-distribution test set is the decoy."""
    rng = np.random.default_rng(seed + seed_offset)
    out: dict[str, list] = {"balls": [], "bin_pos": [], "q0": [], "color_id": [],
                            "tokens": [], "instruction": [],
                            "distractor_colors": [], "distractor_pos": [],
                            "distractor_shapes": []}
    for _ in range(n):
        env = Arm2DPickPlace(cfg, seed=rng.integers(2**31))
        env.reset()
        out["balls"].append(np.array([p for _, p in env.balls], dtype=np.float64))
        out["bin_pos"].append(env.bin_pos.astype(np.float64))
        out["q0"].append(env.q.astype(np.float64))
        out["color_id"].append(np.int64(COLORS.index(env.commanded)))
        out["tokens"].append(env.tokens.copy())
        out["instruction"].append(env.instruction)
        # exactly one decoy: unseen-color ball or square, spaced from everything
        decoy_shape = str(rng.choice(("ball", "square")))
        d_colors: list[str] = []
        d_pos: list[np.ndarray] = []
        for c in rng.choice(DISTRACTOR_COLORS, size=1, replace=False):
            for _ in range(200):
                r = rng.uniform(*cfg.target_r_range)
                th = rng.uniform(0.0, 2.0 * np.pi)
                p = np.array([r * np.cos(th), r * np.sin(th)])
                others = [b for _, b in env.balls]
                if (np.hypot(*(p - env.bin_pos)) > cfg.min_ball_bin_dist
                        and all(np.hypot(*(p - b)) > cfg.min_ball_dist for b in others)):
                    d_colors.append(str(c))
                    d_pos.append(p)
                    break
        out["distractor_colors"].append(np.array(d_colors, dtype=object))
        out["distractor_shapes"].append(np.array([decoy_shape]))
        out["distractor_pos"].append(np.stack(d_pos).astype(np.float64)
                                     if d_pos else np.zeros((0, 2)))
    return {
        "balls": np.stack(out["balls"]),                    # (n,3,2) red/green/blue
        "bin_pos": np.stack(out["bin_pos"]),                # (n,2)
        "q0": np.stack(out["q0"]),                          # (n,2)
        "color_id": np.array(out["color_id"], dtype=np.int64),
        "tokens": np.stack(out["tokens"]),                  # (n,lang_len)
        "instruction": np.array(out["instruction"]),
        "distractor_colors": np.array(out["distractor_colors"], dtype=object),
        "distractor_pos": np.array(out["distractor_pos"], dtype=object),
        "distractor_shapes": np.array(out["distractor_shapes"], dtype=object),
    }


