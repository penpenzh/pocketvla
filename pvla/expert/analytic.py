"""Analytic expert: closed-form IK + dual-rate proportional control + grasp/place state machine."""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..sim.arm2d import wrap_angle


def ik_2r(target, l1: float, l2: float) -> np.ndarray:
    """Closed-form IK for a 2-link planar arm, fixed elbow-down branch."""
    x, y = float(target[0]), float(target[1])
    r = np.hypot(x, y)
    r_lo, r_hi = abs(l1 - l2) + 1e-6, l1 + l2 - 1e-6
    r_c = float(np.clip(r, r_lo, r_hi))
    if r > 0 and r_c != r:
        x, y = x * r_c / r, y * r_c / r
    c2 = float(np.clip((r_c ** 2 - l1 ** 2 - l2 ** 2) / (2.0 * l1 * l2), -1.0, 1.0))
    q2 = -np.arccos(c2)                       # elbow-down branch: q2 in [-pi, 0)
    q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    return np.array([q1, q2])


class AnalyticExpert:
    """Privileged expert controller (dual-rate: low gain inside the critical zone)."""

    def __init__(self, cfg: Config, seed: int | None = None, noise: float | None = None):
        self.cfg = cfg
        self.noise = cfg.expert_noise if noise is None else noise
        self.rng = np.random.default_rng(seed)

    def _drive_to(self, env, target) -> np.ndarray:
        """Dual-rate proportional joint deltas toward a world-coordinate target."""
        cfg = self.cfg
        q_star = ik_2r(target, *env.cfg.link_lengths)
        dq = wrap_angle(q_star - env.q)
        dist = float(np.hypot(*(env.ee - target)))
        gain = cfg.expert_fine_gain if dist < cfg.expert_fine_radius else cfg.expert_gain
        a = np.clip(gain * dq, -cfg.action_max, cfg.action_max)
        if self.noise > 0:
            a = a + self.rng.normal(0.0, self.noise, size=2)
        return np.clip(a, -cfg.action_max, cfg.action_max)

    def act(self, env) -> np.ndarray:
        """Return (dq1, dq2, grip) in R^2 x {-1, +1}."""
        cfg = self.cfg
        if env.held is None:
            # Phase 1: approach the commanded ball (dual-rate)
            a = self._drive_to(env, env.target_pos)
            near = np.hypot(*(env.ee - env.target_pos)) < cfg.expert_grasp_dist
            grip = 1.0 if near else -1.0
        elif env.held == env.commanded:
            # Phase 2: carry to the bin (dual-rate)
            a = self._drive_to(env, env.bin_pos)
            near = np.hypot(*(env.ee - env.bin_pos)) < cfg.expert_release_dist
            grip = -1.0 if near else 1.0
        else:
            # Phase 3 (recover): holding the wrong ball — release and turn to the commanded one
            a = self._drive_to(env, env.target_pos)
            grip = -1.0
        return np.array([a[0], a[1], grip])

    def rollout(self, env, scenario: dict | None = None, **reset_kwargs) -> dict:
        """Run a full episode. reset_kwargs can pin the commanded color/template;
        passing a stored scenario (a dataset test.npz row) replays that fixed episode."""
        if scenario is not None:
            env.reset_scenario(scenario)
        else:
            env.reset(**reset_kwargs)
        q0 = env.q.copy()
        ee_path = [env.ee.copy()]
        ball_path = [env.target_pos.copy()]
        dists = []
        done = False
        while not done:
            a = self.act(env)
            obs, _, done, info = env.step(a)
            ee_path.append(env.ee.copy())
            ball_path.append(env.target_pos.copy())
            dists.append(info["dist"])
        return {
            "ee_path": np.array(ee_path),
            "ball_path": np.array(ball_path),
            "dists": dists,
            "success": info["success"],
            "steps": info["steps"],
            "q0": q0,
        }
