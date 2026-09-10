"""2-DOF planar-arm language-guided pick-and-place simulation environment (pure NumPy)."""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..lang.tokenizer import COLORS, TEMPLATES, TOKENIZER, make_instruction
from .render import SceneRenderer


def wrap_angle(a):
    """Wrap an angle to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def circle_overlap_frac(d: float, r: float, R: float) -> float:
    """Fraction of a circle's area (radius r) lying inside a disk (radius R)
    whose center is at distance d (exact circle-circle lens-area formula)."""
    if d <= R - r:                      # fully inside
        return 1.0
    if d >= R + r:                      # disjoint
        return 0.0
    r2, R2, d2 = r * r, R * R, d * d
    x = float(np.clip((d2 + r2 - R2) / (2.0 * d * r), -1.0, 1.0))
    y = float(np.clip((d2 + R2 - r2) / (2.0 * d * R), -1.0, 1.0))
    a = r2 * np.arccos(x)
    b = R2 * np.arccos(y)
    c = 0.5 * np.sqrt(max(0.0, (-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R)))
    return float((a + b - c) / (np.pi * r2))


def overlap_dist_thresh(frac: float, r: float, R: float) -> float:
    """Invert circle_overlap_frac by bisection: the center distance at which
    the covered fraction equals `frac` (for report / plot annotations)."""
    lo, hi = max(0.0, R - r), R + r
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if circle_overlap_frac(mid, r, R) > frac:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class Arm2DPickPlace:
    def __init__(self, cfg: Config, seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.tokenizer = TOKENIZER
        self._obs_renderer = SceneRenderer(cfg.obs_res, cfg.world_extent)

        self.q = np.zeros(2, dtype=np.float64)
        self.balls: list[tuple[str, np.ndarray]] = []
        self.bin_pos = np.zeros(2, dtype=np.float64)
        self.distractor_balls: list[tuple[str, np.ndarray]] = []   # OOD-only decoy balls
        self.ood_objects: list[tuple[str, np.ndarray, str]] = []   # OOD decoys with shape ("ball"/"square")
        self.held: str | None = None    # color of the currently held ball
        self.grip_cmd: bool = False     # gripper command from the last action (True = closed)
        self.commanded: str | None = None
        self.instruction: str = ""
        self.tokens = np.zeros(cfg.lang_len, dtype=np.int64)
        # Event counters (used for failure attribution in evaluation)
        self.ever_held_commanded = False
        self.ever_held_wrong = False
        self.min_ball_bin_dist = float("inf")
        self.t = 0

    # ---------- Kinematics ----------
    def fk(self, q=None):
        """Forward kinematics: returns (elbow position, EE position)."""
        q = self.q if q is None else q
        l1, l2 = self.cfg.link_lengths
        p1 = np.array([l1 * np.cos(q[0]), l1 * np.sin(q[0])])
        p2 = p1 + np.array([l2 * np.cos(q[0] + q[1]), l2 * np.sin(q[0] + q[1])])
        return p1, p2

    @property
    def ee(self) -> np.ndarray:
        return self.fk()[1]

    @property
    def target_pos(self) -> np.ndarray:
        """Current position of the commanded ball (it may have been moved)."""
        for color, pos in self.balls:
            if color == self.commanded:
                return pos
        raise RuntimeError("target not found")

    @property
    def phase(self) -> str:
        return "carry" if self.held is not None else "reach"

    # ---------- Randomization ----------
    def _sample_balls(self):
        cfg = self.cfg
        pos: list[np.ndarray] = []
        min_dist = cfg.min_ball_dist
        while len(pos) < cfg.n_balls:
            for _ in range(200):
                r = self.rng.uniform(*cfg.target_r_range)
                th = self.rng.uniform(0.0, 2.0 * np.pi)
                p = np.array([r * np.cos(th), r * np.sin(th)])
                if all(np.hypot(*(p - b)) > min_dist for b in pos):
                    pos.append(p)
                    break
            else:
                min_dist *= 0.85  # relax spacing after 200 failed tries
        return list(zip(COLORS, pos))

    def _sample_bin(self, balls):
        """Sample a bin center inside the reachable annulus, keeping its distance
        from all balls."""
        cfg = self.cfg
        for _ in range(500):
            r = self.rng.uniform(*cfg.bin_r_range)
            th = self.rng.uniform(0.0, 2.0 * np.pi)
            p = np.array([r * np.cos(th), r * np.sin(th)])
            if all(np.hypot(*(p - b[1])) > cfg.min_ball_bin_dist for b in balls):
                return p
        return np.array([0.7, 0.0])  # fallback, practically unreachable

    # ---------- Lifecycle ----------
    def reset(self, color_id: int | None = None, template_id: int | None = None):
        cfg = self.cfg
        self.balls = self._sample_balls()
        self.bin_pos = self._sample_bin(self.balls)
        cid = int(self.rng.integers(len(COLORS))) if color_id is None else int(color_id)
        tid = int(self.rng.integers(len(TEMPLATES))) if template_id is None else int(template_id)
        self.commanded = COLORS[cid]
        self.instruction = make_instruction(self.commanded, tid)
        self.tokens = self.tokenizer.encode_padded(self.instruction, cfg.lang_len)
        self.q = np.array([
            self.rng.uniform(*cfg.init_q1_range),
            self.rng.uniform(*cfg.init_q2_range),
        ])
        self.held = None
        self.grip_cmd = False
        self.ever_held_commanded = False
        self.ever_held_wrong = False
        self.min_ball_bin_dist = float("inf")
        self.t = 0
        return self._obs()

    def reset_scenario(self, sc: dict):
        """Replay a stored scenario (balls, bin, joints, instruction) instead of
        sampling: every run evaluates the exact same episodes."""
        self.balls = [(c, np.asarray(p, dtype=np.float64).copy())
                      for c, p in zip(COLORS, sc["balls"])]
        # OOD decoy objects (never interacted with by the task logic)
        self.distractor_balls = [
            (c, np.asarray(p, dtype=np.float64).copy())
            for c, p in zip(sc.get("distractor_colors", ()), sc.get("distractor_pos", []))]
        # OOD decoy objects with a per-object shape ("ball" / "square")
        self.ood_objects = [
            (str(c), np.asarray(p, dtype=np.float64).copy(), str(shp))
            for c, p, shp in zip(sc.get("distractor_colors", ()),
                                 sc.get("distractor_pos", ()),
                                 sc.get("distractor_shapes", ()))]
        self.bin_pos = np.asarray(sc["bin_pos"], dtype=np.float64).copy()
        self.commanded = COLORS[int(sc["color_id"])]
        self.instruction = str(sc["instruction"])
        self.tokens = np.asarray(sc["tokens"], dtype=np.int64).copy()
        self.q = np.asarray(sc["q0"], dtype=np.float64).copy()
        self.held = None
        self.grip_cmd = False
        self.ever_held_commanded = False
        self.ever_held_wrong = False
        self.min_ball_bin_dist = float("inf")
        self.t = 0
        return self._obs()

    def force_hold(self, color: str | None = None):
        """Debug helper: force-attach a ball to the EE (to construct custom start states)."""
        color = color or self.commanded
        for i, (c, _) in enumerate(self.balls):
            if c == color:
                self.balls[i] = (c, self.ee.copy())
        self.held = color
        self.grip_cmd = True
        if color == self.commanded:
            self.ever_held_commanded = True
        else:
            self.ever_held_wrong = True

    def _obs(self) -> dict:
        p1, p2 = self.fk()
        # the renderer never marks the commanded target: no observation leak
        img = self._obs_renderer.render(
            self.q, self.cfg.link_lengths, self.balls + self.distractor_balls,
            bin_pos=self.bin_pos, bin_r=self.cfg.bin_radius,
            grip_closed=self.grip_cmd, held=self.held,
            line_w=2, bg_style=getattr(self, "bg_style", 0),
            extra_objects=getattr(self, "ood_objects", None),
        )
        proprio = np.array([
            self.q[0] / np.pi,
            self.q[1] / np.pi,
            p2[0] / 1.1,
            p2[1] / 1.1,
            1.0 if self.grip_cmd else 0.0,
            1.0 if self.held is not None else 0.0,
        ], dtype=np.float32)
        return {
            "image": img,
            "tokens": self.tokens.copy(),
            "proprio": proprio,
            "instruction": self.instruction,
        }

    def step(self, action):
        """action: (dq1, dq2, grip). grip > 0 closes the gripper, otherwise opens it."""
        cfg = self.cfg
        a = np.asarray(action, dtype=np.float64).reshape(3)
        dq = np.clip(a[:2], -cfg.action_max, cfg.action_max)
        grip = bool(a[2] > 0.0)

        self.q = self.q + dq
        prev_grip = self.grip_cmd
        self.grip_cmd = grip

        # ---- gripper: open->close transition snaps the nearest ball ----
        if self.held is None:
            if grip:
                d = [(np.hypot(*(self.ee - pos)), i) for i, (_, pos) in enumerate(self.balls)]
                dist, idx = min(d)
                if dist < cfg.grasp_thresh:
                    color = self.balls[idx][0]
                    self.held = color
                    if color == self.commanded:
                        self.ever_held_commanded = True
                    else:
                        self.ever_held_wrong = True
        else:
            if not grip:
                self.held = None   # released; the ball stays where it is

        # ---- held ball follows the EE ----
        if self.held is not None:
            for i, (c, _) in enumerate(self.balls):
                if c == self.held:
                    self.balls[i] = (c, self.ee.copy())

        self.t += 1
        ball_bin = float(np.hypot(*(self.target_pos - self.bin_pos)))
        self.min_ball_bin_dist = min(self.min_ball_bin_dist, ball_bin)
        # placed: released, and the ball fully inside the bin rim
        overlap = circle_overlap_frac(ball_bin, cfg.ball_radius, cfg.bin_radius)
        success = bool(self.held is None and overlap >= cfg.place_overlap_frac)
        dist = float(np.hypot(*(self.ee - self.target_pos)))
        done = bool(success or self.t >= cfg.max_steps)
        # shaped reward (informational only; training uses BC/GRPO with its own reward)
        phase_target = self.bin_pos if self.held is not None else self.target_pos
        reward = (1.0 if success else 0.0) - 0.01 * float(np.hypot(*(self.ee - phase_target)))
        info = {
            "dist": dist,                       # EE to commanded-ball distance
            "bin_dist": float(np.hypot(*(self.ee - self.bin_pos))),
            "ball_bin_dist": ball_bin,          # commanded ball to bin center distance
            "success": success,
            "steps": self.t,
            "instruction": self.instruction,
            "commanded": self.commanded,
            "color_id": COLORS.index(self.commanded) if self.commanded else -1,
            "held": self.held,
            "holding": self.held is not None,
            "grip": self.grip_cmd,
            "phase": self.phase,
            "grasped_right": self.ever_held_commanded,
            "grasped_wrong": self.ever_held_wrong,
            "min_ball_bin_dist": self.min_ball_bin_dist,
            "ee": self.ee.copy(),
        }
        return self._obs(), reward, done, info
