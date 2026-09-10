"""Pure-NumPy scene renderer: 64x64 model observations / 256x256 visualization frames / GIF utils."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import BIN_FILL, BIN_RIM, COLOR_RGB

BG = (24, 27, 34)             # default background
GRID = (42, 46, 56)           # grid
# OOD distractor ball colors (never appear in training instructions)
DISTRACTOR_RGB = {
    "yellow": (235, 220, 60),
    "purple": (170, 90, 220),
    "orange": (240, 140, 40),
    "cyan": (60, 210, 220),
    "pink": (235, 110, 180),
    "brown": (150, 100, 60),
    "white": (235, 235, 235),
    "gray": (130, 135, 145),
}
# OOD background styles: (floor color, grid color) — randomized per scenario
OOD_BGS = [
    ((226, 218, 205), (196, 186, 170)),   # light beige floor, warm grid
    ((205, 220, 232), (170, 188, 205)),   # light blue floor, cool grid
    ((228, 210, 224), (200, 180, 196)),   # light pink floor
    ((215, 232, 214), (185, 202, 186)),   # light mint floor
]
ARM = (228, 231, 236)         # arm links
JOINT = (70, 74, 86)          # joint dots
EE_OPEN = (255, 234, 100)     # end effector (gripper open)
EE_CLOSED = (255, 150, 60)    # end effector (gripper closed)


class SceneRenderer:
    def __init__(self, res: int, world_extent: float):
        self.res = res
        self.ext = world_extent
        self.scale = res / (2.0 * world_extent)

    # ---------- Coordinates ----------
    def w2p(self, x: float, y: float):
        """World -> pixel coordinates (y axis flipped)."""
        return (x + self.ext) * self.scale, (self.ext - y) * self.scale

    # ---------- Primitives ----------
    def _fill_circle(self, canvas: np.ndarray, cx: float, cy: float, r: int, color):
        H, W = canvas.shape[:2]
        cx, cy = int(round(cx)), int(round(cy))
        x0, x1 = max(0, cx - r), min(W - 1, cx + r)
        y0, y1 = max(0, cy - r), min(H - 1, cy + r)
        if x1 < x0 or y1 < y0:
            return
        xs = np.arange(x0, x1 + 1)
        ys = np.arange(y0, y1 + 1)
        dx = xs[None, :] - cx
        dy = ys[:, None] - cy
        m = dx * dx + dy * dy <= r * r
        canvas[y0:y1 + 1, x0:x1 + 1][m] = color

    def _fill_square(self, canvas: np.ndarray, cx: float, cy: float, r: int, color):
        """Filled axis-aligned square, half-side r (same pixel bounding as _fill_circle)."""
        H, W = canvas.shape[:2]
        cx, cy = int(round(cx)), int(round(cy))
        x0, x1 = max(0, cx - r), min(W - 1, cx + r)
        y0, y1 = max(0, cy - r), min(H - 1, cy + r)
        if x1 < x0 or y1 < y0:
            return
        canvas[y0:y1 + 1, x0:x1 + 1] = color

    def _ring(self, canvas: np.ndarray, cx: float, cy: float, r: int, w: int, color):
        H, W = canvas.shape[:2]
        cx, cy = int(round(cx)), int(round(cy))
        x0, x1 = max(0, cx - r), min(W - 1, cx + r)
        y0, y1 = max(0, cy - r), min(H - 1, cy + r)
        if x1 < x0 or y1 < y0:
            return
        xs = np.arange(x0, x1 + 1)
        ys = np.arange(y0, y1 + 1)
        d2 = (xs[None, :] - cx) ** 2 + (ys[:, None] - cy) ** 2
        m = (d2 > (r - w) ** 2) & (d2 <= r * r)
        canvas[y0:y1 + 1, x0:x1 + 1][m] = color

    def _line(self, canvas: np.ndarray, x0, y0, x1, y1, w: int, color):
        """Thick line segment: sample along the segment and stamp disks (vectorized)."""
        H, W = canvas.shape[:2]
        n = int(max(abs(x1 - x0), abs(y1 - y0)) * 1.5) + 2
        t = np.linspace(0.0, 1.0, n)
        xs = np.round(x0 + (x1 - x0) * t).astype(np.int64)
        ys = np.round(y0 + (y1 - y0) * t).astype(np.int64)
        r = max(1, w // 2)
        ox, oy = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
        m = ox * ox + oy * oy <= r * r
        ox, oy = ox[m], oy[m]
        X = np.clip(xs[:, None] + ox[None, :], 0, W - 1)
        Y = np.clip(ys[:, None] + oy[None, :], 0, H - 1)
        canvas[Y.ravel(), X.ravel()] = color

    # ---------- Scene ----------
    def render(self, q, link_lengths, balls, bin_pos=None, bin_r: float = 0.20,
               grip_closed: bool = False, held: str | None = None,
               line_w: int = 2, bg_style: int = 0,
               extra_objects: list | None = None) -> np.ndarray:
        """Render one frame.

        balls: [(color, xy)] — COLOR_RGB or DISTRACTOR_RGB colors; bin_pos None
        skips the bin; held is drawn on top of the EE marker; bg_style >0 selects
        an OOD floor style; extra_objects are OOD decoys never touched by the
        task logic."""
        res = self.res
        floor, grid_col = OOD_BGS[(bg_style - 1) % len(OOD_BGS)] if bg_style > 0 else (BG, GRID)
        canvas = np.full((res, res, 3), floor, dtype=np.uint8)
        # Grid
        gx, _ = self.w2p(0.0, 0.0)
        step = 0.5 * self.scale
        for k in range(-2, 3):
            pos = int(round(gx + k * step))
            if 0 <= pos < res:
                canvas[:, pos] = grid_col
                canvas[pos, :] = grid_col
        # Bin (below the balls): dark body + orange rim
        if bin_pos is not None:
            px, py = self.w2p(bin_pos[0], bin_pos[1])
            r_px = int(round(bin_r * self.scale))
            rim_w = max(1, int(round(0.018 * self.scale)))
            self._fill_circle(canvas, px, py, max(2, r_px), BIN_FILL)
            self._ring(canvas, px, py, max(2, r_px), rim_w, BIN_RIM)
        # Balls (dark outline + fill; a held ball already follows the EE)
        ball_r = max(3, int(round(0.075 * self.scale)))
        def _rgb(color: str):
            return COLOR_RGB.get(color) or DISTRACTOR_RGB.get(color) or (200, 200, 200)
        for color, pos in balls:
            px, py = self.w2p(pos[0], pos[1])
            rgb = _rgb(color)
            dark = tuple(int(c * 0.55) for c in rgb)
            self._fill_circle(canvas, px, py, ball_r + 1, dark)
            self._fill_circle(canvas, px, py, ball_r, rgb)
        # OOD decoy objects (same size as a ball; shape = circle or square)
        for color, pos, shape in (extra_objects or []):
            px, py = self.w2p(pos[0], pos[1])
            rgb = _rgb(color)
            dark = tuple(int(c * 0.55) for c in rgb)
            if shape == "square":
                self._fill_square(canvas, px, py, ball_r + 1, dark)
                self._fill_square(canvas, px, py, ball_r, rgb)
            else:
                self._fill_circle(canvas, px, py, ball_r + 1, dark)
                self._fill_circle(canvas, px, py, ball_r, rgb)
        # Arm
        l1, l2 = link_lengths
        bx, by = self.w2p(0.0, 0.0)
        p1 = np.array([l1 * np.cos(q[0]), l1 * np.sin(q[0])])
        p2 = p1 + np.array([l2 * np.cos(q[0] + q[1]), l2 * np.sin(q[0] + q[1])])
        e1x, e1y = self.w2p(*p1)
        e2x, e2y = self.w2p(*p2)
        self._line(canvas, bx, by, e1x, e1y, line_w, ARM)
        self._line(canvas, e1x, e1y, e2x, e2y, line_w, ARM)
        jr = max(1, line_w - 1)
        self._fill_circle(canvas, bx, by, jr, JOINT)
        self._fill_circle(canvas, e1x, e1y, jr, JOINT)
        # EE: color encodes gripper open/closed
        self._fill_circle(canvas, e2x, e2y, max(2, line_w),
                          EE_CLOSED if grip_closed else EE_OPEN)
        # held ball on top of the EE marker (visible from above)
        if held is not None:
            rgb = _rgb(held)
            dark = tuple(int(c * 0.55) for c in rgb)
            self._fill_circle(canvas, e2x, e2y, ball_r + 1, dark)
            self._fill_circle(canvas, e2x, e2y, ball_r, rgb)
        return canvas

    def render_env(self, env, line_w: int = 2) -> np.ndarray:
        return self.render(env.q, env.cfg.link_lengths,
                           env.balls + getattr(env, "distractor_balls", []),
                           bin_pos=env.bin_pos, bin_r=env.cfg.bin_radius,
                           grip_closed=env.grip_cmd, held=env.held,
                           line_w=line_w, bg_style=getattr(env, "bg_style", 0),
                           extra_objects=getattr(env, "ood_objects", None))


# ---------- GIF / annotation utils ----------

def _default_font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def annotate(frame: np.ndarray, lines: list[str], font_size: int = 13) -> np.ndarray:
    """Add a black strip with white text at the bottom of the frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = _default_font(font_size)
    W, H = img.size
    # Estimate line height
    x0, y0, x1, y1 = draw.textbbox((0, 0), "Ag|", font=font)
    line_h = (y1 - y0) + 6
    strip_h = min(H - 2, 8 + len(lines) * line_h)
    draw.rectangle([0, H - strip_h, W, H], fill=(10, 11, 15))
    for i, text in enumerate(lines):
        y = H - strip_h + 4 + i * line_h
        draw.text((8, y), text, fill=(240, 242, 248), font=font)
    return np.array(img)


def save_gif(frames: list[np.ndarray], path: str, fps: int = 12) -> None:
    imgs = [Image.fromarray(f) for f in frames]
    if not imgs:
        return
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
