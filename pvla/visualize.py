"""Summary figures (matplotlib Agg): dataset grid / training curves / eval summary /
trajectory comparison. Called in-flight by the collect / train / evaluate stages."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import BIN_RIM, COLOR_RGB  # noqa: E402


def dataset_grid(data, out_path: str, n: int = 16, seed: int = 0) -> None:
    """Dataset sample grid: observation + instruction + expert action label (with grip)."""
    if isinstance(data, (str, Path)):
        data = dict(np.load(str(data), allow_pickle=False))
    N = len(data["images"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(n, N), replace=False)
    side = int(np.ceil(np.sqrt(len(idx))))
    fig, axes = plt.subplots(side, side, figsize=(2.6 * side, 2.9 * side))
    axes = np.atleast_2d(axes)
    for k, i in enumerate(idx):
        ax = axes[k // side][k % side]
        ax.imshow(data["images"][i])
        a = data["actions"][i]
        grip = "CLOSE" if a[2] > 0 else "open"
        ax.set_title(f'"{str(data["instruction"][i])}"\n'
                     f'a=({a[0]:+.2f},{a[1]:+.2f}) grip={grip}', fontsize=8)
        ax.axis("off")
    for k in range(len(idx), side * side):
        axes[k // side][k % side].axis("off")
    fig.suptitle(f"Dataset samples (N={N})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[viz] dataset grid -> {out_path}")


def loss_curve(log: list[dict], out_path: str) -> None:
    """Training curves: total loss, plus separate panels for the two aux readouts."""
    if isinstance(log, (str, Path)):
        log = json.loads(Path(log).read_text())
    epochs = [d["epoch"] for d in log]
    train = [d["train"] for d in log]
    val = [d["val"] for d in log]
    has_heads = bool(log) and "val_tgt_err" in log[0]
    has_train_heads = has_heads and "train_tgt_err" in log[0]

    if has_heads:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
        axs = list(axs)
    else:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        axs = [ax]

    ax = axs[0]
    ax.plot(epochs, train, lw=1.6, label="train")
    ax.plot(epochs, val, lw=1.6, label="val")
    best_i = int(np.argmin(val))
    ax.axvline(epochs[best_i], color="gray", ls=":", lw=1)
    ax.annotate(f"best val {val[best_i]:.4f}\n(epoch {epochs[best_i]})",
                xy=(epochs[best_i], val[best_i]), xytext=(10, 30),
                textcoords="offset points", fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_title("Total loss")
    ax.set_ylabel("MSE loss")

    if has_heads:
        for ax, title, vkey, tkey in (
            (axs[1], "Ball readout (aux, fused LM states)", "val_tgt_err", "train_tgt_err"),
            (axs[2], "Bin readout (aux, fused LM states)", "val_bin_err", "train_bin_err"),
        ):
            if has_train_heads:
                ax.plot(epochs, [d[tkey] for d in log], lw=1.4, label="train")
            ax.plot(epochs, [d[vkey] for d in log], lw=1.6, label="val")
            ax.set_title(title)
            ax.set_ylabel("offset MSE")

    for ax in axs:
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[viz] training curve -> {out_path}")


def grpo_curve(log: list[dict], out_path: str) -> None:
    """GRPO training curves: rewards (ID vs OOD) + success rate per iteration."""
    if isinstance(log, (str, Path)):
        log = json.loads(Path(log).read_text())
    its = [d["iteration"] for d in log]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axs[0]
    ax.plot(its, [d["mean_reward"] for d in log], lw=1.8, color="crimson")
    ax.set_title("GRPO mean group reward")
    ax.set_ylabel("reward")
    ax = axs[1]
    ax.plot(its, [d["group_success"] for d in log], lw=1.6, color="tab:blue")
    test = [(d["iteration"], d["test_success_rate"]) for d in log
            if "test_success_rate" in d]
    if test:
        ax.plot([i for i, _ in test], [s for _, s in test], "o--", lw=1.4,
                color="tab:green", label="test set (eval_every)")
        ax.legend(fontsize=8)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("GRPO success rate")
    ax.set_ylabel("success rate")
    for ax in axs:
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[viz] grpo curve -> {out_path}")


def eval_summary(report: dict, out_path: str) -> None:
    """Per-color [success + correct-grasp] grouped bars + final ball-bin distance histogram."""
    per_color = report["per_color"]
    colors = list(per_color.keys())
    rates = [per_color[c]["success_rate"] for c in colors]
    grasps = [per_color[c]["grasp_rate"] for c in colors]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    x = np.arange(len(colors) + 1)
    all_rates = rates + [report["success_rate"]]
    all_grasps = grasps + [report["grasp_rate"]]
    b1 = ax.bar(x - 0.19, all_grasps, width=0.38,
                color=[tuple(v / 255 for v in COLOR_RGB[c]) for c in colors] + [(0.6, 0.6, 0.6)],
                label="grasp right ball")
    b2 = ax.bar(x + 0.19, all_rates, width=0.38,
                color=[tuple(v / 255 for v in COLOR_RGB[c]) for c in colors] + [(0.6, 0.6, 0.6)],
                alpha=0.45, label="placed in bin (success)")
    for b, v in list(zip(b1, all_grasps)) + list(zip(b2, all_rates)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}",
                ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(colors + ["overall"])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("rate")
    ax.set_title(f"Grasp / place by commanded color (N={report['episodes']})")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    dists = report["final_dists"]
    ax.hist(dists, bins=30, color=(0.4, 0.6, 0.9))
    v = report.get("place_dist_thresh")
    if v is not None:
        ax.axvline(v, color="crimson", ls="--",
                   label=f"place dist thresh={v:.3f}")
    ax.set_xlabel("final ball-to-bin distance")
    ax.set_title("Final ball-to-bin distance distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[viz] eval summary -> {out_path}")


def traj_compare(report: dict, out_path: str, world_extent: float = 1.25,
                 link_lengths=(0.55, 0.55)) -> None:
    """Expert vs VLA EE trajectories + commanded-ball path + bin (world coordinates)."""
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    # Bin
    bin_pos = np.array(report["bin_pos"])
    bin_rgb = tuple(v / 255 for v in BIN_RIM)
    circ = plt.Circle(bin_pos, report.get("bin_radius", 0.14),
                      facecolor=(0.16, 0.14, 0.12), edgecolor=bin_rgb, lw=2.2)
    ax.add_patch(circ)
    ax.text(bin_pos[0], bin_pos[1] + report.get("bin_radius", 0.14) + 0.06,
            "bin", ha="center", fontsize=9, color=bin_rgb)
    # Balls (initial positions)
    for color, pos in report["balls"]:
        rgb = tuple(v / 255 for v in COLOR_RGB[color])
        circ = plt.Circle((pos[0], pos[1]), 0.075, color=rgb, alpha=0.45)
        ax.add_patch(circ)
        ax.text(pos[0], pos[1] + 0.13, color, ha="center", fontsize=9)
        if color == report["commanded"]:
            ring = plt.Circle((pos[0], pos[1]), 0.11, fill=False,
                              color="k", ls="--", lw=1.2)
            ax.add_patch(ring)
    # Initial arm configuration
    q0 = np.array(report["q0"])
    l1, l2 = link_lengths
    p1 = np.array([l1 * np.cos(q0[0]), l1 * np.sin(q0[0])])
    p2 = p1 + np.array([l2 * np.cos(q0[0] + q0[1]), l2 * np.sin(q0[0] + q0[1])])
    ax.plot([0, p1[0], p2[0]], [0, p1[1], p2[1]], color="0.75", lw=2,
            marker="o", ms=4, label="arm @ start")
    # Trajectories
    ee_vla = np.array(report["vla_ee_path"])
    ee_exp = np.array(report["expert_ee_path"])
    ball_rgb = tuple(v / 255 for v in COLOR_RGB[report["commanded"]])
    if "vla_ball_path" in report:
        ball_vla = np.array(report["vla_ball_path"])
        ax.plot(ball_vla[:, 0], ball_vla[:, 1], "-", color=ball_rgb, lw=2.4,
                alpha=0.85, label=f"{report['commanded']} ball path")
        ax.plot(*ball_vla[-1], marker="o", ms=8, color=ball_rgb)
    ax.plot(ee_exp[:, 0], ee_exp[:, 1], "--", color="0.5", lw=1.5,
            label=f"expert EE ({report['expert_steps']} steps)")
    ax.plot(ee_vla[:, 0], ee_vla[:, 1], "-", color="crimson", lw=1.6,
            label=f"VLA EE ({report['vla_steps']} steps)")
    ax.plot(*ee_vla[-1], marker="*", ms=12, color="crimson")
    ax.set_xlim(-world_extent, world_extent)
    ax.set_ylim(-world_extent, world_extent)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(f'Pick&place: "{report["instruction"]}"  '
                 f'VLA {"SUCCESS" if report["vla_success"] else "FAIL"}')
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[viz] trajectory comparison -> {out_path}")
