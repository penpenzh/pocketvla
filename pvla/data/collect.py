"""Dataset generation: expert demos -> dataset/{train, val, test, grpo, ood_test}.npz

train/val are expert-demo samples (split by episode, frozen by seed); test/grpo/
ood_test are stored initial scenarios from disjoint seed spaces (seed+20,000+i /
seed+40,000+i / seed+80,000+i) — no overlap with demos, fully reproducible.
Runs standalone: `python3 -m pvla collect` (built-in defaults + CLI overrides;
an optional --config is honored).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..config import Config, ensure_dir
from ..expert.analytic import AnalyticExpert
from ..lang.tokenizer import COLORS
from ..sim.arm2d import Arm2DPickPlace
from ..sim.render import SceneRenderer, annotate, save_gif
from .. import visualize

# env seed offsets for scenario sets (disjoint from demos and each other)
TEST_SEED_OFFSET = 20_000
GRPO_SEED_OFFSET = 40_000

# fields of one scenario record (arrays are stacked over episodes)
SCEN_KEYS = ("balls", "bin_pos", "q0", "color_id", "tokens", "instruction")


def scenario_at(scen: dict, i: int) -> dict:
    """Row i of a loaded scenario set, ready for env.reset_scenario()."""
    return {k: scen[k][i] for k in scen}


def _gen_scenarios(cfg: Config, n: int, seed_offset: int = TEST_SEED_OFFSET) -> dict:
    """Generate n fixed scenarios (scenario i uses env seed cfg.seed + offset + i)."""
    out: dict[str, list] = {k: [] for k in SCEN_KEYS}
    for ep in range(n):
        env = Arm2DPickPlace(cfg, seed=cfg.seed + seed_offset + ep)
        env.reset()
        out["balls"].append(np.array([p for _, p in env.balls], dtype=np.float64))
        out["bin_pos"].append(env.bin_pos.astype(np.float64))
        out["q0"].append(env.q.astype(np.float64))
        out["color_id"].append(np.int64(COLORS.index(env.commanded)))
        out["tokens"].append(env.tokens.copy())
        out["instruction"].append(env.instruction)
    return {
        "balls": np.stack(out["balls"]),                    # (n,3,2) red/green/blue order
        "bin_pos": np.stack(out["bin_pos"]),                # (n,2)
        "q0": np.stack(out["q0"]),                          # (n,2)
        "color_id": np.array(out["color_id"], dtype=np.int64),
        "tokens": np.stack(out["tokens"]),                  # (n,lang_len)
        "instruction": np.array(out["instruction"]),         # (n,) unicode
    }


def collect(cfg: Config) -> dict:
    """Dataset generation: expert demos -> train / val / test sets on disk."""
    ds_dir = ensure_dir(cfg.dataset_dir)
    demos_dir = ensure_dir(Path(cfg.dataset_dir) / "demos")
    print(f"[collect] dataset dir: {cfg.dataset_dir}")
    print(f"[collect] collecting {cfg.episodes_collect} expert demo episodes (pick & place) ...")
    env = Arm2DPickPlace(cfg, seed=cfg.seed)
    expert = AnalyticExpert(cfg, seed=cfg.seed + 1)
    pretty = SceneRenderer(cfg.frame_res, cfg.world_extent)
    gif_eps = {0, cfg.episodes_collect // 2} if cfg.episodes_collect >= 2 else {0}

    images, tokens, proprios, actions = [], [], [], []
    targets, bins, color_ids, instructions, ep_ids = [], [], [], [], []
    ep_success, ep_steps = 0, []

    t0 = time.time()
    for ep in range(cfg.episodes_collect):
        obs = env.reset()
        frames: list[np.ndarray] | None = [] if ep in gif_eps else None
        done = False
        while not done:
            a = expert.act(env)
            images.append(obs["image"])
            tokens.append(obs["tokens"])
            proprios.append(obs["proprio"])
            # joint deltas normalized to [-1, 1]; grip stays +/-1
            actions.append(np.array([a[0] / cfg.action_max,
                                     a[1] / cfg.action_max, a[2]]))
            # aux labels: ball/bin offsets from the EE
            targets.append((env.target_pos - env.ee) / cfg.world_extent)
            bins.append((env.bin_pos - env.ee) / cfg.world_extent)
            color_ids.append(COLORS.index(env.commanded))
            instructions.append(env.instruction)
            ep_ids.append(ep)
            obs, _, done, info = env.step(a)
            if frames is not None:
                frame = pretty.render_env(env, line_w=4)
                held = f"holding {env.held}" if env.held else "empty hand"
                frame = annotate(frame, [
                    env.instruction,
                    f"step {env.t}/{cfg.max_steps}  {env.phase}  {held}  "
                    f"ball={info['dist']:.2f}  bin={info['bin_dist']:.2f}"
                    + ("  PLACED!" if info["success"] else ""),
                ])
                frames.append(frame)
        ep_success += int(info["success"])
        ep_steps.append(info["steps"])
        if frames is not None:
            save_gif(frames, str(demos_dir / f"expert_ep{ep}.gif"), fps=cfg.gif_fps)
        if (ep + 1) % 100 == 0:
            print(f"  episode {ep + 1}/{cfg.episodes_collect}  "
                  f"({time.time() - t0:.1f}s)")

    data = {
        "images": np.stack(images).astype(np.uint8),
        "tokens": np.stack(tokens).astype(np.int64),
        "proprio": np.stack(proprios).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32),     # (N,3) normalized
        "target_xy": np.stack(targets).astype(np.float32),   # (N,2) ball offset / extent
        "bin_xy": np.stack(bins).astype(np.float32),         # (N,2) bin offset / extent
        "color_id": np.array(color_ids, dtype=np.int64),
        "instruction": np.array(instructions),
        "ep_id": np.array(ep_ids, dtype=np.int32),
    }
    n_samples = len(data["images"])
    del images, tokens, proprios, actions, targets, bins, color_ids, instructions  # free memory

    # ---- episode-level train/val split (frozen by seed; no adjacent-frame leakage) ----
    eps = np.unique(data["ep_id"])
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(eps)
    n_val_ep = max(1, int(len(perm) * cfg.val_frac))
    val_eps = np.sort(perm[:n_val_ep])
    val_mask = np.isin(data["ep_id"], val_eps)
    train_idx, val_idx = np.where(~val_mask)[0], np.where(val_mask)[0]
    for name, idx, n_eps in (("train", train_idx, len(perm) - n_val_ep),
                             ("val", val_idx, n_val_ep)):
        out_path = ds_dir / f"{name}.npz"
        np.savez_compressed(out_path, **{k: v[idx] for k, v in data.items()})
        print(f"[collect] {name} set -> {out_path}  ({len(idx)} samples / {n_eps} episodes)")

    # ---- fixed closed-loop test set ----
    scen = _gen_scenarios(cfg, cfg.eval_episodes)
    np.savez_compressed(ds_dir / "test.npz", **scen)
    counts = {c: int(np.sum(scen["color_id"] == i)) for i, c in enumerate(COLORS)}

    # ---- GRPO RL scenarios (ID, seed+40,000+i) ----
    rl_scen = _gen_scenarios(cfg, cfg.grpo_episodes, seed_offset=GRPO_SEED_OFFSET)
    np.savez_compressed(ds_dir / "grpo.npz", **rl_scen)
    print(f"[collect] grpo RL scenarios -> {ds_dir / 'grpo.npz'}  "
          f"({len(rl_scen['q0'])} scenarios, seed space seed+{GRPO_SEED_OFFSET}+i)")

    # ---- OOD test set (seed+80,000+i): one decoy object per scene ----
    from ..scenarios import OOD_TEST_SEED_OFFSET, gen_ood_scenarios
    ood_test = gen_ood_scenarios(cfg, cfg.ood_test_episodes,
                                 seed_offset=OOD_TEST_SEED_OFFSET, seed=cfg.seed)
    np.savez_compressed(ds_dir / "ood_test.npz", **ood_test)
    print(f"[collect] OOD test scenarios -> {ds_dir / 'ood_test.npz'}  "
          f"({len(ood_test['q0'])} scenarios, seed space seed+{OOD_TEST_SEED_OFFSET}+i)")

    visualize.dataset_grid(data, str(ds_dir / "dataset_grid.png"), n=16)

    stats = {
        "episodes": cfg.episodes_collect,
        "samples": n_samples,
        "expert_success_rate": ep_success / cfg.episodes_collect,
        "mean_steps": float(np.mean(ep_steps)),
        "max_steps_observed": int(np.max(ep_steps)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "train_episodes": int(len(perm) - n_val_ep),
        "val_episodes": int(n_val_ep),
        "test_scenarios": int(len(scen["q0"])),
        "seconds": time.time() - t0,
    }
    print(f"[collect] expert success rate={stats['expert_success_rate']:.1%}  "
          f"mean steps={stats['mean_steps']:.1f}  longest={stats['max_steps_observed']}")
    print(f"[collect] train/val: {stats['train_samples']} / {stats['val_samples']} samples "
          f"({stats['train_episodes']} / {stats['val_episodes']} episodes, split by episode)")
    print(f"[collect] test: {stats['test_scenarios']} fixed closed-loop scenarios  "
          f"colors {counts}")
    print(f"[collect] dataset ready in '{cfg.dataset_dir}'  elapsed={stats['seconds']:.1f}s")
    return stats
