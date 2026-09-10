"""Closed-loop evaluation on the fixed test set: one sweep produces all
artifacts under eval.output_dir — case/ep{iii}.gif replay GIFs, eval_report.json
(stats + per-episode results + expert comparison), eval_summary.png and
traj_compare.png (zero-noise expert vs VLA trajectories on scenario 0).
"""
from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .config import Config, device_banner, ensure_dir, get_device
from .data.collect import scenario_at
from .expert.analytic import AnalyticExpert
from .lang.tokenizer import COLORS
from .model.vla import PocketVLA
from .sim.arm2d import Arm2DPickPlace, overlap_dist_thresh
from .sim.render import SceneRenderer, annotate, save_gif
from . import visualize


# ---------------- closed-loop machinery ----------------

def obs_to_tensors(obs: dict, device: torch.device):
    img = torch.from_numpy(obs["image"]).permute(2, 0, 1).unsqueeze(0).to(device)
    tok = torch.from_numpy(obs["tokens"]).unsqueeze(0).to(device)
    pad = tok != 0
    prop = torch.from_numpy(obs["proprio"]).unsqueeze(0).to(device)
    return img, tok, pad, prop


@torch.no_grad()
def policy_action(model: PocketVLA, obs: dict, device: torch.device,
                  use_amp: bool = True) -> np.ndarray:
    """Policy forward: obs dict -> action (3,) = (dq1, dq2, grip)."""
    img, tok, pad, prop = obs_to_tensors(obs, device)
    if use_amp and device.type == "cuda":
        ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        ctx = nullcontext()
    with ctx:
        a, _, _ = model(img, tok, pad, prop)
    return a[0].float().cpu().numpy()


def run_episode(env: Arm2DPickPlace, model: PocketVLA, device: torch.device,
                pretty: SceneRenderer, want_frames: bool,
                scenario: dict | None = None):
    """Closed-loop rollout of one episode (scenario != None replays a fixed scenario).
    Returns (frames|None, EE path, ball path, q0, info)."""
    obs = env.reset_scenario(scenario) if scenario is not None else env.reset()
    env.bg_style = int(scenario.get("bg_style", 0)) if scenario is not None else 0
    q0 = env.q.copy()
    ee_path = [env.ee.copy()]
    ball_path = [env.target_pos.copy()]
    frames: list[np.ndarray] | None = [] if want_frames else None
    done = False
    while not done:
        a = policy_action(model, obs, device)
        obs, _, done, info = env.step(a)
        ee_path.append(env.ee.copy())
        ball_path.append(env.target_pos.copy())
        if frames is not None:
            frame = pretty.render_env(env, line_w=4)
            held = f"holding {env.held}" if env.held else "empty hand"
            frame = annotate(frame, [
                obs["instruction"],
                f"step {env.t}/{env.cfg.max_steps}  {env.phase}  {held}  "
                f"ball={info['dist']:.2f}  bin={info['bin_dist']:.2f}"
                + ("  PLACED!" if info["success"] else ""),
            ])
            frames.append(frame)
    if frames is not None:
        # Append a final frame annotating the outcome
        if info["success"]:
            outcome = "RESULT: SUCCESS (ball placed in the bin)"
        elif info["grasped_wrong"] and not info["grasped_right"]:
            outcome = "RESULT: FAIL (grasped wrong ball)"
        elif info["grasped_right"]:
            outcome = "RESULT: FAIL (not placed in the bin)"
        else:
            outcome = "RESULT: FAIL (never grasped right ball)"
        frame = pretty.render_env(env, line_w=4)
        frame = annotate(frame, [
            obs["instruction"], outcome,
            f"steps={env.t}  final ball-bin dist={info['ball_bin_dist']:.3f}",
        ])
        frames.append(frame)
    return frames, np.array(ee_path), np.array(ball_path), q0, info


def _fail_reason(info: dict) -> str:
    if info["success"]:
        return ""
    if not info["grasped_right"] and not info["grasped_wrong"]:
        return "never_grasped"
    if not info["grasped_right"]:
        return "grasped_wrong"
    return "grasped_but_not_placed"


def _resolve_ckpt(cfg: Config, ckpt_path: str | None) -> tuple[str, str]:
    """Resolve the checkpoint: CLI --ckpt wins over eval.ckpt (yaml); no implicit fallback."""
    resolved = ckpt_path or cfg.eval_ckpt
    if not resolved:
        raise ValueError(
            "[eval] checkpoint not specified: set eval.ckpt in the yaml or pass "
            "--ckpt (e.g. --ckpt outputs_sft_3m/train/pocketvla-3m, or the GRPO "
            "model outputs_grpo_3m/pocketvla-3m-grpo)")
    if not Path(resolved).is_dir():
        raise FileNotFoundError(f"[eval] checkpoint directory not found: {resolved}")
    source = "via --ckpt" if ckpt_path else "via eval.ckpt (yaml)"
    return resolved, source


# ---------------- single-sweep evaluation ----------------

def evaluate(cfg: Config, ckpt_path: str | None = None,
             n_episodes: int | None = None, save_gifs: bool = True,
             n_gifs: int | None = None, gif_seed: int = 0,
             test_npz: str | None = None) -> dict:
    """Evaluate the fixed test set once: stats, replay GIFs and summary figures
    in one sweep, all under eval.output_dir. test_npz overrides the test-set file."""
    device = get_device()
    print(f"[eval] {device_banner(device)}")
    test_file = test_npz or cfg.test_npz
    if not Path(test_file).is_file():
        raise FileNotFoundError(
            f"[eval] test set not found: {test_file}\n"
            f"[eval]   run 'python3 -m pvla collect --config <yaml>' first")
    path, how = _resolve_ckpt(cfg, ckpt_path)
    print(f"[eval] checkpoint: {path}  ({how})")
    model, _ = PocketVLA.load_hf(path, device)
    size = model.config.size          # informational: the loaded checkpoint's size
    use_aux = getattr(model.config, "use_aux_heads", True)
    print(f"[eval] model loaded: size={size}  params={model.n_params() / 1e6:.2f}M  "
          f"aux_heads={'on' if use_aux else 'off'}")
    if cfg.eval_dir is None:
        # lightweight in-training mode: stats only, no artifacts on disk
        eval_root = None
        case_dir = None
    else:
        eval_root = Path(cfg.eval_dir)
        case_dir = eval_root / "case"          # one replay GIF per test scenario
        if save_gifs:
            ensure_dir(case_dir)
    print(f"[eval] artifacts -> {cfg.eval_dir}")

    n = cfg.eval_episodes if n_episodes is None else n_episodes
    scen = dict(np.load(test_file, allow_pickle=True))   # object arrays for OOD distractors
    if len(scen["q0"]) < n:
        print(f"[eval] test set holds {len(scen['q0'])} scenarios (< {n} requested); "
              f"using all of them")
    n = min(n, len(scen["q0"]))
    print(f"[eval] fixed test set: {test_file}  ({n} scenarios)")
    env = Arm2DPickPlace(cfg, seed=cfg.seed)   # rng unused: scenarios are replayed
    pretty = SceneRenderer(cfg.frame_res, cfg.world_extent)
    expert = AnalyticExpert(cfg, seed=0, noise=0.0)

    per_color = {c: {"success": 0, "n": 0, "grasp": 0, "steps": [],
                     "ball_bin": []} for c in COLORS}
    final_dists, succ_steps, results = [], [], []
    fails = {"never_grasped": 0, "grasped_wrong": 0, "grasped_but_not_placed": 0}
    comparison = None

    # random subset of scenarios that get a replay GIF (None = every scenario)
    if save_gifs and n_gifs is not None and n_gifs < n:
        gif_eps = set(np.random.default_rng(gif_seed).choice(n, size=n_gifs,
                                                             replace=False).tolist())
        print(f"[eval] saving {n_gifs} random replay GIFs (of {n} scenarios)")
    else:
        gif_eps = None if save_gifs else set()

    for ep in range(n):
        sc = scenario_at(scen, ep)
        want_gif = save_gifs and (gif_eps is None or ep in gif_eps)
        frames, ee_path, ball_path, q0, info = run_episode(
            env, model, device, pretty, want_frames=want_gif, scenario=sc)

        c = info["commanded"]
        per_color[c]["n"] += 1
        per_color[c]["success"] += int(info["success"])
        per_color[c]["grasp"] += int(info["grasped_right"])
        per_color[c]["ball_bin"].append(info["ball_bin_dist"])
        final_dists.append(info["ball_bin_dist"])
        if info["success"]:
            per_color[c]["steps"].append(info["steps"])
            succ_steps.append(info["steps"])
        else:
            reason = _fail_reason(info)
            fails[reason] += 1
        results.append({
            "episode": ep, "instruction": info["instruction"], "commanded": c,
            "steps": info["steps"], "success": info["success"],
            "final_ball_bin_dist": info["ball_bin_dist"],
        })

        if want_gif and case_dir is not None:
            save_gif(frames, str(case_dir / f"ep{ep:03d}.gif"), fps=cfg.gif_fps)
        if ep == 0:
            # zero-noise expert reference on the same fixed scenario
            exp = expert.rollout(env, scenario=sc)
            comparison = {
                "instruction": info["instruction"], "commanded": c,
                "balls": [[col, sc["balls"][i].tolist()] for i, col in enumerate(COLORS)],
                "bin_pos": sc["bin_pos"].tolist(), "bin_radius": cfg.bin_radius,
                "q0": sc["q0"].tolist(),
                "vla_ee_path": ee_path.tolist(), "vla_ball_path": ball_path.tolist(),
                "expert_ee_path": exp["ee_path"].tolist(),
                "vla_steps": info["steps"], "expert_steps": exp["steps"],
                "vla_success": info["success"], "expert_success": exp["success"],
            }
            visualize.traj_compare(comparison, str(eval_root / "traj_compare.png")) if eval_root else None
        if (ep + 1) % 25 == 0:
            print(f"  episode {ep + 1}/{n}")

    n_success = sum(v["success"] for v in per_color.values())
    n_grasp = sum(v["grasp"] for v in per_color.values())
    report = {
        "episodes": n,
        "test_set": test_file,
        "model_size": size,
        "use_aux_heads": bool(getattr(model.config, "use_aux_heads", True)),
        "success_rate": n_success / n,
        "grasp_rate": n_grasp / n,
        "place_overlap_frac": cfg.place_overlap_frac,
        "place_dist_thresh": overlap_dist_thresh(cfg.place_overlap_frac,
                                                 cfg.ball_radius, cfg.bin_radius),
        "mean_final_ball_bin_dist": float(np.mean(final_dists)),
        "mean_steps_if_success": float(np.mean(succ_steps)) if succ_steps else None,
        "final_dists": [float(d) for d in final_dists],
        "failure_taxonomy": fails,
        "per_color": {
            c: {
                "success_rate": v["success"] / v["n"] if v["n"] else 0.0,
                "grasp_rate": v["grasp"] / v["n"] if v["n"] else 0.0,
                "n": v["n"],
                "mean_final_ball_bin_dist": float(np.mean(v["ball_bin"])) if v["ball_bin"] else None,
                "mean_steps": float(np.mean(v["steps"])) if v["steps"] else None,
            }
            for c, v in per_color.items()
        },
        "results": results,
        "comparison": comparison,
        "ckpt": str(path),
    }
    if eval_root is not None:
        out = eval_root / "eval_report.json"
        out.write_text(json.dumps(report, indent=1))
        visualize.eval_summary(report, str(eval_root / "eval_summary.png"))
        print(f"[eval] report -> {out}")

    print(f"[eval] place success rate: {report['success_rate']:.1%}  "
          f"grasp right ball: {report['grasp_rate']:.1%}  "
          f"mean ball-bin dist: {report['mean_final_ball_bin_dist']:.4f}")
    for c in COLORS:
        v = report["per_color"][c]
        dist = f"{v['mean_final_ball_bin_dist']:.4f}" if v["n"] else "n/a"
        print(f"  {c:5s}: success {v['success_rate']:6.1%}  "
              f"grasp right {v['grasp_rate']:6.1%}  "
              f"ball-bin dist {dist}  (N={v['n']})")
    nf = n - n_success
    if nf > 0:
        print(f"[eval] failure attribution ({nf} episodes): "
              f"never grasped={fails['never_grasped']}  "
              f"wrong ball={fails['grasped_wrong']}  "
              f"not placed={fails['grasped_but_not_placed']}")
    if save_gifs and case_dir is not None:
        n_gifs_saved = n if gif_eps is None else len(gif_eps)
        print(f"[eval] artifacts: {n_gifs_saved} scenario GIFs -> {case_dir}  report -> {out}")
    elif eval_root is None:
        print(f"[eval] lightweight stats-only evaluation (no artifacts)")
    else:
        print(f"[eval] report -> {out}  (GIFs disabled)")
    return report
