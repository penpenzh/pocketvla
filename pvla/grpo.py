"""GRPO reinforcement learning: group rollouts + advantage-weighted regression
(AWR/RAFT form, no value network), fully decoupled from SFT.

Each iteration samples G group members on the SAME initial scenario and
group-normalizes rewards into advantages; only above-average members' de-noised
mean actions become regression targets. A KL anchor to the frozen SFT policy
bounds drift; training stops early when mean group reward plateaus.

Artifacts (grpo.output_dir): grpo_log.json, grpo_curve.png,
pocketvla-<size>-grpo/ (best mean reward), eval/ (final full evaluation).
"""
from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .config import Config, device_banner, ensure_dir, get_device
from .model.vla import PocketVLA, PocketVLATokenizer
from .sim.arm2d import Arm2DPickPlace
from .evaluate import evaluate as run_eval
from . import visualize


def _episode_reward(info: dict, cfg: Config) -> float:
    """Scalarize one rollout: success dominates, speed/grasp/distance break ties."""
    r = 0.0
    if info["success"]:
        r += 1.0
        r += 0.1 * (1.0 - min(1.0, info["steps"] / cfg.max_steps))   # faster is better
    if info["grasped_right"]:
        r += 0.2                                                     # at least grasped the right ball
    r -= 0.1 * min(1.0, info["ball_bin_dist"] / cfg.bin_radius)      # closer to the bin is better
    if info["grasped_wrong"]:
        r -= 0.3
    return r


def _rollout_group(env: Arm2DPickPlace, model: PocketVLA, device: torch.device,
                   cfg: Config, use_amp: bool, group_size: int,
                   scenario: dict, seed: int) -> tuple[list[dict], list[list[np.ndarray]]]:
    """Play group_size episodes on the same scenario (exploration noise for diversity).
    Returns (episode summaries, noisy trajectories, de-noised mean trajectories)."""
    episodes = []
    torch_rng = torch.Generator()
    torch_rng.manual_seed(seed)
    trajectories: list[list[np.ndarray]] = []
    denoised: list[list[np.ndarray]] = []
    for _ in range(group_size):
        obs = env.reset_scenario(scenario)
        actions: list[np.ndarray] = []
        means: list[np.ndarray] = []
        done, steps = False, 0
        while not done:
            img = torch.from_numpy(obs["image"]).permute(2, 0, 1).unsqueeze(0).to(device)
            tok = torch.from_numpy(obs["tokens"]).unsqueeze(0).to(device)
            pad = tok != 0
            prop = torch.from_numpy(obs["proprio"]).unsqueeze(0).to(device)
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
            with torch.no_grad(), ctx:
                a, _, _ = model(img, tok, pad, prop)
                # exploration noise on joint dims only; grip stays discrete
                noise = (torch.randn(2, generator=torch_rng) * 0.03).to(a.device)
                a_j = torch.clamp(a[0, :2].float() + noise, -1.0, 1.0)
                action = torch.cat([a_j, a[0, 2:3].float()]).cpu().numpy()
                mean = torch.cat([a[0, :2].float(), a[0, 2:3].float()]).cpu().numpy()
            actions.append(action.copy())
            means.append(mean.copy())
            obs, _, done, info = env.step(action)
            steps += 1
        trajectories.append(actions)
        denoised.append(means)
        episodes.append({"reward": _episode_reward(info, cfg),
                         "success": info["success"], "steps": steps,
                         "ball_bin": info["ball_bin_dist"],
                         "grasped_right": info["grasped_right"]})
    return episodes, trajectories, denoised


def run_grpo(cfg: Config, ckpt_path: str | None = None) -> dict:
    """GRPO fine-tuning entry point: load the SFT checkpoint, train on grpo.npz,
    save the best-by-mean-reward checkpoint and run a final full evaluation."""
    device = get_device()
    torch.manual_seed(cfg.seed)
    print(f"[grpo] {device_banner(device)}")

    if not Path(cfg.grpo_npz).is_file():
        raise FileNotFoundError(
            f"[grpo] RL scenario file not found: {cfg.grpo_npz}\n"
            f"[grpo]   run 'python3 -m pvla collect' first (it writes grpo.npz "
            f"into data.dataset_dir='{cfg.dataset_dir}')")
    # SFT checkpoint: --ckpt (CLI) wins over grpo.ckpt (yaml)
    sft_ckpt = ckpt_path or cfg.grpo_ckpt
    if not sft_ckpt:
        raise ValueError(
            "[grpo] SFT checkpoint not specified: set grpo.ckpt in the yaml "
            "or pass --ckpt (e.g. --ckpt outputs_sft_3m/train/pocketvla-3m)")
    # Test set for the GRPO evals: grpo.test_set overrides data.dataset_dir/test.npz
    test_set = cfg.grpo_test_set or cfg.test_npz
    if not Path(sft_ckpt).is_dir():
        raise FileNotFoundError(
            f"[grpo] SFT checkpoint not found: {sft_ckpt}\n"
            f"[grpo]   run 'python3 -m pvla train --config <yaml>' first, "
            f"or pass --ckpt <sft_checkpoint_dir>")

    if cfg.grpo_train_dir is None:
        raise ValueError("[grpo] yaml must set grpo.output_dir (GRPO artifacts directory)")
    ensure_dir(cfg.grpo_train_dir)
    model, _ = PocketVLA.load_hf(sft_ckpt, device)
    print(f"[grpo] SFT checkpoint: {sft_ckpt}")
    print(f"[grpo] model params: {model.n_params() / 1e6:.2f}M  "
          f"aux_heads={'on' if getattr(model.config, 'use_aux_heads', True) else 'off'}")
    print(f"[grpo] RL scenarios: {cfg.grpo_npz}  iterations={cfg.grpo_iterations}  "
          f"group_size={cfg.grpo_group_size}  lr={cfg.grpo_lr:.1e}")

    # ---- dataset: ID scenarios only (grpo.npz, its own seed space) ----
    scen = dict(np.load(cfg.grpo_npz, allow_pickle=True))
    n_id = len(scen["q0"])
    print(f"[grpo] RL scenarios: {cfg.grpo_npz} ({n_id})")

    device = next(model.parameters()).device
    use_amp = device.type == "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.grpo_lr,
                            weight_decay=cfg.weight_decay)
    # frozen SFT reference for the KL-style anchor (and aux-consistency)
    sft_model, _ = PocketVLA.load_hf(sft_ckpt, device)
    for p in sft_model.parameters():
        p.requires_grad_(False)
    sft_model.eval()
    env = Arm2DPickPlace(cfg, seed=cfg.seed)   # rng unused: scenarios are replayed
    G = cfg.grpo_group_size

    log = []
    best_mean_r, no_improve, t0 = -float("inf"), 0, time.time()
    rng = np.random.default_rng(cfg.seed)
    id_order = rng.permutation(n_id)
    id_pos = 0

    for it in range(1, cfg.grpo_iterations + 1):
        infos = []
        id_reward, id_succ = 0.0, 0
        for _ in range(cfg.grpo_steps_per_domain):
            sc = {k: scen[k][int(id_order[id_pos % n_id])] for k in scen}
            id_pos += 1
            eps, trajs, denoised = _rollout_group(
                env, model, device, cfg, use_amp, G, sc,
                seed=cfg.seed + 100_000 + it * 7 + id_pos)
            rewards = torch.tensor([e["reward"] for e in eps], dtype=torch.float32)
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
            id_reward += float(rewards.mean())
            id_succ += sum(e["success"] for e in eps)
            if float(adv.std()) < 1e-6:
                infos.append({"loss": None, "awr": None, "kl": None, "aux": None, "n": 0})
                continue
            infos.append(_grpo_step(model, sft_model, env, sc, trajs, denoised,
                                    adv.tolist(), cfg, device, opt, use_amp))

        mean_r = id_reward / cfg.grpo_steps_per_domain
        succ = id_succ / (cfg.grpo_steps_per_domain * G)
        step_losses = [i["loss"] for i in infos if i["loss"] is not None]
        loss_val = float(np.mean(step_losses)) if step_losses else None

        if mean_r > best_mean_r:
            best_mean_r = mean_r
            no_improve = 0
            model.save_pretrained(cfg.grpo_ckpt_dir, safe_serialization=True)
            PocketVLATokenizer().save_pretrained(cfg.grpo_ckpt_dir)
            flag = "  <- best, saved"
        else:
            no_improve += 1
            flag = (f"  (no improvement {no_improve}/{cfg.grpo_early_stop})"
                    if cfg.grpo_early_stop > 0 else "")
        log.append({"iteration": it,
                    "mean_reward": mean_r,
                    "group_success": succ,
                    "loss": loss_val,
                    "steps_mean": float(np.mean([i["n"] for i in infos if i["n"]] or [0]))})
        print(f"  grpo {it:3d}/{cfg.grpo_iterations}  mean_reward={mean_r:+.4f}  "
              f"group_success={succ:.2f}"
              + (f"  loss={loss_val:.5f}" if loss_val is not None else "") + flag)
        Path(cfg.grpo_log_path).write_text(json.dumps(log, indent=1))

        # optional early stop (grpo.early_stop: 0 = disabled)
        if cfg.grpo_early_stop > 0 and no_improve >= cfg.grpo_early_stop:
            print(f"  (mean reward stuck for {no_improve} iterations — early stop)")
            break

    visualize.grpo_curve(log, cfg.grpo_curve_path)

    # ---- final full evaluation of the GRPO checkpoint (random GIF subset) ----
    print("\n[grpo] final evaluation of the GRPO checkpoint on the fixed test set ...")
    import copy
    eval_cfg = copy.copy(cfg)
    eval_cfg.eval_dir = str(Path(cfg.grpo_train_dir) / "eval")
    final_rep = run_eval(eval_cfg, ckpt_path=cfg.grpo_ckpt_dir,
                         n_gifs=cfg.grpo_gifs, gif_seed=cfg.seed, test_npz=test_set)
    stats = {"iterations": cfg.grpo_iterations, "group_size": G,
             "best_mean_reward": best_mean_r,
             "test_success_rate": final_rep["success_rate"],
             "test_grasp_rate": final_rep["grasp_rate"],
             "scenarios": n_id, "checkpoint": cfg.grpo_ckpt_dir,
             "eval_dir": str(Path(cfg.grpo_train_dir) / "eval"),
             "seconds": time.time() - t0}
    print(f"[grpo] done: best mean reward={best_mean_r:.4f}  "
          f"test success rate={stats['test_success_rate']:.1%}  "
          f"checkpoint={cfg.grpo_ckpt_dir}  elapsed={stats['seconds']:.1f}s")
    return stats


def _grpo_step(model: PocketVLA, sft_model: PocketVLA, env: Arm2DPickPlace,
               scenario: dict, trajs: list[list[np.ndarray]],
               denoised: list[list[np.ndarray]], advs: list[float],
               cfg: Config, device: torch.device, opt, use_amp: bool) -> dict:
    """One GRPO update (AWR/RAFT form, positives only): replay each above-average
    member's noisy actions to revisit its observations, regress onto its de-noised
    mean actions with normalized advantages, plus a KL anchor to the frozen SFT
    policy and an aux-consistency term. Negative-advantage steps are never
    regressed on."""
    imgs, toks, props, acts, ws = [], [], [], [], []
    for traj, dtraj, adv in zip(trajs, denoised, advs):
        if adv <= 0.0 or not traj:          # positives only (RAFT-style)
            continue
        obs = env.reset_scenario(scenario)
        for action, daction in zip(traj, dtraj):
            imgs.append(obs["image"])
            toks.append(obs["tokens"])
            props.append(obs["proprio"])
            acts.append(daction)            # target: de-noised policy mean
            ws.append(adv)
            obs, _, done, _ = env.step(action)   # replay the noisy action to stay on-trajectory
            if done:
                break
    if not imgs:
        return {"loss": 0.0, "awr": 0.0, "kl": 0.0, "aux": 0.0, "n": 0}
    img = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).to(device).float() / 255.0
    tok = torch.from_numpy(np.stack(toks)).to(device)
    pad = tok != 0
    prop = torch.from_numpy(np.stack(props)).float().to(device)
    act = torch.from_numpy(np.stack(acts)).float().to(device)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    use_aux = getattr(model.config, "use_aux_heads", True)
    with ctx:
        pred, ball_off, bin_off = model(img, tok, pad, prop)
    with torch.no_grad(), ctx:
        sft_pred, sft_ball_off, sft_bin_off = sft_model(img, tok, pad, prop)
    w = torch.tensor(ws, device=device, dtype=torch.float32)
    w = (w - w.mean()) / (w.std() + 1e-4) + 1.0   # mean-1 normalized advantages
    awr = (w * ((pred - act) ** 2).mean(dim=-1)).mean()
    # KL anchor to the frozen SFT policy
    kl = ((pred - sft_pred) ** 2).mean()
    # aux-consistency: aux heads get no reward signal in RL, anchor them to SFT
    if use_aux and ball_off is not None:
        aux = ((ball_off - sft_ball_off) ** 2).mean() + \
              ((bin_off - sft_bin_off) ** 2).mean()
    else:
        aux = torch.zeros((), device=device)
    loss = awr + cfg.grpo_kl_coef * kl + cfg.grpo_aux_weight * aux
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return {"loss": float(loss.item()), "awr": float(awr.item()),
            "aux": float(aux.item()),
            "kl": float(kl.item()), "n": len(imgs)}
