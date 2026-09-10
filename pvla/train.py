"""SFT behavior-cloning training: MSE(action) + class-balanced grip MSE, plus
(with model.use_aux_heads) auxiliary ball/bin offset readout losses. Reads
dataset/train.npz + dataset/val.npz, writes checkpoint + log + curve to
train.output_dir.
"""
from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import Config, device_banner, ensure_dir, get_device
from .lang.tokenizer import TOKENIZER
from .model.vla import PocketVLA, PocketVLAConfig, PocketVLATokenizer
from . import visualize


def _make_loader(data: dict, cfg: Config, shuffle: bool, use_aux: bool) -> DataLoader:
    cols = [
        data["images"],          # uint8 (N,64,64,3)
        data["tokens"],          # int64 (N,10)
        data["proprio"],         # float32 (N,6)
        data["actions"],         # float32 (N,3) normalized actions
    ]
    if use_aux:
        cols += [data["target_xy"], data["bin_xy"]]   # (N,2) aux offset labels
    ds = TensorDataset(*[torch.from_numpy(c) for c in cols])
    g = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=0, generator=g, drop_last=False)


def _denorm_action(act: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Label actions (N,3) -> physical units: joints x action_max, grip stays +/-1."""
    dq = act[:, :2] * cfg.action_max
    return torch.cat([dq, act[:, 2:3]], dim=1)


def _grip_weight(lab: torch.Tensor, holding: torch.Tensor) -> torch.Tensor:
    """Mean-normalized inverse-frequency weights over (holding x label) groups."""

    w = torch.ones_like(lab)
    for hm in (holding, ~holding):
        for lm in (lab > 0, lab <= 0):
            m = hm & lm
            n = int(m.sum())
            if n > 0:
                w[m] = float(np.sqrt(m.numel() / n))
    return w / w.mean().clamp(min=1e-6)


def _ball_dist_weight(off: torch.Tensor) -> torch.Tensor:
    """Mean-normalized inverse-frequency weights over distance buckets."""

    n = off.norm(dim=-1)
    edges = (0.08, 0.2, 0.5, 1.6)          # normalized-distance bucket edges
    bids = torch.zeros_like(n, dtype=torch.long)
    for i, e in enumerate(edges):
        bids = torch.where(n < e, torch.full_like(bids, i + 1), bids)
    w = torch.ones_like(n)
    for b in range(len(edges) + 1):
        m = bids == b
        cnt = int(m.sum())
        if cnt > 0:
            w[m] = float(np.sqrt(m.numel() / cnt))
    return w / w.mean().clamp(min=1e-6)


def _run_epoch(model, loader, device, cfg, opt=None):
    """One epoch (opt=None = validation). Returns (total loss, ball err|None,
    bin err|None, joint MSE, grip MSE)."""
    training = opt is not None
    model.train() if training else model.eval()
    use_aux = getattr(model.config, "use_aux_heads", True)
    use_amp = device.type == "cuda"    # accelerator surfaces via the CUDA layer; bf16 autocast works
    total, tgt_sum, bin_sum, dq_sum, grip_sum, count = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    for batch in loader:
        if use_aux:
            img_nhwc, tok, prop, act, tgt, bin_t = batch
        else:
            img_nhwc, tok, prop, act = batch
        img = img_nhwc.to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
        tok = tok.to(device, non_blocking=True)
        pad = tok != 0
        prop = prop.to(device, non_blocking=True)
        act = _denorm_action(act.to(device, non_blocking=True), cfg)   # rad + grip +/-1
        if use_aux:
            tgt = tgt.to(device, non_blocking=True)
            bin_t = bin_t.to(device, non_blocking=True)
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
        with torch.set_grad_enabled(training), amp:
            pred, ball_off, bin_off = model(img, tok, pad, prop)
            dq_loss = F.mse_loss(pred[:, :2], act[:, :2])
            # class-balanced grip MSE (else "always open/closed" wins)
            if cfg.balanced_losses:
                grip_w = _grip_weight(act[:, 2], prop[:, 5] > 0.5)
            else:
                grip_w = torch.ones_like(act[:, 2])
            grip_loss = (grip_w * (pred[:, 2] - act[:, 2]) ** 2).mean()
            loss = dq_loss + cfg.action_grip_weight * grip_loss
            tgt_err = bin_err = None
            if use_aux:
                # distance-bucketed class balancing (else small offsets dominate)
                if cfg.balanced_losses:
                    ball_w = _ball_dist_weight(tgt).unsqueeze(-1)      # (B,1)
                    ball_loss = (ball_w * (ball_off - tgt) ** 2).mean()
                else:
                    ball_loss = F.mse_loss(ball_off, tgt)
                loss = loss + cfg.aux_target_weight * ball_loss
                loss = loss + cfg.aux_bin_weight * F.mse_loss(bin_off, bin_t)
                tgt_err = F.mse_loss(ball_off, tgt).detach()
                bin_err = F.mse_loss(bin_off, bin_t).detach()
            dq_err = dq_loss.detach()
            grip_err = F.mse_loss(pred[:, 2], act[:, 2]).detach()
        if training:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += loss.item()
        if tgt_err is not None:
            tgt_sum += tgt_err.item()
            bin_sum += bin_err.item()
        dq_sum += dq_err.item()
        grip_sum += grip_err.item()
        count += 1
    k = max(count, 1)
    return (total / k,
            tgt_sum / k if use_aux else None,
            bin_sum / k if use_aux else None,
            dq_sum / k, grip_sum / k)


def train(cfg: Config) -> dict:
    device = get_device()
    torch.manual_seed(cfg.seed)
    print(f"[train] {device_banner(device)}")

    missing = [p for p in (cfg.train_npz, cfg.val_npz) if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            "[train] dataset file(s) not found: " + ", ".join(missing) + "\n"
            "[train]   run 'python3 -m pvla collect --config <yaml>' first")
    if cfg.train_dir is None:
        raise ValueError("[train] yaml must set train.output_dir (SFT artifacts directory)")
    ensure_dir(cfg.train_dir)
    train_data = dict(np.load(cfg.train_npz, allow_pickle=False))
    val_data = dict(np.load(cfg.val_npz, allow_pickle=False))
    use_aux = bool(cfg.model_arch.get("use_aux_heads", True))
    if use_aux:
        missing_cols = {"target_xy", "bin_xy"} - (set(train_data) & set(val_data))
        if missing_cols:
            raise KeyError(
                f"[train] dataset file(s) missing columns {sorted(missing_cols)}\n"
                f"[train]   re-run 'python3 -m pvla collect' to regenerate the dataset "
                f"(auxiliary labels are required when use_aux_heads is true)")
    n = len(train_data["images"]) + len(val_data["images"])
    print(f"[train] dataset: train={cfg.train_npz}  val={cfg.val_npz}")
    print(f"[train] samples={n}  train={len(train_data['images'])}  "
          f"val={len(val_data['images'])}  (episode-level split, "
          f"no adjacent-frame leakage)")

    train_loader = _make_loader(train_data, cfg, shuffle=True, use_aux=use_aux)
    val_loader = _make_loader(val_data, cfg, shuffle=False, use_aux=use_aux)

    hf_cfg = PocketVLAConfig(
        size=cfg.model_size, obs_res=cfg.obs_res, world_extent=cfg.world_extent,
        proprio_dim=cfg.proprio_dim, action_dim=cfg.action_dim,
        lang_len=cfg.lang_len, vocab_size=len(TOKENIZER),
        action_max=cfg.action_max, **cfg.model_arch,
    )
    model = PocketVLA(hf_cfg).to(device)
    print(f"[train] model params: {model.n_params() / 1e6:.2f}M  (size={cfg.model_size}, lr={cfg.lr:.1e})")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    log, best_ctrl = [], float("inf")
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        tr, tr_tgt, tr_bin, tr_dq, tr_grip = _run_epoch(model, train_loader, device, cfg, opt=opt)
        with torch.no_grad():
            va, va_tgt, va_bin, va_dq, va_grip = _run_epoch(model, val_loader, device, cfg, opt=None)
        sched.step()
        lr = sched.get_last_lr()[0]
        use_aux = getattr(model.config, "use_aux_heads", True)
        # selection score: joint MSE + grounding errors; grip excluded (high variance)
        va_ctrl = va_dq + va_tgt + va_bin if use_aux else va_dq
        entry = {"epoch": epoch, "train": tr, "val": va, "lr": lr,
                 "val_dq_mse": va_dq, "val_grip_mse": va_grip, "val_ctrl": va_ctrl,
                 "use_aux_heads": use_aux}
        if use_aux:
            entry.update({"train_tgt_err": tr_tgt, "train_bin_err": tr_bin,
                          "val_tgt_err": va_tgt, "val_bin_err": va_bin})
        log.append(entry)
        flag = ""
        if va_ctrl < best_ctrl:
            best_ctrl = va_ctrl
            model.save_hf(cfg.hf_dir, PocketVLATokenizer(TOKENIZER.vocab))
            flag = "  <- best, saved"
        aux_str = f"  ball={va_tgt:.4f}  bin={va_bin:.4f}" if use_aux else ""
        print(f"  epoch {epoch:3d}/{cfg.epochs}  train={tr:.5f}  val={va:.5f}  "
              f"ctrl={va_ctrl:.5f}  dq={va_dq:.5f}  grip={va_grip:.4f}  "
              f"lr={lr:.2e}{aux_str}{flag}")
        Path(cfg.train_log_path).write_text(json.dumps(log, indent=1))

    visualize.loss_curve(log, cfg.loss_curve_path)
    stats = {"samples": n, "train_samples": len(train_data["images"]),
             "val_samples": len(val_data["images"]), "epochs": cfg.epochs,
             "best_val_ctrl": best_ctrl, "use_aux_heads": use_aux,
             "seconds": time.time() - t0, "params": model.n_params()}
    print(f"[train] done: best val ctrl={best_ctrl:.5f}  "
          f"checkpoint={cfg.hf_dir}  elapsed={stats['seconds']:.1f}s")

    return stats
