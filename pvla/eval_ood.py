"""Out-of-distribution generalization evaluation.

Generates (or loads) a fixed OOD test set: standard scene plus ONE distractor
object in an unseen color (ball or square), from a fourth seed space
(seed + 80,000 + i), persisted to dataset/ood_test.npz. Runs closed-loop
inference of the given checkpoint and writes the standard evaluation artifacts
into eval.output_dir + "_ood".

Usage:
    python3 -m pvla.eval_ood --config configs/sft_3m.yaml
    python3 -m pvla.eval_ood -c configs/sft_3m.yaml \
        --ckpt outputs_grpo_3m/pocketvla-3m-grpo --n 100 --gifs 20
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from .config import Config, device_banner, get_device
from .scenarios import OOD_TEST_SEED_OFFSET, gen_ood_scenarios
DISTRACTOR_COLORS = ("yellow", "purple", "orange", "cyan", "pink",
                     "brown", "white", "gray")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="pvla.eval_ood",
        description="OOD generalization test: unseen distractor objects")
    ap.add_argument("--config", "-c", required=True,
                    help="experiment yaml (defines model + eval.output_dir)")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint directory to evaluate (overrides eval.ckpt)")
    ap.add_argument("--n", type=int, default=300,
                    help="number of OOD scenarios (default 300)")
    ap.add_argument("--gifs", type=int, default=20,
                    help="random replay GIFs to save (default 20; 0 = none)")
    ap.add_argument("--seed", type=int, default=None,
                    help="OOD scenario seed (default: cfg.seed)")
    ap.add_argument("--regenerate", action="store_true",
                    help="regenerate the OOD scenario file even if it exists")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    device = get_device()
    print(f"[ood] {device_banner(device)}")

    seed = cfg.seed if args.seed is None else args.seed
    ood_file = str(Path(cfg.dataset_dir) / "ood_test.npz")
    if Path(ood_file).is_file() and not args.regenerate:
        scen = dict(np.load(ood_file, allow_pickle=True))
        print(f"[ood] loaded existing OOD set: {ood_file} ({len(scen['q0'])} scenarios)")
        if len(scen["q0"]) < args.n:
            scen = gen_ood_scenarios(cfg, args.n, OOD_TEST_SEED_OFFSET, seed)
            np.savez_compressed(ood_file, **scen)
            print(f"[ood] extended/regenerated with n={args.n}")
    else:
        scen = gen_ood_scenarios(cfg, args.n, OOD_TEST_SEED_OFFSET, seed)
        np.savez_compressed(ood_file, **scen)
        print(f"[ood] generated {len(scen['q0'])} OOD scenarios -> {ood_file} "
              f"(seed space seed+{OOD_TEST_SEED_OFFSET}+i)")

    # eval into <eval.output_dir>_ood
    eval_cfg = copy.copy(cfg)
    base = (cfg.eval_dir
            or (f"{cfg.grpo_train_dir}/eval" if cfg.grpo_train_dir
                else f"outputs_sft_{cfg.model_size}/eval"))
    eval_cfg.eval_dir = f"{base.rstrip('/')}_ood"
    eval_cfg.eval_ckpt = args.ckpt or cfg.eval_ckpt
    eval_cfg.eval_episodes = min(args.n, len(scen["q0"]))

    from .evaluate import evaluate as run_eval

    t0 = time.time()
    report = run_eval(eval_cfg, ckpt_path=args.ckpt, save_gifs=args.gifs > 0,
                      n_gifs=args.gifs, gif_seed=cfg.seed, test_npz=ood_file)
    report["ood"] = {"distractor_colors": [list(map(str, x)) for x in scen["distractor_colors"]],
                     "seed_offset": OOD_TEST_SEED_OFFSET}
    out = Path(eval_cfg.eval_dir) / "eval_report.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"[ood] report (with OOD metadata) -> {out}")
    print(f"[ood] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
