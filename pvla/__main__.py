"""Unified CLI: python3 -m pvla <command> [--config <yaml>]

Commands:
  collect    dataset generation (no yaml needed)
  train      SFT behavior cloning (yaml required)
  grpo       GRPO RL fine-tuning of an SFT checkpoint (yaml required)
  evaluate   closed-loop evaluation (with only --ckpt it runs standalone)
  all        train -> evaluate for one yaml (run `collect` first)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import Config, device_banner, get_device


def _add_common(p: argparse.ArgumentParser, config_required: bool = True):
    p.add_argument("--config", "-c", type=str, default=None, required=config_required,
                   help="experiment yaml (one file = one model: arch + dirs + hyperparameters; "
                        "optional for 'collect' and for 'evaluate' with --ckpt)")
    p.add_argument("--out-dir", type=str, default=None,
                   help="smoke-test override: redirect dataset/train/eval outputs to "
                        "<dir>/dataset, <dir>/train, <dir>/eval (yaml stays untouched)")
    p.add_argument("--seed", type=int, default=None,
                   help="override the seed (yaml value or the built-in default)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pvla",
        description="PocketVLA: lightweight VLA pipeline "
                    "(language-guided pick & place: dataset gen / train / evaluate)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("collect",
                       help="dataset generation: expert demos -> dataset/{train,val,test}.npz "
                            "(no yaml needed; built-in defaults + CLI overrides)")
    _add_common(p, config_required=False)
    p.add_argument("--dataset-dir", type=str, default=None,
                   help="override the dataset output directory "
                        "(default: dataset; or data.dataset_dir when --config is given)")
    p.add_argument("--episodes", type=int, default=None,
                   help="override the number of expert demo episodes (default 4500)")
    p.add_argument("--val-frac", type=float, default=None,
                   help="override the validation fraction, split by episode (default 0.10)")
    p.add_argument("--eval-episodes", type=int, default=None,
                   help="override the number of fixed test scenarios (default 100)")

    p = sub.add_parser("train", help="SFT behavior-cloning training")
    _add_common(p)
    p.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    p.add_argument("--batch-size", type=int, default=None, help="override train.batch_size")
    p.add_argument("--lr", type=float, default=None, help="override train.lr")

    p = sub.add_parser("grpo",
                       help="GRPO RL fine-tuning of an SFT checkpoint (own RL "
                            "scenario set grpo.npz + own artifacts directory)")
    _add_common(p)
    p.add_argument("--ckpt", type=str, default=None,
                   help="SFT checkpoint directory to fine-tune "
                        "(overrides grpo.ckpt in the yaml)")
    p.add_argument("--iterations", type=int, default=None,
                   help="override grpo.iterations")

    p = sub.add_parser("evaluate",
                       help="closed-loop eval on the fixed test set (one sweep: stats + "
                            "per-scenario replay GIFs + summary figures); with only "
                            "--ckpt it runs standalone (no yaml): artifacts go to "
                            "<ckpt>/eval (or <ckpt>/eval_ood with --test-npz)")
    _add_common(p, config_required=False)
    p.add_argument("--episodes", type=int, default=None,
                   help="override data.eval_episodes (number of test scenarios)")
    p.add_argument("--ckpt", type=str, default=None,
                   help="checkpoint directory to evaluate (overrides eval.ckpt in "
                        "the yaml; with no --config this is all you need to pass "
                        "for a standalone evaluation, e.g. "
                        "outputs_grpo_3m/pocketvla-3m-grpo)")
    p.add_argument("--gifs", type=int, default=None,
                   help="override eval.gifs (number of random replay GIFs)")
    p.add_argument("--test-npz", type=str, default=None,
                   help="test scenario file to evaluate (default: "
                        "<dataset_dir>/test.npz; pass dataset/ood_test.npz for "
                        "the OOD set — artifacts then go to <ckpt>/eval_ood)")
    p.add_argument("--no-gif", action="store_true",
                   help="skip per-scenario GIF export (summary figures still drawn)")

    p = sub.add_parser("all", help="train -> evaluate for one yaml (no dataset collection; "
                                   "run `collect` first — all sizes share one dataset)")
    _add_common(p)
    p.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    p.add_argument("--eval-episodes", type=int, default=None,
                   help="override data.eval_episodes")
    return parser


def _load_cfg(args) -> Config:
    if args.config is not None:
        cfg = Config.from_yaml(args.config)
    else:
        cfg = Config()   # standalone mode: built-in defaults
    if args.out_dir is not None:
        root = Path(args.out_dir)
        cfg.dataset_dir = str(root / "dataset")
        cfg.train_dir = str(root / "train")
        cfg.eval_dir = str(root / "eval")
    if args.seed is not None:
        cfg.seed = args.seed
    return cfg


def _banner(title: str):
    print("\n" + "=" * 62 + f"\n {title}\n" + "=" * 62)


def main():
    args = build_parser().parse_args()
    cfg = _load_cfg(args)

    # apply per-command CLI overrides before printing the banner
    if args.cmd == "collect":
        if args.episodes is not None:
            cfg.episodes_collect = args.episodes
        if args.eval_episodes is not None:
            cfg.eval_episodes = args.eval_episodes
        if args.val_frac is not None:
            cfg.val_frac = args.val_frac
        if args.dataset_dir is not None:
            cfg.dataset_dir = args.dataset_dir
    elif args.cmd == "train":
        if args.epochs is not None:
            cfg.epochs = args.epochs
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size
        if args.lr is not None:
            cfg.lr = args.lr
    elif args.cmd == "evaluate":
        if args.episodes is not None:
            cfg.eval_episodes = args.episodes
        if args.gifs is not None:
            cfg.eval_gifs = args.gifs
        # standalone mode: derive the artifact dir from the checkpoint
        if args.config is None and args.ckpt is not None:
            ckpt_root = Path(args.ckpt).resolve()
            if cfg.eval_dir is None:
                cfg.eval_dir = str(ckpt_root / ("eval_ood" if args.test_npz else "eval"))
    elif args.cmd == "grpo":
        if args.iterations is not None:
            cfg.grpo_iterations = args.iterations
    elif args.cmd == "all":
        if args.epochs is not None:
            cfg.epochs = args.epochs
        if args.eval_episodes is not None:
            cfg.eval_episodes = args.eval_episodes

    print(device_banner(get_device()))
    if args.config is not None:
        print(f"[cfg] {args.config}:  {cfg.summary()}")
    elif args.cmd == "evaluate":
        print(f"[cfg] no yaml: standalone evaluation on built-in defaults  "
              f"(test set={args.test_npz or cfg.test_npz}  episodes={cfg.eval_episodes}  "
              f"gifs={cfg.eval_gifs}  seed={cfg.seed})")
    else:
        print(f"[cfg] no yaml: dataset generation on built-in defaults  "
              f"(dataset_dir={cfg.dataset_dir}  episodes={cfg.episodes_collect}  "
              f"val_frac={cfg.val_frac}  eval_episodes={cfg.eval_episodes}  "
              f"seed={cfg.seed})")

    if args.cmd == "collect":
        from .data.collect import collect
        _banner("Stage 1/1: dataset generation (train / val / test)")
        collect(cfg)

    elif args.cmd == "train":
        from .train import train
        _banner("VLA SFT training (behavior cloning)")
        train(cfg)

    elif args.cmd == "evaluate":
        from .evaluate import evaluate
        _banner("VLA closed-loop evaluation (pick & place)")
        evaluate(cfg, ckpt_path=args.ckpt, save_gifs=not args.no_gif,
                 n_gifs=cfg.eval_gifs, test_npz=args.test_npz)

    elif args.cmd == "grpo":
        from .grpo import run_grpo
        _banner("GRPO reinforcement learning (decoupled stage: own RL scenarios "
                "+ own artifacts)")
        run_grpo(cfg, ckpt_path=args.ckpt)

    elif args.cmd == "all":
        from .evaluate import evaluate
        from .train import train

        t0 = time.time()
        _banner("Stage 1/2: VLA SFT training (behavior cloning)")
        train(cfg)
        _banner("Stage 2/2: VLA closed-loop evaluation (fixed test set: stats + "
                "per-scenario GIFs + figures, one sweep)")
        evaluate(cfg, n_gifs=cfg.eval_gifs)
        _banner(f"Full pipeline done in {time.time() - t0:.1f}s")
        _list_artifacts(cfg)


def _list_artifacts(cfg: Config):
    print("Artifacts:")
    for root in (cfg.dataset_dir, cfg.train_dir, cfg.eval_dir):
        d = Path(root)
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*")):
            if f.is_file():
                print(f"  {f}  ({f.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
