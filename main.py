"""Command-line entry point.

Examples:
    python main.py gan --preset minimax                 # 2.1
    python main.py gan --preset non-saturating          # 2.2
    python main.py gan --preset conditional             # 2.3
    python main.py adversarial --source 4 --target 9    # 2.4
    python main.py diffusion-1d                         # 2.5, 1D forward/reverse demos
    python main.py diffusion-train --model transformer  # 2.5, unet | unet_res | transformer
    python main.py diffusion-sample --model transformer --digit 7
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GANs, adversarial examples and diffusion models on MNIST")
    parser.add_argument("--seed", type=int, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("gan", help="train a GAN (2.1-2.3)")
    p.add_argument("--preset", choices=["minimax", "non-saturating", "conditional"], default="non-saturating")
    p.add_argument("--iterations", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--sample-every", type=int)

    p = sub.add_parser("adversarial", help="train a CNN and craft a targeted adversarial example (2.4)")
    p.add_argument("--source", type=int, default=4, help="true digit of the image to attack")
    p.add_argument("--target", type=int, default=9, help="digit to make the classifier predict")
    p.add_argument("--epsilon", type=float, default=0.2, help="perturbation size")
    p.add_argument("--epochs", type=int, default=4)

    sub.add_parser("diffusion-1d", help="plot 1D forward and reverse diffusion (2.5)")

    p = sub.add_parser("diffusion-train", help="train a score-based diffusion model (2.5)")
    p.add_argument("--model", choices=["unet", "unet_res", "transformer"], default="unet")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--limit-batches", type=int, default=None, help="cap batches per epoch (quick test)")

    p = sub.add_parser("diffusion-sample", help="generate samples from a trained diffusion model")
    p.add_argument("--model", choices=["unet", "unet_res", "transformer"], default="unet")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--steps", type=int, default=None, help="default: 500 (250 for transformer)")
    p.add_argument("--digit", type=int, default=None, help="digit to generate (transformer only, default 7)")
    p.add_argument("--checkpoint", type=Path, default=None)
    return parser


def _overrides(args, *names):
    return {n: getattr(args, n) for n in names if getattr(args, n, None) is not None}


def main() -> None:
    args = build_parser().parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.command == "gan":
        from genmodels import gan

        cfg = replace(gan.PRESETS[args.preset], **_overrides(args, "iterations", "batch_size", "lr", "sample_every"))
        gan.train(cfg, name=args.preset)

    elif args.command == "adversarial":
        from genmodels import adversarial

        adversarial.run(args.source, args.target, args.epsilon, args.epochs, seed=args.seed)

    elif args.command == "diffusion-1d":
        from genmodels import diffusion

        diffusion.plot_1d_demos()

    elif args.command == "diffusion-train":
        from genmodels import diffusion

        cfg = replace(diffusion.PRESETS[args.model], **_overrides(args, "epochs", "batch_size", "lr", "num_workers"))
        diffusion.train(cfg, limit_batches=args.limit_batches)

    elif args.command == "diffusion-sample":
        from genmodels import diffusion

        diffusion.sample(args.model, args.n, args.steps, args.digit, args.checkpoint)


if __name__ == "__main__":
    main()
