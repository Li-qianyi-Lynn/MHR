#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mhr.mhr import MHR


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dump MHR facial expression (72D) parameter names in order."
    )
    p.add_argument(
        "--assets",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "assets",
        help="Path to the MHR assets folder (default: ./assets).",
    )
    p.add_argument(
        "--lod",
        type=int,
        default=1,
        help="LOD level to load (default: 1).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device string (default: cpu).",
    )
    p.add_argument(
        "--no-pose-correctives",
        action="store_true",
        help="Skip pose correctives (faster load; not needed to print names).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    assets = args.assets
    if not assets.exists():
        raise FileNotFoundError(f"assets folder not found: {assets}")

    m = MHR.from_files(
        folder=assets,
        device=torch.device(args.device),
        lod=args.lod,
        wants_pose_correctives=not args.no_pose_correctives,
    )
    expr_names = m.get_face_expression_names()

    print(f"count={len(expr_names)}")
    for i, n in enumerate(expr_names):
        print(f"{i:02d}\t{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

