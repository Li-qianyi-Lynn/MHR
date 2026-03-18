#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import trimesh

from mhr.mhr import MHR


def _parse_dim_list(s: str, n_dims: int) -> list[int]:
    s = s.strip()
    if s.lower() in {"all", "*"}:
        return list(range(n_dims))

    dims: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a.strip())
            end = int(b.strip())
            if end < start:
                start, end = end, start
            for i in range(start, end + 1):
                dims.add(i)
        else:
            dims.add(int(part))

    out = sorted(dims)
    for d in out:
        if d < 0 or d >= n_dims:
            raise ValueError(f"dim out of range: {d} (expected 0..{n_dims-1})")
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export meshes showing per-dimension facial expression deltas. "
            "This helps you manually label which dims correspond to smile/cry/etc "
            "when blendshape names are generic (blend_XX)."
        )
    )
    p.add_argument(
        "--assets",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "assets",
        help="Path to the MHR assets folder (default: ./assets).",
    )
    p.add_argument("--lod", type=int, default=1, help="LOD level to load (default: 1).")
    p.add_argument(
        "--device", type=str, default="cpu", help="Torch device string (default: cpu)."
    )
    p.add_argument(
        "--no-pose-correctives",
        action="store_true",
        help="Skip pose correctives (faster; not needed for expression deltas).",
    )
    p.add_argument(
        "--dims",
        type=str,
        default="all",
        help='Which expression dims to export. Examples: "0,1,2", "10-20", "all".',
    )
    p.add_argument(
        "--amp",
        type=float,
        default=1.0,
        help="Coefficient amplitude to apply for +/-. Default: 1.0",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("expression_deltas"),
        help="Output directory (default: ./expression_deltas).",
    )
    p.add_argument(
        "--export-neutral",
        action="store_true",
        help="Also export neutral mesh (all expression coeffs = 0).",
    )
    return p.parse_args()


@torch.no_grad()
def _mesh_from_params(
    m: MHR,
    identity_coeffs: torch.Tensor,
    model_parameters: torch.Tensor,
    face_expr_coeffs: torch.Tensor,
) -> trimesh.Trimesh:
    verts, _ = m(identity_coeffs, model_parameters, face_expr_coeffs, apply_correctives=True)
    return trimesh.Trimesh(
        vertices=verts[0].detach().cpu().numpy(),
        faces=m.character.mesh.faces,
        process=False,
    )


def main() -> int:
    args = _parse_args()
    if not args.assets.exists():
        raise FileNotFoundError(f"assets folder not found: {args.assets}")

    m = MHR.from_files(
        folder=args.assets,
        device=torch.device(args.device),
        lod=args.lod,
        wants_pose_correctives=not args.no_pose_correctives,
    )

    n_expr = m.get_num_face_expression_blendshapes()
    dims = _parse_dim_list(args.dims, n_expr)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Neutral identity/pose: keep everything zero so you isolate expression deformation.
    identity = torch.zeros((1, m.get_num_identity_blendshapes()), device=torch.device(args.device))
    model_params = torch.zeros((1, 204), device=torch.device(args.device))
    expr = torch.zeros((1, n_expr), device=torch.device(args.device))

    if args.export_neutral:
        neutral = _mesh_from_params(m, identity, model_params, expr)
        neutral.export(out_dir / "neutral.ply")

    expr_names = m.get_face_expression_names()

    for d in dims:
        name = expr_names[d]

        expr_pos = expr.clone()
        expr_pos[0, d] = float(args.amp)
        mesh_pos = _mesh_from_params(m, identity, model_params, expr_pos)
        mesh_pos.export(out_dir / f"{d:02d}_{name}_plus.ply")

        expr_neg = expr.clone()
        expr_neg[0, d] = -float(args.amp)
        mesh_neg = _mesh_from_params(m, identity, model_params, expr_neg)
        mesh_neg.export(out_dir / f"{d:02d}_{name}_minus.ply")

    print(f"exported_dims={len(dims)} out={out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

