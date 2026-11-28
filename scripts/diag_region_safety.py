"""Are the ALM's convex regions actually collision-free, and do collisions happen INSIDE them?

For one scene this rebuilds the corridor pack, takes the ALM output curve and reports:
  * blocked fraction inside every region polygon (both image-row conventions, so a
    y-flip cannot fake a "0% blocked" result);
  * for each colliding dense point of the ALM curve: is it inside a region or not.
The split tells us whether the region set is unsafe, or the projection just fails to
put the curve inside it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT, base_parser, load  # noqa: E402

import numpy as np  # noqa: E402

from gp_diffuser.evaluation.metrics import dense_points  # noqa: E402
from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.training.real_dataset import RealTrajectoryDataset  # noqa: E402


def clip_polygon(poly, normal, offset):
    if len(poly) == 0:
        return poly
    out = []
    n = len(poly)
    for i in range(n):
        cur, nxt = poly[i], poly[(i + 1) % n]
        dc = float(normal @ cur - offset)
        dn = float(normal @ nxt - offset)
        if dc <= 0:
            out.append(cur)
        if (dc <= 0) != (dn <= 0):
            out.append(cur + dc / (dc - dn) * (nxt - cur))
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def piece_polygons(pack, sample=0):
    """Feasible polygons of the pack's cells (sample pack or historical piece pack).

    The inference pack holds ONE cell per uniform-arc-length sample point, so the
    polygons returned are the unique cells that actually own a sample.
    """
    A_all = (pack.sample_A if hasattr(pack, "sample_A") else pack.piece_A)
    b_all = (pack.sample_b if hasattr(pack, "sample_b") else pack.piece_b)
    face_all = pack.face_mask
    if hasattr(pack, "sample_A"):
        keep_all = pack.sample_mask
    else:
        keep_all = pack.piece_mask
    A = A_all[sample].detach().cpu().numpy()
    b = b_all[sample].detach().cpu().numpy()
    face = face_all[sample].detach().cpu().numpy()
    piece = keep_all[sample].detach().cpu().numpy()
    box = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    out, seen = [], set()
    for p in range(len(piece)):
        if not piece[p] or not face[p].any():
            continue
        key = (tuple(np.round(A[p][face[p]], 9).ravel()),
               tuple(np.round(b[p][face[p]], 9).ravel()))
        if key in seen:                          # one polygon per CELL
            continue
        seen.add(key)
        poly = box
        for f in range(A.shape[1]):
            if not face[p, f]:
                continue
            poly = clip_polygon(poly, A[p, f], b[p, f])
            if len(poly) == 0:
                break
        if len(poly) >= 3:
            out.append(poly)
    return out


def inside_any(points_scene, polys) -> np.ndarray:
    mask = np.zeros(len(points_scene), dtype=bool)
    for poly in polys:
        ok = np.ones(len(points_scene), dtype=bool)
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            edge = b - a
            ok &= (edge[0] * (points_scene[:, 1] - a[1])
                   - edge[1] * (points_scene[:, 0] - a[0])) >= -1e-9
        mask |= ok
    return mask


def main() -> int:
    parser = base_parser("region safety vs projection")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--out", default="outputs/reports/region_safety.json")
    opts = parser.parse_args()

    cfg = load(opts)
    dataset = RealTrajectoryDataset(cfg, "val")
    pipe = build_pipeline(cfg, device=opts.device, checkpoint=opts.checkpoint)
    sampler, transform = pipe["sampler"], pipe["transform"]
    res_m = float(dataset.res_m)

    item = dataset[opts.index]
    occ = item["occupancy"][None].clone()
    sdf = item["sdf"][None].clone()
    sg = item["start_goal"][None].clone()
    hd = item["heading"][None].clone() if item.get("heading") is not None else None
    hv = (item["heading_valid"][None].clone()
          if item.get("heading_valid") is not None else None)
    occ_np = occ[0, 0].numpy()
    H, W = occ_np.shape

    out_map = sampler.sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                             steps=opts.steps, safety_mode="map_only", seed=0)
    raw, _ = sampler._generate_safety(
        occ.to(sampler.device), sdf.to(sampler.device),
        sampler.map_encoder.encode(occ.to(sampler.device), sdf.to(sampler.device)),
        out_map["trajectory_scene"][0:1], sg.to(sampler.device),
        dict(sampler.cfg.get("corridor") or {}))
    polys = piece_polygons(raw["pack"], 0)
    print("regions:", len(polys))

    # 1. blocked fraction inside the regions, both row conventions
    for name, flip in (("rows as-is", False), ("rows flipped", True)):
        fracs = []
        for poly in polys:
            gx = np.linspace(poly[:, 0].min(), poly[:, 0].max(), 16)
            gy = np.linspace(poly[:, 1].min(), poly[:, 1].max(), 16)
            grid = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)
            inside = np.ones(len(grid), dtype=bool)
            for i in range(len(poly)):
                a, b = poly[i], poly[(i + 1) % len(poly)]
                edge = b - a
                inside &= (edge[0] * (grid[:, 1] - a[1])
                           - edge[1] * (grid[:, 0] - a[0])) >= -1e-12
            pts = grid[inside]
            if len(pts) == 0:
                continue
            col = np.clip(((pts[:, 0] + 1) * 0.5 * W).astype(int), 0, W - 1)
            row = ((pts[:, 1] + 1) * 0.5 * H) if not flip else ((1 - (pts[:, 1] + 1) * 0.5) * H)
            row = np.clip(row.astype(int), 0, H - 1)
            fracs.append(float((occ_np[row, col] > 0).mean()))
        print("  %s: blocked mean %.3f  max %.3f  (regions %d)"
              % (name, np.mean(fracs), np.max(fracs), len(fracs)))

    # 2. where do the ALM curve's collisions sit: inside a region or outside?
    out_full = sampler.sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                              steps=opts.steps, safety_mode="full", seed=0)
    curve_m = out_full["trajectory_m"][0].numpy()
    dense = dense_points(curve_m, 0.05)
    scene = transform.to_scene(dense)
    col = np.clip(((scene[:, 0] + 1) * 0.5 * W).astype(int), 0, W - 1)
    row = np.clip(((scene[:, 1] + 1) * 0.5 * H).astype(int), 0, H - 1)
    blocked = occ_np[row, col] > 0
    ins = inside_any(scene, polys)
    print("\nALM curve: %d dense points | blocked %d (%.3f) | inside regions %.3f"
          % (len(dense), int(blocked.sum()), blocked.mean(), ins.mean()))
    print("  blocked AND inside a region : %d  <- the region set itself is unsafe"
          % int((blocked & ins).sum()))
    print("  blocked AND outside regions : %d  <- the projection did not put it inside"
          % int((blocked & ~ins).sum()))
    print("  free    AND inside a region : %d"
          % int((~blocked & ins).sum()))

    out_path = os.path.join(PROJECT_ROOT, opts.out)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"index": opts.index, "regions": len(polys),
                   "dense_points": int(len(dense)),
                   "blocked": int(blocked.sum()),
                   "blocked_inside": int((blocked & ins).sum()),
                   "blocked_outside": int((blocked & ~ins).sum()),
                   "inside_rate": float(ins.mean())}, fh, indent=2)
    print("written:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
