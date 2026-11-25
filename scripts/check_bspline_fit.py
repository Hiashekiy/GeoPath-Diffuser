"""B-spline representation study: fit error vs control count vs path length.

The design document makes ``trajectory.control_points = 32`` conditional on the
reconstruction error of GT trajectories (section 4.1: "正式训练前必须统计 GT
轨迹经 B-spline 拟合后的重建误差，并按路径长度分桶").  This script performs
exactly that measurement, for C in {32, 48, 64}, on

* the synthetic GT paths of the project dataset, and
* synthetic wavy paths of increasing length (a worse case for a cubic spline),

and writes ``outputs/reports/bspline_fit.json`` plus a markdown table.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT, base_parser, load  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gp_diffuser.frames import SceneTransform  # noqa: E402
from gp_diffuser.guidance.search import resample_polyline  # noqa: E402
from gp_diffuser.trajectory.bspline import BSplineCodec  # noqa: E402
from gp_diffuser.training.dataset import SyntheticTrajectoryDataset  # noqa: E402

CONTROL_COUNTS = (32, 48, 64)
BUCKETS = ((0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 1e9))


def fit_error(codec, curve_scene, start_scene, goal_scene) -> float:
    """Max reconstruction error (scene units) of one ``[1,H,2]`` curve."""
    curve_scene = curve_scene.reshape(1, -1, 2)
    controls = codec.fit_curve_to_controls(
        curve_scene, torch.as_tensor(start_scene).reshape(1, 2),
        torch.as_tensor(goal_scene).reshape(1, 2))
    decoded = codec.decode_controls(controls)[0]
    err = (decoded - curve_scene).norm(dim=-1)
    return float(err.max())


def wavy_path(n: int, extent_m: float, amp_m: float, waves: float,
              rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    x = 0.1 * extent_m + t * 0.8 * extent_m
    y = 0.5 * extent_m + amp_m * np.sin(2 * np.pi * waves * t + rng.uniform(0, 6.28))
    return np.stack([x, y], axis=-1)


def main() -> int:
    args = base_parser(__doc__ or "bspline fit study").parse_args()
    cfg = load(args)
    res_m = float(cfg["dataset"]["res_m"])
    grid = int(cfg["dataset"]["grid"])
    transform = SceneTransform(res_m * grid, grid)

    codecs = {c: BSplineCodec(degree=int(cfg["trajectory"]["degree"]),
                             num_controls=c,
                             curve_points=int(cfg["trajectory"]["curve_points"]))
              for c in CONTROL_COUNTS}

    rows = []

    # ---- 1) the project's synthetic GT paths ----------------------------
    dataset = SyntheticTrajectoryDataset(cfg, "test", size=24)
    for i in range(len(dataset)):
        item = dataset[i]
        curve = item["path_gt_scene"].reshape(1, -1, 2)
        length_m = float(torch.linalg.norm(item["path_gt_m"][1:] - item["path_gt_m"][:-1],
                                          dim=-1).sum())
        row = {"source": "dataset", "index": i, "length_m": length_m}
        for count, codec in codecs.items():
            row["err_%d" % count] = fit_error(codec, curve, item["start_goal"][0],
                                              item["start_goal"][1])
        rows.append(row)

    # ---- 2) synthetic wavy paths (long, curved) -------------------------
    rng = np.random.default_rng(7)
    extent_m = res_m * grid
    for k in range(12):
        waves = float(rng.uniform(1.0, 3.5))
        amp = float(rng.uniform(2.0, 8.0))
        n = 128
        path_m = wavy_path(n, extent_m, amp, waves, rng)
        curve = torch.as_tensor(transform.to_scene(path_m),
                                dtype=torch.float32).reshape(1, -1, 2)
        seg = np.linalg.norm(np.diff(path_m, axis=0), axis=1)
        length_m = float(seg.sum())
        # the codec decodes 128 points; compare like with like by resampling
        path_m = resample_polyline(path_m, 128)
        row = {"source": "wavy", "index": k, "length_m": length_m, "waves": waves,
               "amplitude_m": amp}
        for count, codec in codecs.items():
            row["err_%d" % count] = fit_error(codec, curve,
                                              curve[0, 0], curve[0, -1])
        rows.append(row)

    # ---- aggregate by length bucket -------------------------------------
    table = []
    for lo, hi in BUCKETS:
        subset = [r for r in rows if lo <= r["length_m"] < hi]
        if not subset:
            continue
        entry = {"bucket_m": [lo, None if hi > 1e8 else hi], "count": len(subset)}
        for count in CONTROL_COUNTS:
            errors_scene = np.array([r["err_%d" % count] for r in subset])
            entry["max_err_m_%d" % count] = float(errors_scene.max() * transform.scale)
            entry["mean_err_m_%d" % count] = float(errors_scene.mean() * transform.scale)
        table.append(entry)

    os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "reports"), exist_ok=True)
    out = {"control_counts": list(CONTROL_COUNTS), "rows": rows, "buckets": table,
           "scene_scale_m": transform.scale}
    path = os.path.join(PROJECT_ROOT, "outputs", "reports", "bspline_fit.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    lines = ["| path length (m) | n | max err C=32 | max err C=48 | max err C=64 |"
             " mean err C=32 |", "|---|---|---|---|---|---|"]
    for entry in table:
        hi = entry["bucket_m"][1]
        label = "%.0f-%s" % (entry["bucket_m"][0], "inf" if hi is None else "%.0f" % hi)
        lines.append("| %s | %d | %.4f | %.4f | %.4f | %.4f |"
                     % (label, entry["count"], entry["max_err_m_32"],
                        entry["max_err_m_48"], entry["max_err_m_64"],
                        entry["mean_err_m_32"]))
    md = "\n".join(lines)
    with open(path.replace(".json", ".md"), "w", encoding="utf-8") as fh:
        fh.write("# B-spline reconstruction error (metres)\n\n" + md + "\n")
    print(md)
    print("written:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
