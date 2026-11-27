"""Turn ``compare_safety_ablation.py`` reports into markdown tables + sign tests.

Usage::

    python scripts/report_ablation.py outputs/reports/ablation_ep100_1v1.json [...]
    python scripts/report_ablation.py --dir outputs/reports --glob "ablation_*ep100*" --out notes.md

Everything is read back from the JSON payloads (no recomputation), so the tables
always match the reports on disk.  Sign tests are two-sided exact binomials on the
per-sample win/loss counts, ties dropped (the same convention as the earlier
ablation docs).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from math import comb

METRICS = (("collision_rate", "collision frac"), ("chamfer_m", "chamfer"),
           ("control_rmse_m", "control RMSE"), ("length_error", "length error"),
           ("guide_chamfer_to_gt_m", "guide-vs-GT chamfer"))


def sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / 2 ** n)


def _fmt(value, nd: int = 3) -> str:
    return "n/a" if value is None else ("%.*f" % (nd, value))


def table(payload: dict) -> list:
    lines = ["split `%s` / mode `%s` / %d steps / seed %d / %d samples"
             % (payload["split"], payload["safety_mode"], payload["steps"],
                payload["seed"], payload["samples"]), "",
             "| arm | valid | collision-free | collision frac | chamfer (m) | control RMSE (m) | length ratio |",
             "|---|---|---|---|---|---|---|"]
    for a in payload["arms"]:
        lines.append("| `%s` | %.3f | %.3f | %.4f | %.3f | %.3f | %.3f |" % (
            a["arm"], a["valid_rate"], a["collision_free_rate"],
            a["collision_rate"].get("mean", 0.0), a["chamfer_m"].get("mean", 0.0),
            a["control_rmse_m"].get("mean", 0.0), a["length_ratio"].get("mean", 0.0)))
    return lines


def paired_table(payload: dict) -> list:
    arms = payload["arms"]
    records = payload["records"]
    base = arms[0]["arm"]
    lines = ["| pair | metric | better / worse / tie | p (sign test) |",
             "|---|---|---|---|"]
    for other in arms[1:]:
        a = {r["index"]: r for r in records[base]}
        b = {r["index"]: r for r in records[other["arm"]]}
        common = sorted(set(a) & set(b))
        for key, label in METRICS:
            wins = losses = ties = 0
            for idx in common:
                va, vb = a[idx].get(key), b[idx].get(key)
                if va is None or vb is None:
                    continue
                if abs(va - vb) < 1e-12:
                    ties += 1
                elif va < vb:
                    wins += 1
                else:
                    losses += 1
            if wins + losses == 0:
                continue
            p = sign_p(wins, losses)
            mark = "SIGNIFICANT" if p < 0.01 else ("marginal" if p < 0.1 else "n.s.")
            lines.append("| `%s` vs `%s` | %s | %d / %d / %d | %.4f (%s) |"
                         % (base, other["arm"], label, wins, losses, ties, p, mark))
    return lines


def guidance_table(payload: dict) -> list:
    arms = payload["arms"]
    if not any(a.get("guide_chamfer_to_gt_m") for a in arms):
        return []
    lines = ["| arm | guide free rate | guide-vs-GT chamfer | guide-to-final | homotopy witness |",
             "|---|---|---|---|---|"]
    for a in arms:
        lines.append("| `%s` | %.3f | %.3f | %.3f | %s |" % (
            a["arm"], 1.0 - a["guide_collision_rate"].get("mean", 0.0),
            a["guide_chamfer_to_gt_m"].get("mean", 0.0),
            a["guide_chamfer_to_final_m"].get("mean", 0.0),
            _fmt(a.get("homotopy_witness_rate"))))
    return lines


def report(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    out = ["### `%s`" % os.path.relpath(path), ""]
    out += table(payload) + [""]
    if len(payload["arms"]) > 1:
        out += paired_table(payload) + [""]
    extra = guidance_table(payload)
    if extra:
        out += ["guidance quality (only safety-mode runs carry a guide curve)", ""]
        out += extra + [""]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser("markdown tables from ablation reports")
    parser.add_argument("reports", nargs="*")
    parser.add_argument("--dir", default=None)
    parser.add_argument("--glob", default="*.json")
    parser.add_argument("--out", default=None, help="markdown file (default: stdout)")
    opts = parser.parse_args()
    paths = list(opts.reports)
    if opts.dir:
        paths += sorted(glob.glob(os.path.join(opts.dir, opts.glob)))
    if not paths:
        parser.error("give report JSON paths or --dir")
    text = "\n".join(report(path) for path in paths)
    if opts.out:
        with open(opts.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print("written:", opts.out)
    else:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
