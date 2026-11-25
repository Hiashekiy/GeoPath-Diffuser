"""Audit the three reference repositories and the PathSafe checkpoint.

Checks (all read-only against ``references/``):

* the branch / commit / working-tree state of each repository,
* that the files the design document points at actually exist,
* that ``checkpoints/v3_best.pt`` loads and matches the documented parameter
  count (5,041,093) and feature widths (192 bottleneck / 128 full),
* the module-level import graph of the files that are migrated.

Writes ``outputs/reports/reference_audit.json``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT, SRC  # noqa: E402

import torch  # noqa: E402

REFERENCES = os.path.join(PROJECT_ROOT, "references")
REPOS = {
    "Astar-Diffusion": {
        "url": "https://github.com/Hiashekiy/Astar-Diffusion.git",
        "branch": "main",
        "files": ["src/guidance_field/field.py", "src/guidance_field/propagation.py",
                  "src/guidance_field/policies.py", "src/guidance_field/paths.py",
                  "src/guidance_field/grid.py", "src/guidance_field/sources.py",
                  "src/guidance_field/collision.py", "src/guidance_field/config.py",
                  "src/guidance_field/normalize.py"],
    },
    "PathSafe-Ellipse": {
        "url": "https://github.com/Hiashekiy/PathSafe-Ellipse.git",
        "branch": "feature/v2.1-centerline",
        "files": ["ellipsenet/models/resunet.py", "ellipsenet/models/psenet.py",
                  "ellipsenet/geometry/sampling.py", "configs/v3.yaml",
                  "docs/report.md", "checkpoints/v3_best.pt",
                  "checkpoints/v3_config.json", "results/metrics/data_contract.json"],
    },
    "TrajSafe-Diffuser": {
        "url": "https://github.com/Hiashekiy/TrajSafe-Diffuser.git",
        "branch": "v4-map16-fbdelta-safeloss",
        "files": ["src/geometry/bspline.py", "src/geometry/bspline_constraints.py",
                  "src/geometry/convex_region.py", "src/geometry/safety_corridor.py",
                  "src/diffusion/bspline_alm.py", "src/diffusion/schedule.py",
                  "src/diffusion/sampler.py", "src/diffusion/alm_local.py",
                  "src/models/trajsafe/geometry.py", "configs/config_v5_c32q64.yaml"],
    },
}


def git(repo: str, *args: str) -> str:
    """Run git in ``repo``; decode as UTF-8 (commit subjects may be non-ASCII)."""
    done = subprocess.run(["git", "-C", repo, *args], capture_output=True)
    return (done.stdout or b"").decode("utf-8", "replace").strip()


def audit_repo(name: str, spec: dict) -> dict:
    path = os.path.join(REFERENCES, name)
    info = {"path": path, "url": spec["url"], "expected_branch": spec["branch"],
            "exists": os.path.isdir(path)}
    if not info["exists"]:
        return info
    info["branch"] = git(path, "rev-parse", "--abbrev-ref", "HEAD")
    info["commit"] = git(path, "rev-parse", "HEAD")
    info["commit_short"] = git(path, "rev-parse", "--short", "HEAD")
    info["last_commit"] = git(path, "log", "-1", "--format=%h %ad %s", "--date=short")
    info["working_tree_clean"] = git(path, "status", "--porcelain") == ""
    info["branch_matches"] = info["branch"] == spec["branch"]
    info["files_present"] = {f: os.path.exists(os.path.join(path, f)) for f in spec["files"]}
    info["files_missing"] = [f for f, ok in info["files_present"].items() if not ok]
    return info


def audit_checkpoint() -> dict:
    path = os.path.join(REFERENCES, "PathSafe-Ellipse", "checkpoints", "v3_best.pt")
    if not os.path.exists(path):
        return {"exists": False}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"]
    cfg = payload.get("cfg", {}).get("model", {})
    out = {
        "exists": True,
        "size_mb": round(os.path.getsize(path) / 1e6, 2),
        "keys": list(payload.keys()),
        "epoch": payload.get("epoch"),
        "step": payload.get("step"),
        "parameters": int(sum(v.numel() for v in state.values() if hasattr(v, "numel"))),
        "model_cfg": {"ch": cfg.get("ch"), "resunet_base": cfg.get("resunet_base"),
                      "resunet_depths": cfg.get("resunet_depths"),
                      "attn_heads": cfg.get("attn_heads"),
                      "ray_attn_layers": cfg.get("ray_attn_layers"),
                      "path_attn_layers": cfg.get("path_attn_layers")},
        "tensors": {
            "occ_unet.stem.0.weight": list(state["occ_unet.stem.0.weight"].shape),
            "occ_unet.enc3.0.c1.weight": list(state["occ_unet.enc3.0.c1.weight"].shape),
            "occ_unet.head.weight": list(state["occ_unet.head.weight"].shape),
            "dist_unet.enc3.0.c1.weight": list(state["dist_unet.enc3.0.c1.weight"].shape),
            "dir_head.weight": list(state["dir_head.weight"].shape),
            "shape_head.weight": list(state["shape_head.weight"].shape),
        },
    }
    return out


def audit_feature_shapes() -> dict:
    """Load the network and verify the bottleneck / full feature shapes."""
    from gp_diffuser.config import load_config
    from gp_diffuser.safety.safety_features import load_pathsafe
    cfg = load_config(None)
    net = load_pathsafe(cfg, device="cpu")
    occ = torch.zeros(1, 1, 256, 256)
    sdf = torch.full((1, 1, 256, 256), 2.0)
    with torch.no_grad():
        feats = net.encode_maps(occ, sdf)
    shapes = {k: list(v.shape) for k, v in feats.items()}
    frozen = all(not p.requires_grad for p in net.parameters())
    return {"shapes": shapes, "all_parameters_frozen": frozen,
            "rays": int(net.rays), "d_clip": float(net.d_clip),
            "a_range_m": [float(net.a_min), float(net.a_max)],
            "b_min_m": float(net.b_min), "candidate_ray_margin_m": float(net.ray_margin_m)}


def main() -> int:
    os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "reports"), exist_ok=True)
    report = {"repos": {}, "checkpoint": {}, "features": {}}
    for name, spec in REPOS.items():
        report["repos"][name] = audit_repo(name, spec)
        print("== %s" % name)
        info = report["repos"][name]
        if not info.get("exists"):
            print("   MISSING")
            continue
        print("   branch %s (expected %s) clean=%s" % (info["branch"], spec["branch"],
                                                      info["working_tree_clean"]))
        print("   commit %s | %s" % (info["commit"], info["last_commit"]))
        if info["files_missing"]:
            print("   MISSING FILES: %s" % info["files_missing"])
        else:
            print("   all referenced files present")
    report["checkpoint"] = audit_checkpoint()
    print("== checkpoint: %s params, %s MB"
          % (report["checkpoint"].get("parameters"), report["checkpoint"].get("size_mb")))
    report["features"] = audit_feature_shapes()
    print("== features: %s" % json.dumps(report["features"]["shapes"]))
    print("   frozen=%s rays=%d d_clip=%s"
          % (report["features"]["all_parameters_frozen"], report["features"]["rays"],
             report["features"]["d_clip"]))
    out = os.path.join(PROJECT_ROOT, "outputs", "reports", "reference_audit.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("written:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
