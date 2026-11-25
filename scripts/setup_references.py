"""Clone the three read-only reference repositories and verify the checkpoint.

Run once after cloning this project (``references/`` is git-ignored):

    python scripts/setup_references.py            # clone what is missing
    python scripts/setup_references.py --check     # verify only, no network

Exit code 0 means the reference environment is complete.  Nothing here writes
into the reference repositories: they are cloned at the pinned branch and left
untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT  # noqa: E402

REFERENCES = os.path.join(PROJECT_ROOT, "references")

REPOS = (
    ("Astar-Diffusion", "https://github.com/Hiashekiy/Astar-Diffusion.git", "main"),
    ("PathSafe-Ellipse", "https://github.com/Hiashekiy/PathSafe-Ellipse.git",
     "feature/v2.1-centerline"),
    ("TrajSafe-Diffuser", "https://github.com/Hiashekiy/TrajSafe-Diffuser.git",
     "v4-map16-fbdelta-safeloss"),
)
CHECKPOINT = os.path.join(REFERENCES, "PathSafe-Ellipse", "checkpoints", "v3_best.pt")
EXPECTED_PARAMS = 5041093


def run(cmd):
    done = subprocess.run(cmd, capture_output=True)
    return done.returncode, (done.stdout or b"").decode("utf-8", "replace"), \
        (done.stderr or b"").decode("utf-8", "replace")


def sha256(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def clone_missing(verbose: bool = True) -> int:
    os.makedirs(REFERENCES, exist_ok=True)
    failures = 0
    for name, url, branch in REPOS:
        dest = os.path.join(REFERENCES, name)
        if os.path.isdir(os.path.join(dest, ".git")):
            if verbose:
                print("[skip] %-18s already cloned" % name)
            continue
        print("[clone] %-17s %s (%s)" % (name, url, branch))
        code, out, err = run(["git", "clone", "--branch", branch,
                              "--single-branch", url, dest])
        if code != 0:
            failures += 1
            print("        FAILED: %s" % (err.strip().splitlines()[-1] if err else code))
        elif verbose:
            print("        ok")
    return failures


def verify() -> int:
    problems = []
    for name, _url, branch in REPOS:
        dest = os.path.join(REFERENCES, name)
        if not os.path.isdir(os.path.join(dest, ".git")):
            problems.append("%s: not cloned" % name)
            continue
        code, out, _ = run(["git", "-C", dest, "rev-parse", "--abbrev-ref", "HEAD"])
        head = out.strip()
        if code != 0 or head != branch:
            problems.append("%s: branch %s != %s" % (name, head or "?", branch))
        code, out, _ = run(["git", "-C", dest, "status", "--porcelain"])
        if out.strip():
            problems.append("%s: working tree is dirty (must stay read-only)" % name)
        code, out, _ = run(["git", "-C", dest, "rev-parse", "--short", "HEAD"])
        print("[ok]    %-18s branch %-28s commit %s" % (name, branch, out.strip()))

    if not os.path.exists(CHECKPOINT):
        problems.append("PathSafe checkpoint missing: %s" % CHECKPOINT)
    else:
        size_mb = os.path.getsize(CHECKPOINT) / 1e6
        print("[ok]    checkpoint %.2f MB sha256[:16]=%s"
              % (size_mb, sha256(CHECKPOINT)[:16]))
        try:
            import torch
            payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
            params = int(sum(v.numel() for v in payload["model"].values()
                             if hasattr(v, "numel")))
            if params != EXPECTED_PARAMS:
                problems.append("checkpoint has %d parameters, expected %d"
                                % (params, EXPECTED_PARAMS))
            else:
                print("[ok]    checkpoint parameters %d" % params)
        except Exception as error:                       # noqa: BLE001
            problems.append("checkpoint unreadable: %s" % error)

    if problems:
        print("\nINCOMPLETE REFERENCE ENVIRONMENT:")
        for item in problems:
            print("  - %s" % item)
        return 1
    print("\nreference environment OK (%s)" % REFERENCES)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify only, never touch the network")
    parser.add_argument("--quiet", action="store_true")
    opts = parser.parse_args()
    failures = 0
    if not opts.check:
        failures = clone_missing(verbose=not opts.quiet)
    code = verify()
    return 1 if (failures or code) else 0


if __name__ == "__main__":
    raise SystemExit(main())
