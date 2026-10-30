"""Locations of this project, CAPSTONE and Isaac Lab (set by ``run.ps1``)."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("UPHILL_ROOT", Path(__file__).resolve().parents[1]))


def _config() -> dict:
    for name in ("paths.json", "paths.example.json"):
        f = ROOT / "configs" / name
        if f.is_file():
            return json.loads(f.read_text(encoding="utf-8"))
    return {}


_CFG = _config()
CAPSTONE = Path(os.environ.get("UPHILL_CAPSTONE", _CFG.get("capstone_root", "")))
ISAACLAB = Path(os.environ.get("UPHILL_ISAACLAB", _CFG.get("isaaclab_root", "")))
CAPSTONE_COMMIT = os.environ.get("UPHILL_CAPSTONE_COMMIT", _CFG.get("capstone_commit", ""))

TERRAIN = CAPSTONE / "terrain"
SCRIPTS = TERRAIN / "scripts"
RELEASE = TERRAIN / "release"
CROP30 = TERRAIN / "mars_terrain" / "mars_real_terrain_env" / "usd" / "crop30m"

WORK = ROOT / ".work"
MPM_SITE = WORK / "mpm_site"
"""Team versions for the MPM scripts only (Newton 1.6.0, warp 1.17.0, mujoco_warp 3.12.0); see
``scripts/setup_mpm_env.ps1``. Isaac Lab keeps its own (Kit) packages."""
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
SNAPSHOTS = ROOT / "snapshots"
CHECKPOINTS = ROOT / "checkpoints"

H1_PUBLISHED = WORK / "checkpoints" / "h1_published.pt"
"""NVIDIA's Isaac-Velocity-Rough-H1-v0 checkpoint, which CAPSTONE 07/22 load by default from a git-ignored
path; fetched by ``scripts/setup_mpm_env.ps1``."""
H1_PUBLISHED_SHA256 = "054f5de730d8b2fcb56a602864b5cb5d50074946cfa98278f588ea17f30b66e7"

IT25 = RELEASE / "walk" / "walk_irregular_it25.pt"
FLAT_SNAP = RELEASE / "walk" / "flat_snap_irregular_s11.pkl"
STAND_POSE = RELEASE / "fsm" / "stand_pose_orig_it25_3s.npz"


def crop_usd(prefix: str) -> Path:
    """The 30 m crop env USD whose file name starts with ``prefix``."""
    hits = sorted(CROP30.glob(f"{prefix}*_env.usd"))
    if not hits:
        raise FileNotFoundError(f"no crop30m env USD starting with {prefix!r} in {CROP30}")
    return hits[0]
