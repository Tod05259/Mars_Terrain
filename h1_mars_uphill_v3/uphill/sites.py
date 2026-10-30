"""Judge sites (``configs/sites.json``) and the snapshot file each one uses."""

from __future__ import annotations

import json
from pathlib import Path

from uphill import paths

SITES_FILE = paths.ROOT / "configs" / "sites.json"


def load() -> dict:
    return json.loads(SITES_FILE.read_text(encoding="utf-8"))


def usd() -> Path:
    return paths.crop_usd(load()["terrain_prefix"])


def snapshot(name: str, tag: str | None = None) -> Path:
    """Snapshot of site ``name``: the team bed, or a diagnostic bed built with ``--tag``."""
    return paths.SNAPSHOTS / f"uphill_{name}_{tag or load()['bed']['tag']}.pkl"


def site(name: str) -> dict:
    cfg = load()
    if name not in cfg["sites"]:
        raise KeyError(f"unknown site {name!r}; known: {', '.join(cfg['sites'])}")
    return cfg["sites"][name]
