"""Load CAPSTONE's MPM sand environment (26) and its helpers on the team's Newton version.

Import this before anything imports ``newton`` or ``warp``: it puts ``.work/mpm_site`` (Newton
1.6.0, warp 1.17.0, mujoco_warp 3.12.0) first on ``sys.path``, as ``scripts/capstone.py`` does.
"""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

from uphill import paths

if not (paths.MPM_SITE / "newton").is_dir():
    raise SystemExit(f"team Newton not installed in {paths.MPM_SITE}: run scripts\\setup_mpm_env.ps1")
if str(paths.MPM_SITE) not in sys.path:
    sys.path.insert(0, str(paths.MPM_SITE))
if str(paths.SCRIPTS) not in sys.path:
    sys.path.insert(0, str(paths.SCRIPTS))

from uphill import compat  # noqa: E402  (after the path choice)

compat.apply()


def load(name: str, filename: str) -> ModuleType:
    """Import ``terrain/scripts/<filename>`` as module ``name`` (once)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, paths.SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def vec() -> ModuleType:
    """26_escape_env_vec.py"""
    return load("escape_env_vec", "26_escape_env_vec.py")


def record() -> ModuleType:
    """28_record_successes.py (``load_policy``)."""
    return load("record_successes", "28_record_successes.py")


def judge_flags() -> list[str]:
    """The team's judge env flags: ``80_soil_calibration_runs.JUDGE_FLAGS`` (the chain's env_args)."""
    return list(load("soil_calibration_runs", "80_soil_calibration_runs.py").JUDGE_FLAGS)


TYPICAL_SOIL = ["--soil-yield-scale", "2", "--soil-young-scale", "2", "--soil-cohesion-pa", "1000"]
"""TEAM_GUIDE 2026-10-11 09:40: the second test condition S2 "typical soil" (the judge flags carry x1)."""


def override(flags: list[str], name: str, *values: str) -> list[str]:
    """``flags`` with option ``name`` set to ``values`` (replaced in place, or appended)."""
    out = list(flags)
    if name in out:
        i = out.index(name)
        j = i + 1
        while j < len(out) and not out[j].startswith("--"):
            j += 1
        out[i + 1 : j] = list(values)
    else:
        out += [name, *values]
    return out
