"""Run a CAPSTONE ``terrain/scripts`` script unmodified, on the team's Newton version.

    run.ps1 scripts\\capstone.py 51_eval_walk.py --checkpoint ... [script args]
    run.ps1 scripts\\capstone.py --installed-newton 51_eval_walk.py ...   # Kit's Newton 1.2.1 + shims

The MPM scripts use Newton 1.6.0 / warp 1.17.0 / mujoco_warp 3.12.0 from ``.work/mpm_site``
(``scripts/setup_mpm_env.ps1``), the versions the team's judge runs on. ``--installed-newton`` uses
the Kit Python's own Newton with the compatibility shims of :mod:`uphill.compat` instead (kept for
comparison; its robot dynamics differ from the team's, see ``results/status.md``).

The script runs as ``__main__`` from its own location, so its sibling imports work. Relative
paths in its defaults resolve against this project's folder (the working directory), so pass
absolute paths for CAPSTONE inputs.
"""

from __future__ import annotations

import runpy
import sys

from uphill import paths


def main() -> None:
    argv = sys.argv[1:]
    installed = "--installed-newton" in argv
    if installed:
        argv.remove("--installed-newton")
    if not argv:
        raise SystemExit(__doc__)
    if not installed:
        if not (paths.MPM_SITE / "newton").is_dir():
            raise SystemExit(f"team Newton not installed in {paths.MPM_SITE}: run scripts\\setup_mpm_env.ps1")
        sys.path.insert(0, str(paths.MPM_SITE))
    from uphill import compat  # imports newton: after the path choice

    script = paths.SCRIPTS / argv[0]
    if not script.is_file():
        raise SystemExit(f"not found: {script}")
    print(f"[capstone] {script.name}  {', '.join(compat.apply())}", flush=True)
    sys.argv = [str(script)] + argv[1:]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
