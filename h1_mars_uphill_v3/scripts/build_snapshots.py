"""Build the team-standard sand bed snapshot at each uphill judge site.

    run.ps1 scripts\\build_snapshots.py [slope16 slope20 slope22] [--force]

Same recipe as CAPSTONE ``terrain/scripts/build_site_snapshots.sh`` (irregular 0.25 +- 0.12 m,
seed 11, regional material from the USD name, standing robot captured after 5 frames, no settling),
with the 2.0 m bed of the team's current judge (``flat_snap_irregular_s11_w20``) and the robot facing
uphill (``--spawn-yaw-deg``). Output: ``snapshots/uphill_<site>_irregular_s11_w20.pkl`` (git-ignored,
about 5 min per site), plus a ``.json`` with the exact command.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from uphill import paths, sites


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="site names (default: all in configs/sites.json)")
    parser.add_argument("--force", action="store_true", help="rebuild existing snapshots")
    parser.add_argument("--noise-amp", type=float, default=None, help="diagnostic bed: thickness noise [m] (team: 0.12)")
    parser.add_argument("--noise-seed", type=int, default=None, help="diagnostic bed: noise seed (team: 11)")
    parser.add_argument("--tag", default=None, help="file tag of a diagnostic bed (default: the team bed's)")
    args = parser.parse_args()

    cfg = sites.load()
    bed = dict(cfg["bed"])
    if args.noise_amp is not None:
        bed["mpm_depth_noise_amp"] = args.noise_amp
    if args.noise_seed is not None:
        bed["mpm_depth_noise_seed"] = args.noise_seed
    if args.tag is not None:
        bed["tag"] = args.tag
    usd = sites.usd()
    paths.SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    for name in args.names or list(cfg["sites"]):
        site = sites.site(name)
        out = sites.snapshot(name, bed["tag"])
        if out.is_file() and not args.force:
            print(f"[snap] {name}: exists, skip ({out.name})")
            continue
        cmd = [
            sys.executable, str(paths.ROOT / "scripts" / "capstone.py"), "22_escape_env.py",
            "--make-snapshot", "--capture-mode", "settled", "--viewer", "null",
            "--env-usd", str(usd),
            "--spawn-x", str(site["site_xy"][0]), "--spawn-y", str(site["site_xy"][1]),
            "--spawn-yaw-deg", str(site["uphill_yaw_deg"]),
            "--command-x", "0", "--checkpoint", str(paths.H1_PUBLISHED),
            "--mpm-spacing", str(bed["mpm_spacing"]),
            "--mpm-depth", str(bed["mpm_depth"]),
            "--mpm-depth-noise-amp", str(bed["mpm_depth_noise_amp"]),
            "--mpm-depth-noise-seed", str(bed["mpm_depth_noise_seed"]),
            "--mpm-patch-length", str(bed["patch_m"]), "--mpm-patch-width", str(bed["patch_m"]),
            "--mpm-settle-steps", "0", "--snapshot-still", "10.0",
            "--snapshot-hold-seconds", "0.02", "--snapshot-frames", "5",
            "--snapshot-path", str(out),
        ]
        log = out.with_suffix(".log")
        print(f"[snap] {name}: building {out.name} ...", flush=True)
        started = time.time()
        with log.open("w", encoding="utf-8") as f:
            code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        wall = time.time() - started
        ok = code == 0 and out.is_file()
        out.with_suffix(".json").write_text(
            json.dumps({"site": name, **site, "usd": str(usd), "cmd": cmd[2:], "exit": code, "wall_s": wall}, indent=2),
            encoding="utf-8",
        )
        print(f"[snap] {name}: {'ok' if ok else f'FAILED (exit {code}), see {log}'} [{wall:.0f} s]", flush=True)
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
