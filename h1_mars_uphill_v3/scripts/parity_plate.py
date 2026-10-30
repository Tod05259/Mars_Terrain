"""Sand parity with the team: plate presses at the judge's numerics, against the team's fit data.

    run.ps1 scripts\\parity_plate.py [--plates sole_240x120 strip_234x28] [--label name]

Runs CAPSTONE's 09 plate press exactly as 80_soil_calibration_runs.py builds it (C1 group: evaluation
numerics, gusev_center material scales, 0.20 m at 62.5 mm/s plus a 1 s hold on a 0.45 m walled bed),
then reads each curve with 81_fit_soil_law.py's own loader and plateau window, and compares the
plateau and hold forces with the values the team measured (``getup_lab/soil_params_gusev_center.json``
``fit.plateaus``, Newton 1.6.0). The result says whether this PC's MPM sand is the team's sand.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

from uphill import paths


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, paths.SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plates", nargs="+", default=["sole_240x120", "strip_234x28"])
    parser.add_argument("--label", default="plate_parity")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra 09 flags (after the recipe)")
    args = parser.parse_args()

    calib = _load("soil_calibration_runs", "80_soil_calibration_runs.py")
    fit = _load("fit_soil_law", "81_fit_soil_law.py")
    team = json.loads((paths.TERRAIN / "getup_lab" / "soil_params_gusev_center.json").read_text(encoding="utf-8"))
    team_rows = {row["plate_id"]: row for row in team["fit"]["plateaus"]}

    out_dir = paths.RUNS / "parity" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    usd = paths.crop_usd("gusev_spirit_center")
    press_frames = int(round(calib.PRESS_DEPTH_M / (calib.PRESS_SPEED_MPS * calib.FRAME_DT)))
    results = []
    for plate_id in args.plates:
        hl, hw, note = calib.PLATES[plate_id]
        csv = out_dir / f"{plate_id}.csv"
        cmd = [
            sys.executable, str(paths.ROOT / "scripts" / "capstone.py"), "09_mars_mpm_plate_sinkage.py",
            "--env-usd", str(usd), "--viewer", "null",
            *calib.EVAL_NUMERICS, *calib.material_flags(),
            "--plate-half-length", str(hl), "--plate-half-width", str(hw),
            "--penetration-depth", str(calib.PRESS_DEPTH_M),
            "--press-frames", str(press_frames),
            "--num-frames", str(press_frames + calib.HOLD_FRAMES),
            "--output-csv", str(csv),
            *args.extra,
        ]
        log = out_dir / f"{plate_id}.log"
        started = time.time()
        with log.open("w", encoding="utf-8") as f:
            code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        wall = time.time() - started
        if code != 0 or not csv.is_file():
            print(f"[parity] {plate_id}: 09 failed (code {code}), see {log}")
            results.append({"plate_id": plate_id, "error": f"exit {code}", "log": str(log)})
            continue
        curve = fit.load_plate(plate_id, csv, hl, hw, "eval")
        plateau, hold = curve.plateau(), curve.hold_median()
        ref = team_rows.get(plate_id, {})
        row = {
            "plate_id": plate_id,
            "note": note,
            "plateau_n": plateau,
            "hold_n": hold,
            "team_plateau_n": ref.get("force"),
            "team_hold_n": ref.get("hold"),
            "plateau_ratio": plateau / ref["force"] if plateau and ref.get("force") else None,
            "hold_ratio": hold / ref["hold"] if hold and ref.get("hold") else None,
            "force_at_mm": {
                str(mm): float(curve.force[abs(curve.depth - mm / 1000.0).argmin()]) for mm in (10, 25, 50, 100, 150, 200)
            },
            "wall_s": wall,
            "csv": str(csv),
        }
        results.append(row)
        print(
            f"[parity] {plate_id}: plateau {plateau:.1f} N (team {ref.get('force', float('nan')):.1f}, "
            f"x{row['plateau_ratio'] or float('nan'):.2f})  hold {hold:.1f} N (team {ref.get('hold', float('nan')):.1f}, "
            f"x{row['hold_ratio'] or float('nan'):.2f})  [{wall:.0f} s]",
            flush=True,
        )
    import newton
    import warp

    summary = {
        "what": "09 plate press at the judge numerics vs the team's fit.plateaus (Newton 1.6.0)",
        "newton": getattr(newton, "__version__", "?"),
        "warp": warp.config.version,
        "capstone_commit": paths.CAPSTONE_COMMIT,
        "extra_flags": args.extra,
        "results": results,
    }
    (out_dir / "parity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[parity] wrote {out_dir / 'parity.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
