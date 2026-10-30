"""Pick uphill judge sites near target slopes on the seven 30 m Mars crops.

    run.ps1 scripts\\survey_slopes.py [--targets 16.8 20 22.5] [--output configs\\sites_survey.json]

For each candidate site (0.5 m grid), the DTM is sampled the way the sand env does (07
``sample_height_numpy``: x from the column, y flipped from the row). A plane fitted over a 3 m
square gives the site tilt and the uphill direction, as 26's ``_compute_downhill`` probe does. The
climb corridor is then checked along that direction -- from ``--runway-m`` below the site (room to
slide back, 26 ``--arrest-runway-m``) to ``--climb-m`` + 1 m above it (26 ``--descend-target-m``),
``--corridor-half-w`` wide: its plane tilt, its out-of-plane relief, and the tilt of every 1 m
segment along it, so a site whose slope flattens or steepens part-way up is rejected. Sites must
keep the env patch (``--env-pitch``) inside the crop.

Ranking prefers the closest tilt, then the flattest corridor; Gusev crops are listed first because
their sand material is calibrated on two axes (TEAM_GUIDE §3: the others are provisional).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys

import numpy as np

from uphill import paths

CALIBRATED = ("gusev_spirit_center", "gusev_spirit_rough")


def _mars():
    sys.path.insert(0, str(paths.MPM_SITE))
    spec = importlib.util.spec_from_file_location("h1_mars_newton", paths.SCRIPTS / "07_h1_mars_newton.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["h1_mars_newton"] = module
    spec.loader.exec_module(module)
    return module


def fit_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float, float]:
    """Least-squares plane z = a x + b y + c: tilt [deg], uphill yaw [rad], relief (peak-to-peak residual) [m]."""
    design = np.column_stack((x, y, np.ones_like(x)))
    (a, b, c), *_ = np.linalg.lstsq(design, z, rcond=None)
    residual = z - design @ np.array([a, b, c])
    return math.degrees(math.atan(math.hypot(a, b))), math.atan2(b, a), float(np.ptp(residual))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=float, nargs="+", default=[16.8, 20.0, 22.5])
    parser.add_argument("--tolerance", type=float, default=1.5, help="site tilt within target +- this [deg]")
    parser.add_argument("--segment-tolerance", type=float, default=3.5, help="1 m segment tilt within target +- [deg]")
    parser.add_argument("--max-relief", type=float, default=0.15, help="corridor out-of-plane relief [m]")
    parser.add_argument("--climb-m", type=float, default=3.0)
    parser.add_argument("--runway-m", type=float, default=1.0)
    parser.add_argument("--corridor-half-w", type=float, default=0.75)
    parser.add_argument("--env-pitch", type=float, default=8.0)
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--output", default=str(paths.ROOT / "configs" / "sites_survey.json"))
    args = parser.parse_args()

    mars = _mars()
    probe = np.linspace(-1.5, 1.5, 7)
    px, py = (g.ravel() for g in np.meshgrid(probe, probe))
    s_min, s_max = -args.runway_m, args.climb_m + 1.0
    along = np.arange(s_min, s_max + 1e-6, 0.25)
    across = np.linspace(-args.corridor_half_w, args.corridor_half_w, 5)
    cs, ct = (g.ravel() for g in np.meshgrid(along, across))

    found: dict[str, list[dict]] = {f"{t:g}": [] for t in args.targets}
    for usd in sorted(paths.CROP30.glob("*_env.usd")):
        height, dx, dy = mars.load_terrain_grid(usd)
        half_x = 0.5 * (height.shape[1] - 1) * dx
        half_y = 0.5 * (height.shape[0] - 1) * dy
        lim_x = half_x - 0.5 * args.env_pitch - 0.5
        lim_y = half_y - 0.5 * args.env_pitch - 0.5
        key = usd.stem.split("_1024px")[0]
        n = 0
        for x0 in np.arange(-lim_x, lim_x + 1e-6, args.step):
            for y0 in np.arange(-lim_y, lim_y + 1e-6, args.step):
                z = mars.sample_height_numpy(height, dx, dy, x0 + px, y0 + py)
                tilt, yaw, _ = fit_plane(px, py, z)
                target = next((t for t in args.targets if abs(tilt - t) <= args.tolerance), None)
                if target is None:
                    continue
                ux, uy = math.cos(yaw), math.sin(yaw)
                wx = x0 + cs * ux - ct * uy
                wy = y0 + cs * uy + ct * ux
                if np.abs(wx).max() > half_x - 0.5 or np.abs(wy).max() > half_y - 0.5:
                    continue
                wz = mars.sample_height_numpy(height, dx, dy, wx, wy)
                c_tilt, c_yaw, relief = fit_plane(wx - x0, wy - y0, wz)
                if relief > args.max_relief:
                    continue
                seg = []
                for lo in np.arange(s_min, s_max - 1.0 + 1e-6, 1.0):
                    m = (cs >= lo) & (cs <= lo + 1.0)
                    # Rise along the climb direction over the segment, as an angle.
                    rise = np.polyfit(cs[m], wz[m], 1)[0]
                    seg.append(math.degrees(math.atan(rise)))
                if min(seg) < target - args.segment_tolerance or max(seg) > target + args.segment_tolerance:
                    continue
                if min(seg) <= 0.0:
                    continue
                n += 1
                found[f"{target:g}"].append({
                    "terrain": key,
                    "usd": usd.name,
                    "calibrated_2axis": key.startswith(CALIBRATED),
                    "site_xy": [round(float(x0), 2), round(float(y0), 2)],
                    "tilt_deg": round(tilt, 2),
                    "corridor_tilt_deg": round(c_tilt, 2),
                    "segment_tilt_deg": [round(s, 1) for s in seg],
                    "relief_m": round(relief, 3),
                    "uphill_yaw_deg": round(math.degrees(yaw), 1),
                    "corridor_yaw_offset_deg": round(math.degrees(math.atan2(math.sin(c_yaw - yaw), math.cos(c_yaw - yaw))), 1),
                    "site_z_m": round(float(mars.sample_height_numpy(height, dx, dy, np.array([x0]), np.array([y0]))[0]), 3),
                })
        print(f"[survey] {key}: grid {height.shape} dx {dx:.3f} m, {n} sites in the target bands", flush=True)

    best: dict[str, list[dict]] = {}
    for t, rows in found.items():
        target = float(t)
        rows.sort(key=lambda r: (not r["calibrated_2axis"], abs(r["tilt_deg"] - target) + 2.0 * r["relief_m"]))
        # Keep sites at least 2 m apart, so the list is not one face sampled many times.
        picked: list[dict] = []
        for r in rows:
            if all(math.dist(r["site_xy"], p["site_xy"]) >= 2.0 or r["terrain"] != p["terrain"] for p in picked):
                picked.append(r)
            if len(picked) >= args.top * 3:
                break
        best[t] = picked
        print(f"\n== target {target:.1f} deg: {len(rows)} sites")
        for r in picked[: args.top * 2]:
            print(
                f"  {r['terrain'][:34]:<34} xy ({r['site_xy'][0]:+6.2f},{r['site_xy'][1]:+6.2f})  tilt {r['tilt_deg']:5.2f}"
                f"  corridor {r['corridor_tilt_deg']:5.2f}  segs {r['segment_tilt_deg']}  relief {r['relief_m']:.3f}"
                f"  uphill yaw {r['uphill_yaw_deg']:+7.1f}  {'2-axis' if r['calibrated_2axis'] else 'provisional'}"
            )
    out = {"args": vars(args), "candidates": best}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[survey] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
