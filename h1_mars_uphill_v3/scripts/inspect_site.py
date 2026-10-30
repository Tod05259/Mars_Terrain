"""Print the local slope and fall-line direction around a judge site, on a grid in the climb frame.

    run.ps1 scripts\\inspect_site.py slope20 [--half 4.5] [--step 1.0]

For each point (s along the start fall line, t across it) the DTM is sampled the way the sand env does
and a plane is fitted over a 3 m square: tilt [deg] and the uphill direction relative to the start fall
line [deg]. A fall line that turns away from the start direction puts a robot that holds the start
heading on a cross slope.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys

import numpy as np

from uphill import paths, sites


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site")
    parser.add_argument("--half", type=float, default=4.5)
    parser.add_argument("--step", type=float, default=1.0)
    args = parser.parse_args()
    sys.path.insert(0, str(paths.MPM_SITE))
    spec = importlib.util.spec_from_file_location("h1_mars_newton", paths.SCRIPTS / "07_h1_mars_newton.py")
    mars = importlib.util.module_from_spec(spec)
    sys.modules["h1_mars_newton"] = mars
    spec.loader.exec_module(mars)
    site = sites.site(args.site)
    height, dx, dy = mars.load_terrain_grid(sites.usd())
    x0, y0 = site["site_xy"]
    yaw0 = math.radians(site["uphill_yaw_deg"])
    u = np.array([math.cos(yaw0), math.sin(yaw0)])
    v = np.array([-u[1], u[0]])
    probe = np.linspace(-1.5, 1.5, 7)
    px, py = (g.ravel() for g in np.meshgrid(probe, probe))
    design = np.column_stack((px, py, np.ones_like(px)))
    ts = np.arange(-args.half, args.half + 1e-6, args.step)
    ss = np.arange(-1.0, args.half + 1e-6, args.step)
    print(f"{args.site}: site ({x0}, {y0}), start uphill yaw {site['uphill_yaw_deg']} deg; cells: tilt / fall-line turn [deg]")
    print("  s\\t " + " ".join(f"{t:+9.1f}" for t in ts))
    for s in ss[::-1]:
        cells = []
        for t in ts:
            c = np.array([x0, y0]) + s * u + t * v
            z = mars.sample_height_numpy(height, dx, dy, c[0] + px, c[1] + py)
            (a, b, _), *_ = np.linalg.lstsq(design, z, rcond=None)
            tilt = math.degrees(math.atan(math.hypot(a, b)))
            turn = math.degrees(math.remainder(math.atan2(b, a) - yaw0, 2 * math.pi))
            cells.append(f"{tilt:4.1f}/{turn:+4.0f}")
        print(f"{s:+5.1f} " + " ".join(f"{c:>9s}" for c in cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
