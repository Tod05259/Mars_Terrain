# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Construct an entrapment directly, and verify it is a state physics allows.

Walking a robot into the hazard does not produce one. Measured across twenty
configurations of crust depth, crust stiffness, pocket stiffness and radius,
the foot either cannot break in or enters and floats back out within three
seconds, because vertical extraction from regolith costs a few per cent of
body weight against actuators rated at 300 N*m. The moments that do read as
fully buried are collisions -- the robot toppling at 3 m/s, its foot punching
through for a single step.

So the state is built rather than driven into. The robot stands where it
settles, at equilibrium, and material is moved from the surface into the
volume above the trapped foot: the crust caving in on top, which is the Troy
mechanism, with the robot's pose left exactly as the solver produced it.

Constructing a state means it has to be shown to be a state. Two checks:

* **Settling.** Released with no actuation, the burial must persist. A pose
  that unloads itself in three seconds is not an entrapment, and this is the
  check every walked-in candidate failed.
* **Reachability.** The depth and overburden must fall inside the range the
  plate benchmark and the Troy sweep observed, or the state is one the real
  hazard never produces.

Usage
-----
    python terrain/scripts/30_build_buried_state.py --settle-seconds 3.0
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path
from types import ModuleType

import newton.viewer
import numpy as np
import warp as wp


def load_by_path(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(filename)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ESCAPE = load_by_path("escape_env", "22_escape_env.py")
MARS = ESCAPE.MARS


def bury_foot(
    sim, foot_body_id: int, cap_m: float, half_length: float, half_width: float
) -> int:
    """Move surface material into the volume above one foot.

    Particles are taken from the highest part of the bed nearby and packed
    into the column over the foot, which conserves the particle count and
    leaves the rest of the bed as the solve produced it. Nothing is created:
    this is the crust caving in, not sand appearing.

    Args:
        sim: The running H1MarsNewton instance.
        foot_body_id: Body index of the foot to bury.
        cap_m: How far above the foot to fill [m].
        half_length: Footprint half-length [m].
        half_width: Footprint half-width [m].

    Returns:
        Number of particles relocated.
    """
    positions = sim.state_0.particle_q.numpy()
    foot = sim.body_q_torch[foot_body_id].cpu().numpy()

    over_foot = (
        (np.abs(positions[:, 0] - foot[0]) < half_length)
        & (np.abs(positions[:, 1] - foot[1]) < half_width)
        & (positions[:, 2] > foot[2])
        & (positions[:, 2] < foot[2] + cap_m)
    )
    # Donors: the highest material within reach, which is what would slump in.
    reach = (
        (np.abs(positions[:, 0] - foot[0]) < 4.0 * half_length)
        & (np.abs(positions[:, 1] - foot[1]) < 4.0 * half_width)
        & (positions[:, 2] > foot[2] + cap_m)
    )
    donor_index = np.flatnonzero(reach)
    if len(donor_index) == 0:
        return 0
    donor_index = donor_index[np.argsort(-positions[donor_index, 2])]

    # Pack the column on a lattice at the bed's own spacing so the filled
    # volume has the same density as the material around it.
    spacing = float(sim.args.mpm_spacing)
    xs = np.arange(foot[0] - half_length, foot[0] + half_length, spacing)
    ys = np.arange(foot[1] - half_width, foot[1] + half_width, spacing)
    zs = np.arange(foot[2] + 0.5 * spacing, foot[2] + cap_m, spacing)
    lattice = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    want = len(lattice) - int(over_foot.sum())
    if want <= 0:
        return 0
    take = donor_index[: min(want, len(donor_index))]

    free_slots = lattice[: len(take)]
    positions[take] = free_slots.astype(positions.dtype)
    sim.state_0.particle_q.assign(positions)
    # Relocated material starts at rest; carrying its old velocity would give
    # the column a kick it never had.
    velocities = sim.state_0.particle_qd.numpy()
    velocities[take] = 0.0
    sim.state_0.particle_qd.assign(velocities)
    return len(take)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stand-seconds", type=float, default=2.0)
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    parser.add_argument(
        "--cap-m",
        type=float,
        default=0.35,
        help="How far above the foot to fill [m].",
    )
    parser.add_argument(
        "--hold-burial-m",
        type=float,
        default=0.10,
        help="Burial that must survive settling for the state to count.",
    )
    parser.add_argument(
        "--hold-overburden",
        type=float,
        default=0.20,
        help=(
            "Overburden that must survive settling, as a sanity check that "
            "the foot is under material rather than on it. Burial is the "
            "measure that matters: what resists an extraction is the height "
            "of the column above the foot, and the fraction falls simply "
            "because a deeper bed puts more material below the foot. At 0.9 m "
            "a properly buried foot reads about 0.2, where at 0.3 m it read "
            "0.6 for the same column above it."
        ),
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=Path("terrain/output/escape_env/entrapment_snapshot.pkl"),
    )
    cli, passthrough = parser.parse_known_args()

    args = ESCAPE.build_args(["--command-x", "0.0"] + passthrough)
    sim = MARS.H1MarsNewton(newton.viewer.ViewerNull(num_frames=10**9), args)
    env = ESCAPE.EscapeEnv.__new__(ESCAPE.EscapeEnv)
    env.args = args
    env.sim = sim
    env.device = sim.torch_device
    env.action_dim = 19

    dt = 0.02
    stand = int(round(cli.stand_seconds / dt))
    print(f"[build] letting the robot settle on the bed for {cli.stand_seconds} s")
    for _ in range(stand):
        sim.step()

    # Bury the foot that is lower: the one already carrying itself into the bed.
    depths = [ESCAPE._foot_overburden(sim, b)[0] for b in sim.foot_body_ids]
    trapped = int(np.argmax(depths))
    moved = bury_foot(
        sim,
        sim.foot_body_ids[trapped],
        cli.cap_m,
        args.sinkage_foot_half_length,
        args.sinkage_foot_half_width,
    )
    burial, over = ESCAPE._foot_overburden(sim, sim.foot_body_ids[trapped])
    print(
        f"[build] buried foot {trapped}: moved {moved} particles, "
        f"burial {burial * 1000:.0f} mm, overburden {over:.2f}"
    )

    print(f"\n[verify] settling for {cli.settle_seconds} s with no actuation")
    print("  t[s]   burial[mm]  overburden  root speed[m/s]")
    settle = int(round(cli.settle_seconds / dt))
    # Hold the pose the robot settled into rather than commanding zero: a
    # zero target is a different pose, and driving toward it would be the
    # policy acting, which is exactly what this check has to exclude.
    hold = sim.control_torch.clone()
    for step in range(settle):
        sim.control_torch.copy_(hold)
        sim.step()
        if step % 25 == 0 or step == settle - 1:
            burial, over = ESCAPE._foot_overburden(sim, sim.foot_body_ids[trapped])
            speed = float(np.abs(sim.state_0.joint_qd.numpy()[:6]).max())
            print(
                f"  {step * dt:5.2f}  {burial * 1000:10.0f}  {over:10.2f}  "
                f"{speed:14.2f}",
                flush=True,
            )

    burial, over = ESCAPE._foot_overburden(sim, sim.foot_body_ids[trapped])
    held = burial >= cli.hold_burial_m and over >= cli.hold_overburden
    print(
        f"\n[verify] after settling: burial {burial * 1000:.0f} mm "
        f"(needs {cli.hold_burial_m * 1000:.0f}), overburden {over:.2f} "
        f"(needs {cli.hold_overburden:.2f}) -> {'유지' if held else '풀림'}"
    )
    if not held:
        raise SystemExit(
            "The constructed state did not survive settling, so it is not a "
            "state the physics holds. Do not train on it."
        )

    payload = {
        name: array.numpy().copy() for name, array in env._state_arrays().items()
    }
    payload["trapped_index"] = trapped
    payload["reference_q"] = sim.reference_q.numpy().copy()
    path = cli.snapshot_path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    print(f"[build] saved -> {path}")


if __name__ == "__main__":
    main()
