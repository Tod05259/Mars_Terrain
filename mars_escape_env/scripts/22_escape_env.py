# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Escape-training environment: H1 with a foot trapped in a Troy-type pocket.

Wraps the proven physics runtime of ``07_h1_mars_newton.py`` as a gym-style
environment for RSL-RL. The wrapped class keeps everything that was validated —
two-way MPM coupling, regional calibration, the moving soil window, the buried
hazard — and replaces only the control flow:

* the pretrained walking policy is no longer called inside the loop; actions
  come from outside through :meth:`step`;
* episodes start from a cached *entrapment snapshot*: the robot walks into the
  pocket once under the pretrained policy, the full state (joints + particles +
  MPM internals) is saved at maximum sinkage, and every reset restores it.
  Spawning a robot directly in a buried pose injects contact impulses that fold
  the legs (measured on the G1 prototype), while restoring a settled state is
  impulse-free;
* a reward for escaping: raising the trapped foot back to the surface while
  keeping the support foot from sinking — the anchoring budget measured in
  EXTRACTION_HANDOFF 7.5r is the physics this trades against.

Usage
-----
    # build (or rebuild) the entrapment snapshot, then run random actions
    python terrain/scripts/22_escape_env.py --make-snapshot
    python terrain/scripts/22_escape_env.py --episodes 3
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import warp as wp

import newton
import newton.examples
import newton.viewer


def load_mars_module() -> ModuleType:
    """Load the H1 Mars runtime by path (its filename starts with a digit)."""
    script_path = Path(__file__).with_name("07_h1_mars_newton.py")
    spec = importlib.util.spec_from_file_location("h1_mars_newton", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load the H1 Mars runtime: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MARS = load_mars_module()

DEFAULT_ENV_USD = (
    Path("terrain/mars_terrain/mars_real_terrain_env/usd/crop30m")
    / "gusev_spirit_center_columbia_hills_1024px_stride2_crop30m_r530c380_env.usd"
)
_WORKING_SNAPSHOT = Path("terrain/output/escape_env/entrapment_snapshot.pkl")
_RELEASE_SNAPSHOT = Path("terrain/release/entrapment_snapshot.pkl")
# terrain/output/ is not in the repository, so a fresh clone only has the
# released copy; prefer the working one when it exists.
DEFAULT_SNAPSHOT = (
    _WORKING_SNAPSHOT if _WORKING_SNAPSHOT.is_file() else _RELEASE_SNAPSHOT
)

# Episode termination thresholds.
SUPPORT_SINK_LIMIT_M = 0.20   # support-foot depression that counts as failure
TORSO_HEIGHT_MIN_M = 0.55     # below this above local soil top = fallen
UPRIGHT_GRAVITY_MAX = 0.7     # projected gravity along the body up-axis (-1 upright)
ESCAPE_CLEARANCE_M = 0.05     # trapped sole this close to the surface = "foot free"
# Success is leaving the hazard: torso this far from the pocket centre, upright.
EXIT_RADIUS_M = 1.0


class EscapeEnv:
    """Single-environment escape task on the validated H1+MPM runtime."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        # Headless by default; --viewer gl opens the live Newton viewer so a
        # trained policy can be watched.
        mode = getattr(args, "viewer", "null")
        if mode == "gl":
            viewer = newton.viewer.ViewerGL()
        elif mode == "usd":
            # Offline recording: the sim runs at whatever speed it runs, the
            # USD gets clean time samples, and playback in Isaac Sim is smooth
            # at the recorded rate regardless of how choppy the live sim was.
            viewer = newton.viewer.ViewerUSD(
                output_path=str(getattr(args, "usd_output", "terrain/output/escape_env/replay.usd")),
                fps=int(getattr(args, "usd_fps", 15)),
                num_frames=10**9,
            )
        else:
            viewer = newton.viewer.ViewerNull(num_frames=10**9)
        self.render_enabled = mode in ("gl", "usd")
        self.sim = MARS.H1MarsNewton(viewer, args)
        self.device = self.sim.torch_device
        self.action_dim = 19
        self.obs_dim = 256
        self.max_episode_steps = int(args.episode_seconds / MARS.CONTROL_DT)

        self._snapshot = None
        self._step_count = 0
        # Which foot is trapped is decided when the snapshot is made.
        self.trapped_index = 0
        self.support_index = 1

    # ------------------------------------------------------------------
    # Snapshot machinery
    # ------------------------------------------------------------------
    def _state_arrays(self) -> dict[str, wp.array]:
        """Every array that defines the dynamic state of robot and soil."""
        state = self.sim.state_0
        mpm = state.mpm
        arrays = {
            "joint_q": state.joint_q,
            "joint_qd": state.joint_qd,
            "body_q": state.body_q,
            "body_qd": state.body_qd,
            "particle_q": state.particle_q,
            "particle_qd": state.particle_qd,
            "mpm_qd_grad": mpm.particle_qd_grad,
            "mpm_elastic_strain": mpm.particle_elastic_strain,
            "mpm_Jp": mpm.particle_Jp,
            "mpm_stress": mpm.particle_stress,
            "mpm_transform": mpm.particle_transform,
        }
        return arrays

    def save_snapshot(self, path: Path) -> None:
        """Serialise the current dynamic state to disk."""
        payload = {
            name: array.numpy().copy() for name, array in self._state_arrays().items()
        }
        payload["trapped_index"] = self.trapped_index
        payload["reference_q"] = self.sim.reference_q.numpy().copy()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(payload, stream)
        print(f"[EscapeEnv] snapshot saved: {path}")

    def load_snapshot(self, path: Path) -> None:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        self.trapped_index = int(payload.pop("trapped_index"))
        self.support_index = 1 - self.trapped_index
        self._snapshot = payload

    def _restore_snapshot(self) -> None:
        arrays = self._state_arrays()
        for name, array in arrays.items():
            array.assign(self._snapshot[name])
        self.sim.reference_q.assign(self._snapshot["reference_q"])
        # Mirror into state_1 so the double-buffered swap starts consistent.
        self.sim.state_1.joint_q.assign(self._snapshot["joint_q"])
        self.sim.state_1.joint_qd.assign(self._snapshot["joint_qd"])
        self.sim.state_1.body_q.assign(self._snapshot["body_q"])
        self.sim.state_1.body_qd.assign(self._snapshot["body_qd"])
        newton.eval_fk(
            self.sim.model,
            self.sim.state_0.joint_q,
            self.sim.state_0.joint_qd,
            self.sim.state_0,
        )
        self.sim.last_action = torch.zeros(
            (1, self.action_dim), device=self.device
        )

    # ------------------------------------------------------------------
    # Task geometry helpers
    # ------------------------------------------------------------------
    def _foot_heights(self) -> tuple[float, float, float]:
        """Trapped-foot z, support-foot z, and local soil surface z [m]."""
        body_q = self.sim.state_0.body_q.numpy()
        feet = [body_q[i, 2] for i in self.sim.foot_body_ids]
        particles = self.sim.state_0.particle_q.numpy()
        soil_top = float(np.percentile(particles[:, 2], 95))
        return feet[self.trapped_index], feet[self.support_index], soil_top

    def _support_sinkage_m(self) -> float:
        sink = self.sim._soil_sinkage_under_feet()
        return float(sink[self.support_index]) / 1000.0

    def _surface_under_torso(self) -> float:
        """Sand-surface height directly under the torso [m].

        The moving soil window's 95th percentile sits on the window's uphill
        edge, which on this sloped site rises faster than the robot: a normal
        uphill walk then reads as a fall. Judging uprightness against the
        surface under the torso removes that bias (26_escape_env_vec.py
        carries the same fix).
        """
        body_q = self.sim.state_0.body_q.numpy()
        torso_xy = body_q[self.sim.torso_body_id, :2]
        height = self.sim._sample_height_torch(
            torch.tensor([float(torso_xy[0])], device=self.device),
            torch.tensor([float(torso_xy[1])], device=self.device),
        )
        return float(height[0])

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------
    def reset(self) -> torch.Tensor:
        if self._snapshot is None:
            raise RuntimeError(
                "No entrapment snapshot loaded. Run with --make-snapshot first."
            )
        self._restore_snapshot()
        self._step_count = 0
        self._clear_streak = 0
        self._initial_trapped_z, _, self._initial_soil_top = self._foot_heights()
        # Level-match the under-torso upright criterion to the window
        # percentile at the reset pose, so only its slope behaviour differs.
        self._upright_offset = self._initial_soil_top - self._surface_under_torso()
        # Hazard centre: where the trapped foot is at reset. Escape means the
        # robot walks out of this zone, not merely lifts the foot.
        body_q = self.sim.state_0.body_q.numpy()
        self._pocket_xy = body_q[self.sim.foot_body_ids[self.trapped_index], :2].copy()
        torso_xy = body_q[self.sim.torso_body_id, :2]
        self._prev_exit_distance = float(np.hypot(*(torso_xy - self._pocket_xy)))
        return self.sim._observation().detach()

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, float, bool, dict]:
        # Actions are policy-frame joint offsets, exactly like the pretrained
        # policy's output, so a policy trained here stays compatible.
        self.sim.last_action = action.reshape(1, -1).to(self.device)
        self.sim.control_torch.zero_()
        targets = self.sim.default_joint_pos_newton[0].clone()
        targets[self.sim.policy_to_newton_torch] = (
            self.sim.default_joint_pos_policy[0] + 0.5 * self.sim.last_action[0]
        )
        self.sim.control_torch[6:] = targets

        self.sim._simulate_robot()
        self.sim._simulate_mpm()
        self.sim.sim_step += 1
        # 07's own step() advances sim_time; this bypasses it, and the USD
        # recorder stamps time samples with sim_time -- without this line
        # every frame lands on timecode 0 and the recording is static.
        self.sim.sim_time += MARS.CONTROL_DT
        self._step_count += 1

        if self.render_enabled:
            self.sim.render()
        observation = self.sim._observation().detach()
        reward, terminated, info = self._reward_and_termination()
        timeout = self._step_count >= self.max_episode_steps
        return observation, reward, terminated or timeout, info

    def _reward_and_termination(self) -> tuple[float, bool, dict]:
        trapped_z, support_z, soil_top = self._foot_heights()
        support_sink = self._support_sinkage_m()
        torso_z = float(self.sim.state_0.body_q.numpy()[self.sim.torso_body_id, 2])

        # Progress: how much of the initial burial has been recovered.
        burial0 = max(self._initial_soil_top - self._initial_trapped_z, 1e-3)
        burial = max(soil_top - trapped_z, 0.0)
        progress = (burial0 - burial) / burial0

        # The anchoring trade-off: lifting is only worth it if the support
        # foot is not being driven into the ground (7.5r: budget is -6 N).
        reward = (
            1.0 * progress
            - 5.0 * support_sink
            - 0.05 * float(self.sim.last_action.square().mean())
        )
        # Success is walking out of the hazard zone. Distance progress is the
        # primary shaping term; freeing the foot remains rewarded because the
        # robot cannot move away while it is anchored.
        torso_xy = self.sim.state_0.body_q.numpy()[self.sim.torso_body_id, :2]
        exit_distance = float(np.hypot(*(torso_xy - self._pocket_xy)))
        distance_progress = exit_distance - self._prev_exit_distance
        self._prev_exit_distance = exit_distance
        # Capped at 1 m/s so a ballistic leap earns no more than walking.
        reward += 5.0 * float(np.clip(distance_progress, -0.02, 0.02))
        upright_floor = (
            self._surface_under_torso() + self._upright_offset + TORSO_HEIGHT_MIN_M
        )
        if torso_z >= upright_floor:
            reward += 0.01

        # Height alone is not uprightness: a tumbling robot clears a height
        # test while its torso is high mid-flip. Projected gravity along the
        # body's own up-axis is -1 upright and 0 on its side.
        projected_gravity = MARS.quaternion_rotate_inverse(
            self.sim.joint_q_torch[3:7].unsqueeze(0), self.sim.gravity_direction
        )
        fallen = (
            torso_z < upright_floor
            or float(projected_gravity[0, 2]) > -UPRIGHT_GRAVITY_MAX
        )
        # Anchoring only counts as failure inside the hazard: outside it the
        # robot has already walked out, and the implicit-MPM long-hold support
        # decay grows the sink measure under a robot that is merely standing.
        anchored_out = (
            support_sink > SUPPORT_SINK_LIMIT_M and exit_distance < EXIT_RADIUS_M
        )

        # Escaped = out of the zone and upright, held briefly so a fall on the
        # last step does not count.
        out_now = exit_distance >= EXIT_RADIUS_M and not fallen
        self._clear_streak = self._clear_streak + 1 if out_now else 0
        # 1.5 s upright hold: a 0.3 s hold let a leap-and-crash bank the
        # escape bonus mid-flight; 1.5 s requires a landed, stable exit.
        escaped = self._clear_streak >= int(1.5 / MARS.CONTROL_DT)
        if out_now:
            # Dense bridge toward the sparse escape bonus during the hold.
            reward += 0.3
        if escaped:
            reward += 10.0
        if fallen or anchored_out:
            # Falling must never pay: strictly worse than standing still.
            reward -= 20.0

        info = {
            "exit_distance_m": exit_distance,
            "burial_mm": 1000.0 * burial,
            "support_sink_mm": 1000.0 * support_sink,
            "escaped": escaped,
            "fallen": fallen,
            "anchored_out": anchored_out,
        }
        return reward, bool(escaped or fallen or anchored_out), info


# ----------------------------------------------------------------------
# Snapshot construction: walk into the pocket under the pretrained policy.
# ----------------------------------------------------------------------
def _foot_overburden(sim, foot_body_id: int) -> tuple[float, float]:
    """Burial depth [m] and overburden fraction for one foot.

    Returns the height of the material surface directly under the foot minus
    the foot's own height, and the share of the material in the footprint
    that lies above the foot.

    Surface depression is not the same thing and cannot stand in for it. A
    foot can press a 220 mm dimple into the bed and still be sitting on top
    of it with nothing above it to push aside, which is exactly the state the
    previous snapshot captured: 220 mm of sinkage, 12 mm of burial, six per
    cent overburden. What resists an extraction is the material on top.
    """
    import torch

    current = wp.to_torch(sim.state_0.particle_q)
    foot_q = sim.body_q_torch[foot_body_id]
    near = (
        ((current[:, 0] - foot_q[0]).abs() <= sim.args.sinkage_foot_half_length)
        & ((current[:, 1] - foot_q[1]).abs() <= sim.args.sinkage_foot_half_width)
    )
    if not bool(near.any()):
        return 0.0, 0.0
    column = current[near]
    # Ignore material thrown clear of the bed. A 95th percentile over the
    # column reported 905 mm of burial on a 400 mm bed while the foot was
    # punching in, because particles kicked into the air are still inside the
    # footprint. Only material within a bed's depth of the foot can be lying
    # on it.
    lid = float(foot_q[2]) + float(sim.args.mpm_depth)
    settled = column[column[:, 2] <= lid]
    if len(settled) == 0:
        return 0.0, 0.0
    top = float(torch.quantile(settled[:, 2], 0.95))
    above = float((settled[:, 2] > foot_q[2]).sum()) / float(len(settled))
    return top - float(foot_q[2]), above


def make_snapshot(args: argparse.Namespace) -> None:
    """Walk H1 into the pocket and cache the deepest-sinkage state."""
    viewer = newton.viewer.ViewerNull(num_frames=10**9)
    sim = MARS.H1MarsNewton(viewer, args)
    env = EscapeEnv.__new__(EscapeEnv)  # reuse helpers without re-building
    env.args = args
    env.sim = sim
    env.device = sim.torch_device
    env.action_dim = 19

    best = None
    best_burial = 0.0
    best_over = 0.0
    held = 0
    peak_burial = -9.9
    peak_over = 0.0
    peak_held = 0
    need = int(round(args.snapshot_hold_seconds / 0.02))
    for frame in range(args.snapshot_frames):
        sim.step()
        measured = [_foot_overburden(sim, b) for b in sim.foot_body_ids]
        caught = [
            burial >= args.snapshot_burial_m and over >= args.snapshot_overburden
            for burial, over in measured
        ]
        # A foot punching into the bed passes the depth and overburden test at
        # the bottom of its stroke and floats back out afterwards: released
        # from that state with no action at all, the foot rises 209 mm and is
        # free within three seconds. That is a transient penetration, not an
        # entrapment, and capturing the peak captured exactly it.
        #
        # The state has to persist, and the robot has to have stopped moving
        # into it, before it counts.
        settled = float(np.abs(sim.state_0.joint_qd.numpy()[:6]).max()) < args.snapshot_still
        # Track the best seen even when nothing holds, so a failed
        # configuration still says how close it came and in which direction.
        for burial, over in measured:
            if burial > peak_burial:
                peak_burial = burial
            if over > peak_over:
                peak_over = over
        if any(caught) and settled:
            held += 1
            peak_held = max(peak_held, held)
            if held >= need:
                index = 0 if caught[0] else 1
                best_burial, best_over = measured[index]
                env.trapped_index = index
                best = {
                    name: array.numpy().copy()
                    for name, array in env._state_arrays().items()
                }
                break
        else:
            held = 0
        if frame % 50 == 0:
            print(
                f"  frame {frame:4d}  burial = {measured[0][0]*1000:.0f}/"
                f"{measured[1][0]*1000:.0f} mm  overburden = {measured[0][1]:.2f}/"
                f"{measured[1][1]:.2f}  held {held}/{need}",
                flush=True,
            )

    if best is None:
        raise RuntimeError(
            f"RESULT held={peak_held}/{need} "
            f"peak_burial_mm={peak_burial * 1000:.0f} "
            f"peak_overburden={peak_over:.2f} -- no sustained entrapment"
        )
    best["trapped_index"] = env.trapped_index
    best["reference_q"] = sim.reference_q.numpy().copy()
    path = args.snapshot_path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(best, stream)
    print(
        f"[EscapeEnv] RESULT HELD burial_mm={best_burial * 1000:.0f} "
        f"overburden={best_over:.2f} foot={env.trapped_index} -> {path}"
    )


def build_args(extra: list[str] | None = None) -> argparse.Namespace:
    parser = MARS.create_parser()
    parser.add_argument("--make-snapshot", action="store_true")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-seconds", type=float, default=8.0)
    parser.add_argument("--snapshot-frames", type=int, default=400)
    parser.add_argument(
        "--snapshot-hold-seconds",
        type=float,
        default=1.0,
        help=(
            "How long the foot must stay buried before the state counts. "
            "Without it, capture lands on the bottom of the punch-in "
            "stroke, and the foot floats free once released."
        ),
    )
    parser.add_argument(
        "--snapshot-still",
        type=float,
        default=0.35,
        help="Largest root velocity that still counts as settled [m/s, rad/s].",
    )
    parser.add_argument(
        "--snapshot-burial-m",
        type=float,
        default=0.10,
        help="Minimum depth of the foot below the material under it [m].",
    )
    parser.add_argument(
        "--snapshot-overburden",
        type=float,
        default=0.50,
        help=(
            "Minimum share of the material in the footprint lying above "
            "the foot. This is what an extraction has to push aside; "
            "surface depression alone does not imply any of it exists. "
            "A foot resting on the bed reads about 0.05."
        ),
    )
    parser.add_argument("--snapshot-path", type=Path, default=DEFAULT_SNAPSHOT)
    defaults = [
        "--viewer", "null",
        "--env-usd", str(DEFAULT_ENV_USD),
        "--command-x", "0.3",
        "--mpm-pocket-radius", "0.3",
        "--mpm-pocket-scale", "0.1",
        # The crust is what holds. A uniformly soft column sinks a foot and
        # lets it straight back out; the sweep's Troy archetype puts a thin
        # stiff layer over the pocket so the foot breaks through and the
        # broken material comes down on it. 25 mm at x3 is the middle of the
        # band the sweep found consistent with Spirit's outcome.
        "--mpm-crust-depth", "0.025",
        "--mpm-crust-scale", "3.0",
        "--mpm-pocket-dx", "0.45",
        "--mpm-pocket-dy", "0.12",
        # 12.5 mm is where the material was calibrated and where spacing
        # convergence was established (1.0 % against the 3.736 kPa design
        # point, 3.1 % against 6.25 mm). At the 25 mm this used to inherit
        # from 07's default, indentation is 27.7 % off and, worse, the
        # extraction resistance that defines this task is a discretisation
        # artifact -- 1821 N against 0.186 N at 12.5 mm. HANDOFF 3.1 says in
        # as many words not to run experiments at 25 mm.
        #
        # Eight times the points fit in the same volume, so the window comes
        # down with the spacing. Escape is scored at a 1 m radius, so 1.2 m
        # of bed is what the task needs.
        "--mpm-spacing", "0.0125",
        # Deep enough for the leg to go in, not just the foot.
        #
        # The overburden on a buried foot weighs about 36 N -- a 0.24 x 0.12 m
        # sole under 0.2 m of regolith at Mars gravity -- against actuators
        # rated at 300 N*m. No arrangement of crust and pocket stiffness makes
        # that hold a leg, and ten of them were measured not to: with no crust
        # the foot enters and floats back out in three seconds, with a stiff
        # one it never enters, and the eight in between never held for a
        # second.
        #
        # What immobilises a leg is being swallowed to the shin or the thigh,
        # where there is an order of magnitude more material to displace and
        # side friction along the whole limb. That was structurally impossible
        # here: the bed was 400 mm over a rigid patch, so the foot bottomed out
        # at 260 mm and could sink no further. H1's knee sits 430 mm above the
        # sole, so the bed has to be about twice that to reach it.
        #
        # The window narrows to pay for the depth -- 0.9 m square at 0.9 m deep
        # is the same particle count as 1.2 m square at 0.4 m. It follows the
        # robot, so it does not have to span the escape radius.
        "--mpm-patch-length", "0.9",
        "--mpm-patch-width", "0.9",
        "--mpm-depth", "0.90",
        "--render-grains-per-particle", "0",
    ]
    return parser.parse_args(defaults + (extra or []))


def main() -> None:
    args = build_args(sys.argv[1:])
    if args.make_snapshot:
        make_snapshot(args)
        return

    env = EscapeEnv(args)
    env.load_snapshot(args.snapshot_path.expanduser().resolve())
    for episode in range(args.episodes):
        obs = env.reset()
        total = 0.0
        for _ in range(env.max_episode_steps):
            action = 0.1 * torch.randn(env.action_dim, device=env.device)
            obs, reward, done, info = env.step(action)
            total += reward
            if done:
                break
        print(
            f"episode {episode}: steps={env._step_count} return={total:+.2f} "
            f"burial={info['burial_mm']:.0f} mm support_sink={info['support_sink_mm']:.0f} mm "
            f"escaped={info['escaped']} fallen={info['fallen']}"
        )


if __name__ == "__main__":
    main()
