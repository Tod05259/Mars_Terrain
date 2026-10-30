"""MPM judge of an uphill policy: CAPSTONE 26 ``--task slope_ascend`` on a team-standard sand bed.

    run.ps1 scripts\\eval_mpm.py --checkpoint <model.pt> --site slope16 --episodes 20 [--soil typical]
        [--label name] [--command-x 0.3] [--hold-seconds 3] [--scan-dropout 0.3] [-- extra 26 flags]

What one episode is (workflow(3) section 1):

* the env is CAPSTONE 26 with the team's judge flags (``80_soil_calibration_runs.JUDGE_FLAGS``: rated
  torque and speed with saturation ratio 1.5, every link on the sand, pelvis-centred window, surface
  scan, the solver at its documented settings), the site's snapshot, and ``--task slope_ascend``;
* the supervisor gives the policy ``[v_x, 0, yaw_rate]``: v_x from ``--command-x``, yaw rate from the
  team's external heading PI (TEAM_SETTINGS A3: gain 0.5, integral 0.45, +-1 rad/s) towards the fall
  line uphill (26's own downhill probe). The command is written into the observation the policy acts
  on, as 26's walk task does with its internal PI;
* **fallen**: the chain's fall test held 0.2 s -- trunk tilt above 0.65 rad, or the torso less than
  0.50 m above the lowest ankle (slope-independent; see :data:`FALL_HEIGHT_M`). The judge flags turn
  26's own off-feet termination off (``--down-seconds 0``), so the fall test is applied here;
* **reached**: ``--target-m`` gained up the fall line (26's ``worst_drift``: the best progress) while
  not falling and with the torso at least 0.66 m above the lowest ankle. 26's own success latch is
  recorded beside it. 26 ends the episode on sliding back ``--runway-m`` below the best point
  (``slid_back``), on divergence, or on time-out; leaving the env patch is checked here;
* **held**: after reaching, the command goes to zero (the FSM's STAND observation) for
  ``--hold-seconds``; the episode counts as held if the robot is still on its feet at the end;
* **physically valid**: CAPSTONE 51's tests over the episode -- both feet clear of the sand in at most
  5 % of steps, torso within -0.40/+0.25 m of standing height over the lowest foot, no foot within
  20 mm of the rigid DTM, peak planar speed under 1.0 m/s, the reaction clamp on under 5 % of
  foot-steps, no stance foot carried uphill with the body, at least 0.6 stance changes per second.

The headline is "reached, held and valid" over all episodes, with a Wilson 95 % interval. Episodes
are run one at a time (1 env), deterministic actions; MPM is not bit-reproducible, so repeated episodes
from the one snapshot are samples (TEAM_GUIDE 2026-10-04: repeat evaluations, compare by pairs).
Writes ``runs/eval/<label>/result.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from uphill import mpm, paths, sites  # mpm first: puts the team's Newton on sys.path

import numpy as np  # noqa: E402
import torch  # noqa: E402

FALL_TILT_RAD = 0.65
"""Trunk tilt that counts as falling [rad] (the chain's and the FSM's fall test, TEAM_SETTINGS B2)."""
FALL_HEIGHT_M = 0.50
"""Torso link above the lowest ankle link below which the robot counts as falling [m]. The chain reads
the torso against 26's soil top, a 95th percentile over a 2 m window: on a 17-23 deg bed that is the
uphill edge, 0.3-0.4 m above the ground under the robot, which would call a walking robot fallen.
Above the lowest ankle the test does not depend on the slope; a robot lying down reads 0-0.3 m."""
FALL_HOLD_S = 0.2
"""How long the fall test must hold [s] (the chain's --fall-hold-s)."""
REACH_HEIGHT_M = 0.66
"""Torso above the lowest ankle that counts as standing at the target [m]: the lower bound of 51's
physical validity (1.06 - 0.40). 26's own success latch tests the knees against the DTM + 0.30 m
datum, which a normal stride up a 17-23 deg face grazes; it is recorded, not used."""
HEADING_GAIN = 0.5
HEADING_KI = 0.45
HEADING_INT_LIMIT = 2.0
YAW_RATE_LIMIT = 1.0
COMMAND_SLICE = slice(9, 12)  # obs layout (TEAM_SETTINGS A2): lin3 | ang3 | grav3 | command3 | ...


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


class GifRecorder:
    """Headless Newton GL frames of one env, following the robot from the side of the fall line."""

    def __init__(self, env, size: tuple[int, int], headless: bool = True):
        import newton.viewer

        self.env = env
        self.headless = headless
        self.viewer = newton.viewer.ViewerGL(width=int(size[0]), height=int(size[1]), headless=headless)
        self.viewer.set_model(env.model)
        self.viewer.show_particles = True
        self.frames = []
        self.follow = True

    def running(self) -> bool:
        """False once the window of a live view was closed."""
        return self.headless or self.viewer.is_running()

    def capture(self, t: float, up: torch.Tensor, label: str) -> None:
        import warp as wp
        from PIL import Image, ImageDraw

        if not self.headless:
            # live window: follow the robot until the user moves the camera themselves
            self.show(t, up)
            return
        torso = self.env.body_q_t[self.env.torso_ids_t, :3][0].detach().cpu().numpy()
        u = up.detach().cpu().numpy()
        side = np.array([-u[1], u[0], 0.0])
        # nearly a pure side view, so the incline reads as a tilted ground line
        camera = torso + 3.6 * side - 0.6 * np.array([u[0], u[1], 0.0]) + np.array([0.0, 0.0, 0.5])
        d = torso + np.array([0.0, 0.0, -0.35]) - camera
        yaw = math.degrees(math.atan2(d[1], d[0]))
        pitch = math.degrees(math.atan2(d[2], float(np.linalg.norm(d[:2]))))
        self.viewer.set_camera(wp.vec3(*camera.tolist()), pitch, yaw)
        self.viewer.begin_frame(t)
        self.viewer.log_state(self.env.state_0)
        self.viewer.end_frame()
        image = Image.fromarray(self.viewer.get_frame().numpy()).convert("RGB")
        ImageDraw.Draw(image).text((10, 10), label, fill=(255, 255, 255))
        self.frames.append(image)

    def show(self, t: float, up: torch.Tensor) -> None:
        """One frame of the live window, the camera beside the fall line and following the trunk."""
        import warp as wp

        torso = self.env.body_q_t[self.env.torso_ids_t, :3][0].detach().cpu().numpy()
        u = up.detach().cpu().numpy()
        side = np.array([-u[1], u[0], 0.0])
        camera = torso + 3.6 * side - 0.6 * np.array([u[0], u[1], 0.0]) + np.array([0.0, 0.0, 0.5])
        d = torso + np.array([0.0, 0.0, -0.35]) - camera
        self.viewer.set_camera(
            wp.vec3(*camera.tolist()),
            math.degrees(math.atan2(d[2], float(np.linalg.norm(d[:2])))),
            math.degrees(math.atan2(d[1], d[0])),
        )
        self.viewer.begin_frame(t)
        self.viewer.log_state(self.env.state_0)
        self.viewer.end_frame()

    def save(self, path: Path, frame_ms: float) -> None:
        if self.headless and self.frames:
            from PIL import Image

            # one adaptive palette for the whole clip keeps the file small and the colours steady
            palette = self.frames[len(self.frames) // 2].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
            frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in self.frames]
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(frame_ms), loop=0, optimize=True)
        self.frames = []


def build_env(cli, extra: list[str]):
    VEC = mpm.vec()
    site = sites.site(cli.site)
    flags = mpm.judge_flags()
    flags = mpm.override(flags, "--env-pitch", str(cli.env_pitch))
    if cli.soil == "typical":
        for i in range(0, len(mpm.TYPICAL_SOIL), 2):
            flags = mpm.override(flags, mpm.TYPICAL_SOIL[i], mpm.TYPICAL_SOIL[i + 1])
    if cli.scan_dropout > 0:
        flags += ["--scan-dropout", str(cli.scan_dropout), "--scan-dropout-mode", cli.scan_dropout_mode]
    env_argv = (
        ["--num-envs", "1", "--episode-seconds", str(cli.episode_seconds), "--task", "slope_ascend"]
        + flags
        + [
            "--terrain-usd", str(sites.usd()),
            "--site-xy", str(site["site_xy"][0]), str(site["site_xy"][1]),
            "--descend-target-m", str(cli.target_m),
            "--arrest-runway-m", str(cli.runway_m),
            "--command-x", str(cli.command_x),
        ]
        + extra
    )
    args = VEC.build_args(env_argv)
    args.snapshot_path = cli.snapshot or sites.snapshot(cli.site)
    return VEC, VEC.EscapeEnvVec(args), env_argv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--site", default="slope16")
    parser.add_argument("--snapshot", type=Path, default=None, help="default: snapshots/uphill_<site>_*.pkl")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=30.0,
        help="cap per episode [s]; must leave room for the hold after a slow climb",
    )
    parser.add_argument("--command-x", type=float, default=0.3)
    parser.add_argument("--target-m", type=float, default=3.0)
    parser.add_argument("--runway-m", type=float, default=1.0)
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    parser.add_argument(
        "--env-pitch",
        type=float,
        default=10.0,
        help="rigid DTM patch around the site [m]; the episode ends 0.5 m inside its edge (judge flags: 8)",
    )
    parser.add_argument("--soil", choices=("team", "typical"), default="team")
    parser.add_argument("--scan-dropout", type=float, default=0.0)
    parser.add_argument("--scan-dropout-mode", choices=("cell", "block"), default="block")
    parser.add_argument("--max-speed-mps", type=float, default=1.0)
    parser.add_argument(
        "--entry-walker",
        type=Path,
        default=None,
        help="FSM entry test (TEAM_SETTINGS B1/B3-3): this walker holds STAND (command 0) for --entry-seconds, "
        "then a hard switch hands the robot to --checkpoint",
    )
    parser.add_argument("--entry-seconds", type=float, default=3.0)
    parser.add_argument("--label", default=None)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument(
        "--gif",
        action="store_true",
        help="record every episode to runs/eval/<label>/episode_NN_<end>.gif (Newton GL, headless, side view)",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="watch the episodes live in a Newton GL window (every control step; closing the window ends the run)",
    )
    parser.add_argument("--gif-stride", type=int, default=5, help="control steps per GIF frame (5 = 10 fps)")
    parser.add_argument("--gif-size", type=int, nargs=2, default=(560, 315))
    cli, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    label = cli.label or f"{cli.checkpoint.stem}_{cli.site}_{cli.soil}"
    out_dir = paths.RUNS / "eval" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    VEC, env, env_argv = build_env(cli, extra)
    policy, hidden = mpm.record().load_policy(cli.checkpoint, str(env.device))
    walker = mpm.record().load_policy(cli.entry_walker, str(env.device))[0] if cli.entry_walker else None
    entry_steps = int(round(cli.entry_seconds / VEC.CONTROL_DT)) if walker is not None else 0
    dt = VEC.CONTROL_DT
    stand_h = VEC.STANDING_ROOT_HEIGHT_M
    guard = 0.5 * cli.env_pitch - 0.5
    print(f"[judge] {cli.checkpoint} (hidden {hidden}) site {cli.site} soil {cli.soil} -> {out_dir}", flush=True)

    if cli.view and cli.gif:
        parser.error("--view and --gif cannot be combined")
    if cli.view:
        cli.gif_stride = 1
        recorder = GifRecorder(env, (1280, 720), headless=False)
    else:
        recorder = GifRecorder(env, cli.gif_size) if cli.gif else None
    rows = []
    exit_states: list[dict] = []
    for episode in range(cli.episodes):
        obs = env.reset_all()
        env._steps.zero_()  # TEAM_GUIDE 7.4 rule 2
        downhill = env._downhill_t
        up = -downhill / torch.linalg.norm(downhill)
        up_yaw = float(torch.atan2(up[1], up[0]))
        start_xy = env.body_q_t[env.torso_ids_t, :2][0].clone()
        start_yaw = float(env._torso_yaw()[0])
        heading_int = 0.0
        phase, reached_at, hold_steps = "climb", None, 0
        latch26_at, fall_run, tilt_max = None, 0, 0.0
        vb_fwd, vb_lat, cross_signed = [], [], 0.0
        fall_hold = max(1, int(round(FALL_HOLD_S / dt)))
        end = "timeout"
        steps = 0
        best = 0.0
        air = 0
        torso_h, hdg_err, cross_abs = [], [], []
        plate_min, deepest, peak_speed = 1000.0, 0.0, 0.0
        switches, last_stance, last_stance_xy = 0, -1, None
        skid, stance_steps, carried = 0, 0, []
        prev_xy = start_xy.clone()
        sat0 = env.reaction_saturation_counts.numpy().copy()
        foot_ids = env.foot_ids_t.flatten().tolist()

        if entry_steps > 0:
            phase = "entry"

        def command_now() -> torch.Tensor:
            nonlocal heading_int
            yaw = float(env._torso_yaw()[0])
            err = math.remainder(up_yaw - yaw, 2.0 * math.pi)
            if phase == "entry":
                return torch.zeros(3, device=env.device)  # the walker's STAND before the hand-over
            heading_int = max(-HEADING_INT_LIMIT, min(HEADING_INT_LIMIT, heading_int + err * dt))
            yaw_rate = max(-YAW_RATE_LIMIT, min(YAW_RATE_LIMIT, HEADING_GAIN * err + HEADING_KI * heading_int))
            hdg_err.append(abs(math.degrees(err)))
            if phase == "climb":
                return torch.tensor([cli.command_x, 0.0, yaw_rate], device=env.device)
            return torch.zeros(3, device=env.device)  # STAND: the whole command is zero

        cmd = command_now()
        env.command[0] = cmd
        obs[0, COMMAND_SLICE] = cmd
        for _ in range(env.max_episode_steps):
            acting = walker if phase == "entry" else policy
            obs, _reward, done, info = env.step(acting(obs))
            steps += 1
            if phase == "entry" and steps >= entry_steps:
                phase = "climb"  # hard switch, no blending (TEAM_GUIDE 8.3)
            if bool(done[0]):
                # 26 ends a slope episode on divergence, on sliding back past the runway, or on time
                # (the judge flags turn its off-feet fall test off: --down-seconds 0)
                end = "diverged" if bool(info["diverged"][0]) else ("slid_back" if bool(info["swept"][0]) else "timeout")
                best = max(best, float(info["worst_drift_m"][0]))
                break
            best = max(best, float(info["worst_drift_m"][0]))
            if latch26_at is None and bool(info["descended"][0]):
                latch26_at = steps * dt
            # 51's physical readout, with the fall line uphill as "forward".
            feet = env.body_q_t[env.foot_ids_t.flatten(), :3].view(2, 3)
            soil_top = env._soil_top_under_feet()[0]
            torso = env.body_q_t[env.torso_ids_t, :3][0]
            clear = (feet[:, 2] - soil_top) * 1000.0
            air += int(bool((clear > 30.0).all()))
            above_ankle = float(torso[2] - feet[:, 2].min())
            torso_h.append(above_ankle)
            q = env.joint_q_t.view(1, -1)[0, 3:7]  # root quaternion, xyzw
            tilt = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * float(q[0] ** 2 + q[1] ** 2))))
            tilt_max = max(tilt_max, tilt)
            falling = tilt > FALL_TILT_RAD or above_ankle < FALL_HEIGHT_M
            fall_run = fall_run + 1 if falling else 0
            if fall_run >= fall_hold:
                end = "fallen"
                break
            if phase == "climb" and best >= cli.target_m and not falling and above_ankle >= REACH_HEIGHT_M:
                reached_at = steps * dt
                phase = "hold"
            deepest = min(deepest, float(clear.min()))
            fx = feet[:, 0] - env.offsets_t[0, 0]
            fy = feet[:, 1] - env.offsets_t[0, 1]
            plate_min = min(plate_min, float(((feet[:, 2] - env._sample_patch(fx, fy)) * 1000.0).min()))
            stance = int(torch.argmin(clear))
            stance_xy = feet[stance, :2].clone()
            if stance != last_stance:
                switches += int(last_stance >= 0)
                last_stance, last_stance_xy = stance, stance_xy
            elif float(clear.min()) < -50.0 and last_stance_xy is not None:
                stance_steps += 1
                d = (stance_xy - last_stance_xy) / dt
                skid += int(float(torch.linalg.norm(d)) > 0.15)
                carried.append(float(d[0] * up[0] + d[1] * up[1]))
                last_stance_xy = stance_xy
            now_xy = env.body_q_t[env.torso_ids_t, :2][0]
            peak_speed = max(peak_speed, float(torch.linalg.norm(now_xy - prev_xy)) / dt)
            prev_xy = now_xy.clone()
            rel = now_xy - start_xy
            cross_signed = float(rel[0] * -up[1] + rel[1] * up[0])
            cross_abs.append(abs(cross_signed))
            if phase == "climb":
                # root velocity in the trunk's yaw frame: forward and sideways (+ = to the robot's left)
                yaw_now = float(env._torso_yaw()[0])
                v_xy = env.joint_qd_t.view(1, -1)[0, 0:2]
                vb_fwd.append(float(v_xy[0]) * math.cos(yaw_now) + float(v_xy[1]) * math.sin(yaw_now))
                vb_lat.append(-float(v_xy[0]) * math.sin(yaw_now) + float(v_xy[1]) * math.cos(yaw_now))
            local = now_xy - env.offsets_t[0, :2]
            if float(local.abs().max()) > guard:
                end = "off_patch"
                break
            if phase == "hold":
                hold_steps += 1
                if hold_steps * dt >= cli.hold_seconds:
                    end = "held"
                    # exit state for the FSM hand-over (TEAM_SETTINGS B1/B3-4): the state STAND takes over
                    exit_states.append({
                        "episode": episode,
                        "joint_q": env.joint_q_t.view(1, -1)[0].detach().cpu().numpy().copy(),
                        "joint_qd": env.joint_qd_t.view(1, -1)[0].detach().cpu().numpy().copy(),
                        "torso_above_ankle_m": above_ankle,
                        "tilt_rad": tilt,
                    })
                    break
            if recorder is not None and steps % cli.gif_stride == 0:
                recorder.capture(
                    steps * dt,
                    up,
                    f"{cli.site} {sites.site(cli.site)['tilt_deg']:.1f} deg  t {steps * dt:4.1f} s  "
                    f"climbed {best:4.2f} m  {phase}{' (cmd 0)' if phase == 'hold' else ''}",
                )
                if not recorder.running():
                    end = "window_closed"
                    break
            if steps % cli.report_every == 0:
                print(
                    f"    {steps * dt:5.1f} s  climbed {best:5.2f} m  cross {cross_abs[-1]:4.2f} m  "
                    f"heading err {(hdg_err[-1] if hdg_err else 0.0):5.1f} deg  phase {phase}",
                    flush=True,
                )
            cmd = command_now()
            env.command[0] = cmd
            obs[0, COMMAND_SLICE] = cmd

        sat = env.reaction_saturation_counts.numpy() - sat0
        sat_frac = float(sat[foot_ids].sum()) / max(1, 2 * steps)
        air_frac = air / max(1, steps)
        h = np.asarray(torso_h) if torso_h else np.asarray([stand_h])
        cs = np.asarray(carried) if carried else np.zeros(1)
        skid_frac = skid / max(1, stance_steps)
        reasons = []
        if air_frac > 0.05:
            reasons.append(f"airborne {100 * air_frac:.0f}%")
        if float(h.max()) > stand_h + 0.25:
            reasons.append(f"torso high {h.max():.2f} m")
        if float(h.min()) < stand_h - 0.40:
            reasons.append(f"crouch {h.min():.2f} m")
        if plate_min < 20.0:
            reasons.append(f"foot on DTM ({plate_min:.0f} mm)")
        if peak_speed > cli.max_speed_mps:
            reasons.append(f"peak speed {peak_speed:.2f} m/s")
        if sat_frac > 0.05:
            reasons.append(f"clamp {100 * sat_frac:.0f}%")
        if skid_frac > 0.20 and float(cs.mean()) > 0.03:
            reasons.append(f"stance carried uphill {cs.mean():+.2f} m/s")
        if switches < 0.6 * steps * dt:
            reasons.append(f"stance changes {switches}")
        reached = reached_at is not None
        row = {
            "episode": episode,
            "end": end,
            "seconds": steps * dt,
            "reached": reached,
            "reached_at_s": reached_at,
            "held": end == "held",
            "latch26_at_s": latch26_at,
            "tilt_max_deg": math.degrees(tilt_max),
            "climbed_m": best,
            "climb_speed_mps": (cli.target_m / reached_at) if reached else None,
            "cross_max_m": max(cross_abs) if cross_abs else 0.0,
            "cross_end_signed_m": cross_signed,
            "body_v_forward_mean_mps": float(np.mean(vb_fwd)) if vb_fwd else None,
            "body_v_lateral_mean_mps": float(np.mean(vb_lat)) if vb_lat else None,
            "heading_err_mean_deg": float(np.mean(hdg_err)) if hdg_err else 0.0,
            "start_heading_off_deg": math.degrees(math.remainder(up_yaw - start_yaw, 2 * math.pi)),
            "valid": not reasons,
            "invalid_reasons": reasons,
            "airborne": air_frac,
            "torso_h_min": float(h.min()),
            "torso_h_max": float(h.max()),
            "deepest_mm": deepest,
            "plate_min_mm": plate_min,
            "peak_speed_mps": peak_speed,
            "clamp_frac": sat_frac,
            "skid_frac": skid_frac,
            "stance_carried_mps": float(cs.mean()),
            "stance_changes": switches,
        }
        rows.append(row)
        if recorder is not None and recorder.headless:
            gif = out_dir / f"episode_{episode:02d}_{end}{'_valid' if not reasons else '_invalid'}.gif"
            recorder.save(gif, 1000.0 * cli.gif_stride * dt)
            row["gif"] = str(gif)
        at = "-" if not reached else f"{reached_at:4.1f}s"
        print(
            f"  ep {episode:2d}: reached {at:>6}  climbed {best:4.2f} m  end {end:9s} @ {steps * dt:4.1f}s  "
            f"cross {row['cross_max_m']:.2f} m  hdg {row['heading_err_mean_deg']:4.1f}  "
            + ("valid" if not reasons else "INVALID: " + ", ".join(reasons)),
            flush=True,
        )
        if end == "window_closed":
            break

    rows = [r for r in rows if r["end"] != "window_closed"]  # a run cut short by the viewer is not scored
    n = len(rows)
    count = {k: sum(1 for r in rows if pred(r)) for k, pred in {
        "reached": lambda r: r["reached"],
        "held": lambda r: r["held"],
        "valid": lambda r: r["valid"],
        "success": lambda r: r["held"] and r["valid"],
        "fallen": lambda r: r["end"] == "fallen",
        "diverged": lambda r: r["end"] == "diverged",
        "slid_back": lambda r: r["end"] == "slid_back",
        "off_patch": lambda r: r["end"] == "off_patch",
        "timeout": lambda r: r["end"] == "timeout",
    }.items()}
    lo, hi = wilson(count["success"], n)
    summary = {
        "episodes": n,
        **count,
        "success_rate": count["success"] / max(1, n),
        "success_wilson95": [lo, hi],
        "climbed_mean_m": float(np.mean([r["climbed_m"] for r in rows])) if rows else 0.0,
        "reached_at_median_s": float(np.median([r["reached_at_s"] for r in rows if r["reached"]])) if count["reached"] else None,
    }
    print("\n== judge ==")
    print(
        f"  success (reached + held + valid) {count['success']}/{n} = {100 * summary['success_rate']:.0f} % "
        f"(Wilson {100 * lo:.0f}-{100 * hi:.0f} %)"
    )
    print(
        f"  reached {count['reached']}  held {count['held']}  valid {count['valid']}  fallen {count['fallen']}  "
        f"slid back {count['slid_back']}  off patch {count['off_patch']}  timeout {count['timeout']}  "
        f"diverged {count['diverged']}  mean climb {summary['climbed_mean_m']:.2f} m"
    )
    import newton

    result = {
        "checkpoint": str(cli.checkpoint),
        "site": cli.site,
        "site_cfg": sites.site(cli.site),
        "snapshot": str(cli.snapshot or sites.snapshot(cli.site)),
        "soil": cli.soil,
        "cli": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(cli).items()},
        "env_argv": env_argv,
        "newton": getattr(newton, "__version__", "?"),
        "capstone_commit": paths.CAPSTONE_COMMIT,
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": summary,
        "episodes": rows,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if exit_states:
        # Newton joint layout of one env: root pose (x, y, z, qx, qy, qz, qw) then the 19 joints
        np.savez(
            out_dir / "exit_states.npz",
            episode=np.array([s["episode"] for s in exit_states]),
            joint_q=np.stack([s["joint_q"] for s in exit_states]),
            joint_qd=np.stack([s["joint_qd"] for s in exit_states]),
            torso_above_ankle_m=np.array([s["torso_above_ankle_m"] for s in exit_states]),
            tilt_rad=np.array([s["tilt_rad"] for s in exit_states]),
        )
    print(f"[judge] wrote {out_dir / 'result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
