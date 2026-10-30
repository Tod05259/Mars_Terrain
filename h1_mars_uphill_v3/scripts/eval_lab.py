"""Lab scoring of an uphill policy: inverted pyramids of fixed slopes on the soil surrogate.

    run.ps1 scripts\\eval_lab.py --checkpoint <model.pt> [--slopes 0 16.8 20 22.5] [--num_envs 512]
        [--seconds 20] [--command_x 0.3] [--output runs\\lab\\name.json] --headless

``Mars-Ascend-H1-Eval-v1`` with one column group per slope: every env starts upright and at rest on
the bottom platform of an inverted pyramid (random yaw), with the soil at full level and the
training ground mix (a fifth rigid, beds 0.05-0.36 m, 0.5-4 times the fitted sand). The policy acts
deterministically under 0.3 m/s forward and the command term's heading controller holding the
heading each env has after its first step (as ``84_eval_walk_lab.py --heading_hold``; the sand judge's
supervisor does the same towards the fall line). One episode per env.

Per slope: falls, the share that climbed ``--climb_m`` up the face without falling (horizontal
distance from the start beyond the platform edge plus the climb's run), the time it took, speed,
flight share (no foot in contact by the task's soil-aware mask) and the lowest root height above
the lowest ankle (the judge rejects < 0.66 m). Lab numbers are a guide; the decision is the sand's.

The checkpoint is loaded through the task's rsl_rl runner, or else through the actor loader of the
sand environment, so a walker trained on the sand plays here as well.
"""

import argparse
import importlib.metadata as metadata
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--task", default="Mars-Ascend-H1-Eval-v1")
parser.add_argument("--slopes", type=float, nargs="+", default=[0.0, 16.8, 20.0, 22.5])
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--command_x", type=float, default=0.3)
parser.add_argument("--climb_m", type=float, default=3.0)
parser.add_argument("--rigid_share", type=float, default=None, help="share of envs without a bed (task: 0.2)")
parser.add_argument("--loader", choices=("auto", "runner", "actor"), default="auto")
parser.add_argument("--friction", type=float, default=None, help="fix the robot's static and dynamic friction (task: random)")
parser.add_argument("--soil_mu", type=float, nargs=2, default=None, help="soil tangential-friction scale range (task: 0.6 1.5)")
parser.add_argument("--soil_ratio", type=float, nargs=2, default=None, help="soil side/top resistance scale range (task: 0.6 1.5)")
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab.terrains as terrain_gen  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import uphill_lab  # noqa: E402
from getup_lab import walk_mdp  # noqa: E402
from uphill_lab import ascend_env_cfg  # noqa: E402

HEADING_GAIN = 0.5
HEADING_RATE_LIMIT = 1.0
COLS_PER_SLOPE = 4


def load_actor(checkpoint: Path, device: str):
    """Deterministic actor from ``actor_state_dict`` (the loader of CAPSTONE 28/84)."""
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    state = torch.load(str(checkpoint), map_location=device, weights_only=False)["actor_state_dict"]
    layers = sorted({int(k.split(".")[1]) for k in state if k.startswith("mlp.") and k.endswith(".weight")})
    shapes = [tuple(state[f"mlp.{i}.weight"].shape) for i in layers]
    example = TensorDict({"policy": torch.zeros(1, shapes[0][1], device=device)}, batch_size=[1], device=device)
    actor = MLPModel(
        example, {"actor": ["policy"]}, "actor", shapes[-1][0],
        hidden_dims=[rows for rows, _ in shapes[:-1]], activation="elu", obs_normalization=False,
        distribution_cfg={"class_name": "rsl_rl.modules.distribution.GaussianDistribution", "init_std": 1.0,
                          "std_type": "scalar"},
    ).to(device)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    return lambda obs: actor(obs, stochastic_output=False)


def main() -> None:
    uphill_lab.register()
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point"), metadata.version("rsl-rl-lib")
    )
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = args.seconds + 1.0
    generator = env_cfg.scene.terrain.terrain_generator
    generator.curriculum = True  # columns = slope groups, origins spread over the columns in order
    generator.num_cols = COLS_PER_SLOPE * len(args.slopes)
    generator.num_rows = 4
    env_cfg.scene.terrain.max_init_terrain_level = None
    share = 1.0 / len(args.slopes)
    generator.sub_terrains = {
        f"up_{s:g}": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=share, slope_range=(math.tan(math.radians(s)),) * 2,
            platform_width=ascend_env_cfg.PLATFORM_M, border_width=0.25,
        )
        for s in args.slopes
    }
    if args.rigid_share is not None:
        env_cfg.actions.joint_pos.soil.rigid_share = args.rigid_share
    if args.soil_mu is not None:
        env_cfg.actions.joint_pos.soil.friction_scale_range = tuple(args.soil_mu)
    if args.soil_ratio is not None:
        env_cfg.actions.joint_pos.soil.ratio_scale_range = tuple(args.soil_ratio)
    if args.friction is not None:
        env_cfg.events.physics_material.params["static_friction_range"] = (args.friction, args.friction)
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (args.friction, args.friction)
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
    base = env.unwrapped
    device = base.device

    policy, loader = None, "actor"
    if args.loader in ("auto", "runner"):
        try:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(str(args.checkpoint))
            policy, loader = runner.get_inference_policy(device=device), "runner"
        except Exception as error:  # a sand-trained checkpoint has no critic of this task
            if args.loader == "runner":
                raise
            print(f"[lab] runner.load failed ({type(error).__name__}); using the actor loader")
    if policy is None:
        policy = load_actor(args.checkpoint, device)

    robot = base.scene["robot"]
    soil = base.action_manager.get_term("joint_pos").soil
    sensor = base.scene.sensors["contact_forces"]
    feet = [i for i, name in enumerate(sensor.body_names) if name.endswith("ankle_link")]
    feet_robot = [robot.body_names.index(sensor.body_names[i]) for i in feet]
    ankles = robot.find_bodies(".*ankle_link")[0]
    command = base.command_manager.get_term("base_velocity")
    command.cfg.ranges.lin_vel_x = (args.command_x, args.command_x)
    command.cfg.ranges.ang_vel_z = (0.0, 0.0)
    command.cfg.heading_command = False
    n, dt = base.num_envs, base.step_dt
    steps = int(args.seconds / dt)
    # everything from the reset on stays under inference mode (as in 84): tensors the env creates
    # there could not be written by a reset outside it
    with torch.inference_mode():
        obs, _ = env.reset()
        column = base.scene.terrain.terrain_types.clone()
        group = torch.div(column, COLS_PER_SLOPE, rounding_mode="floor").clamp(max=len(args.slopes) - 1)
        rigid = soil.ground == 0.0 if soil.cfg.rigid_share > 0.0 else torch.zeros(n, dtype=torch.bool, device=device)
        slope_rad = torch.tensor([math.radians(s) for s in args.slopes], device=device)[group]
        need = ascend_env_cfg.PLATFORM_M / 2 + args.climb_m * torch.cos(slope_rad)

        alive = torch.ones(n, dtype=torch.bool, device=device)
        fell = torch.zeros_like(alive)
        climbed = torch.zeros_like(alive)
        climb_time = torch.full((n,), float("nan"), device=device)
        flight = torch.zeros(n, device=device)
        count = torch.zeros(n, device=device)
        h_min = torch.full((n,), 9.0, device=device)
        max_dist = torch.zeros(n, device=device)
        last_along = torch.zeros(n, device=device)
        last_cross = torch.zeros(n, device=device)
        lat_sum = torch.zeros(n, device=device)
        slip_sum = torch.zeros(n, device=device)
        slip_n = torch.zeros(n, device=device)
        start_xy = None
        for k in range(steps):
            obs, _, dones, _ = env.step(policy(obs))
            ended = alive & dones.bool()
            fell |= ended & base.termination_manager.terminated
            alive &= ~ended
            xy = robot.data.root_pos_w.torch[:, :2]
            if k == 0:
                start_xy = xy.clone()
                yaw0 = robot.data.heading_w.torch.clone()
                forward = torch.stack([torch.cos(yaw0), torch.sin(yaw0)], dim=1)
                command.cfg.heading_command = True
                command.cfg.heading_control_stiffness = HEADING_GAIN
                command.cfg.ranges.ang_vel_z = (-HEADING_RATE_LIMIT, HEADING_RATE_LIMIT)
                command.heading_target[:] = robot.data.heading_w.torch
                command.is_heading_env[:] = True
                continue
            dist = torch.linalg.norm(xy - start_xy, dim=1)
            max_dist = torch.where(alive, torch.maximum(max_dist, dist), max_dist)
            rel = xy - start_xy
            along_now = (rel * forward).sum(dim=1)
            cross_now = rel[:, 0] * -forward[:, 1] + rel[:, 1] * forward[:, 0]  # + = to the start heading's left
            last_along = torch.where(alive, along_now, last_along)
            last_cross = torch.where(alive, cross_now, last_cross)
            newly = alive & ~climbed & (dist >= need)
            climb_time = torch.where(newly, torch.full_like(climb_time, k * dt), climb_time)
            climbed |= newly
            floor_force = sensor.data.net_forces_w_history.torch[:, :, feet, :].norm(dim=-1).amax(dim=1)
            soil_up = soil.link_forces[:, feet_robot, 2] if soil.level > 0.0 else None
            down = walk_mdp.merge_contact_mask(floor_force, soil_up, walk_mdp.CONTACT_FORCE_N, 48.0).sum(dim=1)
            m = alive.float()
            count += m
            lat_sum += m * robot.data.root_lin_vel_b.torch[:, 1]
            flight += m * (down == 0).float()
            h = robot.data.root_pos_w.torch[:, 2] - robot.data.body_pos_w.torch[:, ankles, 2].min(dim=1)[0]
            h_min = torch.where(alive, torch.minimum(h_min, h), h_min)
            # stance-foot slip along the climb direction (the start heading), the sand judge's
            # "stance carried" read: negative = the planted foot moves back / downhill
            in_contact = walk_mdp.merge_contact_mask(floor_force, soil_up, walk_mdp.CONTACT_FORCE_N, 48.0)
            foot_v = robot.data.body_lin_vel_w.torch[:, feet_robot, :2]
            along = (foot_v * forward.unsqueeze(1)).sum(dim=-1)
            slip_sum += m * (along * in_contact).sum(dim=1)
            slip_n += m * in_contact.sum(dim=1)

    result = {"checkpoint": str(args.checkpoint), "loader": loader, "command_x": args.command_x,
              "seconds": args.seconds, "climb_m": args.climb_m, "slopes": {}}
    print(f"\n[lab] {args.checkpoint} ({loader}), {n} envs, {args.command_x} m/s, {args.seconds:.0f} s, "
          f"climb {args.climb_m} m along the face")
    print(f"  {'slope':>6s} {'n':>4s} {'falls':>6s} {'climbed':>8s} {'t_med':>6s} {'flight':>7s} {'h_min':>6s}"
          f" {'rigid n/climbed':>16s}")
    for g, s in enumerate(args.slopes):
        mask = group == g
        k = int(mask.sum())
        if k == 0:
            continue
        ct = climb_time[mask & climbed]
        row = {
            "n": k,
            "falls": int((mask & fell).sum()),
            "climbed": int((mask & climbed).sum()),
            "climb_time_median_s": float(ct.median()) if ct.numel() else None,
            "flight_share": float((flight[mask] / count[mask].clamp(min=1)).mean()),
            "root_above_ankle_min_m": float(h_min[mask].median()),
            "root_above_ankle_min_p10_m": float(torch.quantile(h_min[mask], 0.1)),
            "rigid_n": int((mask & rigid).sum()),
            "rigid_climbed": int((mask & rigid & climbed).sum()),
            "rigid_falls": int((mask & rigid & fell).sum()),
            "travel_median_m": float(max_dist[mask].median()),
            "stance_slip_along_median_mps": float((slip_sum[mask] / slip_n[mask].clamp(min=1)).median()),
            "along_median_m": float(last_along[mask].median()),
            "cross_median_m": float(last_cross[mask].abs().median()),
            "cross_signed_median_m": float(last_cross[mask].median()),
            "cross_right_share": float((last_cross[mask] < 0).float().mean()),
            "body_v_lateral_median_mps": float((lat_sum[mask] / count[mask].clamp(min=1)).median()),
            "cross_over_along": float((last_cross[mask].abs() / last_along[mask].clamp(min=0.1)).median()),
            "travel_median_rigid_m": float(max_dist[mask & rigid].median()) if bool((mask & rigid).any()) else None,
        }
        result["slopes"][f"{s:g}"] = row
        t = "-" if row["climb_time_median_s"] is None else f"{row['climb_time_median_s']:.1f}"
        print(f"  {s:6.1f} {k:4d} {row['falls']:6d} {row['climbed']:8d} {t:>6s} {row['flight_share']:7.2f}"
              f" {row['root_above_ankle_min_m']:6.2f} {row['rigid_n']:>8d}/{row['rigid_climbed']:<7d}"
              f"  travel {row['travel_median_m']:.2f} m (rigid falls {row['rigid_falls']})"
              f"  stance slip {row['stance_slip_along_median_mps']:+.3f} m/s"
              f"  along {row['along_median_m']:.2f} m cross {row['cross_signed_median_m']:+.2f} m"
              f" (right {row['cross_right_share']:.0%}, body lat {row['body_v_lateral_median_mps']:+.3f} m/s)")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[lab] wrote {args.output}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
