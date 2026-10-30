"""Where do the link-speed spikes of gate G4 come from on the uphill task?

    run.ps1 scripts\\diag_link_speed.py [--task Mars-Ascend-H1-v1] [--num_envs 4096] [--steps 1000]
        [--seed 1] [--threshold 14] --headless

Reproduces gate G4 of CAPSTONE 82 (untrained policy = N(0, 1) actions, soil level 1, the soil scales
at the factorial of their range extremes, re-applied after every step) and records every (env, link)
whose speed exceeds ``--threshold`` after the first 0.3 s of its episode: the link, its speed in the
previous steps, the soil force on it now and one step before, the env's terrain row (slope level)
and column type, its bed (rigid or thickness, stiffness scale), and the trunk's tilt and height
above the lowest ankle (fallen or not). Writes ``runs/gates/diag_link_speed_<task>_s<seed>.json``.
"""

import argparse
import json
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Mars-Ascend-H1-v1")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--threshold", type=float, default=14.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import uphill_lab  # noqa: E402
from uphill import paths  # noqa: E402


def main() -> None:
    uphill_lab.register()
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.actions.joint_pos.soil.level = 1.0
    cfg.curriculum.soil_depth.params["diag_steps"] = 10**8
    if "ramp_steps" in cfg.curriculum.soil_depth.params:
        cfg.curriculum.soil_depth.params["ramp_steps"] = None
    env = gym.make(args.task, cfg=cfg)
    base = env.unwrapped
    base.reset(seed=args.seed)
    robot = base.scene["robot"]
    soil = base.action_manager.get_term("joint_pos").soil
    terrain = base.scene.terrain
    n, dev = base.num_envs, base.device
    soil.cfg.randomize = False
    ids = torch.arange(n, device=dev)
    pattern = {}
    for j, (key, (lo, hi)) in enumerate(soil.scale_ranges().items()):
        pattern[key] = torch.where(((ids >> j) & 1).bool(), float(hi), float(lo)).to(torch.float32)
    soil.set_scales(ids, **pattern)
    soil.reset(ids)
    act_dim = base.action_manager.total_action_dim
    names = list(robot.body_names)
    ankles = robot.find_bodies(".*ankle_link")[0]
    history = []  # speeds of the last 3 steps, [n, bodies]
    prev_force = torch.zeros(n, len(names), device=dev)
    events = []
    with torch.inference_mode():
        for step in range(args.steps):
            base.step(torch.randn(n, act_dim, device=dev))
            soil.set_scales(ids, **pattern)
            speed = torch.linalg.norm(robot.data.body_lin_vel_w.torch, dim=-1)
            force = torch.linalg.norm(soil.link_forces, dim=-1)
            settled = base.episode_length_buf > int(round(0.3 / base.step_dt))
            hot = (speed > args.threshold) & settled.unsqueeze(1)
            if bool(hot.any()):
                up = -robot.data.projected_gravity_b.torch[:, 2]
                tilt = torch.rad2deg(torch.arccos(torch.clamp(up, -1.0, 1.0)))
                above = robot.data.root_pos_w.torch[:, 2] - robot.data.body_pos_w.torch[:, ankles, 2].min(dim=1)[0]
                for e, b in hot.nonzero().tolist():
                    events.append({
                        "step": step,
                        "env": e,
                        "link": names[b],
                        "speed": float(speed[e, b]),
                        "speed_prev": [float(h[e, b]) for h in history],
                        "soil_force_n": float(force[e, b]),
                        "soil_force_prev_n": float(prev_force[e, b]),
                        "terrain_row": int(terrain.terrain_levels[e]),
                        "terrain_col": int(terrain.terrain_types[e]),
                        "ground": float(soil.ground[e]),
                        "thickness_m": float(soil.thickness[e]),
                        "stiffness_scale": float(soil.scales["stiffness"][e]),
                        "cap_scale": float(soil.scales["cap"][e]),
                        "tilt_deg": float(tilt[e]),
                        "root_above_ankle_m": float(above[e]),
                    })
            history = (history + [speed.clone()])[-3:]
            prev_force = force.clone()
    events.sort(key=lambda r: -r["speed"])
    cols = base.cfg.scene.terrain.terrain_generator.num_cols
    props = [(k, v.proportion) for k, v in base.cfg.scene.terrain.terrain_generator.sub_terrains.items()]
    out = {"task": args.task, "seed": args.seed, "threshold": args.threshold, "num_cols": cols,
           "sub_terrains": props, "events": events[:200], "count": len(events)}
    path = paths.RUNS / "gates" / f"diag_link_speed_{args.task}_s{args.seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[diag] {len(events)} (env, link, step) samples over {args.threshold} m/s -> {path}")
    for r in events[:15]:
        print(
            f"  {r['speed']:5.1f} m/s {r['link']:<24s} prev {['%.1f' % s for s in r['speed_prev']]}"
            f" soil {r['soil_force_n']:5.0f}/{r['soil_force_prev_n']:5.0f} N  row {r['terrain_row']} col {r['terrain_col']}"
            f"  bed {'rigid' if r['ground'] == 0 else '%.2f m' % r['thickness_m']} K x{r['stiffness_scale']:.1f}"
            f"  tilt {r['tilt_deg']:5.1f}  above ankle {r['root_above_ankle_m']:.2f}"
        )
    env.close()


if __name__ == "__main__":
    main()
    app.close()
