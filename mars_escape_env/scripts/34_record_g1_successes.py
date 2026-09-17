# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record only the successful escape episodes to USD, one file per episode.

Runs the vectorised escape environment (26_escape_env_vec.py) with a single
environment -- the same environment the policy is trained and scored in -- and
records each episode to its own USD file, keeping the file only when the
episode ends in a sustained escape.

Success cannot be known in advance: the MPM solve is not bit-reproducible, so
the same policy from the same snapshot sometimes escapes and sometimes falls.
Recording per episode and discarding failures is therefore the only way to
collect clean success footage.

Play the results back with 25_open_usd.py (press play in Isaac Sim).

Usage
-----
    python terrain/scripts/28_record_successes.py \\
        --checkpoint terrain/output/escape_training/<run>/model_799.pt \\
        --wanted 3
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import newton.viewer
import numpy as np
import torch
import warp as wp


def load_by_path(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(filename)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VEC = load_by_path("g1_escape_env_vec", "32_g1_escape_env_vec.py")
PLAY = load_by_path("play_escape", "24_play_escape.py")


def load_policy(checkpoint: Path, device: str):
    """Deterministic actor from an RSL-RL checkpoint (mean action, no noise)."""
    saved = torch.load(str(checkpoint), map_location=device, weights_only=False)
    state = saved["actor_state_dict"]
    layers = sorted(
        {
            int(key.split(".")[1])
            for key in state
            if key.startswith("mlp.") and key.endswith(".weight")
        }
    )
    weights = [
        (state[f"mlp.{i}.weight"].to(device), state[f"mlp.{i}.bias"].to(device))
        for i in layers
    ]

    @torch.no_grad()
    def policy(obs: torch.Tensor) -> torch.Tensor:
        x = obs
        for depth, (w_mat, b_vec) in enumerate(weights):
            x = x @ w_mat.T + b_vec
            if depth < len(weights) - 1:
                x = torch.nn.functional.elu(x)
        return x

    return policy, [w.shape[0] for w, _ in weights]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--wanted", type=int, default=3, help="How many successes to collect."
    )
    parser.add_argument(
        "--max-episodes", type=int, default=30, help="Attempt budget."
    )
    parser.add_argument("--episode-seconds", type=float, default=10.0)
    parser.add_argument("--hold-seconds", type=float, default=1.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("terrain/output/escape_env/successes"),
    )
    parser.add_argument("--usd-fps", type=int, default=15)
    parser.add_argument("--snapshot-path", type=Path, default=VEC.DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--no-record",
        action="store_true",
        help=(
            "Score the checkpoint without writing USD. Deterministic scoring "
            "is what the success rate is quoted from; the recording is only "
            "for looking at it."
        ),
    )
    cli, passthrough = parser.parse_known_args()

    checkpoint = (
        cli.checkpoint
        if cli.checkpoint is not None
        else PLAY.latest_checkpoint(Path("terrain/output/escape_training"))
    )
    cli.output_dir.mkdir(parents=True, exist_ok=True)

    env_args = VEC.build_args(
        [
            "--num-envs",
            "1",
            "--episode-seconds",
            str(cli.episode_seconds),
            "--hold-seconds",
            str(cli.hold_seconds),
        ]
        + passthrough
    )
    env_args.snapshot_path = cli.snapshot_path
    env = VEC.EscapeEnvVec(env_args)

    policy, hidden = load_policy(checkpoint, str(env.device))
    print(f"[record] checkpoint: {checkpoint}  (hidden {hidden})")

    surface_index = torch.tensor(
        np.flatnonzero(env.surface_mask_np), device=env.device, dtype=torch.long
    )
    # Material points stand in for a cluster of grains; draw them at half the
    # 25 mm spacing so the surface reads as a continuous bed.
    grain_radius = 0.0125
    print(f"[record] soil points per frame: {len(surface_index)}")

    kept = 0
    attempts = 0
    for episode in range(cli.max_episodes):
        path = cli.output_dir / f"attempt_{episode:02d}.usd"
        viewer = None
        if not cli.no_record:
            viewer = newton.viewer.ViewerUSD(
                output_path=str(path), fps=cli.usd_fps, num_frames=10**9
            )
            viewer.set_model(env.model)

        obs = env.reset_all()
        sim_time = 0.0
        escaped = False
        steps = 0
        exit_m = 0.0
        escape_step = None
        for _ in range(env.max_episode_steps):
            if viewer is not None:
                viewer.begin_frame(sim_time)
                viewer.log_state(env.state_0)
                # log_state draws rigid shapes only, so the soil has to be
                # logged separately or the robot appears to walk on nothing.
                # Only the top layer is drawn: it is what is visible anyway,
                # and logging all 76,800 material points per frame bloats the
                # file sevenfold.
                surface_points = wp.to_torch(env.state_0.particle_q)[surface_index]
                viewer.log_points(
                    "/soil",
                    points=wp.from_torch(surface_points.contiguous(), dtype=wp.vec3),
                    radii=grain_radius,
                    colors=(0.62, 0.35, 0.20),
                )
                viewer.end_frame()
            sim_time += VEC.CONTROL_DT

            obs, _, done, info = env.step(policy(obs))
            steps += 1
            # Peak, not last: the terminating step's distance is read after
            # the env has already restored the snapshot pose.
            exit_m = max(exit_m, float(info["exit_distance"][0]))
            # Escaping no longer ends the episode -- the robot has to keep
            # walking afterwards -- so the moment it happens has to be caught
            # as it passes, not read off the final step.
            if escape_step is None and bool(info["newly_escaped"][0]):
                escape_step = steps
            if bool(done[0]):
                escaped = bool(info["escaped"][0])
                break

        if viewer is not None:
            viewer.close()
        attempts += 1

        if escaped:
            kept += 1
            at_s = "?" if escape_step is None else f"{escape_step * VEC.CONTROL_DT:.1f}"
            keep_path = None
            if viewer is not None:
                keep_path = cli.output_dir / f"success_{kept:02d}.usd"
                path.replace(keep_path)
            print(
                f"episode {episode}: SUCCESS  steps={steps} exit={exit_m:.2f} m "
                f"escape at {at_s} s"
                + (f" -> {keep_path}" if keep_path else ""),
                flush=True,
            )
            if kept >= cli.wanted and not cli.no_record:
                break
        else:
            if viewer is not None:
                path.unlink(missing_ok=True)
            print(
                f"episode {episode}: failed (exit={exit_m:.2f} m), steps={steps}",
                flush=True,
            )

    rate = kept / attempts if attempts else 0.0
    print(
        f"\n[record] deterministic escape rate: {kept}/{attempts} = {100 * rate:.1f}%"
    )
    if kept and not cli.no_record:
        print("[record] view with:")
        print(
            f"    python terrain\\scripts\\25_open_usd.py "
            f"{cli.output_dir}\\success_01.usd"
        )


if __name__ == "__main__":
    main()
