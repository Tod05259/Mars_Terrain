# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a trained escape policy: live, continuous, or recorded to USD.

Three ways to look at a checkpoint from ``23_train_escape.py``:

* default — episode-by-episode in the live GL viewer;
* ``--watch`` — continuous viewing that only resets on physical failure
  (fall / anchored out), so a successful escape keeps walking;
* ``--viewer usd`` — run headless and record clean time samples to a USD
  file. The live sim is choppy because every control step costs real solver
  time; the recording plays back smoothly at the recorded rate in Isaac Sim.

Usage
-----
    # smooth offline recording, then open the file in Isaac Sim
    python terrain/scripts/24_play_escape.py --viewer usd --watch \\
        --watch-seconds 30
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch
from tensordict import TensorDict


def load_by_path(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(filename)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ESCAPE = load_by_path("escape_env", "22_escape_env.py")
TRAIN = load_by_path("train_escape", "23_train_escape.py")


def latest_checkpoint(root: Path) -> Path:
    """Newest model_*.pt under the newest run directory."""
    runs = sorted(p for p in root.iterdir() if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No training runs under {root}")
    ckpts = sorted(
        runs[-1].glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1])
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {runs[-1]}")
    return ckpts[-1]


def pack(obs: torch.Tensor, device) -> TensorDict:
    return TensorDict({"policy": obs.reshape(1, -1).to(device)}, batch_size=[1])


def run_episodes(cli, env, policy) -> None:
    for episode in range(cli.episodes):
        observation = env.get_observations()
        total, steps = 0.0, 0
        while True:
            with torch.inference_mode():
                action = policy(observation)
            observation, reward, done, extras = env.step(action)
            total += float(reward)
            steps += 1
            if bool(done):
                log = extras["log"]
                print(
                    f"episode {episode}: steps={steps} return={total:+.1f} "
                    f"exit={log['exit_distance_m']:.2f} m "
                    f"burial={log['burial_mm']:.0f} mm "
                    f"escaped={bool(log['escaped'])}",
                    flush=True,
                )
                break


def run_watch(cli, core, policy, device) -> None:
    """Continuous run: reset only on physical failure, never on success."""
    obs = core.reset()
    elapsed = 0.0
    while cli.watch_seconds <= 0 or elapsed < cli.watch_seconds:
        with torch.inference_mode():
            action = policy(pack(obs, device))
        obs, _, _, info = core.step(action[0])
        elapsed += 1.0 / 50.0
        if info["fallen"] or info["anchored_out"]:
            print(
                f"  reset ({'fell' if info['fallen'] else 'anchored out'}) "
                f"exit={info['exit_distance_m']:.2f} m  t={elapsed:.1f} s",
                flush=True,
            )
            obs = core.reset()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-seconds", type=float, default=10.0)
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuous viewing; resets only on fall or anchoring failure.",
    )
    parser.add_argument(
        "--watch-seconds",
        type=float,
        default=0.0,
        help="Stop --watch after this long; 0 runs until the window closes.",
    )
    parser.add_argument("--viewer", choices=("gl", "null", "usd"), default="gl")
    parser.add_argument(
        "--usd-output",
        type=Path,
        default=Path("terrain/output/escape_env/replay.usd"),
        help="Where --viewer usd writes the recorded animation.",
    )
    parser.add_argument("--usd-fps", type=int, default=15)
    parser.add_argument(
        "--snapshot-path", type=Path, default=ESCAPE.DEFAULT_SNAPSHOT
    )
    cli, passthrough = parser.parse_known_args()

    checkpoint = (
        cli.checkpoint
        if cli.checkpoint is not None
        else latest_checkpoint(Path("terrain/output/escape_training"))
    )

    # Replay is for watching: unlike training, draw the sand properly. The
    # USD recorder skips the visual grains -- millions of points per time
    # sample make the file unmanageable; material points carry the motion.
    grain_args = (
        ["--render-grains-per-particle", "0"]
        if cli.viewer == "usd"
        else [
            "--render-grains-per-particle", "32",
            "--render-grain-radius", "0.0035",
        ]
    )
    env_args = ESCAPE.build_args(
        ["--episode-seconds", str(cli.episode_seconds)] + grain_args + passthrough
    )
    env_args.viewer = cli.viewer
    env_args.usd_output = str(cli.usd_output)
    env_args.usd_fps = cli.usd_fps
    env_args.snapshot_path = cli.snapshot_path

    cli.usd_output.parent.mkdir(parents=True, exist_ok=True)
    core = ESCAPE.EscapeEnv(env_args)
    core.load_snapshot(cli.snapshot_path.expanduser().resolve())
    env = TRAIN.EscapeVecEnv(core)

    from rsl_rl.runners import OnPolicyRunner

    # The network shape must match the checkpoint being replayed, not the
    # current training default -- read the hidden dims off the checkpoint.
    saved = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    hidden = [
        saved["actor_state_dict"][k].shape[0]
        for k in ("mlp.0.weight", "mlp.2.weight", "mlp.4.weight")
    ]
    cfg = TRAIN.train_cfg(argparse.Namespace(steps_per_env=24, save_interval=10**9))
    cfg["actor"]["hidden_dims"] = hidden
    cfg["critic"]["hidden_dims"] = hidden
    runner = OnPolicyRunner(env, cfg, log_dir=None, device=str(env.device))
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=str(env.device))
    print(f"[play] checkpoint: {checkpoint}  (hidden {hidden})")

    try:
        if cli.watch:
            run_watch(cli, core, policy, env.device)
        else:
            run_episodes(cli, env, policy)
    finally:
        closer = getattr(core.sim.viewer, "close", None)
        if closer is not None:
            closer()
        if cli.viewer == "usd":
            print(f"[play] USD animation written: {cli.usd_output}")


if __name__ == "__main__":
    main()
