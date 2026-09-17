# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the G1 escape policy with RSL-RL PPO on the vectorised entrapment env.

G1 counterpart of ``27_train_escape_vec.py``: identical trainer, pointed at
``32_g1_escape_env_vec.py``. Nothing here is robot-specific -- the adapter
reads its dimensions off the environment -- so the two stay in step.

The warm start is the G1 Mars walking checkpoint. Its actor is the same
[512, 256, 128] elu MLP this trainer builds, so ``runner.load`` transfers the
learned balance directly; what it does not transfer is a sense of scale for
exploration, hence ``--max-action-std``.

Usage
-----
    python terrain/scripts/33_train_g1_escape_vec.py --num-envs 16 \
        --iterations 300 --hold-seconds 0.5 \
        --init-checkpoint logs/rsl_rl/g1_mars/<run>/model_4999.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
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


VEC = load_by_path("g1_escape_env_vec", "32_g1_escape_env_vec.py")
TRAIN_SINGLE = load_by_path("train_escape_single", "23_train_escape.py")


class EscapeVecAdapter:
    """RSL-RL VecEnv adapter over EscapeEnvVec (N environments)."""

    def __init__(self, env: "VEC.EscapeEnvVec"):
        self._env = env
        self.num_envs = env.num_envs
        self.num_actions = VEC.ACTION_DIM
        self.max_episode_length = env.max_episode_steps
        self.device = env.device
        self.episode_length_buf = torch.zeros(
            env.num_envs, dtype=torch.long, device=self.device
        )
        self.cfg = {"task": "mars-escape-vec", "num_obs": VEC.OBS_DIM}
        self._obs: torch.Tensor | None = None

    def _pack(self, obs: torch.Tensor) -> TensorDict:
        return TensorDict({"policy": obs}, batch_size=[self.num_envs])

    def get_observations(self) -> TensorDict:
        if self._obs is None:
            self._obs = self._env.reset_all()
        return self._pack(self._obs)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs, reward, done, info = self._env.step(actions.to(self.device))
        self.episode_length_buf += 1
        self.episode_length_buf[done] = 0
        extras = {
            "log": {
                "exit_distance_m": float(info["exit_distance"].mean()),
                "support_sink_mm": float(1000.0 * info["support_sink"].mean()),
                "on_feet_frac": float(
                    info["on_feet"].float().mean() if "on_feet" in info else 0.0
                ),
                "progress_dist_mm": float(
                    1000.0 * info["progress_dist"].mean()
                    if "progress_dist" in info
                    else 0.0
                ),
                # Count the escape once, on the step it happens. Escaping no
                # longer ends the episode, so the latched flag stays true for
                # every step that follows and would multiply the count by the
                # length of the tail.
                "escaped_per_step": float(
                    info["newly_escaped"].float().sum()
                    if "newly_escaped" in info
                    else info["escaped"].float().sum()
                ),
                "fallen_per_step": float(info["fallen"].float().sum()),
                "anchored_per_step": float(info["anchored"].float().sum()),
                # Optional: older environment revisions do not report these,
                # and the trainer has to stay runnable against them for
                # bisecting a regression.
                "diverged_per_step": float(
                    info["diverged"].float().sum() if "diverged" in info else 0.0
                ),
                "action_nan_per_step": float(
                    info["action_nan"].float().sum() if "action_nan" in info else 0.0
                ),
                "soil_bad_per_step": float(
                    info["soil_bad"].float().sum() if "soil_bad" in info else 0.0
                ),
            },
            "time_outs": info["timeout"].to(self.device),
        }
        self._obs = obs
        return self._pack(obs), reward.float(), done, extras


def _cap_exploration(runner, ceiling: float) -> None:
    """Cap the loaded policy's per-joint exploration standard deviation.

    The published walking checkpoint explores its ankles at 1.15 rad against
    0.07 at the hips, which is reasonable for the task it was trained on and
    is not for this one: the ankle is the joint buried in the material, and
    slamming it through half a radian every 20 ms is what drives the soil
    solve to break. The same checkpoint run without exploration completes
    hundreds of steps untouched.

    Only the extremes are pulled in. The relative structure the walker learned
    is left alone, and nothing here limits what the policy can command -- the
    mean action is untouched, and the actuator's effort limit is still the
    only bound on what the robot may do.

    Args:
        runner: The RSL-RL on-policy runner, already loaded.
        ceiling: Largest per-joint standard deviation to keep [rad].
    """
    actor = runner.alg.actor
    for name, param in actor.named_parameters():
        if not name.endswith("std_param"):
            continue
        with torch.no_grad():
            before = param.detach().clone()
            param.clamp_(max=ceiling)
        print(
            f"[train] exploration std capped at {ceiling}: "
            f"max {float(before.max()):.2f} -> {float(param.max()):.2f}, "
            f"mean {float(before.mean()):.2f} -> {float(param.mean()):.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--steps-per-env", type=int, default=64)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--episode-seconds", type=float, default=10.0)
    parser.add_argument(
        "--max-action-std",
        type=float,
        default=0.3,
        help=(
            "Cap on the warm-started policy's per-joint exploration standard "
            "deviation [rad]. The published walking checkpoint explores its "
            "ankles at 1.15, which is the joint sitting in the soil."
        ),
    )
    parser.add_argument(
        "--log-dir", type=Path, default=Path("terrain/output/escape_training")
    )
    parser.add_argument(
        "--snapshot-path", type=Path, default=VEC.DEFAULT_SNAPSHOT
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help=(
            "Warm-start from this checkpoint (walking or a previous escape "
            "round); the network shape [512, 256, 128] and state-dict layout "
            "match, so learned balance transfers directly."
        ),
    )
    cli, passthrough = parser.parse_known_args()

    env_args = VEC.build_args(
        [
            "--num-envs",
            str(cli.num_envs),
            "--episode-seconds",
            str(cli.episode_seconds),
        ]
        + passthrough
    )
    env_args.snapshot_path = cli.snapshot_path

    core = VEC.EscapeEnvVec(env_args)
    env = EscapeVecAdapter(core)

    from rsl_rl.runners import OnPolicyRunner

    log_dir = cli.log_dir.expanduser().resolve() / time.strftime("%m%d_%H%M")
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg = TRAIN_SINGLE.train_cfg(cli)
    cfg["run_name"] = f"vec_{cli.num_envs}env"
    runner = OnPolicyRunner(
        env, cfg, log_dir=str(log_dir), device=str(env.device)
    )
    if cli.init_checkpoint is not None:
        runner.load(str(cli.init_checkpoint.expanduser().resolve()))
        runner.current_learning_iteration = 0
        print(f"[train] warm-started from {cli.init_checkpoint}")
        _cap_exploration(runner, cli.max_action_std)
    print(f"[train] envs={cli.num_envs}  logging to {log_dir}")
    runner.learn(num_learning_iterations=cli.iterations)
    print("[train] done")


if __name__ == "__main__":
    main()
