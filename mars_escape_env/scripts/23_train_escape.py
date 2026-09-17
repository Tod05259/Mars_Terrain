# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the escape policy with RSL-RL PPO on the entrapment environment.

Wraps :class:`EscapeEnv` (22_escape_env.py) in the ``VecEnv`` contract of the
installed RSL-RL and runs ``OnPolicyRunner``. Starts as a single environment —
sample-inefficient, but it proves the learning loop end to end and lets reward
shaping begin while the vectorised version is built.

Auto-reset semantics follow RSL-RL: when an episode ends, the returned
observation is already the first one of the next episode, and the ``time_outs``
extra marks terminations that were timeouts rather than task outcomes.

Usage
-----
    python terrain/scripts/23_train_escape.py --iterations 50
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


ESCAPE = load_by_path("escape_env", "22_escape_env.py")


class EscapeVecEnv:
    """RSL-RL VecEnv adapter over a single EscapeEnv."""

    def __init__(self, env: "ESCAPE.EscapeEnv"):
        self._env = env
        self.num_envs = 1
        self.num_actions = env.action_dim
        self.max_episode_length = env.max_episode_steps
        self.device = env.device
        self.episode_length_buf = torch.zeros(
            1, dtype=torch.long, device=self.device
        )
        self.cfg = {"task": "mars-escape", "num_obs": env.obs_dim}
        self._obs = None

    def _pack(self, obs: torch.Tensor) -> TensorDict:
        return TensorDict(
            {"policy": obs.reshape(1, -1).to(self.device)}, batch_size=[1]
        )

    def get_observations(self) -> TensorDict:
        if self._obs is None:
            self._obs = self._env.reset()
        return self._pack(self._obs)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs, reward, done, info = self._env.step(actions[0])
        self.episode_length_buf += 1

        timeout = bool(
            done
            and not (info["escaped"] or info["fallen"] or info["anchored_out"])
        )
        extras = {
            "log": {
                "exit_distance_m": info["exit_distance_m"],
                "burial_mm": info["burial_mm"],
                "support_sink_mm": info["support_sink_mm"],
                "escaped": float(info["escaped"]),
            },
            "time_outs": torch.tensor([timeout], device=self.device),
        }
        if done:
            obs = self._env.reset()
            self.episode_length_buf.zero_()
        self._obs = obs
        return (
            self._pack(obs),
            torch.tensor([reward], device=self.device, dtype=torch.float32),
            torch.tensor([done], device=self.device, dtype=torch.bool),
            extras,
        )


def train_cfg(args: argparse.Namespace) -> dict:
    """Minimal PPO configuration matching the installed RSL-RL contract."""
    return {
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "rsl_rl.algorithms.PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 3.0e-4,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "actor": {
            "class_name": "rsl_rl.models.MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            # The stochastic head lives in a distribution config in this
            # RSL-RL, not in a top-level noise_std.
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
                "init_std": 0.3,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "rsl_rl.models.MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "num_steps_per_env": args.steps_per_env,
        "save_interval": args.save_interval,
        "empirical_normalization": False,
        "logger": "tensorboard",
        "experiment_name": "mars_escape",
        "run_name": "single_env",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--steps-per-env", type=int, default=64)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--episode-seconds", type=float, default=8.0)
    parser.add_argument(
        "--log-dir", type=Path, default=Path("terrain/output/escape_training")
    )
    parser.add_argument(
        "--snapshot-path", type=Path, default=ESCAPE.DEFAULT_SNAPSHOT
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help=(
            "Warm-start from this checkpoint. The published H1 walking "
            "checkpoint shares both the network shape [512, 256, 128] and the "
            "state-dict layout, so the balance it already knows transfers "
            "directly instead of being relearned from scratch."
        ),
    )
    cli, passthrough = parser.parse_known_args()

    env_args = ESCAPE.build_args(
        ["--episode-seconds", str(cli.episode_seconds)] + passthrough
    )
    env_args.snapshot_path = cli.snapshot_path

    core = ESCAPE.EscapeEnv(env_args)
    core.load_snapshot(cli.snapshot_path.expanduser().resolve())
    env = EscapeVecEnv(core)

    from rsl_rl.runners import OnPolicyRunner

    log_dir = cli.log_dir.expanduser().resolve() / time.strftime("%m%d_%H%M")
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(
        env, train_cfg(cli), log_dir=str(log_dir), device=str(env.device)
    )
    if cli.init_checkpoint is not None:
        runner.load(str(cli.init_checkpoint.expanduser().resolve()))
        runner.current_learning_iteration = 0
        print(f"[train] warm-started from {cli.init_checkpoint}")
    print(f"[train] logging to {log_dir}")
    runner.learn(num_learning_iterations=cli.iterations)
    print("[train] done")


if __name__ == "__main__":
    main()
