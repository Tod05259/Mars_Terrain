"""CPU tests of the uphill terms (``uphill_lab.ascend_mdp``), on a stub environment.

    run.ps1 -m pytest tests\\test_ascend_mdp.py -q      (no Isaac Sim app, no GPU)
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import torch

from getup_lab import walk_mdp
from uphill_lab import ascend_mdp

BODY_NAMES = ["pelvis", "torso_link", "left_ankle_link", "right_ankle_link"]


def proxy(tensor: torch.Tensor) -> SimpleNamespace:
    """The ``.torch`` side of Isaac Lab's ProxyArray."""
    return SimpleNamespace(torch=tensor)


class StubRobot:
    def __init__(self, n: int):
        self.root = torch.zeros(n, 3)
        self.bodies = torch.zeros(n, len(BODY_NAMES), 3)
        self.gravity = torch.tensor([[0.0, 0.0, -1.0]]).repeat(n, 1)
        self.data = SimpleNamespace(
            root_pos_w=proxy(self.root), body_pos_w=proxy(self.bodies), projected_gravity_b=proxy(self.gravity)
        )

    def find_bodies(self, pattern: str):
        ids = [i for i, name in enumerate(BODY_NAMES) if name.endswith("ankle_link")]
        assert pattern == ".*ankle_link"
        return ids, [BODY_NAMES[i] for i in ids]


class StubTerrain:
    def __init__(self, n: int):
        self.terrain_levels = torch.full((n,), 3, dtype=torch.long)
        self.calls = []

    def update_env_origins(self, env_ids, move_up, move_down):
        self.calls.append((env_ids.clone(), move_up.clone(), move_down.clone()))
        self.terrain_levels[env_ids] += move_up.long() - move_down.long()


class StubScene(dict):
    def __init__(self, n: int):
        super().__init__(robot=StubRobot(n))
        self.env_origins = torch.zeros(n, 3)
        self.terrain = StubTerrain(n)


def make_env(n: int = 4) -> SimpleNamespace:
    return SimpleNamespace(scene=StubScene(n), num_envs=n)


def test_track_travel_keeps_the_farthest_distance():
    env = make_env()
    robot = env.scene["robot"]
    robot.root[:, :2] = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0], [0.0, 0.0]])
    ascend_mdp.track_travel(env, None)
    robot.root[:, :2] = 0.0  # coming back does not lower the record
    ascend_mdp.track_travel(env, None)
    assert torch.allclose(walk_mdp._walk_state(env)["max_travel"], torch.tensor([1.0, 2.0, 5.0, 0.0]))


def test_terrain_levels_climb_promotes_demotes_and_clears_only_reset_envs():
    env = make_env()
    robot = env.scene["robot"]
    robot.root[:, :2] = torch.tensor([[4.0, 0.0], [1.0, 0.0], [2.5, 0.0], [0.0, 3.6]])
    ascend_mdp.track_travel(env, None)
    log = ascend_mdp.terrain_levels_climb(env, torch.tensor([0, 1, 2]), up_distance=3.5, down_distance=1.5)
    (ids, up, down), = env.scene.terrain.calls
    assert ids.tolist() == [0, 1, 2]
    assert up.tolist() == [True, False, False]
    assert down.tolist() == [False, True, False]
    assert env.scene.terrain.terrain_levels.tolist() == [4, 2, 3, 3]
    # the reset envs start their next episode from zero, env 3 keeps its record
    assert torch.allclose(walk_mdp._walk_state(env)["max_travel"], torch.tensor([0.0, 0.0, 0.0, 3.6]))
    assert math.isclose(log["mean"], 3.0)
    assert log["share_row6_up"] == 0.0


def test_terrain_levels_climb_without_a_record_changes_nothing():
    env = make_env()
    ascend_mdp.terrain_levels_climb(env, torch.tensor([0, 1]))
    assert env.scene.terrain.calls == []


def test_crouch_below_is_the_squared_shortfall_of_an_upright_robot():
    env = make_env(3)
    robot = env.scene["robot"]
    robot.root[:, 2] = torch.tensor([1.0, 0.6, 0.6])
    robot.bodies[:, 2, 2] = 0.1  # left ankle
    robot.bodies[:, 3, 2] = 0.0  # right ankle: the lowest
    robot.gravity[2] = torch.tensor([math.sin(0.8), 0.0, -math.cos(0.8)])  # env 2 tilted 0.8 rad
    value = ascend_mdp.crouch_below(env, height=0.8, max_tilt=0.5)
    assert torch.allclose(value, torch.tensor([0.0, 0.04, 0.0]))


def test_flight_counts_no_foot_down_on_upright_robots(monkeypatch):
    env = make_env(3)
    env.scene["robot"].gravity[2] = torch.tensor([math.sin(0.8), 0.0, -math.cos(0.8)])
    contact = torch.tensor([[False, False], [True, False], [False, False]])
    monkeypatch.setattr(walk_mdp, "feet_contact_soil", lambda *args, **kwargs: contact)
    value = ascend_mdp.flight(env, sensor_cfg=None, max_tilt=0.5)
    assert value.tolist() == [1.0, 0.0, 0.0]
