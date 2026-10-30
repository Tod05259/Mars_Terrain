"""MDP terms of the uphill task (``Mars-Ascend-H1-v1``) that the walking task does not have.

Both are aimed at the gait the team's lab walker ``walk1`` learned on the same ground and that the
sand judge rejected (TEAM_GUIDE 2026-10-11 20:04): both feet off the ground in 16-23 % of the
steps and the pelvis lowered to 0.56-0.58 m above the lowest ankle (the judge's physical validity
allows 5 % and 0.66 m). Both are gated on an upright trunk, so that they say nothing while the
robot is falling: a gait penalty that fires during the falls of early training teaches the policy
to keep its feet down instead (lesson recorded in this project's notes, 2026-08-08).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from getup_lab import walk_mdp

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def _upright(asset: Articulation, max_tilt: float) -> torch.Tensor:
    """Trunk within ``max_tilt`` [rad] of vertical, shape [num_envs]."""
    up = -asset.data.projected_gravity_b.torch[:, 2]
    return torch.arccos(torch.clamp(up, -1.0, 1.0)) < max_tilt


def crouch_below(env: ManagerBasedRLEnv, height: float, max_tilt: float = 0.5,
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Squared shortfall of the root below ``height`` above the lowest ankle [m^2], upright robots only.

    The judge reads the same quantity (torso link above the lower ankle link; on the H1 the pelvis
    and the torso link share their origin) and rejects an episode whose minimum is under 0.66 m.

    Args:
        env: The environment.
        height: Root height above the lowest ankle below which the term is non-zero [m].
        max_tilt: Trunk tilt above which the term is zero [rad].
        asset_cfg: The robot.

    Returns:
        ``clamp(height - h, min=0)^2`` [m^2], shape [num_envs].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    state = walk_mdp._walk_state(env)
    if "ankle_ids" not in state:
        state["ankle_ids"] = asset.find_bodies(".*ankle_link")[0]
    lowest = asset.data.body_pos_w.torch[:, state["ankle_ids"], 2].min(dim=1)[0]
    h = asset.data.root_pos_w.torch[:, 2] - lowest
    return torch.clamp(height - h, min=0.0).square() * _upright(asset, max_tilt)


def track_travel(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None,
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Keep, per env, the farthest horizontal distance from its origin reached in this episode [m].

    An interval event run every control step; :func:`terrain_levels_climb` reads and clears it.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    state = walk_mdp._walk_state(env)
    distance = torch.linalg.norm(asset.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2], dim=1)
    if "max_travel" not in state:
        # allocated outside inference mode and only written in place (see walk_mdp.feet_air_time_biped_soil)
        with torch.inference_mode(False):
            state["max_travel"] = torch.zeros(env.num_envs, device=distance.device)
    torch.maximum(state["max_travel"], distance, out=state["max_travel"])


def terrain_levels_climb(env: ManagerBasedRLEnv, env_ids: torch.Tensor, up_distance: float = 3.5,
                         down_distance: float = 1.5) -> dict[str, float]:
    """Terrain-level curriculum on how far up the face a robot got, for the uphill task.

    The stock ``terrain_levels_vel`` promotes a robot that ends its episode more than half a tile (4 m)
    from its origin and demotes one that covered less than half its commanded distance. With this
    task's commands (0-0.5 m/s, a new heading after 10 s) neither the end position nor the commanded
    distance measures a climb: the first run held the mean level at 1.5 (about 5 deg) through 800
    iterations. Here the farthest distance from the origin reached during the episode decides
    (:func:`track_travel`): beyond ``up_distance`` the robot climbed the face (3.5 m is the 0.75 m
    platform plus 2.75 m of run, i.e. 3 m along a 23 deg face) and moves to a steeper row; under
    ``down_distance`` it hardly left the platform and moves down.

    Args:
        env: The environment.
        env_ids: Envs being reset.
        up_distance: Farthest distance from the origin that promotes [m].
        down_distance: Farthest distance from the origin under which the robot is demoted [m].

    Returns:
        The mean terrain level and the share of envs on the rows of the judge's slopes (row 6 and up,
        16 deg and steeper with 10 rows to 26 deg).
    """
    terrain = env.scene.terrain
    state = walk_mdp._walk_state(env)
    if len(env_ids) > 0 and "max_travel" in state:
        reached = state["max_travel"][env_ids]
        move_up = reached > up_distance
        move_down = (reached < down_distance) & ~move_up
        terrain.update_env_origins(env_ids, move_up, move_down)
        state["max_travel"][env_ids] = 0.0
    levels = terrain.terrain_levels.float()
    return {"mean": float(levels.mean()), "share_row6_up": float((levels >= 6).float().mean())}


def flight(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_tilt: float = 0.5,
           soil_threshold: float = walk_mdp.SOIL_SUPPORT_FORCE_N,
           asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """1 when neither foot is on the ground (rigid contact or soil support), upright robots only.

    Args:
        env: The environment.
        sensor_cfg: The contact sensor with the feet as its bodies.
        max_tilt: Trunk tilt above which the term is zero [rad].
        soil_threshold: Upward soil force on a foot that counts as contact [N].
        asset_cfg: The robot.

    Returns:
        Flight indicator, shape [num_envs].
    """
    contact = walk_mdp.feet_contact_soil(env, sensor_cfg, asset_cfg, soil_threshold=soil_threshold)
    asset: Articulation = env.scene[asset_cfg.name]
    return (~contact.any(dim=1)).float() * _upright(asset, max_tilt)
