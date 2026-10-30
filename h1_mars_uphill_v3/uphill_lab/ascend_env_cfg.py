"""H1 uphill walking on the soil surrogate (``Mars-Ascend-H1-v1``), for the team FSM's ASCEND mode.

The config subclasses CAPSTONE's lab walking task :class:`getup_lab.walk_env_cfg.H1WalkSoilEnvCfg`
(itself V8 of the get-up task), so everything the sand judge already accepts stays: the H1 with
every link colliding, the DC motors at the URDF ratings with saturation ratio 1.5, the clamped
joint targets, the 256-wide observation with the soil-surface height scan, the 19 joint targets at
scale 0.5, the soil reaction surrogate with its ground mix (a fifth rigid, beds 0.05-0.36 m at 0.5-4
times the fitted sand) and its level ramp, Mars gravity, 20 s episodes, the fall test of the chain.

What changes, and why (workflow(3) step 3):

* **terrain**: Isaac Lab's terrain-level curriculum on slopes up to 26 deg (the judge's steepest bed
  is 22.5 deg). Half the columns are inverted pyramids -- the robot starts on the bottom platform
  and every heading leads uphill, so walking out of the tile is a 3.25 m climb -- 15 % pyramids
  (descending, so a slip back is not out of distribution), 25 % mild roughness, 10 % flat. A robot
  whose farthest distance from its origin in the episode passed 3.5 m (3 m up a 23 deg face) moves
  to a steeper row, one that stayed within 1.5 m moves down
  (:func:`~uphill_lab.ascend_mdp.terrain_levels_climb`); they start on the four gentlest rows;
* **commands**: one heading per 10 s (the stock rough-terrain setting) instead of 4-8 s, so an
  episode on an inverted pyramid is a sustained climb; a fifth of the commands is a full stop, the
  supervisor's STAND, which on a slope means holding position on it;
* **rewards**: the walking rewards plus two terms against the gait the lab walker ``walk1`` was
  rejected for on the sand judge (flight, crouch; :mod:`uphill_lab.ascend_mdp`), and the feet-slide
  penalty doubled (walk1: stance foot carried forward at 0.4-0.5 m/s);
* **soil height field** cut from the whole terrain, since the curriculum moves env origins to rows
  the initial field would not cover.
"""

from __future__ import annotations

import math

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from getup_lab import walk_env_cfg as walk
from getup_lab import walk_mdp

from . import ascend_mdp

MAX_SLOPE_DEG = 26.0
"""Steepest slope of the curriculum [deg]: above the judge's 22.5 deg bed, below the 30.8 deg repose angle."""

PLATFORM_M = 1.5
"""Flat platform at the centre of a (inverted) pyramid tile [m]; the reset spreads robots +-0.5 m."""

CROUCH_HEIGHT_M = 0.80
"""Root height above the lowest ankle below which the crouch term pays a penalty [m]; the judge
rejects an episode under 0.66 m and the sand-trained walker stays at 0.73-0.99 m on the 16.8 deg bed."""

FEET = SceneEntityCfg("contact_forces", body_names=".*ankle_link")


@configclass
class H1AscendRewards(walk.H1WalkRewards):
    """:class:`getup_lab.walk_env_cfg.H1WalkRewards` with the feet slide doubled and two gait terms."""

    feet_slide = RewTerm(
        func=walk_mdp.feet_slide_soil,
        weight=-0.5,
        params={"sensor_cfg": FEET, "soil_threshold": walk.SOIL_CONTACT_N},
    )
    crouch = RewTerm(func=ascend_mdp.crouch_below, weight=-20.0, params={"height": CROUCH_HEIGHT_M})
    flight = RewTerm(
        func=ascend_mdp.flight,
        weight=-0.5,
        params={"sensor_cfg": FEET, "soil_threshold": walk.SOIL_CONTACT_N},
    )


@configclass
class AscendCurriculum(walk.WalkCurriculum):
    """The walking task's soil level, plus a terrain-level curriculum on the distance climbed.

    (The stock ``terrain_levels_vel`` held the mean level at 1.5 of 9 through 800 iterations of the
    first run: see :func:`uphill_lab.ascend_mdp.terrain_levels_climb`.)
    """

    terrain_levels = CurrTerm(func=ascend_mdp.terrain_levels_climb)


@configclass
class H1AscendSoilEnvCfg(walk.H1WalkSoilEnvCfg):
    """Uphill walking on the soil surrogate; see the module docstring."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards = H1AscendRewards()
        self.curriculum = AscendCurriculum()

        slope = math.tan(math.radians(MAX_SLOPE_DEG))
        generator = self.scene.terrain.terrain_generator
        generator.curriculum = True
        generator.sub_terrains = {
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
            "slope_up": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.5, slope_range=(0.0, slope), platform_width=PLATFORM_M, border_width=0.25
            ),
            "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.15, slope_range=(0.0, slope), platform_width=PLATFORM_M, border_width=0.25
            ),
            "rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.25, noise_range=(0.0, 0.06), noise_step=0.02, border_width=0.25
            ),
        }
        self.scene.terrain.max_init_terrain_level = 3
        # the curriculum moves origins between rows: cut the soil height field from the whole mesh
        self.actions.joint_pos.soil.height_field_margin = 1.0e3

        command = self.commands.base_velocity
        command.resampling_time_range = (10.0, 10.0)
        command.rel_standing_envs = 0.2

        # farthest distance from the origin per episode, read by the terrain curriculum
        self.events.track_travel = EventTerm(
            func=ascend_mdp.track_travel, mode="interval", interval_range_s=(0.02, 0.02), is_global_time=True
        )


CROUCH_HEIGHT_V2_M = 0.88
"""v2: crouch threshold [m]. On the sand judge v1 (``ascend_v2`` run, 1500 and 2500 iterations) lost most
of its invalid episodes to a minimum torso height of 0.58-0.65 m above the lowest ankle (bound 0.66 m),
and in the lab the same minimum sat at 0.67-0.73 m, so the 0.80 m threshold left it no margin."""


@configclass
class H1AscendRewardsV2(H1AscendRewards):
    """v2: the crouch term starts at 0.88 m and weighs -40 (v1: 0.80 m, -20); nothing else changes."""

    crouch = RewTerm(func=ascend_mdp.crouch_below, weight=-40.0, params={"height": CROUCH_HEIGHT_V2_M})


@configclass
class H1AscendSoilEnvCfgV2(H1AscendSoilEnvCfg):
    """``Mars-Ascend-H1-v2``: :class:`H1AscendSoilEnvCfg` with :class:`H1AscendRewardsV2` (one lever)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards = H1AscendRewardsV2()


MIN_SLOPE_V3_DEG = 8.0
"""v3: gentlest slope of the (inverted) pyramid rows [deg] (v1/v2: 0)."""


@configclass
class H1AscendSoilEnvCfgV3(H1AscendSoilEnvCfgV2):
    """``Mars-Ascend-H1-v3``: v2 with the pyramid rows spanning 8-26 deg instead of 0-26 deg (one lever).

    v2 climbed the 16.8 deg sand bed in 11 of 12 trials but the 20 deg one in none: slow (2.0-2.8 m in
    25-29 s) while drifting 4 m sideways out of the patch. In the lab it climbs 20 deg slowly as well
    (median 1.7 m in 20 s), and its curriculum held the mean row near 4 (about 10 deg), with a quarter
    of the robots on the rows of 16 deg and up. Starting the rows at 8 deg puts the same curriculum
    position on steeper ground; flat and rough tiles stay for walking.
    """

    def __post_init__(self):
        super().__post_init__()
        low = math.tan(math.radians(MIN_SLOPE_V3_DEG))
        high = math.tan(math.radians(MAX_SLOPE_DEG))
        terrains = self.scene.terrain.terrain_generator.sub_terrains
        terrains["slope_up"].slope_range = (low, high)
        terrains["slope_down"].slope_range = (low, high)


@configclass
class H1AscendRigidEnvCfg(H1AscendSoilEnvCfg):
    """The uphill recipe with the soil level held at 0: the ablation of the soil surrogate."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.soil_depth.params["max_level"] = 0.0


@configclass
class H1AscendSoilEnvCfg_EVAL(H1AscendSoilEnvCfg):
    """Lab scoring (``scripts/eval_lab.py``): inverted pyramids of one slope, the full ground mix.

    Upright starts at rest on the platform with a random yaw, one fixed command (0.3 m/s forward
    under a heading command, as the judge's supervisor gives it), no observation noise, no pushes,
    no trunk-mass randomisation, curricula off, soil level 1. The eval script sets the slope.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 512
        self.observations.policy.enable_corruption = False
        self.curriculum.soil_depth = None
        self.curriculum.terrain_levels = None
        self.actions.joint_pos.soil.level = 1.0

        generator = self.scene.terrain.terrain_generator
        generator.curriculum = False
        generator.num_rows = 4
        generator.num_cols = 8
        up = generator.sub_terrains["slope_up"]
        generator.sub_terrains = {"slope_up": up.replace(proportion=1.0)}
        self.scene.terrain.max_init_terrain_level = None

        command = self.commands.base_velocity
        command.resampling_time_range = (1.0e6, 1.0e6)
        command.rel_standing_envs = 0.0
        command.ranges.lin_vel_x = (0.3, 0.3)

        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_base.params = {"pose_range": {"yaw": (-math.pi, math.pi)}, "velocity_range": {}}
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
