# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vectorised escape environment: N trapped G1s in one Newton model.

Port of ``26_escape_env_vec.py`` (H1) to the Unitree G1 the Mars walking
policy was trained on. The physics, reward shaping, escape criterion and
reset machinery are carried over unchanged; what differs is the robot, and
the three things that differ about it are worth stating up front.

* **Colliders have to be built, not re-enabled.** ``g1.usd`` carries exactly
  two collision shapes (the feet); every other link is visual-only, so the
  H1 path of clearing ``COLLIDE_SHAPES`` on the links that must not touch
  soil has nothing to clear. Approximate capsules and boxes are added to the
  six links a fallen robot puts on the ground. Without them the robot falls
  through the terrain and can neither crawl nor stand up — the regression
  documented in EXTRACTION_HANDOFF §11.
* **The policy interface is 310/37, not 256/19.** Joint ordering, default
  pose, drive gains and limits are the ones the trained checkpoint resolved,
  read off the registered IsaacLab task rather than guessed from the USD.
* **G1 is lighter with a smaller foot.** Mars weight 120 N against H1's 191,
  on a 0.203 x 0.065 m sole: 9.0 kPa under one foot where H1 puts 3.7. The
  soil reaction clamps are scaled by weight, and the anchoring budget that
  makes the task hard for H1 has to be re-measured for G1 rather than
  extrapolated (see G1_escape.md §2.2).

Usage
-----
    # one-off: seed a G1 snapshot from the released (robot-independent) soil
    python terrain/scripts/32_g1_escape_env_vec.py --bootstrap-snapshot

    # build the real entrapment snapshot by walking G1 into the pocket
    python terrain/scripts/32_g1_escape_env_vec.py --make-snapshot \
        --walk-policy <g1 mars walking checkpoint>

    # smoke
    python terrain/scripts/32_g1_escape_env_vec.py --num-envs 4 --smoke

"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import warp as wp

import newton
import newton.utils
from newton import JointTargetMode
from newton.solvers import SolverImplicitMPM


def load_by_path(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(filename)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MARS = load_by_path("h1_mars_newton", "07_h1_mars_newton.py")

# ----------------------------------------------------------------------
# G1 layout. Every value here was read off the registered IsaacLab task the
# walking checkpoint was trained on (scripts/g1_mars_probe.py), not inferred
# from the USD: the policy's observation and action vectors are indexed by
# this ordering, so a plausible-looking reordering silently feeds the network
# one joint's angle in another's slot.
# ----------------------------------------------------------------------
G1_USD = Path(r"C:\Users\2hj05\repos\Mars_Terrain\usd(completed)\g1.usd")

G1_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_roll_joint",
    "left_five_joint",
    "left_six_joint",
    "left_three_joint",
    "left_four_joint",
    "left_zero_joint",
    "left_one_joint",
    "left_two_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
    "right_five_joint",
    "right_six_joint",
    "right_three_joint",
    "right_four_joint",
    "right_zero_joint",
    "right_one_joint",
    "right_two_joint",
]

G1_DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.20,
    "right_hip_pitch_joint": -0.20,
    "left_knee_joint": 0.42,
    "right_knee_joint": 0.42,
    "left_ankle_pitch_joint": -0.23,
    "right_ankle_pitch_joint": -0.23,
    "left_elbow_pitch_joint": 0.87,
    "right_elbow_pitch_joint": 0.87,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": 0.16,
    "right_shoulder_roll_joint": -0.16,
    "left_one_joint": 1.0,
    "right_one_joint": -1.0,
    "left_two_joint": 0.52,
    "right_two_joint": -0.52,
}


# Finger joints. G1_CFG drives them with the same 40 N·m/rad the arms get, on
# links of 14-60 g -- H1's lightest link is 446 g, so this regime does not
# exist there. Their rotor inertia is what makes the drive solvable.
G1_FINGER_TAGS = ("_zero_", "_one_", "_two_", "_three_", "_four_", "_five_", "_six_")


def g1_joint_drive(
    name: str, finger_effort: float = 300.0
) -> tuple[float, float, float, float]:
    """Position-drive parameters for a G1 joint.

    Armature is not decoration. A position drive on a link with no rotor
    inertia is stiff in the numerical sense, and while it survives a quiet
    rollout it stops being solvable once a policy commands it hard -- which is
    what the first training run did before the soil solve stopped being
    finite. ``G1_CFG`` carries these values and the port dropped them.

    Returns:
        ``(stiffness [N·m/rad], damping [N·m·s/rad], effort limit [N·m],
        armature [kg·m²])`` matching ``G1_CFG``'s implicit actuators.
    """
    if any(tag in name for tag in G1_FINGER_TAGS):
        # G1_CFG's 300 N.m is a blanket limit for the whole "arms" group, not
        # a per-finger specification; a real G1 finger actuator is order 1.
        return 40.0, 10.0, finger_effort, 0.001
    if "ankle" in name:
        return 20.0, 2.0, 20.0, 0.01
    if "hip_pitch" in name or "knee" in name or name == "torso_joint":
        return 200.0, 5.0, 300.0, 0.01
    if "hip_roll" in name or "hip_yaw" in name:
        return 150.0, 5.0, 300.0, 0.01
    return 40.0, 10.0, 300.0, 0.01


# Approximate ground-contact colliders, in each link's own frame, derived from
# the child-joint offsets in the USD (there is no geometry to fit). Boxes are
# (hx, hy, hz); capsules extend along their local +z, so the forearm one is
# rotated a quarter turn about y to lie along +x.
#
# The forearm capsule hangs off ``elbow_pitch_link`` and spans the whole
# 0.22 m through to the palm: ``elbow_roll_joint`` rotates about the same x
# axis the forearm lies on, so the swept volume barely moves with it and one
# shape covers both segments.
G1_CONTACT_BOXES = {
    "pelvis": ((0.0, 0.0, -0.050), (0.060, 0.085, 0.080)),
    "torso_link": ((-0.010, 0.0, 0.140), (0.065, 0.100, 0.160)),
}
G1_CONTACT_CAPSULES = {
    # link suffix: (centre, radius, half_height, axis)
    "knee_link": ((0.0, 0.0, -0.150), 0.045, 0.110, "z"),
    "elbow_pitch_link": ((0.110, 0.0, 0.0), 0.035, 0.075, "x"),
}

# Sole box in the ankle-roll link's frame, measured from the asset's collision
# mesh bounds: 0.2031 x 0.0655 x 0.0185 m, i.e. 0.01330 m^2 of contact. At the
# 120.0 N Mars weight that is 9.0 kPa under one foot against H1's 3.7.
FOOT_BOX_CENTRE_M = (0.03595, 0.0, -0.02515)
FOOT_BOX_HALF_M = (0.101550, 0.032700, 0.009250)

MARS_G = -3.721
CONTROL_DT = 0.02
SIM_SUBSTEPS = 4
# Two-way coupling constants, identical to 07_h1_mars_newton.py defaults: the
# soil calibration (ppc=3, substeps=2) and the anchoring-budget measurements
# were made with these values.
MPM_SUBSTEPS = 2
# Scaled from H1's 400 N / 80 N·m by Mars weight (120.0 N against 191.3 N,
# a factor of 0.63). These are a bound on a physical reaction, not a safety
# net: if the clamp binds on more than a couple of percent of steps the soil
# force the robot feels is no longer the one the solver computed. Measure the
# saturation counter before trusting a run (G1_escape.md Step 1).
REACTION_FORCE_LIMIT_N = 250.0
REACTION_TORQUE_LIMIT_NM = 50.0
# 07's height scan reports the sand surface (DTM + mpm_surface_offset), not
# the bare DTM; the snapshot was made with --mpm-depth 0.30.
SOIL_SURFACE_OFFSET_M = 0.30
PARTICLE_DAMPING = 0.98
PARTICLE_MAX_SPEED = 1.5
ACTION_DIM = 37
OBS_DIM = 310
SCAN_POINTS = 187

_WORKING_SNAPSHOT = Path("terrain/output/escape_env/g1_entrapment_snapshot.pkl")
_BOOTSTRAP_SNAPSHOT = Path("terrain/output/escape_env/g1_bootstrap_snapshot.pkl")
# The soil half of the released H1 snapshot is robot-independent -- the same
# calibrated gusev bed on the same site -- so it seeds the G1 bootstrap.
_H1_RELEASE_SNAPSHOT = Path("terrain/release/entrapment_snapshot.pkl")
DEFAULT_SNAPSHOT = (
    _WORKING_SNAPSHOT if _WORKING_SNAPSHOT.is_file() else _BOOTSTRAP_SNAPSHOT
)

SUPPORT_SINK_LIMIT_M = 0.20
# Functional support test, in place of a torso angle: the robot is on its
# feet when nothing except the feet is near the ground. The knees are the
# lowest non-foot links, so the margin has to clear a deep crouch without
# admitting a robot resting on its shins. G1 stands 0.72 m at the pelvis
# against H1's 1.06, so the H1 margin of 0.12 is scaled with the robot; the
# right way to set it is to log the lowest non-foot link over a few hundred
# steps of normal walking and pick a value under that (G1_escape.md Step 4).
NONFOOT_CLEARANCE_M = 0.08
# Recovered locomotion: planar speed at least half the command. The G1 policy
# was trained on lin_vel_x in [0.5, 1.5], so the escape command is 0.5 -- the
# bottom of its training range rather than H1's 0.3, which is outside it.
WALK_SPEED_MIN_MPS = 0.25
# How long the robot may be off its feet before the episode ends. A leap or a
# flip passes through this state and is allowed to land; only a robot that
# stays down has failed to recover.
DOWN_GRACE_STEPS = int(0.6 / CONTROL_DT)
EXIT_RADIUS_M = 1.0
# Burial that counts as having been caught by the hazard, so a walk-in
# episode cannot be scored as an escape it never needed to make.
TRAPPED_BURIAL_M = 0.10
# Cost of a diverged solve, kept on the scale of an episode return (roughly
# -20 to +10). At 200 it was a ten-fold outlier: value targets and advantages
# blew up and the policy's weights went non-finite mid-update. Divergence is
# kept rare by the action-rate term instead, which removes the violent
# motions that cause it, so the penalty does not have to carry that load.
DIVERGENCE_PENALTY = 20.0
# G1 pelvis height in the nominal standing pose [m], used to seat a walk-in
# start on the sand surface. This is the base_height target the walking
# policy was rewarded against, not the asset's spawn height.
STANDING_ROOT_HEIGHT_M = 0.72
# Gap left under the soles at a walk-in spawn [m]. Small enough that the drop
# is negligible, large enough that no foot starts inside the sand.
WALK_IN_CLEARANCE_M = 0.01
# 1.5 s upright hold past the exit radius: a 0.3 s hold was satisfiable by
# leaping out and crashing after the bonus banked (torso stays high through
# the flight); 1.5 s forces a landed, stabilised exit.
HOLD_STEPS = int(1.5 / CONTROL_DT)


@wp.kernel
def recycle_window_env(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_qd_grad: wp.array(dtype=wp.mat33),
    particle_elastic_strain: wp.array(dtype=wp.mat33),
    particle_jp: wp.array(dtype=float),
    particle_stress: wp.array(dtype=wp.mat33),
    particle_transform: wp.array(dtype=wp.mat33),
    reference_q: wp.array(dtype=wp.vec3),
    rest_z_offset: wp.array(dtype=float),
    recycle_count: wp.array(dtype=int),
    terrain_height: wp.array(dtype=float),
    rows: int,
    cols: int,
    dx: float,
    dy: float,
    center_x: float,
    center_y: float,
    length: float,
    width: float,
    env_offset_x: float,
    env_offset_y: float,
    young_modulus: wp.array(dtype=float),
    yield_pressure: wp.array(dtype=float),
    pristine_young: wp.array(dtype=float),
    pristine_yield: wp.array(dtype=float),
    pocket_x: float,
    pocket_y: float,
    pocket_radius: float,
    pocket_scale: float,
):
    """07's robot-following window recycler with an env-offset terrain lookup.

    Particle and centre coordinates are world coordinates (env offset
    included); the DTM grid is shared, so the height sample shifts back into
    site-local coordinates first.

    Recycled particles also revert to the pristine regional material:
    material is stored per particle index, so without this the soft-pocket
    particles left behind the walking robot teleport ahead of it and lay an
    invisible soft patch directly on the exit path.
    """
    index = wp.tid()
    position = particle_q[index]
    recycled = False
    if position[0] < center_x - 0.5 * length:
        position = wp.vec3(position[0] + length, position[1], position[2])
        recycled = True
    elif position[0] > center_x + 0.5 * length:
        position = wp.vec3(position[0] - length, position[1], position[2])
        recycled = True
    if position[1] < center_y - 0.5 * width:
        position = wp.vec3(position[0], position[1] + width, position[2])
        recycled = True
    elif position[1] > center_y + 0.5 * width:
        position = wp.vec3(position[0], position[1] - width, position[2])
        recycled = True
    if not recycled:
        return

    surface_z = MARS.sample_height_warp(
        terrain_height,
        rows,
        cols,
        dx,
        dy,
        position[0] - env_offset_x,
        position[1] - env_offset_y,
    )
    position = wp.vec3(position[0], position[1], surface_z + rest_z_offset[index])
    identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    particle_q[index] = position
    particle_qd[index] = wp.vec3(0.0)
    particle_qd_grad[index] = wp.mat33(0.0)
    particle_elastic_strain[index] = identity
    particle_jp[index] = 1.0
    particle_stress[index] = wp.mat33(0.0)
    particle_transform[index] = identity
    reference_q[index] = position
    # Material belongs to the ground, not to the particle. Assigning it by
    # index meant the soft pocket travelled with whichever particles the
    # window happened to wrap, so it followed the robot instead of staying
    # in the hazard. Re-derive it from where the particle now sits.
    young = pristine_young[index]
    yield_p = pristine_yield[index]
    radial = wp.sqrt(
        (position[0] - pocket_x) * (position[0] - pocket_x)
        + (position[1] - pocket_y) * (position[1] - pocket_y)
    )
    if radial < pocket_radius:
        young = young * pocket_scale
        yield_p = yield_p * pocket_scale
    young_modulus[index] = young
    yield_pressure[index] = yield_p
    wp.atomic_add(recycle_count, 0, 1)


class _RigidOnlyStateView:
    """State proxy that hides particles from the rigid collision pipeline.

    ``CollisionPipeline.collide`` launches its particle-shape (soft contact)
    pass whenever ``state.particle_q`` is set; that pass is sized
    particles x ALL shapes and cannot scale to many envs. Feet-soil coupling
    runs through the MPM solver's own collider and MuJoCo ignores soft
    contacts, so presenting ``particle_q = None`` skips the pass losslessly.
    """

    def __init__(self, state):
        self._state = state

    def __getattr__(self, name):
        if name == "particle_q":
            return None
        return getattr(self._state, name)


class EscapeEnvVec:
    """N-fold entrapment scenario sharing one robot solver and one MPM solve."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.diag: dict | None = None
        self.reaction_force_limit = float(
            getattr(args, "reaction_force_limit", REACTION_FORCE_LIMIT_N)
        )
        self.reaction_torque_limit = float(
            getattr(args, "reaction_torque_limit", REACTION_TORQUE_LIMIT_NM)
        )
        self.num_envs = int(args.num_envs)
        self.pitch = float(args.env_pitch)
        side = int(np.ceil(np.sqrt(self.num_envs)))
        self.offsets = np.array(
            [
                (
                    (i % side - 0.5 * (side - 1)) * self.pitch,
                    (i // side - 0.5 * (side - 1)) * self.pitch,
                )
                for i in range(self.num_envs)
            ],
            dtype=np.float64,
        )

        with open(args.snapshot_path, "rb") as stream:
            snap = pickle.load(stream)
        self.trapped_index = int(snap.pop("trapped_index"))
        self.support_index = 1 - self.trapped_index
        # Undisturbed particle positions: sinkage is measured as surface-
        # particle depression against these, exactly like 07/22.
        self._reference_q = snap.pop("reference_q")
        self._snap = snap

        # The snapshot site is NOT flat: the real DTM has ~0.6 m of relief
        # across the 2x2 m soil window (particle z spans ~0.9 m). A flat
        # ground plane therefore cannot support the soil, and a constant
        # height scan misleads the policy by up to 0.74. Each env gets a
        # rigid copy of the local DTM patch instead, and the scan samples it.
        self.site_xy = np.array(args.site_xy, dtype=np.float64)
        height_full, self.tdx, self.tdy = MARS.load_terrain_grid(
            Path(args.terrain_usd).expanduser().resolve()
        )
        self.height_full = height_full
        rows, cols = height_full.shape
        half = 0.5 * self.pitch
        grid_x = (np.arange(cols) - (cols - 1) * 0.5) * self.tdx
        grid_y = ((rows - 1) * 0.5 - np.arange(rows)) * self.tdy
        col_keep = np.abs(grid_x - self.site_xy[0]) <= half + self.tdx
        row_keep = np.abs(grid_y - self.site_xy[1]) <= half + self.tdy
        self.patch_h = height_full[np.ix_(row_keep, col_keep)].astype(np.float32)
        self.patch_x = (grid_x[col_keep] - self.site_xy[0]).astype(np.float32)
        self.patch_y = (grid_y[row_keep] - self.site_xy[1]).astype(np.float32)

        # Surface-particle mask, reconstructed from the undisturbed heights:
        # spawn layers sit at DTM + (i + 0.5) * spacing for i in 0..11, so the
        # top layer is the band above DTM + 0.26 m.
        ref = self._reference_q
        rel = ref[:, 2] - MARS.sample_height_numpy(
            height_full, self.tdx, self.tdy, ref[:, 0], ref[:, 1]
        )
        self.surface_mask_np = rel > 0.26
        # Per-particle spawn height above the DTM: the recycler re-seats
        # recycled particles at surface + this offset (07's rest_z_offsets).
        self.rest_z_np = rel.astype(np.float32)
        # Catcher floor for any particle that leaks past the patch edge.
        self.ground_z = float(self.patch_h.min()) - 0.30

        self._build()
        self._init_torch()
        print(
            f"[EscapeVec] envs={self.num_envs} particles={self.model.particle_count} "
            f"bodies={self.model.body_count} pitch={self.pitch} m "
            f"trapped_foot={'left' if self.trapped_index == 0 else 'right'}"
        )

    # ------------------------------------------------------------------
    def _add_g1_shapes(self, builder, body_start: int, material) -> int:
        """Give one G1 the collision geometry the asset does not ship.

        Two jobs, both consequences of ``g1.usd`` carrying nothing but a
        convex-mesh sole on each foot:

        * **Soles.** The imported convex meshes are switched off and replaced
          by the box those meshes already are (0.2031 x 0.0655 x 0.0185 m,
          measured off the asset). The implicit-MPM collider cannot build a
          surface from ``CONVEX_MESH``, and a rectangular sole is also exactly
          the patch shape the soil was characterised with.
        * **Ground contacts.** Coarse capsules and boxes on the six links a
          fallen robot puts on the ground. Without them the robot free-falls
          through the terrain and can neither crawl nor stand up -- the
          regression in EXTRACTION_HANDOFF §11.

        Args:
            builder: The model builder, inside this env's world block.
            body_start: Index of this robot's first body in the builder.
            material: Regional soil material, for the contact stiffnesses.

        Returns:
            Number of shapes added.
        """
        # Density 0: these stand in for geometry the asset never had, and the
        # link masses are already correct.
        def cfg(particles: bool) -> newton.ModelBuilder.ShapeConfig:
            return newton.ModelBuilder.ShapeConfig(
                ke=material.terrain_ke,
                kd=material.terrain_kd,
                kf=0.1 * material.terrain_ke,
                mu=material.friction,
                density=0.0,
                has_particle_collision=particles,
                has_shape_collision=True,
            )

        # Soil stays a feet-only interaction: the soft-contact pass allocates
        # per particle and per shape, and the two-way coupling is calibrated
        # at the foot.
        sole_cfg, contact_cfg = cfg(True), cfg(False)
        bodies = range(body_start, len(builder.body_label))
        feet = [
            b for b in bodies if builder.body_label[b].endswith("ankle_roll_link")
        ]
        for shape_id, body_id in enumerate(builder.shape_body):
            if body_id in feet:
                builder.shape_flags[shape_id] &= ~newton.ShapeFlags.COLLIDE_PARTICLES
                builder.shape_flags[shape_id] &= ~newton.ShapeFlags.COLLIDE_SHAPES

        added = 0
        for body_id in feet:
            builder.add_shape_box(
                body_id,
                xform=wp.transform(wp.vec3(*FOOT_BOX_CENTRE_M), wp.quat_identity()),
                hx=FOOT_BOX_HALF_M[0],
                hy=FOOT_BOX_HALF_M[1],
                hz=FOOT_BOX_HALF_M[2],
                cfg=sole_cfg,
                label="sole",
            )
            added += 1

        quat_z_to_x = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * np.pi)
        for body_id in bodies:
            name = builder.body_label[body_id].rsplit("/", 1)[-1]
            for suffix, (centre, half) in G1_CONTACT_BOXES.items():
                if name.endswith(suffix):
                    builder.add_shape_box(
                        body_id,
                        xform=wp.transform(wp.vec3(*centre), wp.quat_identity()),
                        hx=half[0],
                        hy=half[1],
                        hz=half[2],
                        cfg=contact_cfg,
                        label=f"{name}_ground",
                    )
                    added += 1
            for suffix, (centre, radius, half_h, axis) in G1_CONTACT_CAPSULES.items():
                if name.endswith(suffix):
                    rot = quat_z_to_x if axis == "x" else wp.quat_identity()
                    builder.add_shape_capsule(
                        body_id,
                        xform=wp.transform(wp.vec3(*centre), rot),
                        radius=radius,
                        half_height=half_h,
                        cfg=contact_cfg,
                        label=f"{name}_ground",
                    )
                    added += 1
        return added

    # ------------------------------------------------------------------
    def _build(self) -> None:
        material = MARS.material_for_usd(Path("gusev_center.usd"))
        calibration = MARS.REGIONAL_MATERIAL_CALIBRATIONS["gusev_center"]

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        SolverImplicitMPM.register_custom_attributes(builder)

        # Flat ground shared by all envs.
        builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
            ke=material.terrain_ke,
            kd=material.terrain_kd,
            kf=0.1 * material.terrain_ke,
            mu=material.friction,
            density=0.0,
            has_particle_collision=True,
            has_shape_collision=True,
        )
        # World-attached (body -1): terrain added as a body would get its own
        # free joint and shift every per-env joint index by one.
        # Each env gets a rigid copy of the real DTM patch under its soil
        # window, replicating the single-env support geometry exactly.
        pr, pc = self.patch_h.shape
        verts = np.empty((pr * pc, 3), dtype=np.float32)
        verts[:, 0] = np.tile(self.patch_x, pr)
        verts[:, 1] = np.repeat(self.patch_y, pc)
        verts[:, 2] = self.patch_h.ravel()
        quad_r, quad_c = np.meshgrid(
            np.arange(pr - 1), np.arange(pc - 1), indexing="ij"
        )
        p00 = (quad_r * pc + quad_c).ravel()
        p01 = p00 + 1
        p10 = p00 + pc
        p11 = p10 + 1
        # CCW seen from +z (y decreases with row index): normals point up.
        indices = np.concatenate(
            [
                np.stack([p00, p10, p11], axis=1).ravel(),
                np.stack([p00, p11, p01], axis=1).ravel(),
            ]
        ).astype(np.int32)
        patch_mesh = newton.Mesh(verts, indices, compute_inertia=False)
        for ox, oy in self.offsets:
            builder.add_shape_mesh(
                -1,
                xform=wp.transform(wp.vec3(float(ox), float(oy), 0.0), wp.quat_identity()),
                mesh=patch_mesh,
            )
        # Catcher floor far below for particles that leak past patch edges.
        side_extent = 0.5 * self.pitch * np.ceil(np.sqrt(self.num_envs)) + 3.0
        builder.add_shape_box(
            -1,
            xform=wp.transform(
                wp.vec3(0.0, 0.0, self.ground_z - 0.25), wp.quat_identity()
            ),
            hx=side_extent,
            hy=side_extent,
            hz=0.25,
        )

        robot_usd = Path(self.args.robot_usd).expanduser().resolve()
        if not robot_usd.is_file():
            raise FileNotFoundError(f"G1 USD not found: {robot_usd}")
        snap_root = self._snap["joint_q"][:7].copy()

        self.joint_q_size = None
        self.joint_qd_size = None
        # Each robot lives in its own MJWarp world; separate_worlds batches
        # the identical articulations instead of building one giant tree.
        shapes_added = 0
        for env in range(self.num_envs):
            builder.begin_world()
            body_start = len(builder.body_label)
            builder.add_usd(
                str(robot_usd),
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                enable_self_collisions=False,
                collapse_fixed_joints=True,
                schema_resolvers=[
                    newton.usd.SchemaResolverNewton(),
                    newton.usd.SchemaResolverPhysx(),
                ],
            )
            # Inside the world block, not after it: separate_worlds batches
            # per world, and shapes declared outside leave the MuJoCo geom
            # count disagreeing with Newton's ("number of geoms ... does not
            # match the number of colliding shapes").
            shapes_added += self._add_g1_shapes(builder, body_start, material)
            builder.end_world()
        self._shapes_per_env = shapes_added // self.num_envs

        # Per-joint drives from G1_CFG's implicit actuators.
        joints_per_env = len(builder.joint_label) // self.num_envs
        for joint_id, label in enumerate(builder.joint_label):
            if joint_id % joints_per_env == 0:
                continue  # free base joint of each env
            name = label.rsplit("/", 1)[-1]
            dof = builder.joint_qd_start[joint_id]
            stiffness, damping, effort, armature = g1_joint_drive(
                name, float(getattr(self.args, "finger_effort", 300.0))
            )
            builder.joint_target_mode[dof] = int(JointTargetMode.POSITION)
            builder.joint_target_ke[dof] = stiffness
            builder.joint_target_kd[dof] = damping
            builder.joint_effort_limit[dof] = effort
            builder.joint_armature[dof] = armature
            builder.joint_target_pos[dof] = G1_DEFAULT_JOINT_POS.get(name, 0.0)
            builder.joint_q[builder.joint_q_start[joint_id]] = (
                G1_DEFAULT_JOINT_POS.get(name, 0.0)
            )

        # Feet: only they touch particles.
        self.foot_body_ids = [
            index
            for index, label in enumerate(builder.body_label)
            if label.endswith("ankle_roll_link")
        ]
        per_env_feet = len(self.foot_body_ids) // self.num_envs
        assert per_env_feet == 2, f"expected 2 feet per env, got {per_env_feet}"
        torso_ids = [
            index
            for index, label in enumerate(builder.body_label)
            if label.endswith("torso_link")
        ]
        self.torso_ids = np.array(torso_ids, dtype=np.int64)
        foot_set = set(self.foot_body_ids)
        # Links a robot puts on the ground once it is off its feet. Without
        # them nothing below the ankles exists to the ground: a robot that
        # tipped over fell straight through the terrain -- measured at 1.7 m
        # under the surface after 400 undriven steps, a free fall -- and it
        # had nothing to push against, so crawling out and standing up were
        # not merely unlearned but unavailable. Driving a robot half inside
        # the terrain is also what broke the solve.
        #
        # The H1 asset ships a collider on every link, so the fix there was to
        # switch the unwanted ones off. ``g1.usd`` ships two -- the feet -- and
        # nothing else, so here the shapes have to be created. They are coarse
        # on purpose: the task needs a floor to push against, not an accurate
        # hull, and every extra shape is width in the contact buffers.
        GROUND_CONTACT_LINKS = tuple(G1_CONTACT_BOXES) + tuple(G1_CONTACT_CAPSULES)
        ground_contact_bodies = {
            index
            for index, label in enumerate(builder.body_label)
            if any(label.endswith(suffix) for suffix in GROUND_CONTACT_LINKS)
        }
        if self._shapes_per_env != 8:
            raise RuntimeError(
                f"expected 8 shapes per env (2 soles + 6 ground contacts), "
                f"added {self._shapes_per_env}. Link naming changed; without "
                "them the robot falls through the terrain (HANDOFF §11)."
            )

        for shape_id, body_id in enumerate(builder.shape_body):
            if body_id >= 0 and body_id not in foot_set:
                builder.shape_flags[shape_id] &= ~newton.ShapeFlags.COLLIDE_PARTICLES
                if body_id not in ground_contact_bodies:
                    builder.shape_flags[shape_id] &= ~newton.ShapeFlags.COLLIDE_SHAPES
        self.ground_contact_bodies = sorted(ground_contact_bodies)
        print(
            f"[EscapeVecG1] shapes per env: {self._shapes_per_env} "
            f"(2 soles + 6 ground contacts on "
            f"{sorted({builder.body_label[b].rsplit('/', 1)[-1] for b in ground_contact_bodies})})"
        )

        # Soil: the snapshot's particle block replicated at each env offset.
        base = self._snap["particle_q"].copy()
        base[:, 0] -= self.site_xy[0]
        base[:, 1] -= self.site_xy[1]
        count = len(base)
        self.particles_per_env = count
        positions = np.concatenate(
            [base + np.array([ox, oy, 0.0]) for ox, oy in self.offsets]
        ).astype(np.float32)
        velocities = np.tile(self._snap["particle_qd"], (self.num_envs, 1)).astype(
            np.float32
        )

        total = len(positions)
        spacing = 0.025
        friction = np.full(total, material.friction * calibration.friction, np.float32)
        young = np.full(
            total, material.young_modulus * calibration.young_modulus, np.float32
        )
        yield_p = np.full(
            total, material.yield_pressure * calibration.yield_pressure, np.float32
        )
        density = np.full(total, material.density, np.float32)
        dilatancy = np.full(total, material.dilatancy, np.float32)
        # 07's loose_surface profile: the top 8 cm is looser than the regional
        # base (the snapshot soil was built this way; a uniform bed changes
        # the bearing behaviour). Layer identity comes from the spawn height
        # above the DTM, which recycling preserves.
        loose = np.tile(self.rest_z_np > 0.30 - 0.08, self.num_envs)
        prof = MARS.LOOSE_SURFACE_PROFILE
        density[loose] *= prof.density_scale
        friction[loose] *= prof.friction_scale
        young[loose] *= prof.young_modulus_scale
        yield_p[loose] *= prof.yield_pressure_scale
        dilatancy[loose] = prof.dilatancy
        # Pristine (pre-pocket) per-particle material for the recycler.
        self._pristine_young_arr = young.copy()
        self._pristine_yield_arr = yield_p.copy()
        # Pocket: soft column at each env centre, matching the walk-in scenario.
        for env, (ox, oy) in enumerate(self.offsets):
            sl = slice(env * count, (env + 1) * count)
            radial = np.hypot(
                positions[sl, 0] - (ox + self.args.pocket_dx),
                positions[sl, 1] - (oy + self.args.pocket_dy),
            )
            mask = radial < self.args.pocket_radius
            young[sl][mask] *= self.args.pocket_scale
            yield_p[sl][mask] *= self.args.pocket_scale
        # Pocketed layout for restoring materials on reset (the recycler
        # mutates the runtime arrays).
        self._material_young0 = young.copy()
        self._material_yield0 = yield_p.copy()

        builder.add_particles(
            pos=positions,
            vel=velocities,
            mass=density * spacing**3,
            radius=np.full(total, 0.5 * spacing, np.float32),
            custom_attributes={
                "mpm:friction": friction,
                "mpm:young_modulus": young,
                "mpm:poisson_ratio": np.full(total, material.poisson_ratio, np.float32),
                "mpm:yield_pressure": yield_p,
                "mpm:dilatancy": dilatancy,
            },
        )

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, MARS_G))

        # separate_worlds batches the N identical robots as MJWarp worlds
        # instead of one giant kinematic tree, whose per-thread stack blows up
        # past ~8 H1s (mj_stackAlloc overflow at 16).
        self.robot_solver = newton.solvers.SolverMuJoCo(
            self.model,
            separate_worlds=True,
            solver="newton",
            iterations=50,
            ls_iterations=20,
            njmax=2000,
            nconmax=1000,
            use_mujoco_contacts=False,
        )
        options = SolverImplicitMPM.Config()
        options.voxel_size = 3.0 * spacing
        options.grid_type = "sparse"
        options.transfer_scheme = "pic"
        options.strain_basis = "P0"
        options.max_iterations = self.args.mpm_iterations
        options.tolerance = 1.0e-4
        options.air_drag = 1.0
        options.collider_velocity_mode = "backward"
        self.mpm_solver = SolverImplicitMPM(self.model, options)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        # Newton's soft-contact (particle-shape) pass cannot scale here: its
        # buffers AND its tid bookkeeping are sized particles x ALL shapes
        # (12.7 GB at 16 envs, past int32 at 64), and create_soft_contacts
        # writes soft_contact_tids[tid] unconditionally, so a smaller
        # soft_contact_max corrupts memory. We do not need that pass at all --
        # feet<->soil is handled by the MPM solver's own collider and MuJoCo
        # never consumes soft contacts -- so allocate a token-size buffer and
        # skip the pass by hiding particles from collide() (see step()).
        from newton._src.sim.collide import CollisionPipeline

        pipeline = CollisionPipeline(self.model, soft_contact_max=8)
        self.contacts = self.model.contacts(collision_pipeline=pipeline)
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_0
        )
        # How far the sole sits below the pelvis in the nominal standing pose,
        # measured from the built model instead of assumed. A walk-in start
        # seats the robot on the sand by this offset, and an assumed value
        # that is a few centimetres out spawns the feet already buried --
        # which reads downstream as an entrapment the robot never walked into.
        root_z = float(self.model.joint_q.numpy()[2])
        foot_z = min(
            float(self.state_0.body_q.numpy()[b][2]) for b in self.foot_body_ids[:2]
        )
        sole_z = foot_z + FOOT_BOX_CENTRE_M[2] - FOOT_BOX_HALF_M[2]
        self.sole_below_root = root_z - sole_z
        print(
            f"[EscapeVecG1] standing stance: sole {self.sole_below_root:.3f} m "
            f"below the pelvis (constant was {STANDING_ROOT_HEIGHT_M:.3f})"
        )
        self.mpm_solver.setup_collider(
            body_mass=wp.zeros_like(self.model.body_mass),
            body_q=self.state_0.body_q,
        )
        # Two-way coupling state (ported from 07_h1_mars_newton.py): soil
        # reaction impulses become body wrenches applied on the next frame's
        # robot substeps. Without this the robots fall through the soil and
        # anchor within a few steps.
        self.collider_body_indices = self.mpm_solver.collider_body_index
        self.body_sand_forces = wp.zeros_like(self.state_0.body_f)
        self.foot_body_ids_wp = wp.array(
            self.foot_body_ids, dtype=int, device=self.model.device
        )
        self.all_body_ids = list(range(self.model.body_count))
        self.all_body_ids_wp = wp.array(
            self.all_body_ids, dtype=int, device=self.model.device
        )
        self.reaction_saturation_counts = wp.zeros(
            len(self.all_body_ids), dtype=int, device=self.model.device
        )

        # Per-env joint layout (identical robots).
        jq = self.model.joint_q.numpy()
        self.jq_per_env = len(jq) // self.num_envs
        jqd = self.model.joint_qd.numpy()
        self.jqd_per_env = len(jqd) // self.num_envs
        self._snap_jq = self._snap["joint_q"].copy()
        self._snap_jqd = self._snap["joint_qd"].copy()
        # Snapshot root recentred on the env origin.
        self._snap_jq[0] -= self.site_xy[0]
        self._snap_jq[1] -= self.site_xy[1]

        self._snap_particles_local = base
        self._snap_particle_qd = self._snap["particle_qd"].copy()
        self._mpm_snap = {
            k: self._snap[k]
            for k in (
                "mpm_qd_grad",
                "mpm_elastic_strain",
                "mpm_Jp",
                "mpm_stress",
                "mpm_transform",
            )
        }

    # ------------------------------------------------------------------
    def _init_torch(self) -> None:
        self.device = torch.device(wp.device_to_torch(self.model.device))
        self.joint_q_t = wp.to_torch(self.state_0.joint_q)
        self.joint_qd_t = wp.to_torch(self.state_0.joint_qd)
        self.body_q_t = wp.to_torch(self.state_0.body_q)
        self.control_t = wp.to_torch(self.control.joint_target_pos)
        self.command = torch.tensor(
            [[self.args.command_x, 0.0, 0.0]], device=self.device
        ).repeat(self.num_envs, 1)
        self.gravity_dir = torch.tensor(
            [[0.0, 0.0, -1.0]], device=self.device
        ).repeat(self.num_envs, 1)
        self.last_action = torch.zeros(
            (self.num_envs, ACTION_DIM), device=self.device
        )

        # Policy joint order mapping, computed once from env 0's labels. The
        # Newton order comes out of the USD parser and the policy order out of
        # the IsaacLab articulation; they are not the same, and every joint
        # slot in the 310-vector depends on getting this right.
        labels = [l.rsplit("/", 1)[-1] for l in self.model.joint_label]
        env0 = labels[1 : 1 + ACTION_DIM]
        if sorted(env0) != sorted(G1_JOINT_ORDER):
            missing = sorted(set(G1_JOINT_ORDER) - set(env0))
            extra = sorted(set(env0) - set(G1_JOINT_ORDER))
            raise RuntimeError(
                f"env0 joint labels do not match the policy's G1 joint set "
                f"({len(env0)} joints). missing={missing} extra={extra}"
            )
        self.policy_to_newton = torch.tensor(
            [env0.index(name) for name in G1_JOINT_ORDER],
            device=self.device,
        )
        self.default_pos_policy = torch.tensor(
            [G1_DEFAULT_JOINT_POS.get(n, 0.0) for n in G1_JOINT_ORDER],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        self.default_pos_newton = torch.tensor(
            [G1_DEFAULT_JOINT_POS.get(n, 0.0) for n in env0],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        limits_lo = np.asarray(self.model.joint_limit_lower.numpy(), dtype=np.float32)
        limits_hi = np.asarray(self.model.joint_limit_upper.numpy(), dtype=np.float32)
        actuated = self.default_pos_newton.shape[1]
        self.joint_limit_lo = torch.tensor(
            limits_lo[6 : 6 + actuated], device=self.device
        ).unsqueeze(0)
        self.joint_limit_hi = torch.tensor(
            limits_hi[6 : 6 + actuated], device=self.device
        ).unsqueeze(0)

        # Height-scan grid identical to the walking policy's.
        scan_x = torch.linspace(-0.8, 0.8, 17, device=self.device)
        scan_y = torch.linspace(-0.5, 0.5, 11, device=self.device)
        gx, gy = torch.meshgrid(scan_x, scan_y, indexing="xy")
        self.scan_x = gx.reshape(-1)
        self.scan_y = gy.reshape(-1)

        self.offsets_t = torch.tensor(
            self.offsets, device=self.device, dtype=torch.float32
        )
        self.pocket_xy = self.offsets_t + torch.tensor(
            [[self.args.pocket_dx, self.args.pocket_dy]],
            device=self.device,
            dtype=torch.float32,
        )
        self.foot_ids_t = torch.tensor(
            np.array(self.foot_body_ids).reshape(self.num_envs, 2),
            device=self.device,
        )
        self.torso_ids_t = torch.tensor(self.torso_ids, device=self.device)
        # Every link except the feet, grouped by env, for the contact test.
        #
        # "Foot" here is wider than the MPM coupling's: ``ankle_pitch_link``
        # sits about 52 mm above the sole, so counting it as a non-foot link
        # makes the lowest-link clearance ~0.05 m for a robot standing
        # perfectly normally, and ``on_feet`` -- the gate on the alive credit,
        # the fall timer and the escape test alike -- would never be true.
        # The knees are the lowest links this test should see.
        feet_set = {
            index
            for index, label in enumerate(self.model.body_label)
            if label.endswith(("ankle_roll_link", "ankle_pitch_link"))
        }
        bodies_per_env = self.model.body_count // self.num_envs
        nonfoot = [b for b in range(self.model.body_count) if b not in feet_set]
        nonfoot.sort(key=lambda b: (b // bodies_per_env, b))
        self.nonfoot_ids_t = torch.tensor(nonfoot, device=self.device)
        self.nonfoot_env_t = torch.tensor(
            [b // bodies_per_env for b in nonfoot], device=self.device
        )
        self.max_episode_steps = int(self.args.episode_seconds / CONTROL_DT)
        self._steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._tilt_run = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._was_trapped = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Escape is latched rather than read live. The hold counter resets the
        # moment the robot stops moving, so without a latch an episode that
        # escaped and then paused would report failure, and the escape bonus
        # would be paid again every step the condition happened to hold.
        self._escaped = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Reporting only: time spent tilted past upright and peak torso
        # height. They describe how a given escape was achieved -- a leap
        # shows brief tilt with a high torso, a walk neither, a tumble a long
        # tilt -- without entering the objective.
        self._airborne_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._peak_torso_z = torch.zeros(self.num_envs, device=self.device)
        self._prev_exit = torch.zeros(self.num_envs, device=self.device)
        # Furthest the robot has been from the hazard this episode. The
        # distance reward is paid against it, so an oscillation cannot
        # collect twice for the same ground.
        self._best_exit = torch.zeros(self.num_envs, device=self.device)
        self._prev_burial = torch.zeros(self.num_envs, device=self.device)
        self._prev_action = torch.zeros(
            self.num_envs, ACTION_DIM, device=self.device
        )
        self.hold_steps = int(
            getattr(self.args, "hold_seconds", 1.5) / CONTROL_DT
        )
        self.down_limit_steps = int(
            getattr(self.args, "down_seconds", 2.0) / CONTROL_DT
        )

        # DTM patch (site-local coords) for the height scan, plus the
        # undisturbed reference and surface mask for sinkage measurement.
        self.patch_h_t = torch.tensor(self.patch_h, device=self.device)
        self.patch_x0 = float(self.patch_x[0])
        self.patch_y0 = float(self.patch_y[0])
        self.surface_mask_t = torch.tensor(
            self.surface_mask_np, device=self.device, dtype=torch.bool
        )
        # Sample grid over the 22-equivalent moving soil window (2 x 2 m),
        # plus the 12 spawn-layer depth offsets above the DTM (spacing 25 mm).
        win = torch.linspace(-1.0, 1.0, 17, device=self.device)
        wx, wy = torch.meshgrid(win, win, indexing="xy")
        self.win_x = wx.reshape(-1)
        self.win_y = wy.reshape(-1)
        self.layer_offsets = ((torch.arange(12, device=self.device) + 0.5) * 0.025)
        self._soiltop_bias: torch.Tensor | None = None
        self._burial_ref: float | None = None
        self._snap_soil_p95 = float(
            np.percentile(self._snap["particle_q"][:, 2], 95)
        )

        # Robot-following window recycling state (07 parity): per-env world-
        # coordinate reference positions (updated by the recycler), shared
        # rest offsets, and the full DTM grid for re-seating heights.
        ref_world = np.concatenate(
            [
                (self._reference_q - np.array([*self.site_xy, 0.0]))
                + np.array([ox, oy, 0.0])
                for ox, oy in self.offsets
            ]
        ).astype(np.float32)
        self.reference_wp = wp.array(ref_world, dtype=wp.vec3, device=self.model.device)
        self.rest_z_wp = wp.array(
            np.tile(self.rest_z_np, self.num_envs), dtype=float, device=self.model.device
        )
        self.terrain_wp = wp.array(
            self.height_full.ravel(), dtype=float, device=self.model.device
        )
        self.recycle_count = wp.zeros(1, dtype=int, device=self.model.device)
        self._snap_ref_world = ref_world
        self.pristine_young_wp = wp.array(
            self._pristine_young_arr, dtype=float, device=self.model.device
        )
        self.pristine_yield_wp = wp.array(
            self._pristine_yield_arr, dtype=float, device=self.model.device
        )

        # Walk-in starts: undisturbed soil (the snapshot's reference positions,
        # before the foot dug into it) and a robot standing upstream of the
        # pocket in the nominal pose, so the episode contains the whole arc --
        # walk, sink, escape, walk on -- instead of beginning already trapped.
        self.start_mode = getattr(self.args, "start_mode", "trapped")
        self.approach_m = float(getattr(self.args, "approach_m", 1.5))
        n = self.particles_per_env
        self._pristine_particles = ref_world
        self._fresh_qd = np.zeros((n, 3), dtype=np.float32)
        self._fresh_mat = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
        self._fresh_zero_mat = np.zeros((n, 3, 3), dtype=np.float32)
        self._fresh_jp = np.ones(n, dtype=np.float32)
        self._standing_jq = self._snap_jq.copy()
        self._standing_jq[3:7] = [0.0, 0.0, 0.0, 1.0]
        self._standing_jq[7:] = (
            self.default_pos_newton[0].detach().cpu().numpy().astype(self._snap_jq.dtype)
        )
        print(
            f"[EscapeVec] patch {self.patch_h.shape} "
            f"z=[{self.patch_h.min():.2f},{self.patch_h.max():.2f}] "
            f"surface particles={int(self.surface_mask_np.sum())}/{len(self.surface_mask_np)}"
        )

    def _sample_patch(self, x_local: torch.Tensor, y_local: torch.Tensor) -> torch.Tensor:
        """Bilinear DTM patch height at env-local XY [m]; clamps at the border."""
        rows, cols = self.patch_h_t.shape
        # clamp() propagates NaN, and NaN cast to long is an arbitrary index,
        # which trips a device-side assert deep in an unrelated kernel. A
        # violent landing can put NaN in body_q, so scrub it first (07's
        # sampler does the same).
        col_f = torch.nan_to_num(
            (x_local - self.patch_x0) / self.tdx, nan=0.0, posinf=cols, neginf=0.0
        )
        row_f = torch.nan_to_num(
            (self.patch_y0 - y_local) / self.tdy, nan=0.0, posinf=rows, neginf=0.0
        )
        col_f = torch.clamp(col_f, 0.0, cols - 1.0001)
        row_f = torch.clamp(row_f, 0.0, rows - 1.0001)
        col0 = col_f.floor().long()
        row0 = row_f.floor().long()
        col1 = torch.clamp(col0 + 1, max=cols - 1)
        row1 = torch.clamp(row0 + 1, max=rows - 1)
        tc = col_f - col0
        tr = row_f - row0
        h = self.patch_h_t
        return (
            h[row0, col0] * (1 - tr) * (1 - tc)
            + h[row0, col1] * (1 - tr) * tc
            + h[row1, col0] * tr * (1 - tc)
            + h[row1, col1] * tr * tc
        )

    # ------------------------------------------------------------------
    def reset_all(self) -> torch.Tensor:
        jq = np.concatenate([self._snap_jq] * self.num_envs).astype(np.float32)
        jqd = np.concatenate([self._snap_jqd] * self.num_envs).astype(np.float32)
        for env, (ox, oy) in enumerate(self.offsets):
            jq[env * self.jq_per_env + 0] = self._snap_jq[0] + ox
            jq[env * self.jq_per_env + 1] = self._snap_jq[1] + oy
        self.state_0.joint_q.assign(jq)
        self.state_0.joint_qd.assign(jqd)
        self.state_1.joint_q.assign(jq)
        self.state_1.joint_qd.assign(jqd)

        particles = np.concatenate(
            [
                self._snap_particles_local + np.array([ox, oy, 0.0])
                for ox, oy in self.offsets
            ]
        ).astype(np.float32)
        self.state_0.particle_q.assign(particles)
        def rep(array: np.ndarray) -> np.ndarray:
            # Shape-agnostic replication along axis 0: np.tile mis-broadcasts
            # matrix-valued MPM arrays such as the (n, 3, 3) velocity gradient.
            return np.concatenate([array] * self.num_envs, axis=0)

        self.state_0.particle_qd.assign(rep(self._snap_particle_qd).astype(np.float32))
        mpm = self.state_0.mpm
        mpm.particle_qd_grad.assign(rep(self._mpm_snap["mpm_qd_grad"]))
        mpm.particle_elastic_strain.assign(rep(self._mpm_snap["mpm_elastic_strain"]))
        mpm.particle_Jp.assign(rep(self._mpm_snap["mpm_Jp"]))
        mpm.particle_stress.assign(rep(self._mpm_snap["mpm_stress"]))
        mpm.particle_transform.assign(rep(self._mpm_snap["mpm_transform"]))
        self.reference_wp.assign(self._snap_ref_world)
        self.model.mpm.young_modulus.assign(self._material_young0)
        self.model.mpm.yield_pressure.assign(self._material_yield0)

        # Walk-in envs replace the snapshot placement with a standing pose on
        # pristine soil, upstream of the pocket.
        if self.start_mode != "trapped":
            grads = mpm.particle_qd_grad.numpy()
            strain = mpm.particle_elastic_strain.numpy()
            jp_arr = mpm.particle_Jp.numpy()
            stress = mpm.particle_stress.numpy()
            xform = mpm.particle_transform.numpy()
            pvel = self.state_0.particle_qd.numpy()
            ref_all = self.reference_wp.numpy()
            for env in range(self.num_envs):
                if self._walk_in_env(env):
                    self._place_walk_in(env, jq, jqd, particles, pvel, grads,
                                        strain, jp_arr, stress, xform, ref_all)
            self.state_0.joint_q.assign(jq)
            self.state_0.joint_qd.assign(jqd)
            self.state_1.joint_q.assign(jq)
            self.state_1.joint_qd.assign(jqd)
            self.state_0.particle_q.assign(particles)
            self.state_0.particle_qd.assign(pvel)
            self.reference_wp.assign(ref_all)
            mpm.particle_qd_grad.assign(grads)
            mpm.particle_elastic_strain.assign(strain)
            mpm.particle_Jp.assign(jp_arr)
            mpm.particle_stress.assign(stress)
            mpm.particle_transform.assign(xform)

        newton.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0
        )
        # Same on the full reset: joint coordinates alone leave the second
        # buffer's body transforms untouched.
        newton.eval_fk(
            self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1
        )
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        self._steps.zero_()
        self._hold.zero_()
        self._tilt_run.zero_()
        self._was_trapped.zero_()
        self._escaped.zero_()
        self._airborne_steps.zero_()
        self._peak_torso_z.zero_()
        self.last_action.zero_()
        self._prev_action.zero_()
        self.body_sand_forces.zero_()
        self._refresh_torch_views()
        # Hazard centre. For a trapped start it is the buried foot (22's
        # convention); for a walk-in start the robot is still upstream, so the
        # centre has to be the pocket the soft material was actually built
        # around.
        self.pocket_xy = self.body_q_t[
            self.foot_ids_t[:, self.trapped_index], 0:2
        ].clone()
        if self.start_mode != "trapped":
            built = self.offsets_t + torch.tensor(
                [[self.args.pocket_dx, self.args.pocket_dy]],
                device=self.device,
                dtype=torch.float32,
            )
            walk_in = torch.tensor(
                [self._walk_in_env(env) for env in range(self.num_envs)],
                device=self.device,
            )
            self.pocket_xy = torch.where(
                walk_in.unsqueeze(1), built, self.pocket_xy
            )
        self._prev_exit = self._exit_distances()
        self._best_exit = self._prev_exit.clone()
        # Snapshot restore is identical for every env and every reset, so the
        # reset-time references are constants (22 recomputes them per reset;
        # the value is the same).
        soil_top = self._soil_top_per_env()
        self._initial_soil_top = soil_top.clone()
        feet_z = self.body_q_t[self.foot_ids_t.flatten(), 2].view(self.num_envs, 2)
        self._burial0 = torch.clamp(
            soil_top - feet_z[:, self.trapped_index], min=1e-3
        )
        self._prev_burial = torch.clamp(
            soil_top - feet_z[:, self.trapped_index], min=0.0
        )
        # Fixed reference for the burial-progress term: how deep the snapshot
        # foot was. Trapped starts set it directly; walk-in starts begin on
        # the surface, so they must not define their own scale.
        if self._burial_ref is None:
            trapped_depth = float(self._burial0.max())
            self._burial_ref = max(trapped_depth, 0.05)
        # Level-match the under-torso upright criterion to the window-p95 one
        # at the reset pose (identical there; diverges only away from reset).
        torso_xy = self.body_q_t[self.torso_ids_t, 0:2]
        surface_torso = self._sample_patch(
            torso_xy[:, 0] - self.offsets_t[:, 0],
            torso_xy[:, 1] - self.offsets_t[:, 1],
        ) + SOIL_SURFACE_OFFSET_M
        self._upright_offset = soil_top - surface_torso
        return self._observation()

    def _nonfoot_clearance(self) -> torch.Tensor:
        """Height of the lowest non-foot link above the local surface [m].

        This is the contact test the task actually cares about: a robot
        standing on its soles keeps every other link clear of the ground,
        while one that has gone down rests on a shin, a hip or the torso. It
        needs no notion of a correct pose, so an unusual but recovered
        posture is not penalised.
        """
        z = self.body_q_t[self.nonfoot_ids_t, 2]
        xy = self.body_q_t[self.nonfoot_ids_t, 0:2]
        env_of_body = self.nonfoot_env_t
        surface = self._sample_patch(
            xy[:, 0] - self.offsets_t[env_of_body, 0],
            xy[:, 1] - self.offsets_t[env_of_body, 1],
        ) + SOIL_SURFACE_OFFSET_M
        clearance = (z - surface).view(self.num_envs, -1)
        return clearance.min(dim=1).values

    def _walk_in_env(self, env: int) -> bool:
        """Whether this env's next episode starts upstream, walking in."""
        if self.start_mode == "walk_in":
            return True
        if self.start_mode == "mixed":
            return env % 2 == 1
        return False

    def _place_walk_in(
        self, env, jq, jqd, particles, pvel, grads, strain, jp, stress, xform,
        refs=None,
    ) -> None:
        """Undisturbed soil and a standing robot upstream of the pocket.

        The snapshot's *reference* particle positions are the soil before the
        foot dug into it, so they double as the pristine bed. The robot starts
        ``approach_m`` behind the pocket in the nominal standing pose and
        walks in under its own policy.
        """
        ox, oy = self.offsets[env]
        pocket_x = ox + self.args.pocket_dx
        pocket_y = oy + self.args.pocket_dy
        n = self.particles_per_env
        sl = slice(env * n, (env + 1) * n)
        # Seat the soil window on the robot's start position. Leaving it at
        # the snapshot site means the recycler has to wrap half the window in
        # a single frame once stepping begins, and the re-seated particles
        # land inside the robot's feet -- the contact impulse from that is
        # what was breaking the solve.
        block = self._pristine_particles[sl].copy()
        block[:, 0] += (pocket_x - self.approach_m) - (ox + self.args.pocket_dx)
        # Re-seat the heights: the block moved across sloped terrain, so its
        # original z values would leave particles buried or floating.
        block[:, 2] = (
            MARS.sample_height_numpy(
                self.height_full,
                self.tdx,
                self.tdy,
                block[:, 0] - ox + self.site_xy[0],
                block[:, 1] - oy + self.site_xy[1],
            )
            + self.rest_z_np
        )
        particles[sl] = block
        if refs is not None:
            refs[sl] = block
        pvel[sl] = self._fresh_qd
        grads[sl] = self._fresh_zero_mat
        strain[sl] = self._fresh_mat
        jp[sl] = self._fresh_jp
        stress[sl] = self._fresh_zero_mat
        xform[sl] = self._fresh_mat

        surface = MARS.sample_height_numpy(
            self.height_full,
            self.tdx,
            self.tdy,
            np.array([pocket_x - self.approach_m - ox + self.site_xy[0]]),
            np.array([pocket_y - oy + self.site_xy[1]]),
        )[0] + SOIL_SURFACE_OFFSET_M
        base = env * self.jq_per_env
        jq[base : base + self.jq_per_env] = self._standing_jq
        jq[base + 0] = pocket_x - self.approach_m
        jq[base + 1] = pocket_y
        # Soles just clear of the sand, so the robot settles onto it under
        # gravity instead of spawning inside it.
        jq[base + 2] = surface + self.sole_below_root + WALK_IN_CLEARANCE_M
        jqd[env * self.jqd_per_env : (env + 1) * self.jqd_per_env] = 0.0

    def _soil_top_per_env(self) -> torch.Tensor:
        """95th-percentile sand-surface height over the robot-centred 2x2 m
        window [m].

        22 takes the p95 of the moving soil window's particles; that window is
        centred on the root, so its p95 follows the local terrain as the robot
        walks. A static particle percentile would not, and on this sloped site
        it misclassifies downhill walking as falling. The undisturbed surface
        (DTM patch + soil depth) sampled over the same window reproduces the
        moving percentile: the feet disturb <5 % of the window area, which a
        95th percentile ignores.
        """
        jq = self.joint_q_t.view(self.num_envs, self.jq_per_env)
        root_xy = jq[:, 0:2]
        x_loc = root_xy[:, 0:1] + self.win_x - self.offsets_t[:, 0:1]
        y_loc = root_xy[:, 1:2] + self.win_y - self.offsets_t[:, 1:2]
        surface = self._sample_patch(x_loc, y_loc) + SOIL_SURFACE_OFFSET_M
        # 22's percentile runs over the full particle VOLUME (12 spawn layers
        # below the surface), not the surface alone; reproduce that mixture.
        z = surface.unsqueeze(-1) - SOIL_SURFACE_OFFSET_M + self.layer_offsets
        raw = torch.quantile(z.reshape(self.num_envs, -1), 0.95, dim=1)
        # Level-match to the settled snapshot's particle p95 at reset: the
        # analytic undisturbed model sits ~40 mm above the settled particles.
        if self._soiltop_bias is None:
            self._soiltop_bias = raw - self._snap_soil_p95
        return raw - self._soiltop_bias

    def _clear_solver_warmstart(self) -> None:
        """Drop the MuJoCo solver's carried-over acceleration guess.

        Restoring an environment writes Newton state, and the solver syncs
        only ``qpos`` and ``qvel`` back from it. Its warm-start accumulator is
        never touched, so once that holds a NaN every later solve starts from
        one no matter how clean the restored pose is -- which is the shape the
        logs described: every world diverging on the first step after every
        reset, permanently, while the soil stayed finite.

        The warm start is an initial guess, so clearing it costs at most some
        convergence on the step that follows and nothing physical.
        """
        data = getattr(self.robot_solver, "mjw_data", None)
        if data is None:
            data = getattr(self.robot_solver, "mj_data", None)
        for field in ("qacc_warmstart", "qacc", "qfrc_constraint", "qfrc_smooth"):
            array = getattr(data, field, None)
            if array is None:
                continue
            if hasattr(array, "zero_"):
                array.zero_()
            elif hasattr(array, "fill"):
                array.fill(0.0)

    def _reset_envs(self, mask: torch.Tensor) -> None:
        """Restore the snapshot for the flagged envs only."""
        idx = torch.nonzero(mask).flatten().tolist()
        if not idx:
            return
        jq = self.state_0.joint_q.numpy()
        jqd = self.state_0.joint_qd.numpy()
        particles = self.state_0.particle_q.numpy()
        pvel = self.state_0.particle_qd.numpy()
        refs = self.reference_wp.numpy()
        young_now = self.model.mpm.young_modulus.numpy()
        yield_now = self.model.mpm.yield_pressure.numpy()
        mpm = self.state_0.mpm
        grads = mpm.particle_qd_grad.numpy()
        strain = mpm.particle_elastic_strain.numpy()
        jp = mpm.particle_Jp.numpy()
        stress = mpm.particle_stress.numpy()
        xform = mpm.particle_transform.numpy()
        n = self.particles_per_env
        for env in idx:
            ox, oy = self.offsets[env]
            jq[env * self.jq_per_env : (env + 1) * self.jq_per_env] = self._snap_jq
            jq[env * self.jq_per_env + 0] += ox
            jq[env * self.jq_per_env + 1] += oy
            jqd[env * self.jqd_per_env : (env + 1) * self.jqd_per_env] = self._snap_jqd
            sl = slice(env * n, (env + 1) * n)
            particles[sl] = self._snap_particles_local + np.array([ox, oy, 0.0])
            pvel[sl] = self._snap_particle_qd
            grads[sl] = self._mpm_snap["mpm_qd_grad"]
            strain[sl] = self._mpm_snap["mpm_elastic_strain"]
            jp[sl] = self._mpm_snap["mpm_Jp"]
            stress[sl] = self._mpm_snap["mpm_stress"]
            xform[sl] = self._mpm_snap["mpm_transform"]
            refs[sl] = self._snap_ref_world[sl]
            young_now[sl] = self._material_young0[sl]
            yield_now[sl] = self._material_yield0[sl]
            if self._walk_in_env(env):
                self._place_walk_in(env, jq, jqd, particles, pvel, grads,
                                    strain, jp, stress, xform, refs)
        self.state_0.joint_q.assign(jq)
        self.state_0.joint_qd.assign(jqd)
        # The solver alternates between the two state buffers every substep,
        # so restoring only state_0 leaves the old values -- including a NaN
        # from a broken solve -- to come straight back on the next swap. An
        # environment that diverged once could then never recover: it
        # re-diverged every step, which is why a quarter of them were
        # reporting divergence continuously while the same policy run
        # standalone never diverged at all.
        self.state_1.joint_q.assign(jq)
        self.state_1.joint_qd.assign(jqd)
        self.state_0.particle_q.assign(particles)
        self.state_0.particle_qd.assign(pvel)
        self.reference_wp.assign(refs)
        mpm.particle_qd_grad.assign(grads)
        mpm.particle_elastic_strain.assign(strain)
        mpm.particle_Jp.assign(jp)
        mpm.particle_stress.assign(stress)
        mpm.particle_transform.assign(xform)
        self.model.mpm.young_modulus.assign(young_now)
        self.model.mpm.yield_pressure.assign(yield_now)
        newton.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0
        )
        # Restoring joint coordinates on the second buffer is not enough: the
        # body transforms derived from them are separate arrays, and nothing
        # was recomputing them, so a NaN body pose survived every reset. That
        # is the state a reset does not touch, and it matches what the logs
        # show -- once an environment diverges, all of them diverge on the
        # first step after every reset, for good, while the soil stays finite.
        newton.eval_fk(
            self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1
        )
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        self._clear_solver_warmstart()
        mask_t = mask.to(self.device)
        self._steps[mask_t] = 0
        self._hold[mask_t] = 0
        self._tilt_run[mask_t] = 0
        self._was_trapped[mask_t] = False
        self._escaped[mask_t] = False
        self._airborne_steps[mask_t] = 0
        self._peak_torso_z[mask_t] = 0.0
        self.last_action[mask_t] = 0.0
        self._prev_action[mask_t] = 0.0
        # Stale pre-reset soil wrenches must not push the restored pose.
        sand_t = wp.to_torch(self.body_sand_forces)
        bodies_per_env = sand_t.shape[0] // self.num_envs
        for env in idx:
            sand_t[env * bodies_per_env : (env + 1) * bodies_per_env] = 0.0
        self._prev_exit[mask_t] = self._exit_distances()[mask_t]
        self._best_exit[mask_t] = self._prev_exit[mask_t]
        feet_now = self.body_q_t[self.foot_ids_t.flatten(), 2].view(self.num_envs, 2)
        self._prev_burial[mask_t] = torch.clamp(
            self._soil_top_per_env() - feet_now[:, self.trapped_index], min=0.0
        )[mask_t]

    # ------------------------------------------------------------------
    def _exit_distances(self) -> torch.Tensor:
        # Scrubbed, like the patch sampler: on the step a solve breaks the
        # torso pose is read before the reset clears it, and a distance of
        # 3e9 m then lands in the logs and in ``_best_exit``. 100 m is far
        # past anything reachable in a 10 s episode on a 6 m pitch.
        torso_xy = torch.nan_to_num(
            self.body_q_t[self.torso_ids_t, :2], nan=0.0, posinf=0.0, neginf=0.0
        )
        return torch.clamp(
            torch.linalg.norm(torso_xy - self.pocket_xy, dim=1), max=100.0
        )

    def _observation(self) -> torch.Tensor:
        jq = self.joint_q_t.view(self.num_envs, self.jq_per_env)
        jqd = self.joint_qd_t.view(self.num_envs, self.jqd_per_env)
        root_q = jq[:, 3:7]
        lin = MARS.quaternion_rotate_inverse(root_q, jqd[:, :3])
        ang = MARS.quaternion_rotate_inverse(root_q, jqd[:, 3:6])
        grav = MARS.quaternion_rotate_inverse(root_q, self.gravity_dir)
        joint_pos = jq[:, 7:][:, self.policy_to_newton] - self.default_pos_policy
        joint_vel = jqd[:, 6:][:, self.policy_to_newton]

        torso = self.body_q_t[self.torso_ids_t]
        x, y, z, w = torso[:, 3], torso[:, 4], torso[:, 5], torso[:, 6]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        # Height scan against the rigid DTM patch, exactly as in 07: grid
        # centred on the root XY, rotated by torso yaw, bilinear DTM sample.
        root_xy = jq[:, 0:2]
        scan_wx = root_xy[:, 0:1] + cos_yaw.unsqueeze(1) * self.scan_x - sin_yaw.unsqueeze(1) * self.scan_y
        scan_wy = root_xy[:, 1:2] + sin_yaw.unsqueeze(1) * self.scan_x + cos_yaw.unsqueeze(1) * self.scan_y
        height = self._sample_patch(
            scan_wx - self.offsets_t[:, 0:1], scan_wy - self.offsets_t[:, 1:2]
        )
        scan = torch.clamp(
            torso[:, 2:3] - (height + SOIL_SURFACE_OFFSET_M) - 0.5, -1.0, 1.0
        )
        return torch.cat(
            (lin, ang, grav, self.command, joint_pos, joint_vel, self.last_action, scan),
            dim=1,
        )

    # ------------------------------------------------------------------
    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # A policy whose weights have gone bad emits non-finite actions, and
        # those turn into non-finite joint targets and state, which then read
        # back as a broken solve. Count it separately from a physics
        # divergence and neutralise it, so the two failures stay
        # distinguishable in the logs.
        raw_action = actions.to(self.device)
        self._action_nan = ~torch.isfinite(raw_action).all(dim=1)
        self.last_action = torch.nan_to_num(raw_action)
        targets = self.default_pos_newton.repeat(self.num_envs, 1).clone()
        targets[:, self.policy_to_newton] = (
            self.default_pos_policy + 0.5 * self.last_action
        )
        # Hold targets inside the joint's own range. Commanding past a
        # mechanical stop means driving the actuator into it and holding it
        # there, which is not a motion the robot can perform -- and it is
        # what breaks the solve: measured across action magnitudes, x1 and
        # x3 run clean while x6 diverges 552 times in 100 steps. Every
        # reachable motion survives, including saturating the actuator,
        # since a target at the far limit still commands full torque.
        targets = torch.clamp(targets, self.joint_limit_lo, self.joint_limit_hi)
        control = self.control_t.view(self.num_envs, self.jqd_per_env)
        control.zero_()
        control[:, 6:] = targets

        self.model.collide(_RigidOnlyStateView(self.state_0), self.contacts)
        for _ in range(SIM_SUBSTEPS):
            self.state_0.clear_forces()
            self.state_0.body_f.assign(self.body_sand_forces)
            self.robot_solver.step(
                self.state_0, self.state_1, self.control, self.contacts, CONTROL_DT / SIM_SUBSTEPS
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

        # Robot-following soil windows (07 parity): re-centre each env's
        # 2x2 m particle window on its root before the soil solve, so the
        # robot always walks on soil and the +0.30 scan offset stays true.
        jq_view = wp.to_torch(self.state_0.joint_q).view(
            self.num_envs, self.jq_per_env
        )
        centers = jq_view[:, 0:2].detach().cpu().numpy()
        n = self.particles_per_env
        mpm_state = self.state_0.mpm
        for env in range(self.num_envs):
            s, e = env * n, (env + 1) * n
            wp.launch(
                recycle_window_env,
                dim=n,
                inputs=[
                    self.state_0.particle_q[s:e],
                    self.state_0.particle_qd[s:e],
                    mpm_state.particle_qd_grad[s:e],
                    mpm_state.particle_elastic_strain[s:e],
                    mpm_state.particle_Jp[s:e],
                    mpm_state.particle_stress[s:e],
                    mpm_state.particle_transform[s:e],
                    self.reference_wp[s:e],
                    self.rest_z_wp[s:e],
                    self.recycle_count,
                    self.terrain_wp,
                    self.height_full.shape[0],
                    self.height_full.shape[1],
                    self.tdx,
                    self.tdy,
                    float(centers[env, 0]),
                    float(centers[env, 1]),
                    2.0,
                    2.0,
                    float(self.offsets[env, 0]),
                    float(self.offsets[env, 1]),
                    self.model.mpm.young_modulus[s:e],
                    self.model.mpm.yield_pressure[s:e],
                    self.pristine_young_wp[s:e],
                    self.pristine_yield_wp[s:e],
                    float(self.offsets[env, 0] + self.args.pocket_dx),
                    float(self.offsets[env, 1] + self.args.pocket_dy),
                    float(self.args.pocket_radius),
                    float(self.args.pocket_scale),
                ],
                device=self.model.device,
            )

        # Soil solve with explicit two-way reaction, exactly as in 07: the
        # impulse of every MPM substep is divided by the frame dt and summed,
        # yielding the frame-average wrench applied on the next frame.
        self.body_sand_forces.zero_()
        substep_dt = CONTROL_DT / MPM_SUBSTEPS
        for _ in range(MPM_SUBSTEPS):
            self.mpm_solver.step(self.state_0, self.state_0, None, None, substep_dt)
            self.mpm_solver.project_outside(self.state_0, self.state_0, substep_dt)
            impulses, impulse_positions, collider_ids = (
                self.mpm_solver.collect_collider_impulses(self.state_0)
            )
            wp.launch(
                MARS.compute_body_forces,
                dim=collider_ids.shape[0],
                inputs=[
                    CONTROL_DT,
                    collider_ids,
                    impulses,
                    impulse_positions,
                    self.collider_body_indices,
                    self.state_0.body_q,
                    self.model.body_com,
                    self.body_sand_forces,
                ],
                device=self.model.device,
            )
        # Clamp every body's soil wrench, not only the feet. While a fall
        # ended the episode this never mattered -- nothing but a foot was in
        # the soil for long. Now that a robot may lie in it for the rest of
        # a 16 s episode, a buried shin or torso was taking an unbounded
        # reaction, which is the regime the twenty earlier rounds never
        # entered.
        if self.diag is not None:
            self._diag_before_clamp()
        wp.launch(
            MARS.clamp_foot_wrenches,
            dim=len(self.all_body_ids),
            inputs=[
                self.body_sand_forces,
                self.all_body_ids_wp,
                self.reaction_force_limit,
                self.reaction_torque_limit,
                self.reaction_saturation_counts,
            ],
            device=self.model.device,
        )
        if self.diag is not None:
            self._diag_after_clamp()
        wp.launch(
            MARS.stabilize_particle_velocity,
            dim=self.model.particle_count,
            inputs=[
                self.state_0.particle_qd,
                PARTICLE_DAMPING,
                self.args.particle_speed_cap,
            ],
            device=self.model.device,
        )
        self._refresh_torch_views()
        self._steps += 1

        # Reward and termination: the exact single-env (22) shaping, batched,
        # so a policy warm-started from those rounds sees the same objective.
        exit_d = self._exit_distances()
        progress_dist = exit_d - self._prev_exit
        self._prev_exit = exit_d
        # Pay for ground gained, not for the rate of gaining it. Paying the
        # per-step change under a symmetric clamp was farmable: drifting out
        # at the clamp earns the full rate every step, while dashing back is
        # cut to the same rate over far fewer steps, so an oscillation nets a
        # profit without ever leaving. The measurements show the policy found
        # it -- between iterations 45 and 90 the mean reward rose from 44 to
        # 63 while the distance reached fell from 2.47 m to 1.68 m and time on
        # the feet fell from 0.61 to 0.50.
        #
        # Against the furthest point reached so far, returning earns nothing
        # and costs nothing, and the only way to be paid again is to get
        # further out than the robot has ever been.
        ground_gained = torch.clamp(exit_d - self._best_exit, min=0.0, max=0.02)
        self._best_exit = torch.maximum(self._best_exit, exit_d)
        torso_z = self.body_q_t[self.torso_ids_t, 2]
        soil_top = self._soil_top_per_env()
        torso_xy = self.body_q_t[self.torso_ids_t, 0:2]
        surface_torso = self._sample_patch(
            torso_xy[:, 0] - self.offsets_t[:, 0],
            torso_xy[:, 1] - self.offsets_t[:, 1],
        ) + SOIL_SURFACE_OFFSET_M
        # Functional recovery, not a pose: the robot is back on task when it
        # carries itself on its soles (nothing but the feet near the ground)
        # and is moving again. This says nothing about how it got there, so a
        # leap or a flip is judged by whether it lands and walks off.
        on_feet = self._nonfoot_clearance() > NONFOOT_CLEARANCE_M
        speed = torch.linalg.norm(
            self.joint_qd_t.view(self.num_envs, self.jqd_per_env)[:, 0:2], dim=1
        )
        walking = speed >= WALK_SPEED_MIN_MPS
        upright = on_feet

        feet_z = self.body_q_t[self.foot_ids_t.flatten(), 2].view(self.num_envs, 2)
        burial = torch.clamp(soil_top - feet_z[:, self.trapped_index], min=0.0)
        # Normalise against the snapshot's burial depth, not the episode's own
        # starting depth: a walk-in episode starts on the surface, so dividing
        # by its (near-zero) initial burial turned a normal step into a
        # reward of -200. Clamped, so this term can never dominate.
        # Reward the *change* in burial, not its level. As a level it pays a
        # constant toll every step the foot is still down, and with episodes
        # now running to their full length that toll dwarfs anything else.
        progress_burial = torch.clamp(
            (self._prev_burial - burial) / self._burial_ref, -1.0, 1.0
        )
        self._prev_burial = burial
        # Bound the measured sinkage: when the solve is on the edge of
        # breaking, particle positions can be enormous yet still finite, and
        # an unbounded depression term then swamps the return (a mean of
        # -5e16 was observed) without ever tripping the non-finite check.
        support_sink = torch.clamp(self._support_sink_m(), 0.0, 1.0)

        # What the task pays for: getting the foot free and putting distance
        # between the robot and the hazard. Nothing here prescribes a gait --
        # crawling out on all fours earns the same as walking out. The
        # distance term is capped at 1 m/s so a ballistic leap is worth no
        # more per step than brisk progress.
        # Action rate, the smoothness term standard in locomotion rewards.
        # Freeing the foot pays well, and chasing that reward drove the
        # policy into motions violent enough to break the solve -- the
        # environment itself never diverges under the pretrained walker. A
        # real actuator cannot slam from one extreme to another in 20 ms
        # either, so this costs nothing physical while removing the
        # destructive regime. No strategy is forbidden by it.
        action_rate = (self.last_action - self._prev_action).square().mean(dim=1)
        self._prev_action = self.last_action.clone()
        # A per-step cost, however small, is worth hundreds over a 16 s
        # episode, so ending early always beat finishing and the policy kept
        # finding ways to do it. Surviving now pays instead: the stream of
        # alive credit a diverged episode forfeits is the real cost of
        # breaking the solve, and it carries no outlier for PPO to choke on.
        # Weights back to what ran for twenty rounds without instability.
        # Raising burial to 5 and cutting the action penalty to 0.01 pushed
        # the policy to actions near 20 where normal is 1, which left the
        # physics riding every limiter it has -- 390 N of 400 in sand
        # reaction, particle speed pinned at its cap -- and divergence then
        # appears at random, differently on each run of the same setup.
        # Alive credit is paid for being on the feet, not merely for existing.
        # Paid flat, it was the best-paying thing the robot could do: lying in
        # the sand collects the same 0.5 a step as standing, and it collects
        # it without the sinkage penalty a supporting foot pays, so going down
        # was worth more than staying up. Measured against the untouched
        # walking policy on the same task, the fine-tuned policy fell in 47 of
        # 48 episodes where the walker fell in 10 of 32, and its mean reward
        # climbed the whole time. That is what it had been taught to do.
        #
        # Posture is still not judged. Distance from the hazard pays whatever
        # the robot's shape, so crawling out on all fours earns while it
        # crawls and earns the alive credit back the moment it stands. What no
        # longer earns is lying still.
        reward = (
            0.5 * on_feet.float()
            + 1.0 * progress_burial
            - 1.0 * support_sink
            - 0.05 * self.last_action.square().mean(dim=1)
            - 0.05 * action_rate
            + 25.0 * ground_gained * self._was_trapped.float()
        )

        # "Fallen" must not mean "tilted right now". The task is to escape the
        # hazard; how the robot does it is left to the policy, so an airborne
        # manoeuvre has to be allowed to finish rather than being cut off the
        # instant the torso tips. Only a tilt that persists -- the robot did
        # not recover -- is a fall.
        # Going down is not failure: crawling out on all fours and standing
        # up at the end is a legitimate escape. The run counter is kept only
        # as a description of how the episode went.
        tilted = ~on_feet
        self._tilt_run = torch.where(
            tilted, self._tilt_run + 1, torch.zeros_like(self._tilt_run)
        )
        # A landing hard enough to blow up the solve leaves NaN in the state.
        # Ending that episode keeps the corruption out of the rollout instead
        # of letting it propagate into indices and kernels downstream.
        diverged = ~torch.isfinite(
            self.joint_q_t.view(self.num_envs, self.jq_per_env)
        ).all(dim=1)
        # Separate the soil going bad from the robot going bad. Resetting an
        # environment restores its particles and its joints, so an episode
        # that diverges on its first step after a reset is being poisoned by
        # something a reset does not touch -- and the two candidates, the
        # robot's articulated state and the material state, are told apart
        # only by measuring them apart.
        bad_particle = ~torch.isfinite(
            wp.to_torch(self.state_0.particle_q)
        ).all(dim=1)
        soil_bad = bad_particle.view(
            self.num_envs, self.particles_per_env
        ).any(dim=1)
        # A broken soil solve has to end the episode too, not just be logged.
        #
        # 26 ends an episode on ``diverged`` -- the robot's joint state going
        # non-finite -- and reports ``soil_bad`` beside it without acting on
        # it. For H1 that gap is nearly never exercised. For G1 it is what
        # ends every training run: an env whose particles go non-finite while
        # its joints stay finite is never reset, so the NaN sits in the shared
        # MPM solve for the rest of training. Measured across four runs the
        # signature was identical -- ``soil_bad`` latching at exactly one env
        # and throughput falling from ~90 to ~15 steps/s, because a NaN
        # particle position blows up the sparse grid's extent for everybody.
        #
        # A reset restores the env's particle block and its MPM history, so
        # ending the episode is all that is needed to clear it.
        solve_broken = diverged | soil_bad
        fallen = solve_broken
        if self.down_limit_steps > 0:
            fallen = fallen | (self._tilt_run >= self.down_limit_steps)
        # Anchoring is a failure only inside the hazard: outside it the task
        # is already "walked out", and the implicit-MPM quasi-static creep
        # (documented long-hold support decay) slowly grows the sink measure
        # under a robot that is simply standing.
        # Deep sinkage is reported, not terminal: the robot may still work
        # itself free.
        anchored = torch.zeros_like(fallen)
        deeply_sunk = (support_sink > SUPPORT_SINK_LIMIT_M) & (exit_d < EXIT_RADIUS_M)
        # Escape means out of the hazard, back on the feet, and moving
        # again -- locomotion recovered, not a posture held. A walk-in episode
        # starts outside the radius, so it only counts once the robot has
        # actually been caught: without this it "escapes" by strolling past
        # the pocket it never fell into.
        self._was_trapped |= burial >= TRAPPED_BURIAL_M
        out_now = (
            (exit_d >= EXIT_RADIUS_M) & on_feet & walking & self._was_trapped
        )
        self._hold = torch.where(out_now, self._hold + 1, torch.zeros_like(self._hold))
        # Paid once, on the step the hold completes. Reading the live
        # condition instead would pay it every step it stays true, which is a
        # reward for loitering just outside the radius rather than for the
        # escape itself.
        newly_escaped = (self._hold >= self.hold_steps) & ~self._escaped
        self._escaped |= newly_escaped
        escaped = self._escaped.clone()
        # Falling must never pay: -20 makes leap-and-crash strictly worse
        # than standing still, while the dense out_now bonus accumulates
        # toward the sparse escape reward during the required hold.
        # Falling is free -- crawling out is allowed -- but a diverged solve
        # is not something that happened to the robot, it is the simulator
        # failing to integrate the state it was driven into. Left unpriced,
        # episodes end there for nothing and the policy has no gradient away
        # from the regime that causes it. The same argument applies to the
        # soil solve breaking, so both are priced at the same rate.
        reward = (
            reward
            + 0.3 * out_now.float()
            + 10.0 * newly_escaped.float()
            - DIVERGENCE_PENALTY * solve_broken.float()
        )

        self._airborne_steps += tilted.long()
        self._peak_torso_z = torch.nan_to_num(
            torch.maximum(
                self._peak_torso_z, torso_z - surface_torso - self._upright_offset
            )
        )

        # A diverged solve can leave NaN anywhere upstream of the reward --
        # foot heights, particle percentiles, distances -- not only in
        # joint_q. Catch it on the reward itself, score it as a fall, and end
        # the episode: PPO rejects a NaN reward outright, so letting one
        # through costs the whole run.
        # Same reasoning one level up: keep the return itself in a range the
        # value function can represent.
        reward = torch.clamp(reward, -DIVERGENCE_PENALTY, 50.0)
        bad = ~torch.isfinite(reward)
        if bad.any():
            fallen = fallen | bad
            reward = torch.where(
                bad, torch.full_like(reward, -DIVERGENCE_PENALTY), reward
            )

        timeout = self._steps >= self.max_episode_steps
        # Escaping must not end the episode. It did, and since surviving pays
        # 0.5 a step, an escape at step 100 of 500 threw away four fifths of
        # the return it could still have earned -- against which a one-off
        # bonus of 10 is nothing. The policy was being taught to stay in the
        # hazard, which is what the flat escape rate against a steadily
        # climbing reward was showing. Leaving the episode running makes
        # escaping strictly better than not escaping, with no outlier for the
        # value function to swallow. It also matches the task: the robot has
        # to keep walking afterwards, not stop the moment it is clear.
        done = fallen | anchored | timeout
        info = {
            "escaped": escaped,
            # The transition, not the latched flag: counting the flag would
            # score one escape once for every step that follows it.
            "newly_escaped": newly_escaped,
            "fallen": fallen,
            "anchored": anchored,
            # PPO bootstraps the value of a time-out and does not bootstrap a
            # termination, so this flag decides whether an episode's future is
            # written off. Escape must not appear here: it no longer ends the
            # episode, and excluding it would mark an episode that escaped and
            # then ran out the clock as a hard termination -- throwing away
            # exactly the return the escape was supposed to earn.
            "timeout": timeout & ~(fallen | anchored),
            "exit_distance": exit_d,
            "support_sink": support_sink,
            # The gate on both the alive credit and the escape test. Without
            # it in the logs, every explanation of a falling return was a
            # guess between "off its feet" and "penalties dominate".
            "on_feet": on_feet,
            "progress_dist": progress_dist,
            # How the escape was achieved (reporting only).
            "tilted_seconds": self._airborne_steps.float() * CONTROL_DT,
            "peak_torso_height": self._peak_torso_z.clone(),
            "walk_speed": speed,
            "deeply_sunk": deeply_sunk,
            "was_trapped": self._was_trapped.clone(),
            "diverged": diverged,
            "soil_bad": soil_bad,
            "action_nan": self._action_nan,
        }
        if done.any():
            self._reset_envs(done.cpu())
        # Last line of defence: the reset restores a clean state, but a stale
        # NaN reaching the policy would poison the update the same way.
        obs = torch.nan_to_num(self._observation())
        return obs, reward, done, info

    def _support_sink_m(self) -> torch.Tensor:
        """Support-foot surface depression per env [m], the 07/22 measure:
        95th percentile of (reference z - current z) over surface particles
        inside the foot footprint, in the foot's local frame."""
        cur = wp.to_torch(self.state_0.particle_q)
        ref_z = wp.to_torch(self.reference_wp)[:, 2]
        depression = torch.clamp(ref_z - cur[:, 2], min=0.0)
        n = self.particles_per_env
        support_ids = self.foot_ids_t[:, self.support_index]
        out = torch.zeros(self.num_envs, device=self.device)
        for env in range(self.num_envs):
            sl = slice(env * n, (env + 1) * n)
            foot_q = self.body_q_t[support_ids[env]]
            relative = cur[sl] - foot_q[:3]
            local = MARS.quaternion_rotate_inverse(
                foot_q[3:7].unsqueeze(0).expand(n, -1), relative
            )
            under = (
                self.surface_mask_t
                & (local[:, 0].abs() <= 0.16)
                & (local[:, 1].abs() <= 0.08)
            )
            if torch.any(under):
                out[env] = torch.quantile(depression[sl][under], 0.95)
        return out

    # ------------------------------------------------------------------
    # Diagnostics. Off unless enable_diagnostics() is called: they copy
    # particle velocities and the saturation counter to the host every step,
    # which is far too expensive to leave on during training.
    # ------------------------------------------------------------------
    def enable_diagnostics(self) -> None:
        self.diag = {
            "steps": 0,
            "foot_sat": 0,
            "other_sat": 0,
            "raw_force_max": 0.0,
            "raw_torque_max": 0.0,
            "raw_force_p99": [],
            "speed_bound": 0.0,
            "speed_max": 0.0,
            "soil_bad_steps": 0,
        }
        self._sat_prev = self.reaction_saturation_counts.numpy().copy()

    def _diag_before_clamp(self) -> None:
        """Raw soil reaction and particle speed, before either is limited."""
        wrench = wp.to_torch(self.body_sand_forces)
        force = torch.linalg.norm(wrench[:, :3], dim=1)
        torque = torch.linalg.norm(wrench[:, 3:], dim=1)
        self.diag["raw_force_max"] = max(
            self.diag["raw_force_max"], float(force.max())
        )
        self.diag["raw_torque_max"] = max(
            self.diag["raw_torque_max"], float(torque.max())
        )
        feet = self.foot_ids_t.flatten()
        self.diag["raw_force_p99"].append(float(force[feet].max()))

        speed = torch.linalg.norm(wp.to_torch(self.state_0.particle_qd), dim=1)
        cap = float(self.args.particle_speed_cap)
        self.diag["speed_bound"] += float((speed > cap).float().mean())
        self.diag["speed_max"] = max(self.diag["speed_max"], float(speed.max()))
        bad = ~torch.isfinite(wp.to_torch(self.state_0.particle_q)).all(dim=1)
        self.diag["soil_bad_steps"] += int(bad.any())

    def _diag_after_clamp(self) -> None:
        """How often the clamp actually bound, split feet vs everything else."""
        now = self.reaction_saturation_counts.numpy()
        delta = now - self._sat_prev
        self._sat_prev = now.copy()
        foot_positions = {
            self.all_body_ids.index(b) for b in self.foot_body_ids
        }
        for position, count in enumerate(delta):
            if count <= 0:
                continue
            if position in foot_positions:
                self.diag["foot_sat"] += int(count)
            else:
                self.diag["other_sat"] += int(count)
        self.diag["steps"] += 1

    def diagnostics_report(self) -> str:
        d = self.diag
        steps = max(d["steps"], 1)
        feet_per_step = 2 * self.num_envs
        other_per_step = len(self.all_body_ids) - feet_per_step
        foot_rate = d["foot_sat"] / (steps * feet_per_step)
        other_rate = d["other_sat"] / (steps * max(other_per_step, 1))
        speed_rate = d["speed_bound"] / steps
        p99 = float(np.percentile(d["raw_force_p99"], 99)) if d["raw_force_p99"] else 0.0
        gate = "PASS" if foot_rate < 0.02 else "FAIL"
        return (
            f"[diag] {steps} steps, {self.num_envs} envs\n"
            f"  reaction clamp   {self.reaction_force_limit:.0f} N / "
            f"{self.reaction_torque_limit:.0f} N.m\n"
            f"  foot saturation   {100 * foot_rate:6.2f} %   <- gate < 2 %: {gate}\n"
            f"  other-link sat.   {100 * other_rate:6.2f} %\n"
            f"  raw force  max    {d['raw_force_max']:8.1f} N   (feet p99 {p99:.1f} N)\n"
            f"  raw torque max    {d['raw_torque_max']:8.1f} N.m\n"
            f"  particle speed cap {self.args.particle_speed_cap:.1f} m/s: "
            f"bound on {100 * speed_rate:.2f} % of particles, max "
            f"{d['speed_max']:.2f} m/s\n"
            f"  steps with non-finite soil: {d['soil_bad_steps']}/{steps}"
        )

    def capture_snapshot(self) -> dict:
        """Env 0's full state in the on-disk snapshot format.

        Positions are written back in site coordinates -- env offset removed,
        ``site_xy`` added -- because that is what :meth:`__init__` expects to
        read and re-place per env.
        """
        shift = np.array([*self.site_xy, 0.0], dtype=np.float32)
        origin = np.array([*self.offsets[0], 0.0], dtype=np.float32)
        n = self.particles_per_env
        mpm = self.state_0.mpm
        joint_q = self.state_0.joint_q.numpy()[: self.jq_per_env].copy()
        joint_q[0] += self.site_xy[0] - self.offsets[0][0]
        joint_q[1] += self.site_xy[1] - self.offsets[0][1]
        return {
            "joint_q": joint_q,
            "joint_qd": self.state_0.joint_qd.numpy()[: self.jqd_per_env].copy(),
            "particle_q": self.state_0.particle_q.numpy()[:n] - origin + shift,
            "particle_qd": self.state_0.particle_qd.numpy()[:n].copy(),
            "reference_q": self.reference_wp.numpy()[:n] - origin + shift,
            "mpm_qd_grad": mpm.particle_qd_grad.numpy()[:n].copy(),
            "mpm_elastic_strain": mpm.particle_elastic_strain.numpy()[:n].copy(),
            "mpm_Jp": mpm.particle_Jp.numpy()[:n].copy(),
            "mpm_stress": mpm.particle_stress.numpy()[:n].copy(),
            "mpm_transform": mpm.particle_transform.numpy()[:n].copy(),
            "trapped_index": int(self.trapped_index),
        }

    def _refresh_torch_views(self) -> None:
        # state_0 identity changes each frame because of the substep swap.
        self.joint_q_t = wp.to_torch(self.state_0.joint_q)
        self.joint_qd_t = wp.to_torch(self.state_0.joint_qd)
        self.body_q_t = wp.to_torch(self.state_0.body_q)


def build_args(extra: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4)
    # 6 m pitch gives each env a +-3 m DTM patch, so the robot-following
    # 2x2 m soil window (root can reach ~1.5 m before termination) never
    # recycles particles past its own patch onto a neighbour's.
    parser.add_argument("--env-pitch", type=float, default=6.0)
    parser.add_argument("--episode-seconds", type=float, default=10.0)
    # The G1 walking policy was trained on lin_vel_x in [0.5, 1.5]; 0.3 (H1's
    # value) is outside the command distribution it ever saw.
    parser.add_argument("--command-x", type=float, default=0.5)
    parser.add_argument("--robot-usd", type=Path, default=G1_USD)
    parser.add_argument(
        "--particle-speed-cap",
        type=float,
        default=PARTICLE_MAX_SPEED,
        help=(
            "Ceiling on material point speed [m/s]. At the inherited 1.5 this "
            "is not a safety net: measured under a driven policy it binds on "
            "246 of 250 steps, so the velocity field is being cut back every "
            "frame and no longer matches the stress state it was solved with. "
            "At 6.0 it binds on 18 of 250. Raise it before blaming anything "
            "else for a soil solve that will not stay finite."
        ),
    )
    parser.add_argument("--pocket-radius", type=float, default=0.3)
    parser.add_argument("--pocket-scale", type=float, default=0.1)
    parser.add_argument("--pocket-dx", type=float, default=0.45)
    parser.add_argument("--pocket-dy", type=float, default=0.12)
    # 200 matches 07/22 and the soil calibration; the implicit solve is
    # highly iteration-sensitive (plate tests: 0.96 kPa at 50 vs 3.27 kPa at
    # 200), so fewer iterations silently soften the soil.
    parser.add_argument("--mpm-iterations", type=int, default=200)
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.5,
        help=(
            "Upright time past the exit radius required for escape success. "
            "Lower values are curriculum stages; the graded criterion is 1.5 s."
        ),
    )
    parser.add_argument(
        "--start-mode",
        choices=("trapped", "walk_in", "mixed"),
        default="trapped",
        help=(
            "'trapped' restores the entrapment snapshot (dense escape practice, "
            "but the episode contains no walking). 'walk_in' starts the robot "
            "upstream on undisturbed soil so it walks into the hazard itself, "
            "giving the full walk-trap-escape-walk arc. 'mixed' alternates, "
            "which keeps escape practice dense while walking stays exercised."
        ),
    )
    parser.add_argument(
        "--approach-m",
        type=float,
        default=1.5,
        help="Walk-in start distance upstream of the pocket [m].",
    )
    parser.add_argument(
        "--down-seconds",
        type=float,
        default=2.0,
        help=(
            "How long the robot may stay off its feet before the episode "
            "ends. 0 lets it keep trying for the whole episode."
        ),
    )
    parser.add_argument("--site-xy", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--snapshot-path", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--terrain-usd",
        type=Path,
        default=Path("terrain/mars_terrain/mars_real_terrain_env/usd/crop30m")
        / "gusev_spirit_center_columbia_hills_1024px_stride2_crop30m_r530c380_env.usd",
        help="Env USD whose DTM the snapshot was made on (patch + scan source).",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--bootstrap-snapshot",
        action="store_true",
        help=(
            "Write a G1 start state onto the released H1 snapshot's soil and "
            "exit. The soil (bed, calibration, site) is robot-independent; "
            "only the articulation state and the dug hole are not, so the hole "
            "is discarded and the robot is stood on the undisturbed surface."
        ),
    )
    parser.add_argument(
        "--make-snapshot",
        action="store_true",
        help=(
            "Walk G1 into the pocket under its own walking policy and save the "
            "deepest-burial frame as the entrapment snapshot."
        ),
    )
    parser.add_argument(
        "--walk-policy",
        type=Path,
        default=None,
        help="RSL-RL checkpoint of the G1 Mars walking policy (for --make-snapshot).",
    )
    parser.add_argument("--snapshot-frames", type=int, default=500)
    parser.add_argument("--snapshot-target-mm", type=float, default=180.0)
    parser.add_argument(
        "--reaction-force-limit", type=float, default=REACTION_FORCE_LIMIT_N
    )
    parser.add_argument(
        "--reaction-torque-limit", type=float, default=REACTION_TORQUE_LIMIT_NM
    )
    parser.add_argument(
        "--diagnose",
        type=int,
        default=0,
        help=(
            "Run N steps under the walking policy plus training-scale action "
            "noise and report how often the soil reaction clamp and the "
            "particle speed cap bind. The clamp is a bound on a physical "
            "reaction: if it binds often, the force the robot feels is not "
            "the one the solver computed."
        ),
    )
    parser.add_argument("--diagnose-noise", type=float, default=0.3)
    parser.add_argument(
        "--finger-effort",
        type=float,
        default=300.0,
        help="Effort limit on the 14 finger joints [N.m]. G1_CFG's 300 is a "
        "group-wide default applied to 14 g links.",
    )
    parser.add_argument("--h1-snapshot", type=Path, default=_H1_RELEASE_SNAPSHOT)
    parser.add_argument("--bootstrap-out", type=Path, default=_BOOTSTRAP_SNAPSHOT)
    parser.add_argument("--snapshot-out", type=Path, default=_WORKING_SNAPSHOT)
    parser.add_argument(
        "--drop-test",
        type=int,
        default=0,
        help=(
            "Run N undriven steps and report the lowest link relative to the "
            "surface. The gate is > -300 mm: a robot without ground-contact "
            "colliders free-falls through the terrain (HANDOFF §11)."
        ),
    )
    return parser.parse_args(extra if extra is not None else [])


def bootstrap_snapshot(args: argparse.Namespace) -> None:
    """Seed a G1 snapshot from the released H1 one.

    Only the soil carries over. It is the same calibrated gusev bed on the
    same site, built by the same protocol, and none of that depends on which
    robot was standing on it -- but the hole in it does, so the *reference*
    (undisturbed) positions are used as the particle state and the history-
    dependent MPM fields are reset to pristine. The robot is stood on the
    surface in its nominal pose; :func:`make_snapshot` then walks it in.
    """
    source = Path(args.h1_snapshot).expanduser().resolve()
    with open(source, "rb") as stream:
        h1 = pickle.load(stream)

    reference = h1["reference_q"].astype(np.float32)
    count = len(reference)
    identity = np.tile(np.eye(3, dtype=np.float32), (count, 1, 1))
    surface_z = float(np.percentile(reference[:, 2], 95))

    joint_q = np.zeros(7 + ACTION_DIM, dtype=np.float32)
    # Stand the robot where the H1 snapshot's robot was, on the undisturbed
    # surface, facing along +x -- the direction the pocket offset points.
    joint_q[0] = float(h1["joint_q"][0])
    joint_q[1] = float(h1["joint_q"][1])
    joint_q[2] = surface_z + STANDING_ROOT_HEIGHT_M
    joint_q[3:7] = [0.0, 0.0, 0.0, 1.0]
    joint_q[7:] = [G1_DEFAULT_JOINT_POS.get(n, 0.0) for n in G1_JOINT_ORDER]

    payload = {
        "joint_q": joint_q,
        "joint_qd": np.zeros(6 + ACTION_DIM, dtype=np.float32),
        "particle_q": reference.copy(),
        "particle_qd": np.zeros((count, 3), dtype=np.float32),
        "reference_q": reference.copy(),
        "mpm_qd_grad": np.zeros((count, 3, 3), dtype=np.float32),
        "mpm_elastic_strain": identity.copy(),
        "mpm_Jp": np.ones(count, dtype=np.float32),
        "mpm_stress": np.zeros((count, 3, 3), dtype=np.float32),
        "mpm_transform": identity.copy(),
        "trapped_index": 0,
    }
    out = Path(args.bootstrap_out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as stream:
        pickle.dump(payload, stream)
    print(
        f"[bootstrap] {count} particles from {source.name}; "
        f"surface p95={surface_z:.3f} m, root z={joint_q[2]:.3f} m -> {out}"
    )
    print(
        "[bootstrap] NOTE: the robot is standing, not trapped. Run "
        "--make-snapshot next to produce the real entrapment state."
    )


def load_walk_policy(path: Path, device: torch.device) -> torch.nn.Module:
    """Rebuild the deterministic actor from an RSL-RL checkpoint.

    The trained runs were saved by ``OnPolicyRunner``, whose actor is a plain
    MLP under ``actor_state_dict``; the exported TorchScript copy only exists
    for one older run, so the weights are loaded directly instead.
    """
    from torch import nn

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    state = checkpoint["actor_state_dict"]
    hidden = [512, 256, 128]
    layers: list[nn.Module] = []
    dims = [OBS_DIM] + hidden
    for index in range(len(hidden)):
        layers += [nn.Linear(dims[index], dims[index + 1]), nn.ELU()]
    layers += [nn.Linear(hidden[-1], ACTION_DIM)]
    mlp = nn.Sequential(*layers)
    weights = {k[len("mlp.") :]: v for k, v in state.items() if k.startswith("mlp.")}
    missing, unexpected = mlp.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"walking checkpoint does not match a {OBS_DIM}->{ACTION_DIM} MLP "
            f"{hidden}: missing={missing} unexpected={unexpected}"
        )
    return mlp.to(device).eval()


def make_snapshot(args: argparse.Namespace) -> None:
    """Walk G1 into the pocket and save the deepest-burial frame.

    Mirrors ``22_escape_env.py --make-snapshot``: the entrapment state has to
    be produced by the physics rather than posed by hand, or the soil around
    the foot carries no stress history and the escape it is asked to perform
    is not the one it fell into.
    """
    if args.walk_policy is None:
        raise SystemExit("--make-snapshot needs --walk-policy <checkpoint>")
    args.num_envs = 1
    args.start_mode = "walk_in"
    env = EscapeEnvVec(args)
    obs = env.reset_all()
    policy = load_walk_policy(Path(args.walk_policy).expanduser().resolve(), env.device)

    best_depth = -1.0
    best: dict | None = None
    for frame in range(args.snapshot_frames):
        with torch.no_grad():
            action = policy(obs)
        obs, _, _, _ = env.step(action)
        feet = env.body_q_t[env.foot_ids_t.flatten(), 2].view(env.num_envs, 2)
        soil_top = env._soil_top_per_env()
        sink = (soil_top.unsqueeze(1) - feet)[0]
        depth = float(sink.max())
        if depth > best_depth:
            best_depth = depth
            env.trapped_index = int(torch.argmax(sink))
            env.support_index = 1 - env.trapped_index
            best = env.capture_snapshot()
        if frame % 25 == 0:
            print(
                f"  frame {frame:4d}  deepest foot {1000 * depth:6.1f} mm  "
                f"best {1000 * best_depth:6.1f} mm"
            )
        if best_depth >= args.snapshot_target_mm / 1000.0 and frame > 100:
            break

    if best is None or best_depth <= 0.0:
        raise SystemExit("no burial recorded; deepen the pocket or walk further")
    out = Path(args.snapshot_out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as stream:
        pickle.dump(best, stream)
    print(
        f"[snapshot] foot {best['trapped_index']} buried "
        f"{1000 * best_depth:.1f} mm -> {out}"
    )
    if best_depth < args.snapshot_target_mm / 1000.0:
        print(
            f"[snapshot] WARNING: {1000 * best_depth:.1f} mm is below the "
            f"{args.snapshot_target_mm:.0f} mm target. G1 may not be breaking "
            "the crust -- see G1_escape.md §2.2 before training on this."
        )


def drop_test(args: argparse.Namespace) -> None:
    """Undriven fall test: the gate that says the colliders actually exist.

    A robot whose only shapes are its feet falls straight through the terrain
    (H1 measured -1745 mm below the surface after 400 undriven steps, which is
    free fall at Mars gravity) and can neither crawl nor push itself up.

    Three numbers are reported because they answer different questions. The
    **root** height is the free-fall test: the pelvis carries a collider, so
    if the robot has a floor its trunk stays on it. The **shape-bearing**
    minimum is what those colliders actually do. The **any-link** minimum
    includes hands, head and hip links, which carry no geometry in this asset
    and therefore pass through everything -- it is reported for honesty, not
    used as a gate.
    """
    env = EscapeEnvVec(args)
    env.reset_all()
    zero = torch.zeros(env.num_envs, ACTION_DIM, device=env.device)
    shaped = torch.tensor(
        sorted(env.ground_contact_bodies), device=env.device
    )
    bodies_per_env = env.model.body_count // env.num_envs
    shaped_env = torch.tensor(
        [int(b) // bodies_per_env for b in sorted(env.ground_contact_bodies)],
        device=env.device,
    )

    def surface_at(ids: torch.Tensor, env_of: torch.Tensor) -> torch.Tensor:
        xy = env.body_q_t[ids, 0:2]
        return (
            env._sample_patch(
                xy[:, 0] - env.offsets_t[env_of, 0],
                xy[:, 1] - env.offsets_t[env_of, 1],
            )
            + SOIL_SURFACE_OFFSET_M
        )

    worst_root = float("inf")
    worst_shaped = float("inf")
    worst_any = float("inf")
    for step in range(args.drop_test):
        env.step(zero)
        jq = env.joint_q_t.view(env.num_envs, env.jq_per_env)
        root_surface = env._sample_patch(
            jq[:, 0] - env.offsets_t[:, 0], jq[:, 1] - env.offsets_t[:, 1]
        ) + SOIL_SURFACE_OFFSET_M
        root = float((jq[:, 2] - root_surface).min())
        shaped_clear = float(
            (env.body_q_t[shaped, 2] - surface_at(shaped, shaped_env)).min()
        )
        any_clear = float(env._nonfoot_clearance().min())
        worst_root = min(worst_root, root)
        worst_shaped = min(worst_shaped, shaped_clear)
        worst_any = min(worst_any, any_clear)
        if step % 50 == 0:
            print(
                f"  step {step:4d}  root {1000 * root:+8.1f} mm  "
                f"shaped {1000 * shaped_clear:+8.1f} mm  "
                f"any {1000 * any_clear:+8.1f} mm"
            )
    verdict = "PASS" if worst_root > -0.300 else "FAIL"
    print(
        f"[drop-test] {args.drop_test} undriven steps:\n"
        f"   root (pelvis) minimum      {1000 * worst_root:+8.1f} mm   <- gate > -300 mm: {verdict}\n"
        f"   collider-bearing minimum   {1000 * worst_shaped:+8.1f} mm\n"
        f"   any-link minimum           {1000 * worst_any:+8.1f} mm   (links without geometry)"
    )


def diagnose(args: argparse.Namespace) -> None:
    """Measure what the limiters are doing under a training-like action stream.

    The training collapse showed up as a soil solve that stopped being finite,
    and the two candidates -- a reaction clamp set too low for this robot, and
    a foot pressure the bed cannot carry -- are told apart only by measuring
    them. Actions are the walking policy's mean plus Gaussian noise at the
    exploration scale PPO was running, so the distribution matches the one
    that broke it rather than a quiet deterministic rollout.
    """
    env = EscapeEnvVec(args)
    obs = env.reset_all()
    policy = None
    if args.walk_policy is not None:
        policy = load_walk_policy(
            Path(args.walk_policy).expanduser().resolve(), env.device
        )
    env.enable_diagnostics()
    burial = []
    for step in range(args.diagnose):
        if policy is None:
            action = args.diagnose_noise * torch.randn(
                env.num_envs, ACTION_DIM, device=env.device
            )
        else:
            with torch.no_grad():
                action = policy(obs)
            action = action + args.diagnose_noise * torch.randn_like(action)
        obs, _, _, info = env.step(action)
        burial.append(float(info["burial"].mean()) if "burial" in info else 0.0)
        if step and step % 100 == 0:
            print(f"  step {step}", flush=True)
    print(env.diagnostics_report())
    feet = env.body_q_t[env.foot_ids_t.flatten(), 2].view(env.num_envs, 2)
    sink = (env._soil_top_per_env().unsqueeze(1) - feet).max(dim=1).values
    print(
        f"  final foot burial [mm]: "
        f"{[int(1000 * v) for v in sink.tolist()][:8]}"
    )


def main() -> None:
    args = build_args(sys.argv[1:])
    if args.diagnose:
        diagnose(args)
        return
    if args.bootstrap_snapshot:
        bootstrap_snapshot(args)
        return
    if args.make_snapshot:
        make_snapshot(args)
        return
    if args.drop_test:
        drop_test(args)
        return
    env = EscapeEnvVec(args)
    obs = env.reset_all()
    print(f"obs shape: {tuple(obs.shape)}")
    if args.smoke:
        import time

        for warm in range(3):
            env.step(0.1 * torch.randn(env.num_envs, ACTION_DIM, device=env.device))
        wp.synchronize()
        started = time.perf_counter()
        frames = 20
        escapes = 0
        for _ in range(frames):
            _, reward, done, info = env.step(
                0.1 * torch.randn(env.num_envs, ACTION_DIM, device=env.device)
            )
            escapes += int(info["escaped"].sum())
        wp.synchronize()
        dt = (time.perf_counter() - started) / frames
        feet = env.body_q_t[env.foot_ids_t.flatten(), 2].view(env.num_envs, 2)
        soil_top = env._initial_soil_top
        trapped = feet[:, env.trapped_index] - soil_top
        support = feet[:, env.support_index] - soil_top
        print(
            f"smoke: {frames} frames  {dt:.3f} s/frame  "
            f"{env.num_envs / dt:.1f} env-steps/s  reward[0]={float(reward[0]):+.3f}"
        )
        print(
            "  trapped foot vs soil top [mm]:",
            [int(1000 * v) for v in trapped.tolist()][:8],
        )
        print(
            "  support foot vs soil top [mm]:",
            [int(1000 * v) for v in support.tolist()][:8],
        )
        print(
            "  support local sink [mm]:",
            [int(1000 * v) for v in env._support_sink_m().tolist()][:8],
        )
        print(f"  escapes in smoke: {escapes}")


if __name__ == "__main__":
    main()
