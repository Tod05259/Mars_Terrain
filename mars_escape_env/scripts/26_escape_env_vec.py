# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vectorised escape environment: N trapped H1s in one Newton model.

Every training round so far ran a single environment — 25k steps per round,
eight minutes of experience. This builds ``--num-envs`` copies of the
entrapment scenario into one model, stepped by one MuJoCo solver and one
implicit-MPM solve, which the scaling measurements showed is nearly free
(2.1 M particles across 256 windows stepped at 0.57 s/frame).

Two deliberate simplifications versus the single-env wrapper (22):

* **Static soil windows.** The escape task moves the robot about a metre, so
  each env gets a fixed 1.6 m window around its pocket instead of the moving,
  particle-recycling window of the walking runtime. Outside the window is
  rigid ground — consistent with the task definition, where outside the
  hazard zone the terrain is trafficable.
* **Flat ground at training time.** Real-DTM sites differ in height and slope
  per env, which breaks snapshot replication. Training runs on flat ground
  with the calibrated gusev soil; evaluation stays on the real-terrain
  single-env (22/24), doubling as a sim-to-sim generalisation check.

Episodes start from the same entrapment snapshot as the single-env version,
replicated to every env with its spatial offset applied.

Usage
-----
    python terrain/scripts/26_escape_env_vec.py --num-envs 4 --smoke
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

MARS_G = -3.721
CONTROL_DT = 0.02
SIM_SUBSTEPS = 4
# Two-way coupling constants, identical to 07_h1_mars_newton.py defaults: the
# soil calibration (ppc=3, substeps=2) and the anchoring-budget measurements
# were made with these values.
MPM_SUBSTEPS = 2
REACTION_FORCE_LIMIT_N = 400.0
REACTION_TORQUE_LIMIT_NM = 80.0
# 07's height scan reports the sand surface (DTM + mpm_surface_offset), not
# the bare DTM; the snapshot was made with --mpm-depth 0.30.
SOIL_SURFACE_OFFSET_M = 0.30
PARTICLE_DAMPING = 0.98
PARTICLE_MAX_SPEED = 1.5
ACTION_DIM = 19
OBS_DIM = 256
SCAN_POINTS = 187

_WORKING_SNAPSHOT = Path("terrain/output/escape_env/entrapment_snapshot.pkl")
_RELEASE_SNAPSHOT = Path("terrain/release/entrapment_snapshot.pkl")
# terrain/output/ is not in the repository, so a fresh clone only has the
# released copy; prefer the working one when it exists.
DEFAULT_SNAPSHOT = (
    _WORKING_SNAPSHOT if _WORKING_SNAPSHOT.is_file() else _RELEASE_SNAPSHOT
)

SUPPORT_SINK_LIMIT_M = 0.20
# Functional support test, in place of a torso angle: the robot is on its
# feet when nothing except the feet is near the ground. The knees are the
# lowest non-foot links, so the margin has to clear a deep crouch without
# admitting a robot resting on its shins.
NONFOOT_CLEARANCE_M = 0.12
# H1 sole footprint, used to read the material directly under a foot
# rather than over a window the site's slope dominates.
FOOT_HALF_LENGTH_M = 0.12
FOOT_HALF_WIDTH_M = 0.06
# Recovered locomotion: planar speed at least half the 0.3 m/s command.
WALK_SPEED_MIN_MPS = 0.15
# How long the robot may be off its feet before the episode ends. A leap or a
# flip passes through this state and is allowed to land; only a robot that
# stays down has failed to recover.
DOWN_GRACE_STEPS = int(0.6 / CONTROL_DT)
EXIT_RADIUS_M = 1.0
# Burial that counts as having been caught by the hazard, so a walk-in
# episode cannot be scored as an escape it never needed to make.
# What counts as a foot being caught, as opposed to standing on soft ground.
#
# Insufficient bearing reaction is what makes a foot sink, but it cannot test
# for being caught: a foot that has come to rest is by definition carrying its
# share of the weight, so the reaction balances exactly when the robot is most
# stuck. Spirit sat still at Troy with its wheels in equilibrium.
#
# Being caught is about extraction: the force to free the foot exceeds what
# the robot can apply while holding itself up. The plate measurements put that
# budget at -6 N for H1 -- 93 N to pull a foot out of 226 mm of material,
# against a support foot that tops out 6 N short of the 284 N it would have to
# carry. Two things have to be true of the material for that to be the case,
# and both are measured under the foot itself:
#   * it is deep enough to swallow the foot (TRAPPED_BURIAL_M), and
#   * enough of it lies on top of the foot to resist lifting
#     (TRAPPED_OVERBURDEN).
# A foot resting on the surface reads about 0.05 for the second.
TRAPPED_BURIAL_M = 0.10
TRAPPED_OVERBURDEN = 0.50
# Cost of a diverged solve, kept on the scale of an episode return (roughly
# -20 to +10). At 200 it was a ten-fold outlier: value targets and advantages
# blew up and the policy's weights went non-finite mid-update. Divergence is
# kept rare by the action-rate term instead, which removes the violent
# motions that cause it, so the penalty does not have to carry that load.
DIVERGENCE_PENALTY = 20.0
# H1 pelvis height in the nominal standing pose [m], used to seat a walk-in
# start on the sand surface.
STANDING_ROOT_HEIGHT_M = 1.06
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

        robot_usd = (
            newton.utils.download_asset("unitree_h1") / "usd_structured" / "h1.usda"
        )
        snap_root = self._snap["joint_q"][:7].copy()

        self.joint_q_size = None
        self.joint_qd_size = None
        # Each robot lives in its own MJWarp world; separate_worlds batches
        # the identical articulations instead of building one giant tree.
        for env in range(self.num_envs):
            builder.begin_world()
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
            builder.end_world()

        # Per-joint drives copied from the walking runtime's gains.
        joints_per_env = len(builder.joint_label) // self.num_envs
        for joint_id, label in enumerate(builder.joint_label):
            if joint_id % joints_per_env == 0:
                continue  # free base joint of each env
            name = label.rsplit("/", 1)[-1]
            name = "torso" if name == "torso_1" else name
            dof = builder.joint_qd_start[joint_id]
            stiffness, damping = MARS.H1MarsNewton._joint_gains(name)
            builder.joint_target_mode[dof] = int(JointTargetMode.POSITION)
            builder.joint_target_ke[dof] = stiffness
            builder.joint_target_kd[dof] = damping
            builder.joint_effort_limit[dof] = 100.0 if "ankle" in name else 300.0

        # Feet: only they touch particles.
        self.foot_body_ids = [
            index
            for index, label in enumerate(builder.body_label)
            if label.endswith("ankle_link")
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
        GROUND_CONTACT_LINKS = (
            "pelvis",
            "torso_link",
            "knee_link",
            "elbow_link",
        )
        ground_contact_bodies = {
            index
            for index, label in enumerate(builder.body_label)
            if any(label.endswith(suffix) for suffix in GROUND_CONTACT_LINKS)
        }
        for shape_id, body_id in enumerate(builder.shape_body):
            if body_id >= 0 and body_id not in foot_set:
                # Soil stays a feet-only interaction: the soft-contact pass
                # allocates per particle and per shape, so opening it to every
                # link is what exhausts memory, and the two-way coupling is
                # calibrated at the foot anyway.
                builder.shape_flags[shape_id] &= ~newton.ShapeFlags.COLLIDE_PARTICLES
                if body_id not in ground_contact_bodies:
                    builder.shape_flags[shape_id] &= ~newton.ShapeFlags.COLLIDE_SHAPES
        self.ground_contact_bodies = sorted(ground_contact_bodies)

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

        order = [
            MARS.ISAACLAB_H1_JOINT_ORDER.index(
                "torso" if n == "torso_1" else n
            )
            for n in (
                "torso" if n == "torso_1" else n
                for n in [
                    l.rsplit("/", 1)[-1]
                    for l in self.model.joint_label[1 : self.jq_per_env and 20]
                ]
            )
        ] if False else None
        # Policy joint order mapping, computed once from env 0's labels.
        labels = [
            l.rsplit("/", 1)[-1] for l in self.model.joint_label
        ]
        env0 = labels[1:20]
        env0 = ["torso" if n == "torso_1" else n for n in env0]
        if len(env0) != 19 or 'right_elbow' not in env0:
            raise RuntimeError(f'env0 joint labels wrong ({len(env0)}): {env0}')
        self.policy_to_newton = torch.tensor(
            [env0.index(name) for name in MARS.ISAACLAB_H1_JOINT_ORDER],
            device=self.device,
        )
        self.default_pos_policy = torch.tensor(
            [
                MARS.H1_DEFAULT_JOINT_POS[
                    "torso_1" if n == "torso" else n
                ]
                for n in MARS.ISAACLAB_H1_JOINT_ORDER
            ],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        self.default_pos_newton = torch.tensor(
            [MARS.H1_DEFAULT_JOINT_POS["torso_1" if n == "torso" else n] for n in env0],
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
        feet_set = set(self.foot_body_ids)
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
        # Sample grid over the moving soil window, plus the spawn-layer depth
        # offsets above the DTM. Both follow the snapshot's own geometry: the
        # window and the spacing changed together when the bed moved to the
        # 12.5 mm the material was calibrated at, and hard-coding either here
        # silently mismatches the measurement against the bed it measures.
        ref = self._reference_q
        half = 0.5 * float(
            max(ref[:, 0].max() - ref[:, 0].min(), ref[:, 1].max() - ref[:, 1].min())
        )
        win = torch.linspace(-half, half, 17, device=self.device)
        wx, wy = torch.meshgrid(win, win, indexing="xy")
        self.win_x = wx.reshape(-1)
        self.win_y = wy.reshape(-1)
        # Spawn heights above the DTM, sampled from the snapshot rather than
        # assumed. The bed's depth and spacing are properties of the file being
        # loaded; writing them here as constants is how a 12.5 mm bed ends up
        # measured on a 25 mm ruler. Quantiles rather than distinct values:
        # the bed is draped over sloped ground, so its heights form a
        # continuum, and one level per two hundred particles describes the
        # profile without carrying thousands of them through every step.
        levels = np.quantile(self.rest_z_np.astype(np.float64), np.linspace(0, 1, 24))
        self.layer_offsets = torch.tensor(
            levels, device=self.device, dtype=torch.float32
        )
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
        # Under the foot, not over the window: see _soil_top_under_feet.
        foot_top0 = self._soil_top_under_feet()[:, self.trapped_index]
        self._burial0 = torch.clamp(
            foot_top0 - feet_z[:, self.trapped_index], min=1e-3
        )
        self._prev_burial = torch.clamp(
            foot_top0 - feet_z[:, self.trapped_index], min=0.0
        )
        over0 = self._overburden_fraction()[:, self.trapped_index]
        print(
            f"[EscapeVec] reset burial {float(self._burial0.mean()) * 1000:.0f} mm, "
            f"overburden {float(over0.mean()):.2f} "
            f"(a foot resting on the surface reads about 0.05)"
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
        jq[base + 2] = surface + STANDING_ROOT_HEIGHT_M
        jqd[env * self.jqd_per_env : (env + 1) * self.jqd_per_env] = 0.0

    def _soil_top_under_feet(self) -> torch.Tensor:
        """Height of the material surface directly under each foot [m].

        Shape ``[num_envs, 2]``, ordered as :attr:`foot_ids_t`.

        Burial has to be read against the ground the foot is standing on, and
        a percentile over a two-metre window is not that. The site falls 620 mm
        across that window, so the percentile returns the height of the uphill
        edge; subtracting a downhill foot from it reports the slope as burial.
        That is where the snapshot's advertised 220 mm came from -- measured
        against the particles the foot is actually in, it is 12 mm.

        The analytic surface is no substitute either: the settled bed is
        thinner than its spawn depth wherever material has run downhill, by
        about 110 mm at the trapped foot. Only the particles say where the
        surface is, so this reads them, and falls back to the analytic height
        for a foot with no material beneath it at all.
        """
        cur = wp.to_torch(self.state_0.particle_q)
        n = self.particles_per_env
        out = torch.zeros(self.num_envs, 2, device=self.device)
        for env in range(self.num_envs):
            block = cur[env * n : (env + 1) * n]
            for slot in range(2):
                foot = self.body_q_t[self.foot_ids_t[env, slot]]
                near = (
                    (torch.abs(block[:, 0] - foot[0]) < FOOT_HALF_LENGTH_M)
                    & (torch.abs(block[:, 1] - foot[1]) < FOOT_HALF_WIDTH_M)
                )
                if bool(near.any()):
                    out[env, slot] = torch.quantile(block[near, 2], 0.95)
                else:
                    out[env, slot] = self._sample_patch(
                        foot[0:1] - self.offsets_t[env, 0],
                        foot[1:2] - self.offsets_t[env, 1],
                    ).squeeze() + SOIL_SURFACE_OFFSET_M
        return out

    def _overburden_fraction(self) -> torch.Tensor:
        """Fraction of the material under each foot that lies above it.

        Shape ``[num_envs, 2]``. This is what has to be pushed aside to lift
        the foot, so it is the measure that says whether a foot is buried
        rather than merely resting on soft ground. A foot standing on the
        surface reads near zero; the snapshot currently reads 0.06.
        """
        cur = wp.to_torch(self.state_0.particle_q)
        n = self.particles_per_env
        out = torch.zeros(self.num_envs, 2, device=self.device)
        for env in range(self.num_envs):
            block = cur[env * n : (env + 1) * n]
            for slot in range(2):
                foot = self.body_q_t[self.foot_ids_t[env, slot]]
                near = (
                    (torch.abs(block[:, 0] - foot[0]) < FOOT_HALF_LENGTH_M)
                    & (torch.abs(block[:, 1] - foot[1]) < FOOT_HALF_WIDTH_M)
                )
                total = int(near.sum())
                if total:
                    out[env, slot] = (block[near, 2] > foot[2]).sum() / total
        return out

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
            self._soil_top_under_feet()[:, self.trapped_index]
            - feet_now[:, self.trapped_index],
            min=0.0,
        )[mask_t]

    # ------------------------------------------------------------------
    def _exit_distances(self) -> torch.Tensor:
        torso_xy = self.body_q_t[self.torso_ids_t, :2]
        return torch.linalg.norm(torso_xy - self.pocket_xy, dim=1)

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
        wp.launch(
            MARS.clamp_foot_wrenches,
            dim=len(self.all_body_ids),
            inputs=[
                self.body_sand_forces,
                self.all_body_ids_wp,
                REACTION_FORCE_LIMIT_N,
                REACTION_TORQUE_LIMIT_NM,
                self.reaction_saturation_counts,
            ],
            device=self.model.device,
        )
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
        # Read against the material under this foot, not a window percentile.
        # On a site that falls 620 mm across the window the percentile is the
        # uphill edge, and subtracting a downhill foot from it reports the
        # slope as burial -- which is how a foot 12 mm into the sand came to
        # be described as 220 mm buried.
        foot_soil_top = self._soil_top_under_feet()
        overburden = self._overburden_fraction()
        burial = torch.clamp(
            foot_soil_top[:, self.trapped_index] - feet_z[:, self.trapped_index],
            min=0.0,
        )
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
        # Staying down is allowed in principle -- crawling out and standing
        # up counts -- but it puts limbs in the soil for the rest of a long
        # episode, a regime the earlier rounds never ran. --down-seconds
        # bounds how long that lasts; 0 disables the cut-off entirely.
        fallen = diverged
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
        self._was_trapped |= (burial >= TRAPPED_BURIAL_M) & (
            overburden[:, self.trapped_index] >= TRAPPED_OVERBURDEN
        )
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
        # from the regime that causes it.
        reward = (
            reward
            + 0.3 * out_now.float()
            + 10.0 * newly_escaped.float()
            - DIVERGENCE_PENALTY * diverged.float()
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
            # Burial read under the foot, and the share of the material
            # there that sits above it -- the part that has to be pushed
            # aside to lift the foot. A foot resting on the surface reads
            # near zero for both.
            "burial_m": burial,
            "overburden": overburden[:, self.trapped_index],
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
    parser.add_argument("--command-x", type=float, default=0.3)
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
    return parser.parse_args(extra if extra is not None else [])


def main() -> None:
    args = build_args(sys.argv[1:])
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
