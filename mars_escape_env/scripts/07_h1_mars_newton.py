# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run pretrained H1, Mars terrain, and two-way MPM using Newton physics only."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import newton
import newton.examples
import newton.utils
import numpy as np
import torch
import torch.nn as nn
import warp as wp
from newton import JointTargetMode
from newton.solvers import SolverImplicitMPM
from pxr import Usd, UsdGeom

MARS_GRAVITY_MPS2 = -3.721
CONTROL_DT = 0.02
SIM_SUBSTEPS = 4

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / ".pretrained_checkpoints"
    / "rsl_rl"
    / "Isaac-Velocity-Rough-H1-v0"
    / "Assets"
    / "Isaac"
    / "6.0"
    / "Isaac"
    / "IsaacLab"
    / "PretrainedCheckpoints"
    / "rsl_rl"
    / "Isaac-Velocity-Rough-H1-v0"
    / "checkpoint.pt"
)

H1_DEFAULT_JOINT_POS = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_pitch": -0.28,
    "left_knee": 0.79,
    "left_ankle": -0.52,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_pitch": -0.28,
    "right_knee": 0.79,
    "right_ankle": -0.52,
    "torso_1": 0.0,
    "left_shoulder_pitch": 0.28,
    "left_shoulder_roll": 0.0,
    "left_shoulder_yaw": 0.0,
    "left_elbow": 0.52,
    "right_shoulder_pitch": 0.28,
    "right_shoulder_roll": 0.0,
    "right_shoulder_yaw": 0.0,
    "right_elbow": 0.52,
}

ISAACLAB_H1_JOINT_ORDER = [
    "left_hip_yaw",
    "right_hip_yaw",
    "torso",
    "left_hip_roll",
    "right_hip_roll",
    "left_shoulder_pitch",
    "right_shoulder_pitch",
    "left_hip_pitch",
    "right_hip_pitch",
    "left_shoulder_roll",
    "right_shoulder_roll",
    "left_knee",
    "right_knee",
    "left_shoulder_yaw",
    "right_shoulder_yaw",
    "left_ankle",
    "right_ankle",
    "left_elbow",
    "right_elbow",
]


@dataclass(frozen=True)
class SandMaterial:
    """MPM sand calibration for one Mars region."""

    key: str
    density: float
    friction: float
    young_modulus: float
    poisson_ratio: float
    yield_pressure: float
    dilatancy: float
    terrain_ke: float
    terrain_kd: float


@dataclass(frozen=True)
class SandLayerProfile:
    """Relative material properties for a loose surface layer."""

    key: str
    density_scale: float
    friction_scale: float
    young_modulus_scale: float
    yield_pressure_scale: float
    dilatancy: float


@dataclass(frozen=True)
class MaterialCalibrationScales:
    """Dimensionless calibration scales applied to one regional material."""

    friction: float
    young_modulus: float
    yield_pressure: float


LOOSE_SURFACE_PROFILE = SandLayerProfile(
    key="loose_surface",
    density_scale=0.85,
    friction_scale=0.90,
    young_modulus_scale=0.35,
    yield_pressure_scale=0.35,
    dilatancy=0.0,
)

# Refit against the same design point as before -- an H1 foot at 3.736 kPa
# penetrating 30 mm -- but on a converged MPM patch. The previous scales were
# fitted on a 0.8 m patch, which leaves the free soil face only 1.5 plate-lengths
# from the plate edge; soil squeezed out sideways instead of bearing the load, so
# the fit compensated with stiffer material. Sinkage at body weight converges by
# 1.6 m (13 / 7 / 5 / 4 mm at 0.8 / 1.2 / 1.6 / 2.0 m), and refitting there drops
# every strong region by 6-18x. Fit error at the design point is within 4 %.
#
# Friction is unchanged: only Young's modulus and yield pressure were refitted.
#
# gale is the exception at 1.06x, and the reason is physical rather than
# numerical: it is weak enough that the plate punches locally instead of
# mobilising a shear wedge wide enough to reach the patch boundary, so the
# boundary never affected it. The artifact scaled with material strength, which
# means the weakest -- and most hazardous -- soils were measured correctly all
# along and the strong ones were not.
#
# See EXTRACTION_HANDOFF.md sections 7.5d and 7.5e.
REGIONAL_MATERIAL_CALIBRATIONS = {
    "gusev_center": MaterialCalibrationScales(1.0, 4.5, 4.5),
    "gusev_highland": MaterialCalibrationScales(0.8, 2.55, 2.55),
    "mawrth_center": MaterialCalibrationScales(2.03, 16.0, 16.0),
    "mawrth_layered": MaterialCalibrationScales(1.56, 5.5, 5.5),
    "oxia_clay": MaterialCalibrationScales(1.75, 9.6, 9.6),
    "jezero": MaterialCalibrationScales(1.3, 25.5, 25.5),
    "gale": MaterialCalibrationScales(2.0, 18.0, 18.0),
}


def calibration_for_usd(env_usd: Path) -> MaterialCalibrationScales:
    """Return the validated calibration scales available for a Mars region."""
    material = material_for_usd(env_usd)
    return REGIONAL_MATERIAL_CALIBRATIONS[material.key]


def material_for_usd(env_usd: Path) -> SandMaterial:
    """Select the same regional material calibration used by the mixed backend."""
    name = env_usd.stem.lower()
    if "gusev_spirit_rough" in name or "gusev_highland" in name:
        return SandMaterial(
            "gusev_highland", 1800.0, 0.55, 5.0e5, 0.28, 3000.0, 0.10, 1.2e5, 600.0
        )
    if "gusev_spirit_center" in name or "gusev_center" in name:
        return SandMaterial(
            "gusev_center", 1700.0, 0.52, 2.0e5, 0.30, 2000.0, 0.08, 8.0e4, 400.0
        )
    if "mawrth_image_center" in name or "mawrth_center" in name:
        return SandMaterial(
            "mawrth_center", 1200.0, 0.32, 5.0e4, 0.35, 400.0, 0.0, 2.5e4, 180.0
        )
    if "mawrth_layered" in name or "mawrth_highland" in name:
        return SandMaterial(
            "mawrth_layered", 1500.0, 0.44, 1.5e5, 0.32, 1200.0, 0.04, 6.0e4, 320.0
        )
    if "oxia" in name:
        return SandMaterial(
            "oxia_clay", 1400.0, 0.40, 8.0e4, 0.33, 700.0, 0.02, 4.5e4, 260.0
        )
    if "jezero" in name:
        return SandMaterial(
            "jezero", 1400.0, 0.43, 4.0e4, 0.35, 300.0, 0.0, 2.0e4, 150.0
        )
    if "gale" in name:
        return SandMaterial(
            "gale", 1500.0, 0.42, 5.0e4, 0.34, 350.0, 0.01, 2.2e4, 160.0
        )
    raise ValueError(f"Could not infer the Mars sand material from: {env_usd.name}")


def load_terrain_grid(env_usd: Path) -> tuple[np.ndarray, float, float]:
    """Load the regular DTM height grid and XY sample spacing [m]."""
    stage = Usd.Stage.Open(str(env_usd))
    if stage is None:
        raise RuntimeError(f"Could not open Mars environment USD: {env_usd}")
    mesh = UsdGeom.Mesh.Get(stage, "/World/Terrain")
    if not mesh:
        raise RuntimeError(f"Missing /World/Terrain in: {env_usd}")
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
    x_values = np.unique(points[:, 0])
    y_values = np.unique(points[:, 1])
    rows, cols = len(y_values), len(x_values)
    if rows * cols != len(points):
        raise RuntimeError("The Mars terrain must be a regular DTM grid.")
    return (
        points[:, 2].reshape(rows, cols),
        float(np.mean(np.diff(x_values))),
        float(np.mean(np.diff(y_values))),
    )


def sample_height_numpy(
    height: np.ndarray, dx: float, dy: float, world_x: np.ndarray, world_y: np.ndarray
) -> np.ndarray:
    """Bilinearly sample DTM heights at world XY positions [m]."""
    rows, cols = height.shape
    col_f = np.clip(world_x / dx + (cols - 1) * 0.5, 0.0, cols - 1.0001)
    row_f = np.clip((rows - 1) * 0.5 - world_y / dy, 0.0, rows - 1.0001)
    col0 = np.floor(col_f).astype(np.int32)
    row0 = np.floor(row_f).astype(np.int32)
    col1 = np.minimum(col0 + 1, cols - 1)
    row1 = np.minimum(row0 + 1, rows - 1)
    tc = col_f - col0
    tr = row_f - row0
    return (
        height[row0, col0] * (1.0 - tr) * (1.0 - tc)
        + height[row0, col1] * (1.0 - tr) * tc
        + height[row1, col0] * tr * (1.0 - tc)
        + height[row1, col1] * tr * tc
    )


def quaternion_rotate_inverse(
    quaternion: torch.Tensor, vector: torch.Tensor
) -> torch.Tensor:
    """Rotate vectors by inverse XYZW quaternions."""
    scalar = quaternion[..., 3]
    xyz = quaternion[..., :3]
    first = vector * (2.0 * scalar.square() - 1.0).unsqueeze(-1)
    second = torch.cross(xyz, vector, dim=-1) * scalar.unsqueeze(-1) * 2.0
    third = xyz * torch.sum(xyz * vector, dim=-1, keepdim=True) * 2.0
    return first - second + third


class H1Actor(nn.Module):
    """Deterministic actor matching the published RSL-RL H1 checkpoint."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 19),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return deterministic joint actions for a 256-value observation."""
        return self.mlp(observation)


@wp.func
def sample_height_warp(
    height: wp.array(dtype=float),
    rows: int,
    cols: int,
    dx: float,
    dy: float,
    world_x: float,
    world_y: float,
):
    """Bilinearly sample a regular DTM from a Warp kernel [m]."""
    col_f = wp.clamp(world_x / dx + float(cols - 1) * 0.5, 0.0, float(cols) - 1.0001)
    row_f = wp.clamp(float(rows - 1) * 0.5 - world_y / dy, 0.0, float(rows) - 1.0001)
    col0 = int(wp.floor(col_f))
    row0 = int(wp.floor(row_f))
    col1 = wp.min(col0 + 1, cols - 1)
    row1 = wp.min(row0 + 1, rows - 1)
    tc = col_f - float(col0)
    tr = row_f - float(row0)
    return (
        height[row0 * cols + col0] * (1.0 - tr) * (1.0 - tc)
        + height[row0 * cols + col1] * (1.0 - tr) * tc
        + height[row1 * cols + col0] * tr * (1.0 - tc)
        + height[row1 * cols + col1] * tr * tc
    )


@wp.kernel
def compute_body_forces(
    dt: float,
    collider_ids: wp.array(dtype=int),
    collider_impulses: wp.array(dtype=wp.vec3),
    collider_impulse_positions: wp.array(dtype=wp.vec3),
    collider_body_indices: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_forces: wp.array(dtype=wp.spatial_vector),
):
    """Convert MPM collider impulses to rigid-body forces and torques."""
    index = wp.tid()
    collider_id = collider_ids[index]
    if collider_id < 0 or collider_id >= collider_body_indices.shape[0]:
        return
    body_id = collider_body_indices[collider_id]
    if body_id < 0:
        return
    force = collider_impulses[index] / dt
    center = wp.transform_point(body_q[body_id], body_com[body_id])
    torque = wp.cross(collider_impulse_positions[index] - center, force)
    wp.atomic_add(body_forces, body_id, wp.spatial_vector(force, torque))


@wp.kernel
def clamp_foot_wrenches(
    body_forces: wp.array(dtype=wp.spatial_vector),
    foot_body_ids: wp.array(dtype=int),
    force_limit: float,
    torque_limit: float,
    saturation_counts: wp.array(dtype=int),
):
    """Clamp explicit MPM reaction wrenches on each H1 ankle."""
    foot_index = wp.tid()
    body_id = foot_body_ids[foot_index]
    wrench = body_forces[body_id]
    force = wp.spatial_top(wrench)
    torque = wp.spatial_bottom(wrench)
    force_norm = wp.length(force)
    torque_norm = wp.length(torque)
    saturated = 0
    if force_norm > force_limit:
        force *= force_limit / force_norm
        saturated = 1
    if torque_norm > torque_limit:
        torque *= torque_limit / torque_norm
        saturated = 1
    body_forces[body_id] = wp.spatial_vector(force, torque)
    if saturated != 0:
        wp.atomic_add(saturation_counts, foot_index, 1)


@wp.kernel
def stabilize_particle_velocity(
    velocities: wp.array(dtype=wp.vec3), damping: float, max_speed: float
):
    """Damp particles and cap MPM projection velocity [m/s]."""
    index = wp.tid()
    velocity = velocities[index] * damping
    speed = wp.length(velocity)
    if speed > max_speed:
        velocity *= max_speed / speed
    velocities[index] = velocity


@wp.kernel
def recycle_particle_window(
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
):
    """Recycle particles around H1 and reset history-dependent MPM state."""
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

    surface_z = sample_height_warp(
        terrain_height, rows, cols, dx, dy, position[0], position[1]
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
    wp.atomic_add(recycle_count, 0, 1)


class H1MarsNewton:
    """Pure-Newton H1 walking preview with DTM-aware two-way MPM sand."""

    def __init__(self, viewer, args: argparse.Namespace):
        self.viewer = viewer
        self.args = args
        self.frame_dt = CONTROL_DT
        self.sim_dt = CONTROL_DT / SIM_SUBSTEPS
        self.sim_time = 0.0
        self.sim_step = 0
        self.env_usd = args.env_usd.expanduser().resolve()
        self.checkpoint = args.checkpoint.expanduser().resolve()
        if not self.env_usd.is_file():
            raise FileNotFoundError(f"Mars environment USD not found: {self.env_usd}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"H1 checkpoint not found: {self.checkpoint}")

        calibration = calibration_for_usd(self.env_usd)
        if args.mpm_friction_scale is None:
            args.mpm_friction_scale = calibration.friction
        if args.mpm_young_modulus_scale is None:
            args.mpm_young_modulus_scale = calibration.young_modulus
        if args.mpm_yield_pressure_scale is None:
            args.mpm_yield_pressure_scale = calibration.yield_pressure

        self.height, self.terrain_dx, self.terrain_dy = load_terrain_grid(self.env_usd)
        self.terrain_half_length = 0.5 * (self.height.shape[1] - 1) * self.terrain_dx
        self.terrain_half_width = 0.5 * (self.height.shape[0] - 1) * self.terrain_dy
        self.stop_reason = None
        if args.mpm_spacing <= 0.0:
            raise ValueError("--mpm-spacing must be positive.")
        if args.mpm_depth <= 0.0:
            raise ValueError("--mpm-depth must be positive.")
        if (
            min(args.mpm_iterations, args.mpm_substeps, args.mpm_particles_per_cell)
            <= 0
        ):
            raise ValueError(
                "MPM iterations, substeps, and particles per cell must be positive."
            )
        if not 0.0 <= args.particle_jitter_fraction < 0.5:
            raise ValueError("--particle-jitter-fraction must be in [0, 0.5).")
        if args.render_grains_per_particle < 0:
            raise ValueError("--render-grains-per-particle cannot be negative.")
        if args.stats_interval <= 0:
            raise ValueError("--stats-interval must be positive.")
        if (
            min(
                args.mpm_friction_scale,
                args.mpm_young_modulus_scale,
                args.mpm_yield_pressure_scale,
            )
            <= 0.0
        ):
            raise ValueError("MPM material calibration scales must be positive.")
        self.surface_z = float(
            sample_height_numpy(
                self.height,
                self.terrain_dx,
                self.terrain_dy,
                np.asarray([0.0]),
                np.asarray([0.0]),
            )[0]
        )
        if args.disable_mpm:
            self.mpm_surface_offset = 0.0
        else:
            # Newton derives each MPM point's represented volume as (2r)^3.
            particle_radius = 0.5 * args.mpm_spacing
            particle_layers = max(1, int(round(args.mpm_depth / args.mpm_spacing)))
            self.mpm_surface_offset = (
                2.0 * particle_radius + (particle_layers - 1) * args.mpm_spacing
            )
        self.material = material_for_usd(self.env_usd)
        self._build_model()
        self._build_policy()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0
        )
        self.contacts = self.model.contacts()
        self.body_sand_forces = wp.zeros_like(self.state_0.body_f)
        self.collider_impulses = None
        self.collider_impulse_positions = None
        self.collider_ids = None

        self.joint_q_torch = wp.to_torch(self.state_0.joint_q)
        self.joint_qd_torch = wp.to_torch(self.state_0.joint_qd)
        self.body_q_torch = wp.to_torch(self.state_0.body_q)
        self.control_torch = wp.to_torch(self.control.joint_target_pos)
        self.policy_to_newton_torch = torch.tensor(
            self.policy_to_newton, device=self.torch_device
        )
        self.default_joint_pos_newton = (
            self.joint_q_torch[7:].detach().clone().unsqueeze(0)
        )
        self.default_joint_pos_policy = self.default_joint_pos_newton[
            :, self.policy_to_newton_torch
        ]
        self.last_action = torch.zeros((1, 19), device=self.torch_device)
        self.command = torch.tensor(
            [[args.command_x, 0.0, 0.0]], device=self.torch_device
        )
        self.gravity_direction = torch.tensor(
            [[0.0, 0.0, -1.0]], device=self.torch_device
        )
        self.height_torch = torch.tensor(self.height.copy(), device=self.torch_device)
        scan_x = torch.linspace(-0.8, 0.8, 17, device=self.torch_device)
        scan_y = torch.linspace(-0.5, 0.5, 11, device=self.torch_device)
        # IsaacLab GridPatternCfg(ordering="xy") uses X as the inner loop.
        grid_x, grid_y = torch.meshgrid(scan_x, scan_y, indexing="xy")
        self.scan_x = grid_x.reshape(-1)
        self.scan_y = grid_y.reshape(-1)

        self.mpm_solver = None
        self.reference_q = None
        self.recycle_count = None
        self.reaction_saturation_counts = None
        self.render_grains = None
        self.render_grain_radii = None
        self.render_grain_colors = None
        self.render_grain_previous_state = None
        if not args.disable_mpm:
            self._initialize_mpm()

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "show_particles"):
            self.viewer.show_particles = (
                not args.disable_mpm and self.render_grains is None
            )

        print(f"[Pure Newton] terrain={self.env_usd}")
        print(f"[Pure Newton] checkpoint={self.checkpoint}")
        print(
            f"[Pure Newton] backend=Newton rigid+terrain+MPM gravity=(0, 0, {MARS_GRAVITY_MPS2}) m/s^2 "
            f"spawn_z={self.surface_z + self.mpm_surface_offset + 1.05:.4f} m "
            f"soil_top_offset={self.mpm_surface_offset:.3f} m command_x={args.command_x:.2f} m/s"
        )
        robot_mass = float(np.sum(self.model.body_mass.numpy()))
        print(
            f"[Pure Newton] H1 mass={robot_mass:.1f} kg Mars weight={robot_mass * abs(MARS_GRAVITY_MPS2):.1f} N "
            f"foot_support={'MPM only' if args.mpm_only_foot_support else 'rigid DTM + MPM'}"
        )
        if self.mpm_solver is None:
            print("[Pure Newton MPM] disabled")
        else:
            print(
                f"[Pure Newton MPM] particles={self.model.particle_count} material={self.material.key} "
                f"window={args.mpm_patch_length:.1f}x{args.mpm_patch_width:.1f}x{args.mpm_depth:.2f} m "
                f"layers={args.mpm_layer_profile} ppc={args.mpm_particles_per_cell:g} "
                f"scales=mu:{args.mpm_friction_scale:g}/E:{args.mpm_young_modulus_scale:g}/"
                f"yield:{args.mpm_yield_pressure_scale:g} substeps={args.mpm_substeps} "
                f"coupling={'one-way' if args.one_way else 'two-way explicit'}"
            )
            if self.render_grains is not None:
                print(
                    f"[Pure Newton MPM] render_grains={self.render_grains.size} "
                    f"radius={self.render_grain_radius * 1000.0:.2f} mm (visual only)"
                )

    def _build_model(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=MARS_GRAVITY_MPS2)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        SolverImplicitMPM.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.1,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
            friction=1.0e-5,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        h1_asset = (
            newton.utils.download_asset("unitree_h1") / "usd_structured" / "h1.usda"
        )
        builder.add_usd(
            str(h1_asset),
            # The H1 USD already stores its free-joint root at z=1.05 m.
            xform=wp.transform(
                wp.vec3(0.0, 0.0, self.surface_z + self.mpm_surface_offset),
                wp.quat_identity(),
            ),
            ignore_paths=["/GroundPlane"],
            enable_self_collisions=False,
        )
        newton_joint_names = [
            label.rsplit("/", 1)[-1] for label in builder.joint_label[1:]
        ]
        newton_joint_names = [
            "torso" if name == "torso_1" else name for name in newton_joint_names
        ]
        if set(newton_joint_names) != set(ISAACLAB_H1_JOINT_ORDER):
            raise RuntimeError(
                f"Newton H1 joints do not match the pretrained policy: {newton_joint_names}"
            )
        self.policy_to_newton = [
            newton_joint_names.index(name) for name in ISAACLAB_H1_JOINT_ORDER
        ]
        builder.approximate_meshes("bounding_box")
        self.foot_body_ids = [
            index
            for index, label in enumerate(builder.body_label)
            if label.endswith(("left_ankle_link", "right_ankle_link"))
        ]
        torso_body_ids = [
            index
            for index, label in enumerate(builder.body_label)
            if label.endswith("torso_link")
        ]
        if len(self.foot_body_ids) != 2:
            raise RuntimeError(
                f"Expected two H1 ankle bodies, got: {self.foot_body_ids}"
            )
        if len(torso_body_ids) != 1:
            raise RuntimeError(f"Expected one H1 torso body, got: {torso_body_ids}")
        self.torso_body_id = torso_body_ids[0]

        for joint_id, label in enumerate(builder.joint_label[1:], start=1):
            name = label.rsplit("/", 1)[-1]
            builder.joint_q[builder.joint_q_start[joint_id]] = H1_DEFAULT_JOINT_POS[
                name
            ]
            dof = builder.joint_qd_start[joint_id]
            builder.joint_target_pos[dof] = H1_DEFAULT_JOINT_POS[name]
            builder.joint_target_mode[dof] = int(JointTargetMode.POSITION)
            stiffness, damping = self._joint_gains(name)
            builder.joint_target_ke[dof] = stiffness
            builder.joint_target_kd[dof] = damping
            builder.joint_effort_limit[dof] = 100.0 if "ankle" in name else 300.0

        for shape_id, body_id in enumerate(builder.shape_body):
            if body_id not in self.foot_body_ids:
                builder.shape_flags[shape_id] &= ~newton.ShapeFlags.COLLIDE_PARTICLES

        builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
            ke=self.material.terrain_ke,
            kd=self.material.terrain_kd,
            kf=0.1 * self.material.terrain_ke,
            mu=self.material.friction,
            density=0.0,
            restitution=0.0,
            has_particle_collision=True,
            has_shape_collision=True,
            margin=0.005,
            gap=0.01,
        )
        terrain_shape_begin = len(builder.shape_flags)
        builder.add_usd(
            str(self.env_usd),
            ignore_paths=["/World/Rocks", "/World/Sand", "/World/Materials"],
            skip_mesh_approximation=True,
            schema_resolvers=[
                newton.usd.SchemaResolverNewton(),
                newton.usd.SchemaResolverPhysx(),
            ],
        )
        if self.args.mpm_only_foot_support:
            for shape_id in range(terrain_shape_begin, len(builder.shape_flags)):
                builder.shape_flags[shape_id] &= ~newton.ShapeFlags.COLLIDE_SHAPES

        self.rest_z_offsets_np = np.empty(0, dtype=np.float32)
        if not self.args.disable_mpm:
            self._add_particles(builder)

        self.model = builder.finalize(device=self.args.device)
        self.model.set_gravity((0.0, 0.0, MARS_GRAVITY_MPS2))
        self.torch_device = torch.device(wp.device_to_torch(self.model.device))
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            iterations=50,
            ls_iterations=20,
            njmax=2000,
            nconmax=self.args.nconmax,
            use_mujoco_contacts=False,
        )

    @staticmethod
    def _joint_gains(name: str) -> tuple[float, float]:
        if "ankle" in name:
            return 20.0, 4.0
        if "shoulder" in name or "elbow" in name:
            return 40.0, 10.0
        if "hip_yaw" in name or "hip_roll" in name:
            return 150.0, 5.0
        return 200.0, 5.0

    def _add_particles(self, builder: newton.ModelBuilder) -> None:
        spacing = self.args.mpm_spacing
        radius = 0.5 * spacing
        center_x = self.args.mpm_forward_offset
        xs = np.arange(
            -0.5 * self.args.mpm_patch_length + 0.5 * spacing,
            0.5 * self.args.mpm_patch_length,
            spacing,
        )
        ys = np.arange(
            -0.5 * self.args.mpm_patch_width + 0.5 * spacing,
            0.5 * self.args.mpm_patch_width,
            spacing,
        )
        layers = max(1, int(round(self.args.mpm_depth / spacing)))
        grid_x, grid_y = np.meshgrid(xs + center_x, ys, indexing="xy")
        rng = np.random.default_rng(7)
        positions = []
        rest_z_offsets = []
        layer_indices = []
        for layer in range(layers):
            jitter = self.args.particle_jitter_fraction * spacing
            x = grid_x.ravel() + rng.uniform(-jitter, jitter, grid_x.size)
            y = grid_y.ravel() + rng.uniform(-jitter, jitter, grid_y.size)
            z_offset = radius + layer * spacing
            z = (
                sample_height_numpy(self.height, self.terrain_dx, self.terrain_dy, x, y)
                + z_offset
            )
            positions.append(np.column_stack((x, y, z)))
            rest_z_offsets.append(np.full(x.shape, z_offset, dtype=np.float32))
            layer_indices.append(np.full(x.shape, layer, dtype=np.int32))
        particle_positions = np.concatenate(positions).astype(np.float32)
        self.rest_z_offsets_np = np.concatenate(rest_z_offsets)
        layer_indices_np = np.concatenate(layer_indices)
        self.surface_particle_mask_np = layer_indices_np == layers - 1
        count = len(particle_positions)

        density = np.full(count, self.material.density, dtype=np.float32)
        friction = np.full(count, self.material.friction, dtype=np.float32)
        young_modulus = np.full(count, self.material.young_modulus, dtype=np.float32)
        yield_pressure = np.full(count, self.material.yield_pressure, dtype=np.float32)
        dilatancy = np.full(count, self.material.dilatancy, dtype=np.float32)
        if (
            self.args.mpm_layer_profile == "loose_surface"
            and self.args.mpm_loose_layer_depth > 0.0
        ):
            loose_layer_count = min(
                layers, max(1, int(np.ceil(self.args.mpm_loose_layer_depth / spacing)))
            )
            loose_mask = layer_indices_np >= layers - loose_layer_count
            density[loose_mask] *= LOOSE_SURFACE_PROFILE.density_scale
            friction[loose_mask] *= LOOSE_SURFACE_PROFILE.friction_scale
            young_modulus[loose_mask] *= LOOSE_SURFACE_PROFILE.young_modulus_scale
            yield_pressure[loose_mask] *= LOOSE_SURFACE_PROFILE.yield_pressure_scale
            dilatancy[loose_mask] = LOOSE_SURFACE_PROFILE.dilatancy
            self.loose_particle_count = int(np.count_nonzero(loose_mask))
        else:
            self.loose_particle_count = 0

        if self.args.mpm_pocket_radius > 0.0:
            # Troy-type buried hazard under the robot spawn, placed at the
            # spawn point plus an offset so one foot lands on it and the other
            # does not.
            #
            # This used to omit the crust -- "already broken" -- leaving a
            # column of uniformly soft material. Measured, that cannot hold a
            # biped: over 24 s of walk-in the foot never stayed buried for
            # even one second, entering and floating back out, because soft
            # sand sinks a foot without gripping it and vertical extraction
            # from ordinary regolith is 8 % of body weight. A wheel that must
            # keep rolling is stopped by ground that cannot carry it; a leg
            # is simply lifted out.
            #
            # What held Spirit was structure, not softness: a thin cemented
            # crust over the pocket, which breaks under the foot and comes
            # down on top of it. That collapsed material is the extraction
            # resistance. It is modelled here as a stiff surface layer of
            # --mpm-crust-depth, the same construction the plate benchmark
            # sweeps in 09.
            pocket_x = self.args.mpm_pocket_dx
            pocket_y = self.args.mpm_pocket_dy
            radial = np.hypot(
                particle_positions[:, 0] - pocket_x,
                particle_positions[:, 1] - pocket_y,
            )
            in_column = radial < self.args.mpm_pocket_radius
            # Depth is counted in spawn layers, as the loose-surface profile
            # above does: the bed is draped over sloped ground, so a height
            # threshold would cut across it while a layer index follows it.
            crust_depth = float(self.args.mpm_crust_depth)
            crust_layers = (
                min(layers, max(1, int(np.ceil(crust_depth / spacing))))
                if crust_depth > 0.0
                else 0
            )
            crust = layer_indices_np >= layers - crust_layers
            if crust_layers > 0:
                young_modulus[crust] *= self.args.mpm_crust_scale
                yield_pressure[crust] *= self.args.mpm_crust_scale
            in_pocket = in_column & ~crust
            young_modulus[in_pocket] *= self.args.mpm_pocket_scale
            yield_pressure[in_pocket] *= self.args.mpm_pocket_scale
            print(
                f"[Pure Newton MPM] pocket: {int(in_pocket.sum())} particles "
                f"x{self.args.mpm_pocket_scale:g} at "
                f"({pocket_x:.2f}, {pocket_y:.2f}) r={self.args.mpm_pocket_radius:.2f} m"
                f"; crust {1000.0 * crust_depth:.0f} mm "
                f"x{self.args.mpm_crust_scale:g} ({int(crust.sum())} particles)"
            )

        friction *= self.args.mpm_friction_scale
        young_modulus *= self.args.mpm_young_modulus_scale
        yield_pressure *= self.args.mpm_yield_pressure_scale

        builder.add_particles(
            pos=particle_positions,
            vel=np.zeros_like(particle_positions),
            mass=density * spacing**3,
            radius=np.full(count, radius, dtype=np.float32),
            custom_attributes={
                "mpm:friction": friction,
                "mpm:young_modulus": young_modulus,
                "mpm:poisson_ratio": np.full(
                    count, self.material.poisson_ratio, dtype=np.float32
                ),
                "mpm:yield_pressure": yield_pressure,
                "mpm:dilatancy": dilatancy,
            },
        )
        builder.default_particle_radius = radius

    def _build_policy(self) -> None:
        checkpoint = torch.load(
            self.checkpoint, map_location=self.torch_device, weights_only=False
        )
        self.policy = H1Actor().to(self.torch_device).eval()
        incompatible = self.policy.load_state_dict(
            checkpoint["actor_state_dict"], strict=False
        )
        if incompatible.missing_keys or incompatible.unexpected_keys != [
            "distribution.std_param"
        ]:
            raise RuntimeError(
                f"Unexpected H1 checkpoint structure: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

    def _initialize_mpm(self) -> None:
        options = SolverImplicitMPM.Config()
        options.voxel_size = self.args.mpm_particles_per_cell * self.args.mpm_spacing
        options.grid_type = "sparse"
        options.transfer_scheme = "pic"
        options.strain_basis = "P0"
        options.max_iterations = self.args.mpm_iterations
        options.tolerance = 1.0e-4
        options.air_drag = 1.0
        options.collider_velocity_mode = "backward"
        self.mpm_solver = SolverImplicitMPM(self.model, options)
        self.mpm_solver.setup_collider(
            body_mass=wp.zeros_like(self.model.body_mass), body_q=self.state_0.body_q
        )
        self.collider_body_indices = self.mpm_solver.collider_body_index
        self.foot_body_ids_wp = wp.array(
            self.foot_body_ids, dtype=int, device=self.model.device
        )
        self.terrain_height_wp = wp.array(
            self.height.ravel(), dtype=float, device=self.model.device
        )
        self.rest_z_offsets = wp.array(
            self.rest_z_offsets_np, dtype=float, device=self.model.device
        )
        self.surface_particle_mask_torch = torch.tensor(
            self.surface_particle_mask_np, dtype=torch.bool, device=self.torch_device
        )

        for _ in range(self.args.mpm_settle_steps):
            self.mpm_solver.step(
                self.state_0, self.state_0, control=None, contacts=None, dt=0.005
            )
            self.mpm_solver.project_outside(self.state_0, self.state_0, 0.005)
        self.reference_q = wp.clone(self.state_0.particle_q)
        self.recycle_count = wp.zeros(1, dtype=int, device=self.model.device)
        self.reaction_saturation_counts = wp.zeros(
            2, dtype=int, device=self.model.device
        )
        if self.args.render_grains_per_particle > 0:
            # Newton draws render grains as spheres standing in for the volume of
            # one material point, so the radius has to follow the grain count:
            # voxel_size / (3 * N). Pinning it independently overstates the solid
            # fraction and makes the sand read as gravel.
            voxel_size = self.args.mpm_particles_per_cell * self.args.mpm_spacing
            self.render_grain_radius = (
                self.args.render_grain_radius
                if self.args.render_grain_radius is not None
                else voxel_size / (3.0 * self.args.render_grains_per_particle)
            )
            self.render_grain_previous_state = self.model.state()
            self.render_grains = self.mpm_solver.sample_render_grains(
                self.state_0, self.args.render_grains_per_particle
            )
            self.render_grain_radii = wp.full(
                self.render_grains.size,
                value=self.render_grain_radius,
                dtype=float,
                device=self.model.device,
            )
            self.render_grain_colors = wp.full(
                self.render_grains.size,
                value=wp.vec3(0.58, 0.20, 0.075),
                dtype=wp.vec3,
                device=self.model.device,
            )
        self._collect_mpm_impulses()

    def _sample_height_torch(
        self, world_x: torch.Tensor, world_y: torch.Tensor
    ) -> torch.Tensor:
        rows, cols = self.height.shape
        col_f = world_x / self.terrain_dx + (cols - 1) * 0.5
        row_f = (rows - 1) * 0.5 - world_y / self.terrain_dy
        col_f = torch.nan_to_num(col_f, nan=0.0, posinf=cols - 1.0001, neginf=0.0)
        row_f = torch.nan_to_num(row_f, nan=0.0, posinf=rows - 1.0001, neginf=0.0)
        col_f = torch.clamp(col_f, 0.0, cols - 1.0001)
        row_f = torch.clamp(row_f, 0.0, rows - 1.0001)
        col0 = torch.floor(col_f).long()
        row0 = torch.floor(row_f).long()
        col1 = torch.clamp(col0 + 1, max=cols - 1)
        row1 = torch.clamp(row0 + 1, max=rows - 1)
        tc = col_f - col0
        tr = row_f - row0
        terrain_height = (
            self.height_torch[row0, col0] * (1.0 - tr) * (1.0 - tc)
            + self.height_torch[row0, col1] * (1.0 - tr) * tc
            + self.height_torch[row1, col0] * tr * (1.0 - tc)
            + self.height_torch[row1, col1] * tr * tc
        )
        return terrain_height + self.mpm_surface_offset

    def _observation(self) -> torch.Tensor:
        root_q = self.joint_q_torch[3:7].unsqueeze(0)
        base_linear_velocity = quaternion_rotate_inverse(
            root_q, self.joint_qd_torch[:3].unsqueeze(0)
        )
        base_angular_velocity = quaternion_rotate_inverse(
            root_q, self.joint_qd_torch[3:6].unsqueeze(0)
        )
        projected_gravity = quaternion_rotate_inverse(root_q, self.gravity_direction)
        joint_position = (
            self.joint_q_torch[7:].unsqueeze(0)[:, self.policy_to_newton_torch]
            - self.default_joint_pos_policy
        )
        joint_velocity = self.joint_qd_torch[6:].unsqueeze(0)[
            :, self.policy_to_newton_torch
        ]

        scanner_q = self.body_q_torch[self.torso_body_id, 3:7]
        x, y, z, w = scanner_q
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        scan_world_x = (
            self.joint_q_torch[0] + cos_yaw * self.scan_x - sin_yaw * self.scan_y
        )
        scan_world_y = (
            self.joint_q_torch[1] + sin_yaw * self.scan_x + cos_yaw * self.scan_y
        )
        scan_height = torch.clamp(
            self.body_q_torch[self.torso_body_id, 2]
            - self._sample_height_torch(scan_world_x, scan_world_y)
            - 0.5,
            -1.0,
            1.0,
        ).unsqueeze(0)
        return torch.cat(
            (
                base_linear_velocity,
                base_angular_velocity,
                projected_gravity,
                self.command,
                joint_position,
                joint_velocity,
                self.last_action,
                scan_height,
            ),
            dim=1,
        )

    def _apply_policy(self) -> None:
        with torch.inference_mode():
            observation = self._observation()
            if observation.shape != (1, 256):
                raise RuntimeError(
                    f"Expected H1 observation shape (1, 256), got {tuple(observation.shape)}"
                )
            self.last_action = self.policy(observation)
            self.control_torch.zero_()
            targets_newton = self.default_joint_pos_newton[0].clone()
            targets_newton[self.policy_to_newton_torch] = (
                self.default_joint_pos_policy[0] + 0.5 * self.last_action[0]
            )
            self.control_torch[6:] = targets_newton

    def _accumulate_mpm_reaction(self) -> None:
        if self.mpm_solver is None or self.args.one_way or self.collider_ids is None:
            return
        wp.launch(
            compute_body_forces,
            dim=self.collider_ids.shape[0],
            inputs=[
                self.frame_dt,
                self.collider_ids,
                self.collider_impulses,
                self.collider_impulse_positions,
                self.collider_body_indices,
                self.state_0.body_q,
                self.model.body_com,
                self.body_sand_forces,
            ],
            device=self.model.device,
        )

    def _clamp_mpm_reaction(self) -> None:
        if self.mpm_solver is None or self.args.one_way:
            return
        wp.launch(
            clamp_foot_wrenches,
            dim=2,
            inputs=[
                self.body_sand_forces,
                self.foot_body_ids_wp,
                self.args.mpm_reaction_force_limit,
                self.args.mpm_reaction_torque_limit,
                self.reaction_saturation_counts,
            ],
            device=self.model.device,
        )

    def _simulate_robot(self) -> None:
        self.model.collide(self.state_0, self.contacts)
        for _ in range(SIM_SUBSTEPS):
            self.state_0.clear_forces()
            self.state_0.body_f.assign(self.body_sand_forces)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _window_center(self) -> tuple[float, float]:
        root_q = self.joint_q_torch[3:7]
        x, y, z, w = root_q
        forward_x = 1.0 - 2.0 * (y.square() + z.square())
        forward_y = 2.0 * (x * y + w * z)
        return (
            float(
                (
                    self.joint_q_torch[0] + self.args.mpm_forward_offset * forward_x
                ).item()
            ),
            float(
                (
                    self.joint_q_torch[1] + self.args.mpm_forward_offset * forward_y
                ).item()
            ),
        )

    def _simulate_mpm(self) -> None:
        if self.mpm_solver is None:
            return
        center_x, center_y = self._window_center()
        state_mpm = self.state_0.mpm
        wp.launch(
            recycle_particle_window,
            dim=self.model.particle_count,
            inputs=[
                self.state_0.particle_q,
                self.state_0.particle_qd,
                state_mpm.particle_qd_grad,
                state_mpm.particle_elastic_strain,
                state_mpm.particle_Jp,
                state_mpm.particle_stress,
                state_mpm.particle_transform,
                self.reference_q,
                self.rest_z_offsets,
                self.recycle_count,
                self.terrain_height_wp,
                self.height.shape[0],
                self.height.shape[1],
                self.terrain_dx,
                self.terrain_dy,
                center_x,
                center_y,
                self.args.mpm_patch_length,
                self.args.mpm_patch_width,
            ],
            device=self.model.device,
        )
        if self.render_grain_previous_state is not None:
            self.render_grain_previous_state.particle_q.assign(self.state_0.particle_q)
            self.render_grain_previous_state.mpm.particle_transform.assign(
                state_mpm.particle_transform
            )
        self.body_sand_forces.zero_()
        substep_dt = self.frame_dt / self.args.mpm_substeps
        for _ in range(self.args.mpm_substeps):
            self.mpm_solver.step(
                self.state_0,
                self.state_0,
                control=None,
                contacts=None,
                dt=substep_dt,
            )
            self.mpm_solver.project_outside(self.state_0, self.state_0, substep_dt)
            self._collect_mpm_impulses()
            self._accumulate_mpm_reaction()
        self._clamp_mpm_reaction()
        if self.render_grains is not None:
            self.mpm_solver.update_particle_frames(
                self.render_grain_previous_state, self.state_0, self.frame_dt
            )
            self.mpm_solver.update_render_grains(
                self.render_grain_previous_state,
                self.state_0,
                self.render_grains,
                self.frame_dt,
            )
            # The moving window teleports recycled particles ahead of the
            # robot, but advection cannot follow a teleport: the grains of a
            # recycled particle stay stranded in mid-air at the old location.
            # Periodic resampling clears the strays within a few frames.
            if (
                self.args.render_grain_resample > 0
                and self.sim_step % self.args.render_grain_resample == 0
            ):
                self.render_grains = self.mpm_solver.sample_render_grains(
                    self.state_0, self.args.render_grains_per_particle
                )
        wp.launch(
            stabilize_particle_velocity,
            dim=self.model.particle_count,
            inputs=[self.state_0.particle_qd, 0.98, self.args.mpm_max_particle_speed],
            device=self.model.device,
        )

    def _collect_mpm_impulses(self) -> None:
        self.collider_impulses, self.collider_impulse_positions, self.collider_ids = (
            self.mpm_solver.collect_collider_impulses(self.state_0)
        )

    def step(self) -> None:
        """Advance one pretrained-policy control step."""
        self._apply_policy()
        self._simulate_robot()
        self._simulate_mpm()
        self.sim_time += self.frame_dt
        self.sim_step += 1
        if self.sim_step % self.args.stats_interval == 0:
            self._print_stats()

    def _print_stats(self) -> None:
        root = self.joint_q_torch[:3].detach().cpu().tolist()
        message = f"[Pure Newton] step={self.sim_step} root_pos={root}"
        if self.mpm_solver is not None:
            current = wp.to_torch(self.state_0.particle_q)
            reference = wp.to_torch(self.reference_q)
            displacement = torch.linalg.vector_norm(current - reference, dim=1)
            body_forces = wp.to_torch(self.body_sand_forces)
            left = torch.linalg.vector_norm(
                body_forces[self.foot_body_ids[0], :3]
            ).item()
            right = torch.linalg.vector_norm(
                body_forces[self.foot_body_ids[1], :3]
            ).item()
            recycled = int(self.recycle_count.numpy()[0])
            sinkage = self._soil_sinkage_under_feet()
            saturation = self.reaction_saturation_counts.numpy()
            denominator = max(1, self.sim_step)
            message += (
                f" mpm_disp_mean={displacement.mean().item():.3f} m"
                f" reaction_N(L/R)={left:.1f}/{right:.1f} recycled={recycled}"
                f" soil_dep_p95_mm(L/R)={sinkage[0]:.1f}/{sinkage[1]:.1f}"
                f" reaction_sat_pct(L/R)={100.0 * saturation[0] / denominator:.1f}/"
                f"{100.0 * saturation[1] / denominator:.1f}"
            )
        print(message)

    def _soil_sinkage_under_feet(self) -> tuple[float, float]:
        """Measure positive surface-particle depression below each foot [mm]."""
        current = wp.to_torch(self.state_0.particle_q)
        reference = wp.to_torch(self.reference_q)
        depressions = torch.clamp(reference[:, 2] - current[:, 2], min=0.0)
        values = []
        for body_id in self.foot_body_ids:
            foot_q = self.body_q_torch[body_id]
            relative = current - foot_q[:3]
            local = quaternion_rotate_inverse(
                foot_q[3:7].unsqueeze(0).expand(len(current), -1), relative
            )
            under_foot = (
                self.surface_particle_mask_torch
                & (local[:, 0].abs() <= self.args.sinkage_foot_half_length)
                & (local[:, 1].abs() <= self.args.sinkage_foot_half_width)
            )
            if torch.any(under_foot):
                values.append(
                    1000.0 * torch.quantile(depressions[under_foot], 0.95).item()
                )
            else:
                values.append(0.0)
        return values[0], values[1]

    def render(self) -> None:
        """Render the current Newton state when a non-null viewer is selected."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.render_grains is not None:
            self.viewer.log_points(
                "/model/mpm_render_grains",
                points=self.render_grains.flatten(),
                radii=self.render_grain_radii,
                colors=self.render_grain_colors,
                hidden=False,
            )
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()
        self._stop_gl_viewer_if_needed()

    def _stop_gl_viewer_if_needed(self) -> None:
        """Stop GL before H1 leaves the finite DTM or exceeds the requested frame count."""
        if self.args.viewer != "gl" or self.stop_reason is not None:
            return

        root = self.joint_q_torch[:3].detach().cpu().numpy()
        if not np.isfinite(root).all():
            self.stop_reason = "H1 state became non-finite"
        elif (
            abs(root[0]) >= self.terrain_half_length - 1.0
            or abs(root[1]) >= self.terrain_half_width - 1.0
        ):
            self.stop_reason = "H1 reached the finite DTM boundary"
        elif self.args.num_frames > 0 and self.sim_step >= self.args.num_frames:
            self.stop_reason = f"requested {self.args.num_frames} frames completed"

        if self.stop_reason is not None:
            print(f"[Pure Newton] stopping GL viewer: {self.stop_reason}")
            self.viewer.renderer.close()

    def test_final(self) -> None:
        """Check the headless preview for finite states and terrain tunneling."""
        root = self.joint_q_torch[:3]
        self._print_stats()
        if (
            not torch.isfinite(self.joint_q_torch).all()
            or not torch.isfinite(self.joint_qd_torch).all()
        ):
            raise AssertionError("H1 state contains NaN or infinity.")
        if root[2].item() < self.surface_z + self.mpm_surface_offset - 0.2:
            raise AssertionError(
                f"H1 tunneled below terrain: root_z={root[2].item():.3f} m"
            )


def create_parser() -> argparse.ArgumentParser:
    """Create the pure-Newton preview command-line parser."""
    parser = newton.examples.create_parser()
    parser.description = "Pretrained H1 on Mars using only Newton rigid and MPM physics"
    parser.add_argument(
        "--env-usd", type=Path, required=True, help="Mars environment USD."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Published H1 RSL-RL checkpoint.",
    )
    parser.add_argument(
        "--command-x", type=float, default=0.5, help="Forward command [m/s]."
    )
    parser.add_argument(
        "--render-grain-resample",
        type=int,
        default=8,
        help=(
            "Resample render grains every N frames so grains stranded by "
            "particle recycling do not accumulate as floating sand; 0 disables."
        ),
    )
    parser.add_argument(
        "--mpm-crust-depth", type=float, default=0.0,
        help="Stiff surface layer over the pocket [m]. Zero leaves it broken.",
    )
    parser.add_argument(
        "--mpm-crust-scale", type=float, default=3.0,
        help="Stiffness multiplier applied within --mpm-crust-depth.",
    )
    parser.add_argument(
        "--mpm-pocket-radius", type=float, default=0.0,
        help="Troy-type soft pocket radius [m] near the spawn; 0 disables.",
    )
    parser.add_argument(
        "--mpm-pocket-scale", type=float, default=0.3,
        help="Strength multiplier inside the pocket.",
    )
    parser.add_argument(
        "--mpm-pocket-dx", type=float, default=0.25,
        help="Pocket centre offset from spawn, along +x (walking direction).",
    )
    parser.add_argument(
        "--mpm-pocket-dy", type=float, default=0.12,
        help="Pocket centre offset from spawn, along +y (left foot side).",
    )
    parser.add_argument(
        "--disable-mpm",
        action="store_true",
        help="Run Newton rigid terrain without MPM sand.",
    )
    parser.add_argument(
        "--one-way", action="store_true", help="Do not apply MPM impulses back to H1."
    )
    parser.add_argument(
        "--mpm-patch-length",
        type=float,
        default=1.0,
        help="Moving MPM window length [m].",
    )
    parser.add_argument(
        "--mpm-patch-width",
        type=float,
        default=1.0,
        help="Moving MPM window width [m].",
    )
    parser.add_argument(
        "--mpm-depth", type=float, default=0.08, help="MPM layer depth [m]."
    )
    parser.add_argument(
        "--mpm-spacing", type=float, default=0.025, help="MPM particle spacing [m]."
    )
    parser.add_argument(
        "--mpm-layer-profile",
        choices=("uniform", "loose_surface"),
        default="loose_surface",
        help="Use a uniform bed or a loose upper layer over the regional base material.",
    )
    parser.add_argument(
        "--mpm-loose-layer-depth",
        type=float,
        default=0.025,
        help="Loose surface-layer thickness [m].",
    )
    parser.add_argument(
        "--mpm-friction-scale",
        type=float,
        default=None,
        help="MPM friction scale; defaults to the validated regional calibration.",
    )
    parser.add_argument(
        "--mpm-young-modulus-scale",
        type=float,
        default=None,
        help="MPM Young's modulus scale; defaults to the validated regional calibration.",
    )
    parser.add_argument(
        "--mpm-yield-pressure-scale",
        type=float,
        default=None,
        help="MPM yield pressure scale; defaults to the validated regional calibration.",
    )
    parser.add_argument(
        "--mpm-only-foot-support",
        action="store_true",
        help="Disable rigid DTM contacts so H1 feet are supported only by MPM soil.",
    )
    parser.add_argument(
        "--mpm-forward-offset",
        type=float,
        default=0.0,
        help="Window offset ahead of H1 [m].",
    )
    parser.add_argument(
        "--mpm-iterations",
        type=int,
        default=200,
        help="Maximum implicit MPM iterations.",
    )
    parser.add_argument(
        "--mpm-particles-per-cell",
        type=float,
        default=3.0,
        help="Particle samples per MPM grid cell along each axis.",
    )
    parser.add_argument(
        "--particle-jitter-fraction",
        type=float,
        default=0.0,
        help="XY jitter as a fraction of material-point spacing.",
    )
    parser.add_argument(
        "--mpm-substeps",
        type=int,
        default=2,
        help="MPM integration substeps per 20 ms policy frame.",
    )
    parser.add_argument(
        "--mpm-settle-steps",
        type=int,
        default=30,
        help="Initial 5 ms MPM settling steps.",
    )
    parser.add_argument(
        "--mpm-max-particle-speed",
        type=float,
        default=1.5,
        help="Particle speed cap [m/s].",
    )
    parser.add_argument(
        "--mpm-reaction-force-limit",
        type=float,
        default=400.0,
        help="Per-foot force cap [N].",
    )
    parser.add_argument(
        "--mpm-reaction-torque-limit",
        type=float,
        default=80.0,
        help="Per-foot torque cap [N m].",
    )
    parser.add_argument(
        "--sinkage-foot-half-length",
        type=float,
        default=0.16,
        help="Footprint measurement half-length [m].",
    )
    parser.add_argument(
        "--sinkage-foot-half-width",
        type=float,
        default=0.08,
        help="Footprint measurement half-width [m].",
    )
    parser.add_argument(
        "--render-grains-per-particle",
        type=int,
        default=128,
        help=(
            "Visual grains sampled per MPM material point; zero disables them. "
            "Newton sizes them as voxel_size / (3 * N), so 8 gives 3.1 mm grains "
            "-- twenty times coarser than Mars sand -- while 128 gives 0.20 mm, "
            "the middle of the measured 50-500 um range, for 4 % more frame time. "
            "Set 0 for training: the grains are advected on the GPU but nothing "
            "reads them, so they are pure waste without a viewer."
        ),
    )
    parser.add_argument(
        "--render-grain-radius",
        type=float,
        default=None,
        help=(
            "Visual-only render grain radius [m]. Default derives it the way "
            "Newton's own grain-rendering example does, voxel_size / (3 * N), so "
            "the drawn grains conserve the volume they stand for. Setting it "
            "independently of --render-grains-per-particle misrepresents that "
            "volume: a fixed 2 mm with N=128 overstates it sixteenfold."
        ),
    )
    parser.add_argument(
        "--nconmax", type=int, default=2000, help="Newton/MJWarp contact capacity."
    )
    parser.add_argument(
        "--stats-interval",
        type=int,
        default=100,
        help="Runtime metric print interval [frames].",
    )
    return parser


if __name__ == "__main__":
    viewer, cli_args = newton.examples.init(create_parser())
    newton.examples.run(H1MarsNewton(viewer, cli_args), cli_args)
