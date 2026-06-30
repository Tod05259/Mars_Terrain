# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Import a completed Mars terrain USD in Isaac Lab and bind a contact (physics) material.

This visualizes the Gale Crater / Aeolis Palus-Mount Sharp terrain USD and binds a PhysX
rigid-body contact material to the terrain collider at runtime. ``TerrainImporterCfg`` does
not apply material properties for ``terrain_type="usd"``, so the material is created and bound
manually with :func:`~isaaclab.sim.utils.bind_physics_material`.

Run from the IsaacLab repository root::

    .\\isaaclab.bat -p .\\mars_terrain\\scripts\\visualize_gale_terrain_material.py

Visualize the terrain in the Isaac Sim GUI

    .\isaaclab.bat -p .\mars_terrain\scripts\visualize_gale_terrain_material.py --viz kit


Headless verification (finite steps, no window)::

    .\\isaaclab.bat -p .\\mars_terrain\\scripts\\visualize_gale_terrain_material.py --headless --num_steps 60
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# Default to the Gale Crater terrain USD. The terrain directory holds several Mars USDs, so a
# concrete file is used rather than the directory to keep the target unambiguous.
DEFAULT_USD = r"C:\IsaacLab-3.0.0-beta2\mars_terrain\usd\gale_aeolis_mount_sharp_aeolis_palus_center_1024px_stride2.usd"

# Mars terrain contact-material values (see GALE_TERRAIN_IMPORTER_MATERIAL_WORKFLOW.md).
STATIC_FRICTION = 0.54
DYNAMIC_FRICTION = 0.42
COMPLIANT_CONTACT_STIFFNESS = 2.2e4  # [N/m]
COMPLIANT_CONTACT_DAMPING = 160.0  # [N*s/m]


def resolve_usd_path(path_or_dir: str) -> str:
    """Resolve a direct USD file path, or a directory containing exactly one USD file."""
    path = Path(path_or_dir)
    if path.is_file():
        if path.suffix.lower() not in (".usd", ".usda", ".usdc"):
            raise ValueError(f"Target file is not a USD file: {path}")
        return str(path)
    if path.is_dir():
        usd_files = sorted(path.glob("*.usd")) + sorted(path.glob("*.usda")) + sorted(path.glob("*.usdc"))
        if not usd_files:
            raise FileNotFoundError(f"No USD files found in directory: {path}")
        if len(usd_files) > 1:
            listing = "\n".join(f"  - {c}" for c in usd_files)
            raise RuntimeError(
                f"Multiple USD files found in {path}. Pass the target file explicitly with"
                f" --usd_path.\n{listing}"
            )
        return str(usd_files[0])
    raise FileNotFoundError(f"USD path does not exist: {path}")


# Parse arguments before launching Isaac Sim.
parser = argparse.ArgumentParser(
    description="Import Gale Crater / Aeolis Palus-Mount Sharp USD terrain and bind a Mars contact material."
)
parser.add_argument(
    "--usd_path",
    type=str,
    default=DEFAULT_USD,
    help="Path to the Gale terrain USD file, or a directory containing exactly one USD file.",
)
parser.add_argument(
    "--terrain_prim_path",
    type=str,
    default="/World/terrain",
    help="Root prim path where TerrainImporter places the terrain.",
)
parser.add_argument(
    "--env_spacing",
    type=float,
    default=10.0,
    help="Environment spacing used by TerrainImporterCfg for usd terrain.",
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=60,
    help="Number of simulation steps to run when --headless is set. Ignored in GUI mode.",
)
parser.add_argument(
    "--export_scene",
    type=str,
    default="",
    help="Optional path to export the stage (terrain + material binding) after setup.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim / Omniverse.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab / Omniverse imports must come after AppLauncher.
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.spawners.materials import PhysxRigidBodyMaterialCfg  # noqa: E402
from isaaclab.sim.utils import bind_physics_material  # noqa: E402
from isaaclab.sim.utils.stage import get_current_stage  # noqa: E402
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402


def frame_camera(sim: sim_utils.SimulationContext, prim_path: str) -> None:
    """Point the camera at the terrain by computing its world-space bounding box."""
    stage = get_current_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bound = cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path))
    rng = bound.ComputeAlignedRange()
    bb_min, bb_max = rng.GetMin(), rng.GetMax()
    if rng.IsEmpty():
        sim.set_camera_view(eye=[30.0, -30.0, 20.0], target=[0.0, 0.0, 0.0])
        return
    center = Gf.Vec3d(0.5 * (bb_min + bb_max))
    ext = max(bb_max[0] - bb_min[0], bb_max[1] - bb_min[1])
    eye = [center[0] - 0.9 * ext, center[1] - 0.9 * ext, center[2] + 0.7 * ext]
    target = [center[0], center[1], center[2]]
    print(f"[INFO] Terrain bbox min={tuple(round(v, 2) for v in bb_min)} max={tuple(round(v, 2) for v in bb_max)}")
    print(f"[INFO] Camera eye={[round(v, 1) for v in eye]} target={[round(v, 1) for v in target]}")
    sim.set_camera_view(eye=eye, target=target)


def report_terrain(stage: Usd.Stage, root_path: str, material_path: str) -> None:
    """Print mesh prims, collision prims, and the bound physics material under the terrain root."""
    mesh_paths, collision_paths = [], []
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if not (p == root_path or p.startswith(root_path + "/")):
            continue
        if prim.IsA(UsdGeom.Mesh):
            mesh_paths.append(p)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_paths.append(p)

    print(f"[INFO] Mesh prims under {root_path}: {len(mesh_paths)}")
    for p in mesh_paths[:10]:
        print(f"  mesh: {p}")
    print(f"[INFO] Collision prims under {root_path}: {len(collision_paths)}")
    for p in collision_paths[:10]:
        bound_mat, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(p)).ComputeBoundMaterial("physics")
        bound_path = bound_mat.GetPath() if bound_mat else "<none>"
        ok = "OK" if str(bound_path) == material_path else "MISMATCH"
        print(f"  collision: {p} -> physics material: {bound_path} [{ok}]")

    if not collision_paths:
        print("[WARNING] No collision-enabled prims under the terrain root.")
        print("[WARNING] Terrain will visualize, but contact material cannot affect physics until a collider exists.")


def restore_masked_mesh_type(stage: Usd.Stage, prim_paths: list[str]) -> list[str]:
    """Restore the ``Mesh`` type on terrain prims whose geometry was flattened onto an Xform.

    These Mars terrain USDs author the geometry directly on the default prim (``/terrain`` is the
    Mesh). ``TerrainImporter`` references that file onto a prim created with ``prim_type="Xform"``,
    and the local Xform type masks the referenced Mesh type. The mesh attributes (points, indices)
    still compose onto the prim, but Hydra dispatches render prims by type, so an Xform carrying
    mesh data is not drawn. Setting the type back to ``Mesh`` makes it render. The original USD file
    is untouched; collider/material bindings are applied schemas and are unaffected.
    """
    fixed = []
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid() and not prim.IsA(UsdGeom.Mesh) and prim.HasAttribute("points"):
            prim.SetTypeName("Mesh")
            fixed.append(path)
    return fixed


def main():
    usd_path = resolve_usd_path(args_cli.usd_path)
    print(f"[INFO] Target USD: {usd_path}")

    # Simulation context.
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Lighting for viewport visibility.
    dome_light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0))
    dome_light_cfg.func("/World/DomeLight", dome_light_cfg)

    # Import terrain USD. Material is NOT auto-applied for terrain_type="usd"; it is bound below.
    terrain_cfg = TerrainImporterCfg(
        prim_path=args_cli.terrain_prim_path,
        terrain_type="usd",
        usd_path=usd_path,
        env_spacing=args_cli.env_spacing,
        debug_vis=True,
    )
    terrain = TerrainImporter(terrain_cfg)
    print(f"[INFO] Terrain imported. Terrain prim paths: {terrain.terrain_prim_paths}")

    # Restore Mesh type so the terrain renders (see restore_masked_mesh_type for the rationale).
    stage = get_current_stage()
    restored = restore_masked_mesh_type(stage, terrain.terrain_prim_paths)
    if restored:
        print(f"[INFO] Restored 'Mesh' type for renderability on: {restored}")

    # Create the PhysX rigid-body contact material for the Mars surface.
    material_path = "/World/gale_mars_surface_physics_material"
    material_cfg = PhysxRigidBodyMaterialCfg(
        static_friction=STATIC_FRICTION,
        dynamic_friction=DYNAMIC_FRICTION,
        restitution=0.0,
        friction_combine_mode="average",
        restitution_combine_mode="average",
        compliant_contact_stiffness=COMPLIANT_CONTACT_STIFFNESS,
        compliant_contact_damping=COMPLIANT_CONTACT_DAMPING,
    )
    material_cfg.func(material_path, material_cfg)
    print(f"[INFO] Physics material created: {material_path}")
    print(
        f"[INFO]   static_friction={STATIC_FRICTION} dynamic_friction={DYNAMIC_FRICTION}"
        f" k_e={COMPLIANT_CONTACT_STIFFNESS} k_d={COMPLIANT_CONTACT_DAMPING}"
    )

    # Bind the material to the terrain root; apply_nested propagates it to the collider descendant.
    bind_result = bind_physics_material(args_cli.terrain_prim_path, material_path, stronger_than_descendants=True)
    print(f"[INFO] bind_physics_material returned: {bind_result}")

    # Diagnostics: meshes, colliders, and the resolved physics-material binding.
    if not stage.GetPrimAtPath(args_cli.terrain_prim_path).IsValid():
        raise RuntimeError(f"Terrain root prim is invalid: {args_cli.terrain_prim_path}")
    report_terrain(stage, args_cli.terrain_prim_path, material_path)

    # Frame the camera on the terrain (it is not centered at the origin).
    frame_camera(sim, args_cli.terrain_prim_path)

    # Reset after scene construction.
    sim.reset()

    if args_cli.export_scene:
        stage.GetRootLayer().Export(args_cli.export_scene)
        print(f"[INFO] Exported material-bound scene: {args_cli.export_scene}")

    # Step the simulation: finite steps when headless, otherwise until the window is closed.
    if args_cli.headless:
        print(f"[INFO] Headless: running {args_cli.num_steps} steps then exiting.")
        for _ in range(args_cli.num_steps):
            sim.step()
    else:
        print("[INFO] Simulation running. Close the Isaac Sim window to exit.")
        while simulation_app.is_running():
            sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
