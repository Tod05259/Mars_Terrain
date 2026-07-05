""" Citation

@inproceedings{mortensen2024rlroverlab,
  title={RLRoverLAB: An Advanced Reinforcement Learning Suite for Planetary Rover Simulation and Training},
  author={Mortensen, Anton Bj{\o}rndahl and B{\o}gh, Simon},
  booktitle={2024 International Conference on Space Robotics (iSpaRo)},
  pages={273--277},
  year={2024},
  organization={IEEE}
}

URL: https://github.com/abmoRobotics/MAPs

--------------------------------------------------------------------------------
NOTE ON STONE COLOR
--------------------------------------------------------------------------------
The original version reassigned every cloned stone to a height-based debug
material ("Generated_Green" / "Generated_Red"). Those materials only set
`diffuse_color`, which is ignored by EEVEE/Cycles when a material uses nodes, so
the clones rendered near-white instead of the recolored tan of the source stones.

This version removes that debug-material override. `copy_object_with_data()`
already duplicates the source object together with its material slots, so each
clone now inherits the exact material (and Base Color) of the stone it was copied
from. The per-object dimension/position data is still recorded in
`placed_objects` and saved to the stone-info file, unchanged.
"""

import bpy
import random
import re
import pickle
import math
import numpy as np
import mathutils
from mathutils import Vector


# Settings


TERRAIN_NAME = "terrainTest"
TARGET_TOTAL_OBJECTS = 1000

CHALLENGE_COUNT = 10
CHALLENGE_INDEX = 1
CHALLENGE_SCALE_MIN = 0.005
CHALLENGE_SCALE_MAX = 0.025
CHALLENGE_BASE_SCALE = 0.025

STONE_SCALE_MIN = 0.001
STONE_SCALE_MAX = 0.01

EXTRA_SMALL_STONE_COUNT = 7
HEIGHT_HALF_THRESHOLD = 0.25

VERTICES_SAVE_PATH = "./vertices.pkl"
STONE_INFO_SAVE_PATH = "./stone_info"


# Utility Function


def delete_generated_obj():

    for obj in list(bpy.data.objects):
        if re.match(r"^.*_generated.*$", obj.name):
            print("Deleted " + obj.name)
            bpy.data.objects.remove(obj, do_unlink=True)


def pickle_vertices_world(terrain, path=VERTICES_SAVE_PATH):

    v_x = []
    v_y = []

    for v in terrain.data.vertices:
        world_co = terrain.matrix_world @ v.co
        v_x.append(world_co.x)
        v_y.append(world_co.y)

    vs = [v_x, v_y]

    with open(path, "wb") as file:
        pickle.dump(vs, file)


def get_object_world_dimensions(obj):

    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]

    dim_x = max(xs) - min(xs)
    dim_y = max(ys) - min(ys)
    dim_z = max(zs) - min(zs)

    return Vector((dim_x, dim_y, dim_z))


def get_pos_info(obj):

    loc = obj.matrix_world.translation
    dim = get_object_world_dimensions(obj)

    return np.array(
        [loc.x, loc.y, loc.z, dim.x, dim.y, dim.z],
        dtype=float
    )


def is_tall(info):
    """Height classification kept for reference (e.g. deciding extra-stone spawns).

    Returns True when the object's half-height exceeds HEIGHT_HALF_THRESHOLD.
    No longer used to swap the material — clones keep their inherited color.
    """
    return (info[5] / 2.0) > HEIGHT_HALF_THRESHOLD


def terrain_vertex_world_location(terrain, vertex_index):

    return terrain.matrix_world @ terrain.data.vertices[vertex_index].co


def get_random_unused_terrain_vertex(terrain, used_vertex_indices, max_try=10000):

    vert_count = len(terrain.data.vertices)

    if len(used_vertex_indices) >= vert_count:
        return None, None

    for _ in range(max_try):
        idx = random.randrange(0, vert_count)

        if idx not in used_vertex_indices:
            used_vertex_indices.add(idx)
            world_loc = terrain_vertex_world_location(terrain, idx)
            return idx, world_loc

    return None, None


def closest_point_on_terrain_world(terrain, world_location):

    local_location = terrain.matrix_world.inverted() @ Vector(world_location)

    result, local_hit, local_normal, face_index = terrain.closest_point_on_mesh(local_location)

    if not result:
        return Vector(world_location)

    world_hit = terrain.matrix_world @ local_hit
    return world_hit


def get_challenge_child_offsets(challenge_parent):

    offsets = []
    parent_world_loc = challenge_parent.matrix_world.translation

    for child in challenge_parent.children:
        child_world_loc = child.matrix_world.translation
        offset = child_world_loc - parent_world_loc
        offsets.append((child, offset))

    return offsets


def copy_object_with_data(src_obj, new_name):

    new_obj = src_obj.copy()

    if src_obj.data is not None:
        new_obj.data = src_obj.data.copy()

    new_obj.name = new_name
    new_obj.animation_data_clear()

    # NOTE: new_obj keeps src_obj's material slots (and therefore its Base Color),
    # so the clone renders with the same recolored material as the source stone.
    return new_obj


# Initialization


delete_generated_obj()

terrain = None
stones = []
challenges = []
placed_objects = []
used_vertex_indices = set()

stone_name_pattern = re.compile(r"^(?!R)[a-zA-Z]*_[0-9]\.[0-9]{3}$")
challenge_name_pattern = re.compile(r"^Challenge_object[0-9]+$")

for obj in bpy.data.objects:
    if obj.name == TERRAIN_NAME:
        terrain = obj

    if challenge_name_pattern.match(obj.name):
        challenges.append(obj)

    if stone_name_pattern.match(obj.name):
        stones.append(obj)


if terrain is None:
    raise RuntimeError(f"'{TERRAIN_NAME}' Object not found.")

if len(challenges) == 0:
    raise RuntimeError("Challenge family object not found.")

if len(stones) == 0:
    raise RuntimeError("Stone object not found, check the stone name pattern.")


# No debug materials are created here anymore. Clones inherit the source stone's
# material, keeping the color identical to the original stones.

pickle_vertices_world(terrain)


# Set challenge


if len(challenges) > CHALLENGE_INDEX:
    challenge_template = challenges[CHALLENGE_INDEX]
else:
    challenge_template = challenges[0]

challenge_child_offsets = get_challenge_child_offsets(challenge_template)

for challenge_id in range(CHALLENGE_COUNT):
    if len(placed_objects) >= TARGET_TOTAL_OBJECTS:
        break

    vertex_index, location = get_random_unused_terrain_vertex(terrain, used_vertex_indices)

    if location is None:
        break

    rad = random.random() * 2.0 * math.pi

    scale_value = random.uniform(CHALLENGE_SCALE_MIN, CHALLENGE_SCALE_MAX)
    scale_ratio = scale_value / CHALLENGE_BASE_SCALE

    # parent 복사 (inherits the challenge template's material)
    new_challenge_parent = copy_object_with_data(
        challenge_template,
        f"{challenge_template.name}_{challenge_id}_parent_generated"
    )

    new_challenge_parent.location = location
    new_challenge_parent.scale = (scale_value, scale_value, scale_value)

    new_challenge_parent.rotation_euler = challenge_template.rotation_euler.copy()
    new_challenge_parent.rotation_euler.z += rad

    bpy.context.collection.objects.link(new_challenge_parent)

    parent_info = get_pos_info(new_challenge_parent)
    placed_objects.append(parent_info)
    # Material left as inherited from the source (no color override).

    # Copy child
    rotation_matrix = mathutils.Matrix.Rotation(rad, 4, "Z")

    for child_id, (child_template, offset) in enumerate(challenge_child_offsets):
        if len(placed_objects) >= TARGET_TOTAL_OBJECTS:
            break

        new_challenge_child = copy_object_with_data(
            child_template,
            f"{challenge_template.name}_{challenge_id}_child_{child_id}_generated"
        )

        rotated_offset = rotation_matrix @ (offset * scale_ratio)
        child_location = location + rotated_offset

        child_location = closest_point_on_terrain_world(terrain, child_location)

        new_challenge_child.location = child_location
        new_challenge_child.scale = (scale_value, scale_value, scale_value)

        new_challenge_child.rotation_euler = child_template.rotation_euler.copy()
        new_challenge_child.rotation_euler.z += rad

        bpy.context.collection.objects.link(new_challenge_child)

        child_info = get_pos_info(new_challenge_child)
        placed_objects.append(child_info)
        # Material left as inherited from the source (no color override).



# Place stone


stone_counter = 0

while len(placed_objects) < TARGET_TOTAL_OBJECTS:
    vertex_index, location = get_random_unused_terrain_vertex(terrain, used_vertex_indices)

    if location is None:
        print("There are no more terrain vertex available.")
        break

    stone_template = random.choice(stones)

    new_stone = copy_object_with_data(
        stone_template,
        f"{stone_template.name}_{stone_counter}_generated"
    )

    new_stone.location = location
    new_stone.rotation_euler = (0, 0, random.random() * 2.0 * math.pi)

    scale_value = random.uniform(STONE_SCALE_MIN, STONE_SCALE_MAX)
    new_stone.scale = (scale_value, scale_value, scale_value)

    bpy.context.collection.objects.link(new_stone)

    stone_info = get_pos_info(new_stone)
    placed_objects.append(stone_info)
    # Material left as inherited from the source stone (recolored tan).

    half_height = stone_info[5] / 2.0

    if half_height <= HEIGHT_HALF_THRESHOLD:
        for extra_id in range(EXTRA_SMALL_STONE_COUNT):
            if len(placed_objects) >= TARGET_TOTAL_OBJECTS:
                break

            extra_vertex_index, extra_location = get_random_unused_terrain_vertex(
                terrain,
                used_vertex_indices
            )

            if extra_location is None:
                break

            new_extra_stone = copy_object_with_data(
                new_stone,
                f"{stone_template.name}_{stone_counter}_extra_{extra_id}_generated"
            )

            new_extra_stone.location = extra_location
            new_extra_stone.rotation_euler = (0, 0, random.random() * 2.0 * math.pi)
            new_extra_stone.scale = new_stone.scale

            bpy.context.collection.objects.link(new_extra_stone)

            extra_info = get_pos_info(new_extra_stone)
            placed_objects.append(extra_info)
            # Material left as inherited from new_stone (recolored tan).

    stone_counter += 1



# Save


np_arr = np.array(placed_objects, dtype=float)

print(np_arr)
print(f"Saved object count: {len(np_arr)}")

np.save(STONE_INFO_SAVE_PATH, np_arr, allow_pickle=False)
