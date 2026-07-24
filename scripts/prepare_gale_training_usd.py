r"""Gale_stone_env_color.usd -> 학습용 USD 생성 (단일 메시 병합 + 충돌체 + 마찰 재질 + 재중심화).

Newton MJWarp는 USD로 참조된 501개 분할/인스턴스 메시 충돌체를 제대로 해석하지 못한다
(로봇이 지형을 뚫고 가라앉음 — 2026-07-14 진단: 공식 G1도 동일 증상, 절차 생성
단일-트라이메시 지형에서는 정상). 검증된 패턴에 맞춰 지형+돌 전체를 하나의 삼각
메시로 병합해 새 스테이지에 굽는다. 재중심화는 정점 좌표에 직접 베이크한다.

실행:  .\isaaclab.bat -p C:\Users\2hj05\repos\Mars_Terrain\scripts\prepare_gale_training_usd.py
"""

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

SRC = r"C:\Users\2hj05\repos\Mars_Terrain\usd(completed)\Gale_stone_env_color.usd"
DST = r"C:\Users\2hj05\repos\Mars_Terrain\usd(completed)\Gale_stone_env_color_train.usd"
GROUND_MESH = "/root/Terrain/Terrain_mesh"

# 화성 표면 접촉 물성 (기존 mars_terrain 워크플로 값)
STATIC_FRICTION = 0.54
DYNAMIC_FRICTION = 0.42

# 병합 메시 표시 색 (화성 표토 톤; 학습은 headless라 시각 품질은 중요하지 않음)
DISPLAY_COLOR = (0.45, 0.28, 0.18)


def world_points(prim: Usd.Prim) -> np.ndarray:
    """메시 정점을 월드 좌표로 변환해 반환한다 (미러/스케일 포함 전체 4x4 사용)."""
    xf = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    pts = np.asarray(prim.GetAttribute("points").Get(), dtype=np.float64)
    pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    return (pts_h @ xf)[:, :3]


def triangulate(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """폴리곤(faceVertexCounts/Indices)을 팬 방식으로 삼각형 인덱스 배열로 변환한다."""
    tris = []
    offset = 0
    for c in counts:
        for k in range(1, c - 1):
            tris.append((indices[offset], indices[offset + k], indices[offset + k + 1]))
        offset += c
    return np.asarray(tris, dtype=np.int64)


def fix_winding(pts: np.ndarray, tris: np.ndarray, is_ground: bool) -> np.ndarray:
    """삼각형 감김 방향을 바깥(outward)으로 교정한다.

    Blender에서 내보낸 원본은 지면 법선이 100% 아래를 향해 Newton MJWarp의 단면
    (one-sided) 트라이메시 충돌이 성립하지 않는다 (2026-07-14 진단). 지면(열린 시트)은
    법선 z의 다수결, 돌(닫힌 볼륨)은 부호 있는 부피로 판정해 메시 단위로 뒤집는다.
    """
    v0, v1, v2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    if is_ground:
        flip = float(np.cross(v1 - v0, v2 - v0)[:, 2].mean()) < 0.0
    else:
        signed_volume = float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)
        flip = signed_volume < 0.0
    return tris[:, ::-1] if flip else tris


def main():
    src = Usd.Stage.Open(SRC)

    # 1) 모든 메시를 월드 좌표 단일 삼각메시로 병합 (감김 방향 교정 포함)
    all_pts, all_tris = [], []
    vert_offset = 0
    n_meshes = 0
    n_flipped = 0
    for prim in src.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        pts = world_points(prim)
        counts = np.asarray(prim.GetAttribute("faceVertexCounts").Get(), dtype=np.int64)
        indices = np.asarray(prim.GetAttribute("faceVertexIndices").Get(), dtype=np.int64)
        tris = triangulate(counts, indices)
        fixed = fix_winding(pts, tris, is_ground=(str(prim.GetPath()) == GROUND_MESH))
        if fixed is not tris:
            n_flipped += 1
        all_pts.append(pts)
        all_tris.append(fixed + vert_offset)
        vert_offset += len(pts)
        n_meshes += 1
    pts = np.concatenate(all_pts)
    tris = np.concatenate(all_tris)
    print(f"[1/3] Merged {n_meshes} meshes ({n_flipped} winding-flipped) -> {len(pts)} verts, {len(tris)} triangles")

    # 2) 재중심화를 정점에 베이크: XY 중심 -> 원점, 중심부 지면 높이 -> z=0
    ground = src.GetPrimAtPath(GROUND_MESH)
    gpts = world_points(ground)
    cx = 0.5 * (pts[:, 0].min() + pts[:, 0].max())
    cy = 0.5 * (pts[:, 1].min() + pts[:, 1].max())
    near = gpts[(np.abs(gpts[:, 0] - cx) < 10.0) & (np.abs(gpts[:, 1] - cy) < 10.0)]
    z0 = float(np.median(near[:, 2])) if len(near) else float(np.median(gpts[:, 2]))
    pts -= np.array([cx, cy, z0])
    print(f"[2/3] Recenter baked into vertices: shift=({-cx:.2f}, {-cy:.2f}, {-z0:.2f})")

    # 3) 새 스테이지에 단일 Mesh + 충돌체 + 마찰 재질 작성
    dst = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(dst, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(dst, 1.0)
    world = UsdGeom.Xform.Define(dst, "/World")
    dst.SetDefaultPrim(world.GetPrim())

    mesh = UsdGeom.Mesh.Define(dst, "/World/Terrain")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tris), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tris.astype(np.int32).ravel()))
    ext = np.stack([pts.min(axis=0), pts.max(axis=0)]).astype(np.float32)
    mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(ext))
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*DISPLAY_COLOR)]))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_col.CreateApproximationAttr().Set("none")  # static triangle mesh

    mat = UsdShade.Material.Define(dst, "/World/MarsPhysicsMaterial")
    pmat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pmat.CreateStaticFrictionAttr().Set(STATIC_FRICTION)
    pmat.CreateDynamicFrictionAttr().Set(DYNAMIC_FRICTION)
    pmat.CreateRestitutionAttr().Set(0.0)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    dst.GetRootLayer().Export(DST)
    print(f"[3/3] Saved single-mesh terrain: {DST}")

    # 검증
    check = Usd.Stage.Open(DST)
    n_mesh = sum(1 for p in check.Traverse() if p.IsA(UsdGeom.Mesh))
    n_col = sum(1 for p in check.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    r = cache.ComputeWorldBound(check.GetDefaultPrim()).ComputeAlignedRange()
    print(
        f"[VERIFY] meshes={n_mesh}, collision prims={n_col},"
        f" bbox min={tuple(round(v, 2) for v in r.GetMin())}, max={tuple(round(v, 2) for v in r.GetMax())}"
    )


if __name__ == "__main__":
    main()
