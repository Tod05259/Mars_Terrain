# G1 휴머노이드 × Gale 화성 지형 보행 강화학습 가이드

> 목표: `g1.usd`(Unitree G1, 37관절)를 `Gale_stone_env_color.usd`(100 m × 100 m, 돌 430여 개) 위에서
> 속도 명령을 추종하며 걷도록 rsl_rl PPO로 학습시킨다.
>
> 작성일: 2026-07-14 · 대상: IsaacLab 3.0.0-beta2 (`C:\IsaacLab-3.0.0-beta2`)

---

## 0. 전략 요약

처음부터 환경을 만들지 않는다. IsaacLab에 이미 있는 **`Isaac-Velocity-Rough-G1-v0`**
(거친 지형 G1 보행, 보상/관측/커리큘럼 완성품)을 상속받아 딱 세 가지만 바꾼다:

1. **지형**: 절차적 생성 지형 → `terrain_type="usd"`로 Gale 지형 USD 로드
2. **로봇 USD**: Nucleus의 g1_minimal.usd → 내 `g1.usd`
3. **종료 조건·센서**: 내 USD의 특성(아래 진단 참고)에 맞게 교체

물리 백엔드는 이 PC의 PhysX GPU 관절 구동 버그 때문에 **반드시 `physics=newton_mjwarp`**
프리셋으로 학습한다 (rough 계열 env에는 이미 프리셋이 정의되어 있어 토큰만 붙이면 됨).

---

## 1. 사전 진단 결과 (2026-07-14 실측)

두 USD를 pxr로 열어 확인한 사실. 아래 파일 설계가 전부 여기서 나왔다.

### `Gale_stone_env_color.usd` (usd(completed) 폴더)

| 항목 | 값 | 영향 |
|---|---|---|
| defaultPrim | `/root` (Xform), 메시가 하위에 정상 중첩 | 예전 mars_terrain USD의 "Mesh 타입 가려짐" 문제 **없음** |
| 지면 메시 | `/root/Terrain/Terrain_mesh` (263,169 points) | 레이캐스트·충돌 대상 |
| 돌 메시 | 약 430개 (`*_generated/Mesh_*`, 각 ~100 points) | 총 메시 501개 |
| **CollisionAPI** | **0개 — 충돌체가 하나도 없음** | 그대로 쓰면 로봇이 지형을 통과해 추락 → **전처리 필수** |
| bbox | X [0.2, 100.4], Y [0.0, 100.6], Z [-0.9, 2.4] | 원점이 모서리 → **재중심화 필수** (로봇은 원점 주변 격자에 스폰됨) |
| 기타 | DomeLight 1개, 컬러 머티리얼 11개 내장 | 조명은 그대로 활용 가능 |

### `g1.usd` (usd(completed) 폴더)

| 항목 | 값 | 영향 |
|---|---|---|
| defaultPrim | `/g1`, ArticulationRoot는 `/g1/pelvis` | 정상 |
| 관절 | Revolute 37개 + 드라이브 37개, 관절명이 공식 Unitree G1과 동일 (`left_hip_pitch_joint`, `torso_joint`, `.*_ankle_roll_joint`, 손가락 `.*_one_joint`…) | IsaacLab `G1_CFG`의 액추에이터/보상 정규식과 **그대로 호환** |
| 충돌체 | **발바닥 큐브 2개뿐** (`left/right_ankle_roll_link/Cube`) | 서고 걷는 건 가능. 단 몸통 접촉 감지가 불가능 → **`base_contact` 종료 조건을 자세 기반으로 교체 필수** |
| PhysicsScene | `/physicsScene` — `/g1` **바깥**(형제 프림) | defaultPrim만 참조되므로 학습 시 안 딸려 옴. 수정 불필요 |

### IsaacLab 쪽 제약

- 기본 `RayCasterCfg`(height scanner)는 **prim 하나당 첫 번째 Mesh 1개만** 읽는다
  (`base_ray_caster.py:161-178`). 돌 430개짜리 지형에서는 엉뚱한 돌 하나만 스캔하게 됨
  → **`MultiMeshRayCasterCfg`로 교체** (기본값 `merge_prim_meshes=True`가 전체 메시를 병합해 줌).
- `terrain_type="usd"`에서는 `TerrainImporterCfg.physics_material`이 적용되지 않음
  → 전처리 단계에서 마찰 재질을 USD에 직접 바인딩.
- usd 지형에는 terrain curriculum(난이도 레벨)이 없음 → `curriculum.terrain_levels = None`.
- env origin은 `env_spacing` 간격의 원점 중심 격자로 계산됨
  → num_envs 512면 약 57 m × 57 m, **1024 초과 시 100 m 지형을 벗어나기 시작**하므로 512~1024 권장.

---

## 2. 만들 파일 (총 5개)

```
C:\Users\2hj05\repos\Mars_Terrain\scripts\
└── prepare_gale_training_usd.py          ← [파일 1] 지형 전처리 (1회 실행)

C:\IsaacLab-3.0.0-beta2\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\
└── g1_mars\                              ← 새 폴더 (만들면 자동으로 태스크 등록됨)
    ├── __init__.py                       ← [파일 2] gym.register
    ├── mars_env_cfg.py                   ← [파일 3] 환경 설정 (핵심)
    └── agents\
        ├── __init__.py                   ← [파일 4]
        └── rsl_rl_ppo_cfg.py             ← [파일 5] PPO 러너 설정
```

`config/` 아래 폴더는 `isaaclab_tasks`가 import 시 자동 스캔하므로 별도 등록 절차가 없다.

---

### [파일 1] `prepare_gale_training_usd.py` — 지형 전처리 (1회 실행)

원본은 건드리지 않고 학습용 사본 `Gale_stone_env_color_train.usd`를 만든다.

> **⚠ 최종 구현은 아래 예시에서 더 발전했다 — 저장소의 실제 스크립트를 사용할 것.**
> 실제 학습 과정에서 두 가지 치명적 사실이 추가로 밝혀졌다 (2026-07-14):
>
> 1. **지형 메시의 삼각형 법선이 100% 아래를 향해 있었다** (Blender 내보내기 산물).
>    PhysX는 정적 삼각메시를 양면 충돌로 처리해 시각화에서는 문제가 없었지만,
>    **Newton MJWarp는 단면(one-sided) 충돌**이라 로봇이 지형을 뚫고 가라앉는다
>    (공식 G1 에셋으로도 재현, 절차 생성 지형에서는 정상 → 지형 메시 데이터 문제로 확정).
>    → 스크립트가 메시별로 감김 방향(winding)을 판정해 교정한다 (지면: 법선 z 다수결,
>    돌: 부호 있는 부피).
> 2. **분할/인스턴스 501개 메시 구조 대신 단일 병합 메시로 굽는다** — 검증된
>    generator 지형 패턴과 동일한 구조 (단일 트라이메시 + CollisionAPI), 레이캐스터·
>    충돌 파이프라인 모두에 가장 안전하다. 재중심화는 xformOp 대신 정점에 직접 베이크.
>
> 최종 스크립트가 하는 일: ① 501개 메시(지형+돌)를 월드 좌표 단일 삼각메시로 병합
> (감김 방향 교정 포함) ② 충돌체(triangle mesh) 부여 ③ 화성 마찰 재질 생성·바인딩
> ④ 재중심화를 정점에 베이크.

아래는 초기 설계 버전이다 (참고용 — 병합/법선 교정 이전):

```python
r"""Gale_stone_env_color.usd -> 학습용 USD 생성 (충돌체 + 마찰 재질 + 재중심화).

실행:  .\isaaclab.bat -p C:\Users\2hj05\repos\Mars_Terrain\scripts\prepare_gale_training_usd.py
"""

import numpy as np

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

SRC = r"C:\Users\2hj05\repos\Mars_Terrain\usd(completed)\Gale_stone_env_color.usd"
DST = r"C:\Users\2hj05\repos\Mars_Terrain\usd(completed)\Gale_stone_env_color_train.usd"
GROUND_MESH = "/root/Terrain/Terrain_mesh"

# 화성 표면 접촉 물성 (기존 mars_terrain 워크플로 값)
STATIC_FRICTION = 0.54
DYNAMIC_FRICTION = 0.42


def main():
    stage = Usd.Stage.Open(SRC)
    root = stage.GetDefaultPrim()

    # 1) 모든 Mesh에 정적 삼각메시 충돌체 적용
    n_col = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_col.CreateApproximationAttr().Set(UsdPhysics.Tokens.none)  # triangle mesh
            n_col += 1
    print(f"[1/3] CollisionAPI applied to {n_col} meshes")

    # 2) 마찰 재질 생성 후 충돌 메시 전체에 physics 바인딩
    mat = UsdShade.Material.Define(stage, root.GetPath().AppendChild("MarsPhysicsMaterial"))
    pmat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pmat.CreateStaticFrictionAttr().Set(STATIC_FRICTION)
    pmat.CreateDynamicFrictionAttr().Set(DYNAMIC_FRICTION)
    pmat.CreateRestitutionAttr().Set(0.0)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                mat, UsdShade.Tokens.weakerThanDescendants, "physics"
            )
    print(f"[2/3] Mars physics material bound (mu_s={STATIC_FRICTION}, mu_d={DYNAMIC_FRICTION})")

    # 3) 재중심화: XY 중심 -> 원점, 중심부 지면 높이 -> z=0
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rng = cache.ComputeWorldBound(root).ComputeAlignedRange()
    bb_min, bb_max = rng.GetMin(), rng.GetMax()
    cx = 0.5 * (bb_min[0] + bb_max[0])
    cy = 0.5 * (bb_min[1] + bb_max[1])

    ground = stage.GetPrimAtPath(GROUND_MESH)
    xf_mat = UsdGeom.Xformable(ground).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    pts = np.asarray(ground.GetAttribute("points").Get())
    rot = np.asarray(xf_mat.ExtractRotationMatrix()).T
    pts_w = pts @ rot.T + np.asarray(xf_mat.ExtractTranslation())
    near = pts_w[(np.abs(pts_w[:, 0] - cx) < 10.0) & (np.abs(pts_w[:, 1] - cy) < 10.0)]
    z0 = float(np.median(near[:, 2])) if len(near) else float(np.median(pts_w[:, 2]))

    xf = UsdGeom.Xformable(root)
    op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "recenter")
    op.Set(Gf.Vec3d(-cx, -cy, -z0))
    ops = xf.GetOrderedXformOps()
    xf.SetXformOpOrder([op] + [o for o in ops if o.GetOpName() != op.GetOpName()])  # 월드 기준 이동
    print(f"[3/3] Recentered: shift=({-cx:.2f}, {-cy:.2f}, {-z0:.2f})")

    stage.GetRootLayer().Export(DST)
    print(f"[DONE] Saved: {DST}")


if __name__ == "__main__":
    main()
```

---

### [파일 2] `g1_mars/__init__.py` — 태스크 등록

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Velocity-Mars-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mars_env_cfg:G1MarsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MarsPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Mars-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mars_env_cfg:G1MarsEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MarsPPORunnerCfg",
    },
)
```

---

### [파일 3] `g1_mars/mars_env_cfg.py` — 환경 설정 (핵심)

> **구현 노트 (2026-07-14):** 전처리 후 지형을 실측해 보니 512-env 스폰 구역(±28 m)의
> 고저차가 z −0.93 ~ +1.20 m로 커서, 고정 스폰 높이로는 일부 로봇이 파묻히거나 낙하한다.
> 그래서 `TerrainImporterCfg.class_type` 확장 포인트로 **env origin의 z를 지면 높이에
> 스냅하는 `MarsTerrainImporter`**를 추가했다 (아래 코드에 포함, 실제 적용된 버전).

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 velocity-tracking locomotion on the Gale crater Mars terrain (USD)."""

import numpy as np
import torch

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import MultiMeshRayCasterCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg

# 전처리된 학습용 지형(prepare_gale_training_usd.py 산출물)과 로봇 USD
MARS_TERRAIN_USD = r"C:\Users\2hj05\repos\Mars_Terrain\usd(completed)\Gale_stone_env_color_train.usd"
G1_USD = r"C:\Users\2hj05\repos\Mars_Terrain\usd(completed)\g1.usd"


class MarsTerrainImporter(TerrainImporter):
    """USD 지형용 TerrainImporter: env origin의 z를 각 지점의 지면 높이에 맞춘다."""

    def __init__(self, cfg: TerrainImporterCfg):
        super().__init__(cfg)
        self._snap_env_origins_to_ground()

    def _snap_env_origins_to_ground(self):
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import Usd, UsdGeom

        stage = get_current_stage()
        # 지면 메시 탐색: 지형 루트 아래에서 정점이 가장 많은 메시(= 지면, 나머지는 돌)
        ground_prim, max_pts = None, 0
        for root_path in self.terrain_prim_paths:
            root = stage.GetPrimAtPath(root_path)
            for prim in Usd.PrimRange(root):
                if prim.IsA(UsdGeom.Mesh):
                    pts_attr = prim.GetAttribute("points").Get()
                    if pts_attr is not None and len(pts_attr) > max_pts:
                        ground_prim, max_pts = prim, len(pts_attr)
        if ground_prim is None:
            return

        xf = UsdGeom.Xformable(ground_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pts = np.asarray(ground_prim.GetAttribute("points").Get(), dtype=np.float64)
        rot = np.asarray(xf.ExtractRotationMatrix())
        pts_w = pts @ rot + np.asarray(xf.ExtractTranslation())

        # 각 env origin의 xy 최근접 지면 정점 z를 취한다 (chunk 단위 brute-force).
        origins = self.env_origins.cpu().numpy()
        ground_z = np.empty(len(origins))
        pts_xy = pts_w[:, :2]
        for start in range(0, len(origins), 64):
            chunk = origins[start : start + 64, :2]
            d2 = ((pts_xy[None, :, :] - chunk[:, None, :]) ** 2).sum(axis=-1)
            ground_z[start : start + 64] = pts_w[d2.argmin(axis=1), 2]

        self.env_origins[:, 2] = torch.from_numpy(ground_z).to(self.env_origins)
        print(
            "[INFO] MarsTerrainImporter: snapped env origins to ground"
            f" (z range [{ground_z.min():.2f}, {ground_z.max():.2f}], mesh={ground_prim.GetPath()})"
        )


@configclass
class G1MarsEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # -- 지형: 절차 생성 지형 -> Gale USD (100m x 100m, 재중심화됨)
        self.scene.terrain = TerrainImporterCfg(
            class_type=MarsTerrainImporter,
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path=MARS_TERRAIN_USD,
            collision_group=-1,
            env_spacing=2.5,
        )
        # usd 지형에는 난이도 레벨이 없음 -> terrain curriculum 비활성화
        self.curriculum.terrain_levels = None

        # -- 로봇: 내 g1.usd (관절명이 공식 G1과 동일해 액추에이터/보상 설정 재사용 가능)
        self.scene.robot.spawn.usd_path = G1_USD
        # env origin이 지면 높이로 보정되므로 서있는 높이(0.74) + 여유고만 준다
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.85)

        # -- height scanner: 기본 RayCaster는 메시 1개만 읽음 -> 멀티 메시(지형+돌 병합)로 교체
        self.scene.height_scanner = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            mesh_prim_paths=["/World/ground"],
            debug_vis=False,
        )

        # -- 종료 조건: g1.usd는 발에만 충돌체가 있어 몸통 접촉 감지가 불가능
        #    -> 몸통 기울기(중력 방향과의 각도) 기반 전도 감지로 교체
        self.terminations.base_contact = None
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation, params={"limit_angle": 1.0}
        )

        # -- 스폰 격자(env_spacing 2.5m)가 지형(100m)을 벗어나지 않도록 기본 env 수 제한
        self.scene.num_envs = 512

        # -- (옵션) 화성 중력. 우선 지구 중력으로 보행을 확보한 뒤 켜는 것을 권장
        # self.sim.gravity = (0.0, 0.0, -3.71)


@configclass
class G1MarsEnvCfg_PLAY(G1MarsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
```

---

### [파일 4] `g1_mars/agents/__init__.py`

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
```

### [파일 5] `g1_mars/agents/rsl_rl_ppo_cfg.py`

기존 G1 rough PPO 설정을 그대로 물려받고 실험 이름과 반복 수만 바꾼다.

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1RoughPPORunnerCfg,
)


@configclass
class G1MarsPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_mars"
        self.max_iterations = 5000
```

---

## 3. 실행 순서

모든 명령은 `C:\IsaacLab-3.0.0-beta2` 루트에서 실행한다.

```powershell
# STEP 1. 지형 전처리 (1회) — Gale_stone_env_color_train.usd 생성
.\isaaclab.bat -p C:\Users\2hj05\repos\Mars_Terrain\scripts\prepare_gale_training_usd.py

# STEP 2. 스모크 테스트 — 적은 env로 짧게 돌려 등록/스폰/충돌 확인 (GUI로 보려면 --viz kit 추가)
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py `
  --task Isaac-Velocity-Mars-G1-v0 --num_envs 16 --max_iterations 10 `
  physics=newton_mjwarp --headless

# STEP 3. 본 학습 (512 envs, TensorBoard 로그: logs\rsl_rl\g1_mars\)
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py `
  --task Isaac-Velocity-Mars-G1-v0 --num_envs 512 `
  physics=newton_mjwarp --headless

# 학습 곡선 확인
.\isaaclab.bat -p -m tensorboard.main --logdir logs\rsl_rl\g1_mars

# STEP 4. 학습 결과 재생 (최신 체크포인트 자동 로드)
# 주의: 이 설치본에서 --viz kit(Isaac Sim GUI)은 Kit 확장 DLL 로드 실패로
# "RuntimeError: Caught an unknown exception!"과 함께 죽는다 (2026-07-14 확인).
# Newton 백엔드에는 Kit이 필요 없는 자체 OpenGL 뷰어가 있으니 --viz newton을 쓸 것.
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py `
  --task Isaac-Velocity-Mars-G1-Play-v0 --num_envs 32 `
  physics=newton_mjwarp --viz newton
```

스모크 테스트에서 확인할 것:
- `Isaac-Velocity-Mars-G1-v0`가 정상 등록되어 학습이 시작되는가
- 로봇이 지형을 **뚫고 떨어지지 않는가** (뚫으면 STEP 1 전처리가 안 된 USD를 읽고 있는 것)
- 콘솔에 height scanner 메시 로드 로그가 찍히고 에러가 없는가

---

## 4. 학습이 시작된 뒤 튜닝 포인트

| 증상 | 조치 |
|---|---|
| 제자리 정지/주저앉기만 함 | `rewards.track_lin_vel_xy_exp.weight` ↑ (1.0→2.0), 명령 범위 `lin_vel_x=(0.3, 1.0)`으로 하한 부여 |
| 돌에 걸려 자주 넘어짐 | `commands.base_velocity.ranges.lin_vel_x` 상한 ↓ (0.6), `feet_air_time.weight` ↑ |
| 발을 끄는 걸음 | `feet_slide.weight` 페널티 강화 (-0.1 → -0.25) |
| 뒤뚱거림/진동 | `action_rate_l2`, `dof_acc_l2` 페널티 강화 |
| 화성 중력 적응 | 지구 중력으로 안정 보행 확보 → `sim.gravity=(0,0,-3.71)` 켜고 이어서 fine-tune (`--resume` + `--load_run`) |

보상 가중치는 파일 수정 없이 Hydra 오버라이드로도 실험 가능:
`... env.rewards.feet_slide.weight=-0.25 env.commands.base_velocity.ranges.lin_vel_x="(0.3, 0.8)"`

---

## 5. 알려진 제약 & 트러블슈팅

- **`physics=newton_mjwarp`를 빼먹으면**: 이 PC의 PhysX GPU는 관절 토크를 조용히 무시한다.
  에피소드 길이가 자유낙하 시간에 고정되고 action std만 커지는 "가짜 정체"가 나타남.
- **`Unknown preset(s): newton_mjwarp`**: rough 계열 env는 프리셋이 이미 있어 발생하지 않지만,
  다른 env를 만들 때는 env cfg에 `PresetCfg`(newton_mjwarp 필드 포함)를 정의해야 한다.
- **로봇이 공중에서 시작하거나 언덕에 파묻힘**: `MarsTerrainImporter`가 env origin z를 지면
  높이에 스냅해 해결한다. 그래도 특정 지점(돌 위 스폰 등)이 이상하면 `init_state.pos`의 z
  여유고(기본 0.85)를 조금 올린다.
- **num_envs는 512~1024 유지**: env origin 격자(spacing 2.5 m)가 원점 중심 정사각형으로 퍼지므로
  2048 이상이면 지형 경계를 벗어난 로봇이 허공에 스폰된다.
- **g1.usd는 발 외 충돌체가 없음**: 넘어지면 몸통이 지형을 파고드는 모습이 보이는 게 정상이며
  `bad_orientation` 종료가 그 전에 에피소드를 끊는다. 몸통 충돌까지 원하면 Isaac Sim에서
  pelvis/torso 링크에 충돌 프리미티브를 추가한 사본을 만들 것.
- **로봇이 지형을 뚫고 가라앉음 / feet_air_time이 학습 내내 0 / 학습 정체 후 NaN 크래시**
  (2026-07-14에 전부 실제 발생 — 같은 뿌리였다): **지형 메시 법선이 아래를 향해 Newton의
  단면 트라이메시 충돌이 성립하지 않는 것이 근본 원인.** 진단 방법: 액션 0으로 로봇을
  세워두는 스탠딩 테스트 — 정상이면 root z가 서있는 높이로 유지되고 발 접촉력이 체중
  (~340 N) 수준으로 지속 측정된다. 가라앉으면 지형 충돌 문제. `prepare_gale_training_usd.py`의
  감김 교정으로 해결됨. PhysX 시각화에서 멀쩡해 보여도 Newton에서는 뚫릴 수 있음에 주의.
- **학습 중반 이후 `observation contains NaN` 크래시 + action std 발산** (2026-07-14 1차 학습에서
  iteration 4528/5000에 실제 발생): 근본 원인은 위의 지형 법선 문제였지만(로봇이 물리적으로
  걸을 수 없는 환경 → 정책 퇴화 → 물리 발산), 재발 방지용 안전망 세 가지도 함께 적용했다:
  ① agents cfg에 `clip_actions = 6.0` (액션 안전망),
  ② `bad_orientation`을 1.0 → 0.8 rad로 조여 기어다니는 퇴화 정책 조기 차단 +
  `root_height_below_minimum`(-3.0 m) 종료 추가 (추락/관통 시 상태 폭주 차단),
  ③ 커리큘럼이 없는 돌밭 과제의 난이도 완화 — `lin_vel_x=(0.0, 0.8)`, `ang_vel_z=(±0.5)`,
  `track_lin_vel_xy_exp.weight=1.5`. 크래시가 나도 `save_interval=50`마다 체크포인트가 남으니
  `logs\rsl_rl\g1_mars\<run>\model_*.pt`에서 복구 가능하다.
- **`nefc overflow - please increase njmax to N` 경고**: MJWarp 솔버의 제약 버퍼가 넘친 것.
  rough 프리셋 기본값(njmax=200, nconmax=100)이 돌밭 접촉이 몰릴 때 부족하다. 넘치면 접촉
  제약이 조용히 누락되어 물리가 왜곡되므로 경고를 무시하지 말 것. env cfg `__post_init__`에서
  `self.sim.physics.newton_mjwarp.solver_cfg.njmax = 400`, `nconmax = 200`으로 확대해 해결
  (2026-07-14 실측: 512 envs에서 203 요구).
- **속도가 느릴 때**: 지형 삼각형이 ~52만 개라 Newton 충돌 파이프라인 부담이 있다.
  `RoughPhysicsCfg`의 `max_triangle_pairs`(기본 2.5M) 초과 에러가 나면 값을 올리고,
  그래도 느리면 전처리에서 지형을 50 m × 50 m로 잘라 쓰는 것도 방법.

---

## 부록: 왜 이 설계인가

- **Manager 기반 velocity env 상속**: 보행 RL의 어려운 부분(보상 셰이핑, 관측 노이즈, 푸시 이벤트,
  발 공중시간 보상 등)이 이미 G1용으로 튜닝되어 있다. 우리가 다루는 건 "지형과 로봇 에셋 교체"라는
  환경 차원의 변경뿐이므로 상속 + `__post_init__` 오버라이드가 최소 침습적이다.
- **전처리로 USD를 굽는 이유**: `terrain_type="usd"`는 물리 재질을 적용해 주지 않고, 런타임 패치는
  학습 스크립트 수정을 요구한다. 충돌체/재질/원점을 파일에 구워 두면 학습·재생·시각화 어디서든
  같은 지형이 보장된다.
- **MultiMeshRayCaster**: 정책 관측(height_scan 187차원)이 지형 요철을 보는 유일한 통로다.
  기본 RayCaster는 메시 1개만 읽어 돌들이 관측에서 누락되고, 그러면 정책이 돌을 "보지 못한 채"
  발끝 접촉으로만 학습하게 된다.
