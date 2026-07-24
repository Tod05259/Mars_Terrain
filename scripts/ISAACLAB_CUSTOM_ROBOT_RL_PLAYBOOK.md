# IsaacLab 커스텀 로봇 × 커스텀 지형 강화학습 플레이북

> **목적**: 임의의 로봇 USD를 임의의 지형 USD 위에서 걷/움직이도록 IsaacLab에서 RL로
> 학습시키기 위한 **재사용 가능한 절차·진단법·교훈** 모음.
>
> **작성 근거**: Unitree G1 휴머노이드를 Gale 크레이터 화성 지형(`Gale_stone_env_color.usd`)에서
> 보행 학습시킨 실제 사례 (2026-07-14, IsaacLab 3.0.0-beta2, Windows).
> 구체적 구현·수치는 자매 문서 [`G1_MARS_RL_GUIDE.md`](./G1_MARS_RL_GUIDE.md) 참조.
>
> 이 문서는 "G1이 아닌 다른 로봇/지형"에 응용할 때 **무엇을 그대로 쓰고, 무엇을 바꿔야 하는지**에
> 초점을 둔다. 각 절 끝의 **▶ 일반화** 블록이 응용 포인트다.

---

## 0. 핵심 철학 3가지

1. **처음부터 만들지 말고 상속하라.** IsaacLab에는 이미 검증된 태스크(보상 셰이핑, 관측 노이즈,
   커리큘럼, 이벤트)가 많다. 우리가 다루는 건 대개 "에셋 교체"라는 환경 차원의 변경뿐이므로,
   기존 태스크를 상속해 `__post_init__`에서 필요한 것만 덮어쓰는 게 최소 침습적이고 안전하다.
2. **에셋을 의심하라, 하이퍼파라미터보다 먼저.** 이번 사례의 두 대형 사고(관절 무동작, 지형 관통)는
   모두 **에셋/설치 환경 문제**였지 RL 문제가 아니었다. 학습이 이상하면 물리·에셋부터 격리 진단한다.
3. **격리 진단 스크립트를 자산으로 남겨라.** "액션 0으로 세워보기", "USD 구조 덤프", "법선 검사"
   같은 소형 스크립트가 문제 절반을 잡는다. 3장에 템플릿을 모아뒀다 — 로봇이 바뀌어도 재사용된다.

---

## 1. 전체 워크플로 (한눈에)

```
[1] 기존 태스크 선정        어떤 IsaacLab 태스크를 상속할지 (로봇 형태·과제 유형 기준)
        │
[2] 에셋 사전 진단          로봇 USD + 지형 USD를 pxr로 열어 구조 검사 (★ 학습 전 필수)
        │                   → 관절명, ArticulationRoot, 충돌체 위치, 법선, 원점, 메시 구조
        │
[3] 지형 전처리            충돌체 + 마찰 재질 + 단일 메시 병합 + 재중심화 + 법선 교정 → *_train.usd
        │
[4] 로봇 에셋 처리          관절명 호환성 확인, 필요 시 충돌체 근사 추가
        │
[5] 태스크 패키지 작성      __init__ / env_cfg / agents (5개 파일) — 상속 + 오버라이드
        │
[6] 물리 백엔드 설정        PresetCfg(newton_mjwarp), njmax 등
        │
[7] 스탠딩 테스트          ★ 학습 전에 로봇이 지형 위에 '서는지' 물리 검증 (액션 0)
        │
[8] 스모크 테스트          16 envs × 10 iters — 등록·스폰·크래시 확인
        │
[9] 본 학습 + 모니터링      512 envs, 마일스톤마다 지표 점검
        │
[10] 재생/시각화 + 보상 반복  걸음새 관찰 → 퇴화 걸음새면 5장으로 돌아가 보상 재설계
```

**[2]와 [7]을 건너뛰지 말 것.** 이번 사례에서 낭비된 시간의 대부분은 이 두 단계를 생략하고
바로 학습에 들어갔다가 뒤늦게 원인을 역추적한 데서 나왔다.

---

## 2. 단계별 절차

### 2.1 기존 태스크 선정

로봇 형태와 과제로 매핑한다:

| 로봇/과제 | 상속 후보 태스크 | 위치 |
|---|---|---|
| 4족 보행 | `Isaac-Velocity-Rough-Anymal-C-v0` 등 | `manager_based/locomotion/velocity/config/anymal_*` |
| 2족(휴머노이드) 보행 | `Isaac-Velocity-Rough-G1-v0`, `...-H1-v0` | `.../velocity/config/g1`, `.../h1` |
| 팔 조작 | `Isaac-Reach-*`, `Isaac-Lift-*` | `manager_based/manipulation/*` |
| 단순 제어 | `Isaac-Cartpole-*` (Direct) | `direct/cartpole` |

**Manager-based** 태스크(보상/관측/이벤트가 선언적 Term으로 분리)가 상속·오버라이드에 유리하다.
**Direct** 태스크는 보상이 파이썬 함수에 하드코딩돼 있어 파일을 직접 고쳐야 한다.

> ▶ **일반화**: 목표 로봇과 **형태(morphology)가 같은** 기존 로봇의 태스크를 고르는 게 핵심.
> 관절 구성·발 개수가 비슷할수록 보상·종료 항을 그대로 재사용할 수 있다.

### 2.2 에셋 사전 진단 (★ 가장 중요)

로봇/지형 USD를 pxr로 열어 아래 체크리스트를 채운다. 3.1의 `inspect_usd.py` 템플릿 사용.

**로봇 USD 체크리스트**
- [ ] `defaultPrim`과 `ArticulationRootAPI`가 있는 prim 경로
- [ ] **관절 이름** — 상속할 태스크의 정규식(`.*_hip_.*`, `.*_ankle_roll_link` 등)과 맞는가?
      맞으면 액추에이터·보상 설정을 그대로 재사용, 다르면 전부 다시 매핑해야 함
- [ ] **충돌체가 어느 링크에 있는가** — 발에만? 전신? 이게 종료 조건 설계를 좌우한다 (아래 2.4)
- [ ] `PhysicsScene`이 `defaultPrim` 안/밖 어디에 있는가 (밖이면 참조 시 안 딸려와 문제없음)

**지형 USD 체크리스트**
- [ ] **CollisionAPI가 하나라도 있는가** — 없으면 로봇이 관통·추락 (전처리 필수)
- [ ] **삼각형 법선 방향** — 위를 향하는가? (3.3 법선 검사) ★ Newton은 단면 충돌이라 치명적
- [ ] bbox 중심이 **원점 근처인가** — 모서리에 있으면 재중심화 필요 (로봇은 원점 격자에 스폰됨)
- [ ] 메시 구조 — 단일 메시인가, 수백 개 분할/인스턴스인가 (병합 권장)
- [ ] 스폰 구역의 **고저차** — 크면 env origin을 지면에 스냅해야 함 (2.5 커스텀 임포터)

> ▶ **일반화**: 이 두 체크리스트는 로봇/지형이 바뀌어도 그대로다. Blender·타 DCC에서 나온
> USD는 **법선·충돌체·원점**이 IsaacLab 관례와 어긋나는 경우가 흔하니 항상 의심할 것.

### 2.3 지형 전처리 → `*_train.usd` 굽기

원본은 건드리지 말고 학습용 사본을 만든다. 이번에 [`prepare_gale_training_usd.py`](./prepare_gale_training_usd.py)가
수행한 작업 = 재사용 템플릿:

1. **모든 메시를 단일 삼각메시로 병합** (월드 좌표로 변환하며) — 검증된 generator 지형과 동일 구조
2. **삼각형 감김 방향(winding) 교정** — 지면은 법선 z 다수결, 닫힌 볼륨(돌)은 부호 있는 부피로 판정해 뒤집기
3. **CollisionAPI + MeshCollisionAPI(approximation="none", 정적 삼각메시)** 부여
4. **마찰 재질**(UsdPhysics.MaterialAPI) 생성 후 physics purpose로 바인딩
5. **재중심화를 정점 좌표에 직접 베이크** (xformOp보다 안전)

> ▶ **일반화**: 지형이 바뀌면 `SRC`/`DST`/`GROUND_MESH` 경로와 마찰값만 바꾸면 된다.
> **같은 DCC 파이프라인에서 나온 다른 지형(예: 다른 크레이터)은 법선이 똑같이 뒤집혀 있을 확률이
> 높으니** 이 전처리를 항상 거칠 것. 마찰값은 표면 재질에 맞게 조정 (화성 표토: μs 0.54, μd 0.42).

### 2.4 로봇 에셋 처리

- **관절명이 호환되면** 로봇 USD는 대개 그대로 쓰고, env cfg에서 `robot.spawn.usd_path`만 교체.
- **충돌체 위치가 종료 조건을 결정한다**:
  - 전신에 충돌체가 있으면 → 스톡의 `base_contact`(몸통이 땅에 닿으면 종료) 사용 가능.
  - **발에만 있으면**(이번 G1 케이스) → `base_contact` 불가 → **자세 기반**(`bad_orientation`) +
    **높이 기반**(`base_height` 보상, `root_height_below_minimum` 종료)으로 대체.
  - 필요하면 발 큐브에 했던 것처럼 `pelvis`/`torso`에 convexHull 충돌체를 추가한 사본을 구워
    `base_contact`를 복원할 수도 있다 (가장 견고하지만 USD 편집 필요).

> ▶ **일반화**: "이 로봇은 어디에 충돌체가 있나?"가 종료 조건 설계의 출발점. 접촉 센서 기반
> 종료를 쓰려면 그 링크에 충돌체가 반드시 있어야 한다.

### 2.5 태스크 패키지 작성 (5개 파일)

`manager_based/locomotion/velocity/config/<robot>_<terrain>/` 아래에 새 폴더를 만들면
IsaacLab이 import 시 자동 등록한다. 파일 구성:

```
<robot>_<terrain>/
├── __init__.py              # gym.register (train/play 2개 id)
├── <name>_env_cfg.py        # 상속 + __post_init__ 오버라이드 (핵심)
└── agents/
    ├── __init__.py
    └── rsl_rl_ppo_cfg.py     # 러너 설정 상속 (experiment_name, max_iterations)
```

env_cfg의 `__post_init__`에서 오버라이드하는 **표준 항목**(이번 사례 기준):
- `scene.terrain` → `TerrainImporterCfg(terrain_type="usd", usd_path=...)`
- `curriculum.terrain_levels = None` (usd 지형엔 난이도 레벨이 없음)
- `scene.robot.spawn.usd_path` → 내 로봇 USD
- `scene.height_scanner` → **`MultiMeshRayCasterCfg`** (아래 2.5.1)
- 종료 조건 교체 (2.4)
- `scene.num_envs` 제한 (지형 크기에 맞게, env_spacing 격자가 지형을 벗어나지 않도록)

#### 2.5.1 다중 메시 지형에는 MultiMeshRayCaster

기본 `RayCasterCfg`(height scanner)는 **prim 하나당 첫 Mesh 1개만** 읽는다. 돌 수백 개가 있는
지형에선 엉뚱한 돌 하나만 스캔하므로, `MultiMeshRayCasterCfg`로 교체해야 정책이 지형을 제대로
관측한다 (기본값 `merge_prim_meshes=True`가 전체 메시를 병합). **단일 메시로 병합했다면(2.3)
기본 RayCaster로도 충분**하지만, 안전하게 MultiMesh를 권장.

#### 2.5.2 env origin을 지면 높이에 스냅 (기복 큰 지형)

usd 지형의 env origin은 z=0 평면 격자로 계산된다. 스폰 구역 고저차가 크면 로봇이 파묻히거나
낙하하므로, `TerrainImporterCfg.class_type`를 커스텀 임포터로 교체해 각 origin의 z를 최근접
지면 정점 높이로 보정한다 (이번 `MarsTerrainImporter`). 평탄 지형이면 생략 가능.

> ▶ **일반화**: 폴더명·id·experiment_name만 로봇/지형에 맞게 바꾸면 패키지 골격은 그대로 재사용.
> 4족이면 상속 태스크를 Anymal 계열로 바꾸는 것 외엔 구조가 동일하다.

### 2.6 물리 백엔드 설정

**이 설치본(Windows, IsaacLab 3.0.0-beta2)의 특수 제약**:
- **PhysX GPU 파이프라인이 관절 토크를 조용히 무시한다** → 반드시 **`physics=newton_mjwarp`**.
- `physics=newton_mjwarp` 토큰은 env cfg에 `newton_mjwarp` 필드를 가진 **`PresetCfg`가 정의돼
  있어야** 먹는다. 상속한 rough 계열엔 이미 있지만, 없는 태스크(예: cart_double_pendulum)는
  직접 추가해야 한다.
- **`nefc overflow - please increase njmax to N`** 경고 = MJWarp 제약 버퍼 초과. 접촉이 조용히
  누락되니 무시 말고 `sim.physics.newton_mjwarp.solver_cfg.njmax`/`nconmax`를 키운다
  (이번 512 envs 돌밭: njmax 200→400, nconmax 100→200).

> ▶ **일반화**: PhysX-GPU 무동작은 이 설치 환경 고유 문제일 수 있다. 새 로봇에서도 관절이
> 안 움직이면 `physics=newton_mjwarp`로 전환해 격리 확인. njmax는 접촉 복잡도(다리 수·지형)에
> 비례해 키운다.

### 2.7 스탠딩 테스트 (★ 학습 전 물리 검증)

**학습을 돌리기 전에** 로봇이 지형 위에 실제로 서는지 확인한다. 3.2의 `diag_feet_contact.py` 사용:
액션 0으로 몇십 스텝 두고 —
- **정상**: root z가 서 있는 높이로 유지되고, 발 접촉력이 체중(~로봇 무게×g) 수준으로 지속 측정.
- **비정상(관통/침몰)**: root z가 계속 내려가고 발 접촉력이 0 또는 산발적 스파이크 → **지형 충돌
  문제**(법선/충돌체). 이 상태로 학습하면 로봇이 걸을 수 없는 환경이라 반드시 퇴화하거나 NaN.

이번 사례에서 이 테스트 하나가 "지형 법선 뒤집힘"이라는 근본 원인을 확정했다. 공식 G1로도
재현하고, 절차 생성 지형에선 정상임을 대조해 "지형 데이터 문제"로 좁혔다.

> ▶ **일반화**: 로봇이 바뀌면 스크립트의 발 링크 정규식(`.*_ankle_roll_link`)과 예상 체중만
> 바꾼다. 4족이면 4개 발 접촉력의 합이 체중에 근접하는지 확인.

### 2.8 스모크 테스트 → 본 학습

```powershell
# 스모크: 등록·스폰·크래시만 확인 (수십 초)
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py `
  --task <TASK> --num_envs 16 --max_iterations 10 physics=newton_mjwarp --headless

# 본 학습 (512 envs). 로그: logs\rsl_rl\<experiment_name>\<timestamp>\
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py `
  --task <TASK> --num_envs 512 physics=newton_mjwarp --headless
```

**모니터링**: 마일스톤(500·1000·2000...)마다 지표를 본다. `save_interval`마다 체크포인트가
남으므로 크래시해도 복구 가능. TensorBoard: `.\isaaclab.bat -p -m tensorboard.main --logdir logs\rsl_rl\<exp>`.

### 2.9 재생/시각화

```powershell
# ★ 이 설치본에서 --viz kit(Isaac Sim GUI)은 Kit 확장 DLL 로드 실패로 죽는다.
#   Newton 백엔드의 자체 OpenGL 뷰어(--viz newton)를 쓸 것.
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py `
  --task <TASK>-Play-v0 --num_envs 32 physics=newton_mjwarp --viz newton
```

---

## 3. 진단 도구 모음 (재사용 핵심 자산)

이 스크립트들이 로봇이 바뀌어도 쓰이는 실질 자산이다. 스크래치 폴더가 아니라
프로젝트에 보관해 두길 권한다.

### 3.1 USD 구조 덤프 (`inspect_usd.py` 패턴)
pxr로 USD를 열어 `defaultPrim`/타입, 메시 수·정점 수, CollisionAPI 유무, 관절
(RevoluteJoint/PrismaticJoint) 이름, ArticulationRoot 경로, bbox를 출력. **AppLauncher 없이**
순수 pxr만 쓰므로 `./isaaclab.bat -p`로 수 초 만에 실행된다.

### 3.2 스탠딩 테스트 + Newton 모델 검사 (`diag_feet_contact.py` 패턴)
env를 만들어 액션 0으로 스텝하며 발 접촉력·root z를 출력. 추가로 `NewtonManager._model`의
`shape_body`(정적 shape=−1), `shape_type`, `shape_flags`를 덤프해 **지형 shape이 실제로
모델에 들어갔는지 / 로봇 shape에 충돌 플래그가 있는지** 확인. 핵심 주의:
`parse_env_cfg`는 프리셋을 **기본값(physx)**으로 해석하므로, Newton 진단은
`load_cfg_from_registry` + `resolve_presets(cfg, selected=("newton_mjwarp",))`로 로드하고
로그에 `Registered backend 'newton'`이 찍히는지 확인할 것.

### 3.3 법선/winding 검사 (`check_winding.py` 패턴)
지형 메시 삼각형의 법선 z 부호 분포를 계산. "normals up" 비율이 낮으면(특히 0%) 법선이
뒤집힌 것 → Newton 단면 충돌 실패 → 전처리에서 winding 교정 필요.

### 3.4 학습 지표 해석표 (퇴화 걸음새 진단)

| 관찰된 지표 | 해석 |
|---|---|
| `feet_air_time` 보상이 내내 ≈0 | 로봇이 발을 안 뗌 (앉기/끌기). 걸음이 아예 형성 안 됨 |
| 에피소드 길이 만점 + `bad_orientation` 낮음 | 매우 안정 = 넘어지진 않음 (앉기와 양립) |
| `mean action std`가 계속 상승 | 정책 발산 — 과제가 너무 어렵거나 물리가 깨짐 |
| 보상 정체 + ep_len이 자유낙하 시간에 고정 | 관절이 안 움직임 (PhysX-GPU 무동작 의심) |
| success_rate 0에서 안 오름 | 명령 추종 실패 — 난이도·가중치·물리 점검 |

---

## 4. 트러블슈팅 (이번에 실제로 겪은 것들)

| 증상 | 근본 원인 | 해결 |
|---|---|---|
| 관절에 토크를 줘도 안 움직임 | 이 설치본 PhysX-GPU 무동작 | `physics=newton_mjwarp` |
| `Unknown preset(s): newton_mjwarp` | env cfg에 PresetCfg 미정의 | cfg에 `newton_mjwarp` 필드 가진 PresetCfg 추가 |
| skrl `ModuleNotFoundError` | Kit python에 미설치 | `./isaaclab.bat -p -m pip install "skrl[torch]"` |
| 로봇이 지형을 뚫고 침몰 / 발 접촉력 0 | **지형 법선이 아래로 뒤집힘** (Newton 단면 충돌) | 전처리에서 winding 교정. 스탠딩 테스트로 진단 |
| `nefc overflow ... increase njmax` | MJWarp 제약 버퍼 초과 (접촉 누락) | `solver_cfg.njmax`/`nconmax` 확대 |
| 학습 중반 `observation contains NaN` 크래시 | 대개 물리 발산(위 지형 문제 등)의 2차 증상 | 근본 원인 수정 + 안전망: `clip_actions`, `root_height_below_minimum` 종료 |
| 로봇이 공중 스폰/언덕에 파묻힘 | usd 지형 env origin z=0 고정 | 커스텀 임포터로 origin z를 지면에 스냅 |
| 로봇이 앉아서 끌고 다님 (퇴화 걸음새) | "서 있어라" 보상 부재 + 걸음 보상 미점화 | 5장 참조 (base_height 보상 등) |
| `--viz kit` 실행 시 `Caught an unknown exception!` | 이 설치본 Kit GUI 확장 DLL 로드 실패 | `--viz newton` (Newton 자체 뷰어) 또는 `--viz none` |

---

## 5. 보상 설계 원칙 — 퇴화 걸음새(degenerate gait) 방지

**퇴화 걸음새**란 과제를 "요령"으로 푸는 안정적 국소 최적점이다. 대표적으로 **앉아서 끌기**,
**기어다니기**, **제자리 정지**. 물리·에셋이 정상인데도 나타나면 **보상 구조의 구멍** 때문이다.

이번에 G1이 "앉아서 끌기"에 빠진 원인과 교정(재사용 체크리스트):

- [ ] **자세를 강제하는 보상이 있는가?** 골반/몸통을 서 있는 높이로 유지하는 `base_height_l2`
      보상(지형 상대: `sensor_cfg=height_scanner` 필수). **이게 없으면 로봇은 CoM이 낮은
      앉은 자세로 수렴**하고, 그 상태에선 걸음 보상이 구조적으로 안 켜지는 악순환에 갇힌다.
- [ ] **걸음(발 들기)을 보상하는가?** `feet_air_time_positive_biped`는 "한 발 지지 + 명령속도>0.1"
      일 때만 보상. 앉으면 이 조건이 영영 성립 안 함 → 자세 보상으로 **일단 세운 뒤** 이 가중치를
      키워 걷기를 강화 (0.25→0.5).
- [ ] **정지 최적점을 없앴는가?** 명령 하한이 0이면 명령≈0 구간에서 "가만히 있기"가 만점.
      상한을 올려(0.8→1.0) 이동을 요구하되, 하한 0은 유지하고 **base_height 보상으로 v=0에서도
      '앉기'가 아니라 '서서 정지'가 최적**이 되게 한다.
- [ ] **종료로 걸러지는가?** 전신 충돌체가 있으면 `base_contact`로 앉기/넘어짐을 즉시 종료.
      없으면 `bad_orientation`(기울기)만으로는 '똑바로 앉기'를 못 잡으니 높이 기반 항이 필수.

**중요**: 퇴화 걸음새를 깊이 학습한 정책은 이어학습(fine-tune)으로 빠져나오기 어렵다.
보상을 고쳤으면 **처음부터 재학습**하고, 초기부터 `feet_air_time`/`base_height` 지표가
개선되는지 확인한다.

> ▶ **일반화**: 자세 보상·걸음 보상·정지 최적점 제거는 **모든 다리 로봇 보행**에 공통. 4족은
> 앉기 대신 "배 깔고 끌기"가 나오는데 처방은 동일(base_height + 걸음 보상 + 접촉 종료).

---

## 6. 다른 로봇으로 응용할 때 — 무엇을 바꾸는가

| 구성요소 | 그대로 재사용 | 바꿔야 하는 것 |
|---|---|---|
| 워크플로(1장) / 진단 스크립트(3장) | ✅ 전부 | 발 링크 정규식, 예상 체중 정도 |
| 지형 전처리(2.3) | ✅ 로직 전부 | USD 경로, 마찰값 |
| 태스크 패키지 골격(2.5) | ✅ 구조 | 상속 대상 태스크, 폴더/id/experiment_name |
| 물리 백엔드(2.6) | ✅ 원칙 | njmax(다리 수·접촉 복잡도에 비례) |
| 종료 조건(2.4) | 원칙 | **로봇 충돌체 위치**에 따라 base_contact vs 자세/높이 |
| 보상 항(5장) | ✅ 체크리스트 | 관절 그룹 정규식(로봇 관절명), 목표 높이 |

**로봇 형태별 상속 대상 바꾸기**가 응용의 8할이다:
- **4족**: Anymal 계열 velocity 태스크 상속. 발 4개 → `feet_air_time`이 2족과 다른 함수일 수 있음
  (single-stance 조건이 다름). base 링크명이 다르니(`base` vs `torso_link`) 종료·센서 경로 수정.
- **다른 휴머노이드(H1 등)**: G1 케이스를 거의 그대로. 관절명 정규식과 목표 골반 높이만 조정.
- **팔 조작 로봇**: 지형·보행 부분은 무관. Reach/Lift 태스크 상속하고 로봇 USD·EE 링크만 교체.

**가장 먼저 할 일은 항상 2.2 에셋 진단**이다. 관절명이 상속 태스크의 정규식과 맞으면 작업량이
급감하고, 안 맞으면 액추에이터·보상의 관절 그룹을 전부 다시 매핑해야 한다.

---

## 부록 A. 이번 G1 × 화성 사례 최종 스펙

- **로봇**: `g1.usd` (Unitree G1, 37관절, 관절명이 공식 G1과 동일 → IsaacLab `G1_CFG` 정규식 호환,
  충돌체는 발 큐브 2개뿐)
- **지형**: `Gale_stone_env_color.usd` → 전처리 → `Gale_stone_env_color_train.usd`
  (501메시 단일 병합, 법선 15개 교정, 100×100 m, 원점 재중심)
- **태스크**: `Isaac-Velocity-Mars-G1-v0` (`Isaac-Velocity-Rough-G1-v0` 상속)
- **백엔드**: newton_mjwarp, njmax 400 / nconmax 200
- **학습**: rsl_rl PPO, 512 envs, 5000 iters, ~1시간 (RTX, 약 17k steps/s)
- **1차 결과(보상 결함)**: 앉아서 끌기. → base_height_l2(-10, 지형상대) + feet_air_time(0.5) +
  lin_vel_x(0,1.0) 추가 후 재학습
- **주요 산출물**:
  - `prepare_gale_training_usd.py` — 지형 전처리
  - `source/.../velocity/config/g1_mars/` — 태스크 패키지 (`MarsTerrainImporter` 포함)
  - `G1_MARS_RL_GUIDE.md` — G1/화성 특정 구현 상세서

## 부록 B. 자주 쓴 명령 모음

```powershell
# 인라인 파이썬 / USD 진단 (AppLauncher 불필요한 것은 수 초)
.\isaaclab.bat -p <script.py>
.\isaaclab.bat -p -c "..."

# 패키지 설치 (Kit python 대상)
.\isaaclab.bat -p -m pip install "<pkg>"

# 학습 / 재생 / 텐서보드
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task <TASK> --num_envs 512 physics=newton_mjwarp --headless
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py  --task <TASK>-Play-v0 --num_envs 32 physics=newton_mjwarp --viz newton
.\isaaclab.bat -p -m tensorboard.main --logdir logs\rsl_rl\<experiment_name>

# 멀티에이전트(skrl) 태스크
.\isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task <TASK> --num_envs 512 --algorithm IPPO physics=newton_mjwarp --headless
```
