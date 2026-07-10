# 화성 지형 위 모래 시뮬레이션 워크플로 (Newton Implicit MPM)

Isaac Lab에서 화성 로봇을 **실제 입상(granular) 모래** 위에서 테스트하기 위한 절차.
Newton 엔진의 Implicit MPM 솔버를 사용해 침하·미끄러짐·발자국을 물리적으로 재현한다.

- **환경**: Windows / NVIDIA GPU / Isaac Lab `C:\IsaacLab` (v3.0.0-beta2)
- **접근**: 로봇(강체 솔버) + 모래(implicit MPM 솔버) 2-솔버 커플링
- **레퍼런스 예제**: Newton 저장소의 `mpm_anymal` (사족 로봇이 MPM 모래 위를 보행)

> ⚠️ 아래 코드 조각은 Apache-2.0 라이선스인 Newton `mpm_anymal` 예제를 화성용으로 각색한 **예시**다.
> Newton은 아직 베타라 API가 바뀔 수 있으니, 메서드/파라미터 이름은 설치된 버전의 예제 소스와 대조해 확정할 것.

---

## 0. 시작 전 반드시 알아둘 두 가지

### (1) MPM은 대규모 RL 학습용이 아니라 고정밀 검증용이다
- 실측: **RTX 4090 / Windows에서 약 150만 MPM 파티클 → 8~12 FPS**.
- 즉 수천 병렬 환경 학습용이 아니라, 소수 환경에서 침하·슬립을 정밀 재현하는 **검증/데모/파인튜닝**용.
- **핵심 전략: 지형 전체를 모래로 덮지 말고, 로봇 발밑 국소 영역에만 얇은 모래층을 깐다.**

### (2) v3.0.0-beta2에서 MPM은 아직 Isaac Lab 매니저로 1급 노출되지 않았다
- Isaac Lab 3.0 백엔드: `isaaclab_physx`(기본, 전체 기능) / `isaaclab_newton`(신규).
- 이 버전의 Newton 백엔드가 노출하는 솔버: **MJWarp · XPBD · Featherstone + VBD(천) 커플링**, rough-terrain 프리셋.
- **입상 MPM 커플링은 아직 Isaac Lab 프리셋에 없음** → 현재 확실한 경로는 **Newton 엔진 레벨에서 직접 구성**(`mpm_anymal` 템플릿 활용).
- Isaac Lab 네이티브 MPM 프리셋은 개발 중이므로 `develop` 브랜치를 주시할 것.

---

## 1. 전체 구조: 2-솔버 커플링

```
[로봇]  ─ MuJoCo-Warp 솔버 (강체/articulation)
                  │  로봇 바디 = kinematic collider
                  ▼
[모래]  ─ Implicit MPM 솔버 (라그랑주 material point + 배경 격자)
                  ▲
[화성 지형] ─ static collider (원본 삼각형 메시 그대로 사용 가능)
```

- 로봇은 강체 솔버로 서브스텝, 모래는 MPM 솔버로 프레임 dt 스텝. 둘을 collider로 커플링.
- MPM은 material point에 상태를 담고 매 스텝 배경 격자를 리셋 → 큰 변형(침하·밀림)에도 안정적.
- Newton의 implicit MPM 구현은 Gilles Daviet, SIGGRAPH 2016 기반.
- **장점**: implicit MPM은 convex decomposition 없이 원본 고해상도 삼각형 메시를 collider로 사용 가능
  → 만들어 둔 화성 지형 USD 메시를 그대로 활용 가능. (MuJoCo Warp 솔버는 convex 분해 필요)

---

## 2. 사전 준비 (환경 세팅)

### 2.1 Newton + 예제 설치

Isaac Lab 번들 Python 환경에 설치하고, Windows에서는 `isaaclab.bat` 래퍼로 실행한다.

```bat
:: Isaac Lab 환경의 Python에 Newton 설치
C:\IsaacLab\isaaclab.bat -p -m pip install "newton[examples]"
```

GPU 필수(CPU 불가). 드라이버 545 이상 / CUDA 12 계열 권장.

### 2.2 레퍼런스 예제부터 실행 (화성 각색 전에)

원본 예제가 리그에서 정상 동작하는지 먼저 확인해 감을 잡는다.

```bat
C:\IsaacLab\isaaclab.bat -p -m newton.examples mpm_anymal --viewer gl
```

성능·정확도에 직결되는 주요 인자:
- `--voxel-size` / `-dx` : 격자 해상도 (작을수록 정밀·느림, 기본 0.03)
- `--particles-per-cell` / `-ppc` : 셀당 파티클 수 (기본 3.0)
- `--grid-type` / `-gt` : `sparse` / `dense` / `fixed`

### 2.3 작업 스크립트 위치

커스텀 워크플로 스크립트는 예를 들어 아래에 둔다.

```
C:\IsaacLab\scripts\mars_mpm\mars_mpm_workflow.py
```

실행:

```bat
C:\IsaacLab\isaaclab.bat -p C:\IsaacLab\scripts\mars_mpm\mars_mpm_workflow.py
```

---

## 3. 절차 (단계별)

### 3단계-A. 모델 빌드: 로봇 + 화성 지형 collider

```python
import numpy as np
import warp as wp
import newton
from newton.solvers import SolverImplicitMPM, SolverMuJoCo

builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
builder.default_shape_cfg.mu = 0.75          # 로봇-지면/모래 접촉 마찰

# 로봇 로드 (URDF 또는 USD)
builder.add_urdf(ROBOT_URDF_PATH, floating=True)   # 또는 builder.add_usd(...)

# 화성 지형 메시를 static collider로 추가
# (implicit MPM은 원본 삼각형 메시를 그대로 사용 가능)
# builder.add_usd(MARS_TERRAIN_USD_PATH, ...)  또는 mesh collider 추가 API 사용
```

### 3단계-B. MPM 커스텀 속성 등록 — 파티클 추가 **전에** 반드시

```python
SolverImplicitMPM.register_custom_attributes(builder)
```

> 순서 주의: 이 호출이 파티클 추가보다 먼저여야 한다.

### 3단계-C. 모래 뿌리기 — 지형 위 "얇은 층"으로만

비용을 좌우하는 핵심 단계. 지형 전체가 아니라 **로봇 주변 좁은 영역**에만 emit한다.
(원본 예제도 대략 1m × 3m × 0.15m 상자 정도만 모래를 깐다.)

```python
density = 1400.0                              # 화성 레골리스 겉보기 밀도(kg/m³) — 4절 참고
particle_lo = np.array([-0.5, -0.5, 0.00])    # emit 하한
particle_hi = np.array([ 0.5,  2.5, 0.15])    # emit 상한 (z 두께 15cm 얇은 층)

voxel_size = 0.03
particles_per_cell = 3.0
# 격자 해상도 계산 후 add_particle_grid 로 뿌림
# res = ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size)
builder.add_particle_grid(
    pos=...,          # emit 원점
    dim_x=..., dim_y=..., dim_z=...,   # 셀 개수
    cell_x=..., cell_y=..., cell_z=..., # 셀 크기
    mass=...,         # 파티클 질량 (density 기반)
    jitter=...,       # 위치 지터
    radius_mean=...,  # 파티클 반경
)
```

### 3단계-D. MPM 솔버 설정

Newton 권장 초기값에서 출발해 조정하는 것이 안전한 베이스라인.

```python
cfg = SolverImplicitMPM.Config()
cfg.voxel_size = voxel_size          # 작을수록 정밀·느림
cfg.transfer_scheme = "pic"          # PIC/FLIP 전이 방식
cfg.grid_type = "sparse"             # sparse / dense / fixed
cfg.max_iterations = 50
cfg.tolerance = 1.0e-6
cfg.air_drag = 1.0
cfg.critical_fraction = 0.0
cfg.collider_velocity_mode = "backward"

model = builder.finalize()

robot_solver = SolverMuJoCo(model, ls_iterations=50, njmax=50)
mpm_solver   = SolverImplicitMPM(model, config=cfg)
```

물질별 거동(모래 vs 무른 레골리스)은 파티클 그룹의 물질 파라미터(**cohesion, friction, elasticity**)로 구분한다.

### 3단계-E. collider 설정 (로봇 바디를 MPM 격자에 등록)

```python
state_0 = model.state()
newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

mpm_solver.setup_collider(
    body_mass=wp.zeros_like(model.body_mass),  # kinematic collider 취급
    body_q=state_0.body_q,
)
```

### 3단계-F. 2-솔버 스텝 루프

로봇은 서브스텝으로, 모래는 프레임 dt로 한 번 스텝. CUDA graph capture로 감싸면 성능↑.

```python
state_1 = model.state()
control = model.control()

for frame in range(num_frames):
    # --- 로봇 ---
    for _ in range(sim_substeps):
        state_0.clear_forces()
        robot_solver.step(state_0, state_1, control, contacts=None, dt=sim_dt)
        state_0, state_1 = state_1, state_0

    # --- 모래 ---
    mpm_solver.step(state_0, state_0, contacts=None, control=None, dt=frame_dt)
```

---

## 4. 화성 특화 튜닝 (sim-to-real 충실도의 핵심)

물리 근사와 달리 MPM은 실제 물질 거동을 재현하므로, 아래 세 값이 침하 깊이·발자국·슬립을 결정한다.

| 항목 | 지구 기준 | 화성 설정 | 비고 |
|---|---|---|---|
| **중력** | 9.81 m/s² | **≈ 3.72 m/s²** | `ModelBuilder`/`Model`의 gravity에 설정. 침하·비산 거동을 좌우하는 최우선 변수 |
| **밀도** | 모래 ~2500 kg/m³ | **~1200–1600 kg/m³** | 화성 레골리스는 다공성이 커 겉보기 밀도가 낮음. 예제 2500에서 낮춰 실험 |
| **물질 파라미터** | — | 내부 마찰각·약한 cohesion | 레골리스의 안식각/약한 응집력 반영 (cohesion, friction) |

```python
# 예: 화성 중력 설정 (축은 up_axis=Z 기준)
builder.gravity = wp.vec3(0.0, 0.0, -3.72)
```

---

## 5. 성능 최적화 (필수)

150만 파티클이 8~12 FPS인 만큼 최적화는 선택이 아니라 필수.

### 5.1 접촉 바디 외에는 파티클 충돌 끄기
발/정강이 등 실제로 모래에 닿는 바디만 파티클과 충돌시키고, 나머지는 `COLLIDE_PARTICLES` 플래그를 제거한다.

```python
for body in range(builder.body_count):
    if "FOOT" not in builder.body_label[body]:   # 접촉 바디 이름 규칙에 맞게 수정
        for shape in builder.body_shapes[body]:
            builder.shape_flags[shape] &= ~newton.ShapeFlags.COLLIDE_PARTICLES
```

### 5.2 모래 영역 최소화
- 로봇을 따라다니는 **이동 패치**로 모래 영역을 한정.
- `voxel_size`를 키워 격자를 성기게.

### 5.3 CUDA Graph 캡처
- `grid_type="fixed"`로 두면 모래 스텝도 CUDA graph로 캡처 가능 → 속도 향상.

---

## 6. 권장 워크플로 (하이브리드)

```
[학습]  마찰 근사(강체 지형 + physics material) 로 수천 병렬 환경 → 정책 빠르게 학습
   │
   ▼
[검증]  학습된 정책을 이 MPM 모래 패치 환경에 올려 실제 침하·슬립 재현/검증
   │
   ▼
[개선]  (선택) Newton은 미분 가능 → gradient 기반 파인튜닝/최적화 활용
```

- 대규모 RL은 마찰 근사로, **정밀 검증·sim-to-real 갭 축소는 MPM으로** 분담.
- Isaac Lab `develop` 브랜치에서 MPM 프리셋이 매니저로 1급 통합되면 위 Newton-레벨 수작업이 크게 줄어듦.

---

## 참고 자료

- Newton 저장소 / MPM 예제: `newton-physics/newton` — `newton/examples/mpm/example_mpm_anymal.py`
- Newton 문서: https://newton-physics.github.io/newton/
- Isaac Lab + Newton 통합 개요: NVIDIA Technical Blog "Train a Quadruped Locomotion Policy ... with Isaac Lab and Newton"
- Lightwheel–Newton 입상 지형 에셋/성능 사례 (Windows·RTX 4090, 1.5M 파티클, 8–12 FPS)
- Implicit MPM 이론: Gilles Daviet & Florence Bertails-Descoubes, SIGGRAPH 2016
