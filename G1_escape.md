# G1 화성 발빠짐 탈출 학습 가이드라인

> **목표**: CAPSTONE이 H1으로 완성한 "화성 레골리스 발빠짐 → 자력 탈출" RL 파이프라인
> (측정 → MPM 환경 → 벡터 학습 → 판정 → 녹화)을 **Unitree G1으로 이식**하기 위한 실행 절차.
>
> **전제**: 이 문서는 처음부터 만드는 계획이 아니다. H1 파이프라인은 이미 동작하고
> (결정론 탈출 37.5%), 우리 쪽은 Mars 중력 rigid 지형에서 G1 보행 정책을 확보했다.
> 남은 일은 **두 자산을 잇는 것**과, **G1에서만 새로 성립시켜야 하는 물리 조건**을
> 재측정하는 것이다.
>
> **참조 문서** (읽는 순서 권장)
> - `CAPSTONE/PROJECT_PLAN.md` — 과제 정의, 확정 측정 프로토콜, 커리큘럼 손잡이
> - `CAPSTONE/terrain/release/README.md` — 릴리스 자산과 실행 명령
> - `CAPSTONE/terrain/mars_terrain/mars_real_terrain_env/EXTRACTION_HANDOFF.md`
>   — 마스터 기록. 특히 **§7.5r**(앵커링 수지), **§7.5ab**(Spirit 검증 + G1 관통 실패),
>   **§7.5ac~af**(벡터화·패리티 결함), **§10~§11**(학습 붕괴 회귀와 정정)
> - `Mars_Terrain/mpm_learning.md` — rigid↔MPM sim-to-sim 로드맵 (본 문서의 상위 계획)
> - `Mars_Terrain/scripts/G1_MARS_RL_GUIDE.md` — G1 rigid 학습 구현
>
> 최초 작성: 2026-08-05

---

## 0. 한 장 요약

```
Step 0  선행 차단 요인 해소          2~3일   콜라이더 복구 + 정책 export      ← 여기서 막히면 나머지 전부 무의미
Step 1  G1 MPM 런타임 포팅           3~5일   07 → 30_g1_mars_newton.py
Step 2  G1 발빠짐 성립 재측정        3~5일   29 스윕 + 09/11 재측정           ← G1 고유, 본 프로젝트의 새 기여
Step 3  발빠짐 스냅샷 생성기         1~2일   22 --make-snapshot 이식
Step 4  벡터 탈출 env 포팅           4~7일   26 → 32_g1_escape_env_vec.py
Step 5  학습 (웜스타트 + 커리큘럼)   1~2주   27 재사용
Step 6  평가·베이스라인·녹화         3~5일   28 재사용
```

**핵심 판단**: Step 2가 이 프로젝트의 성패를 가른다. G1은 H1보다 가볍지만 발이 작아
접지압이 더 높고, 그럼에도 **정적 하중으로는 Troy형 껍질을 뚫지 못한다**는 것이
이미 관측됐다(HANDOFF §7.5ab). 즉 **"G1이 빠지는 조건"을 먼저 물리적으로 성립시켜야
탈출 학습이라는 과제 자체가 존재한다.** Step 2를 건너뛰고 Step 4로 가면
H1의 스냅샷 파라미터를 그대로 써서 "빠지지도 않는 로봇의 탈출"을 학습시키게 된다.

---

## 1. 출발점 — 실측 인벤토리

이 절의 수치는 전부 이 저장소/설치본에서 직접 확인한 값이다.

### 1.1 우리 쪽 (Mars_Terrain + IsaacLab)

| 자산 | 위치 | 상태 |
|---|---|---|
| G1 로봇 USD | `Mars_Terrain/usd(completed)/g1.usd` | 44 강체, **콜라이더 2개(발만)**, 총질량 **32.24 kg** |
| G1 화성 보행 태스크 | `IsaacLab-3.0.0-beta2/.../velocity/config/g1_mars/` | `Isaac-Velocity-Mars-G1-v0` / `-Play-v0` 등록됨 |
| Mars 중력 + soft DR 적용 | `mars_env_cfg.py:194-201` | 중력 −3.721, 마찰 DR 0.4~0.9 / 0.3~0.75, push 6~10 s |
| **최신 보행 체크포인트** | `logs/rsl_rl/g1_mars/2026-07-31_14-14-14_mars_gait_v2/model_4999.pt` | Mars 중력 + 걸음새 교정 v2 |
| 대안 체크포인트 | `.../2026-07-31_12-44-14_mars_gravity_soft_dr/model_4999.pt` | 걸음새 교정 전 |
| exported policy.pt | `.../2026-07-24_14-10-37/exported/policy.pt` | ⚠ **지구 중력 시절 것** — 최신 런은 export 안 됨 |
| 지형 USD | `Mars_Terrain/usd(completed)/*_env.usd` | Gale 학습용 + HiRISE crop100m |

정책 인터페이스 (`params/env.yaml` 확인):

| 항목 | 값 |
|---|---|
| 관측 | **310차원** = lin_vel 3 + ang_vel 3 + gravity 3 + cmd 3 + joint_pos 37 + joint_vel 37 + last_action 37 + height_scan 187 |
| 행동 | **37차원**, `scale=0.5`, `use_default_offset=true` → `target = default + 0.5 · a` |
| 네트워크 | MLP [512, 256, 128] elu, `empirical_normalization=False`, `clip_actions=6.0` |
| height scan | `torso_link` 부착, GridPattern res 0.1 / size [1.6, 1.0] = 17×11 = **187**, yaw 정렬 |
| 명령 범위 | 학습 `lin_vel_x ∈ (0.5, 1.5)`, play 고정 1.0 |
| 종료 | `bad_orientation 0.8 rad` + `root_below_terrain −3.0` (**base_contact는 콜라이더가 없어 비활성**) |

### 1.2 CAPSTONE 쪽 (이식 대상)

| 자산 | 위치 | 역할 |
|---|---|---|
| MPM 런타임 | `terrain/scripts/07_h1_mars_newton.py` | H1 + DTM + 양방향 MPM 핵심 루프 |
| 판 압입 벤치 | `terrain/scripts/09_mars_mpm_plate_sinkage.py` | 지지력 측정 (`--plate-half-*`) |
| 발 인발 측정 | `terrain/scripts/11_mars_mpm_foot_extraction.py` | 압입/유지/전단/인발 4상 |
| Troy 대역 스윕 | `terrain/scripts/29_troy_parameter_sweep.py` | **`g1_foot` 프리셋 이미 존재** |
| 단일 탈출 env | `terrain/scripts/22_escape_env.py` | 스냅샷 생성기(`--make-snapshot`) 포함 |
| **벡터 탈출 env** | `terrain/scripts/26_escape_env_vec.py` (1795줄) | 이식의 주 대상 |
| 벡터 트레이너 | `terrain/scripts/27_train_escape_vec.py` | **로봇 비의존** — 그대로 재사용 가능 |
| 성공 녹화 | `terrain/scripts/28_record_successes.py` | 로봇 비의존에 가까움 |
| G1 시제품 (보관) | `archive/archived_scripts/21_g1_buried_single_env.py` | 관절 게인·자세·포켓 배치 참고 |

H1 기준 확정 물리값 (G1 대응값을 Step 2에서 새로 채울 표):

| 항목 | H1 값 | 출처 |
|---|---:|---|
| 화성 체중 | 191.3 N | PROJECT_PLAN §3 |
| 발 접촉 면적 | 0.32 × 0.16 = 0.0512 m² | 29번 `h1_foot` 프리셋 |
| 한 발 접지압 | 3.73 kPa | 물성 보정 설계점(3.736 kPa)과 일치 |
| 7지역 균질 침하 | 29~56 mm | PROJECT_PLAN §3.1 |
| 226 mm 매장 시 인발력 | 93 N | HANDOFF §7.5r |
| 지지발 최대 지지력 | 278 N | HANDOFF §7.5r |
| **앵커링 수지** | 191 + 93 − 278 = **−6 N (탈출 불가)** | HANDOFF §7.5r |
| 침하 ↔ 포켓 바닥 | 침하 ≈ 0.85~0.91 × 포켓 바닥 깊이 | HANDOFF §7.5r (5조건 일관) |
| 커리큘럼 대역 | 포켓 바닥 150 mm(쉬움) → 270 mm(불가) | PROJECT_PLAN §3.4 |

---

## 2. G1이 H1과 다른 세 가지 — 선행 차단 요인

### 2.1 [치명] `g1.usd`에 콜라이더가 발 2개뿐이다

직접 확인한 결과:

```
RIGID BODIES 44
COLLIDERS      2      /g1/left_ankle_roll_link/Cube, /g1/right_ankle_roll_link/Cube
```

`mars_env_cfg.py:134`도 같은 사실을 기록하고 있다 — "g1.usd는 발에만 충돌체가 있어
몸통 접촉 감지가 불가능" → `terminations.base_contact = None`.

rigid 보행 학습에서는 기울기 종료로 우회했으므로 문제가 없었다. **탈출 과제에서는
치명적이다.** 이것은 CAPSTONE이 5일에 걸쳐 잘못된 원인을 쫓다가 발견한
HANDOFF **§11**과 완전히 동일한 상황이다:

> 발을 제외한 모든 링크에 형상 충돌이 꺼져 있었다. 넘어진 로봇은 지형을 그대로
> 통과한다(액션 0으로 400스텝 → 지면 아래 **−1745 mm**). 손·무릎으로 땅을 짚을 수
> 없으므로 기어가기도, 일어서기도 물리적으로 불가능했다. … "네발로 기어 나가서
> 일어서면 성공"이라는 과제 정의를 판정 로직에는 반영했지만 **환경에는 기어 다닐
> 바닥이 없었다.** 그 상태에서 보상만 반복해 수정하고 있었다.

그리고 이 버그 아래에서 잰 탈출률·난이도 사다리 수치는 **전부 폐기**됐다.

**할 일**: G1의 6개 링크에 콜라이더를 추가한다 — `pelvis`, `torso_link`,
`left/right_knee_link`, `left/right_elbow_pitch_link`(또는 `elbow_roll_link`).
H1에서 전 링크(20개)를 켰을 때 16 env 접촉 버퍼가 12.7 GB였지만 **6개만 켜면 GPU
메모리 증가가 측정되지 않았고 처리량은 오히려 86 → 106 steps/s로 올랐다**(§11).

권장 방식은 USD 원본을 고치는 것이다(`usd(completed)/g1_escape.usd`로 사본 생성 후
`UsdPhysics.CollisionAPI` + 근사 형상(capsule/box) 부여). 런타임에서
`builder.add_usd` 뒤에 shape을 붙이는 방식도 가능하지만, rigid 학습 태스크와 자산을
공유하려면 USD 쪽이 깔끔하다.

> 주의: 콜라이더를 추가하면 rigid 보행 태스크의 `base_contact` 종료를 되살릴 수
> 있게 되어 **보행 정책의 종료 조건이 달라진다.** 기존 체크포인트를 그대로 쓰려면
> `g1_mars` 태스크 cfg는 건드리지 말고 탈출 env에서만 새 USD를 쓴다.

### 2.2 [과제 성립] G1은 정적으로는 Troy 껍질을 뚫지 못한다

HANDOFF §7.5ab, Spirit 바퀴 검증의 각주:

> 얇은 판은 아래가 무르면 휨-펀칭으로 재료 강도보다 훨씬 낮은 하중에서 파괴된다.
> 112 N이 그 임계 아래였다. **(같은 이유로 정적 6 kPa의 G1 발은 못 뚫었다 — §21
> 시제품에서 관찰. 관통 여부는 압력만이 아니라 접촉 형상과 하중 이력에 달려 있다.)**

접지압 수치가 문서마다 다르므로 정리한다.

| 기준 | 체중 | 발 면적 | 한 발 접지압 | 출처 |
|---|---:|---:|---:|---|
| H1 | 191.3 N | 0.0512 m² | **3.73 kPa** | 29번 `h1_foot` |
| G1 (공식 자산) | 128.0 N | 0.170 × 0.060 = 0.0102 m² | **12.5 kPa** | 29번 `g1_foot`, PROJECT_PLAN §5 |
| **G1 (우리 `g1.usd`)** | **120.0 N** (32.24 kg) | **0.2031 × 0.0655 = 0.01330 m²** | **9.02 kPa** | 본 문서 실측 |

**우리 학습 자산의 발은 공식 G1보다 크다.** 29번의 `g1_foot` 프리셋(12.5 kPa)을
그대로 인용하면 안 되고, 스윕도 우리 자산의 실제 값으로 다시 돌려야 한다.

이 절의 결론: G1은 H1보다 접지압이 2.4배 높은데도 총 하중이 작아 껍질을 못 뚫는다.
**"가벼우면 안 빠진다"도, "접지압이 높으니 잘 빠진다"도 둘 다 틀렸다.** 그래서
Step 2에서 다음 셋 중 하나(또는 조합)로 발빠짐을 성립시켜야 한다.

1. **허용 대역 안에서 껍질을 약화** — 29번 스윕은 crust_scale 2.0/3.0/4.0 ×
   crust_depth 0.01/0.02/0.03 × pocket_scale 0.1/0.2/0.3 = 27조합이 **전부 Spirit
   고착과 모순 없음**을 이미 보였다. G1이 뚫는 조합이 이 대역 안에 있으면
   **물리적 타당성을 잃지 않고** 시나리오가 성립한다. ← **1순위**
2. **동적 진입** — 정적 6 kPa로 안 뚫려도 착지 충격은 그보다 훨씬 크다.
   walk-in(`--start-mode walk_in`)에서 실제 보행 착지가 뚫는지 측정.
3. **껍질이 이미 깨진 포켓** — 21번 시제품이 쓴 방식(포켓 기둥이 표면까지 무름).
   가장 확실하지만 "왜 깨져 있는가"를 논문에서 별도로 정당화해야 한다.

### 2.3 [인터페이스] 관측 310 / 행동 37 vs H1의 256 / 19

26번은 H1 전용 상수로 하드코딩돼 있다: `ACTION_DIM = 19`, `OBS_DIM = 256`,
`SCAN_POINTS = 187`, `MARS.ISAACLAB_H1_JOINT_ORDER` 리맵.

바꿔야 할 것:

| 상수/로직 | H1 | G1 |
|---|---|---|
| `ACTION_DIM` | 19 | **37** |
| `OBS_DIM` | 256 | **310** |
| `SCAN_POINTS` | 187 | 187 (동일) |
| 관측 블록 구성 | lin/ang/grav/cmd/qpos/qvel/act/scan | **동일 순서** — 차원만 다름 |
| 관절 순서 | `ISAACLAB_H1_JOINT_ORDER` | `logs/.../params/env.yaml`의 해석된 순서에서 복사 |
| 기본 자세 | `H1_DEFAULT_JOINT_POS` | `G1_CFG.init_state.joint_pos` (hip_pitch −0.20, knee 0.42, ankle_pitch −0.23, elbow 0.87 …) |
| 행동→타겟 | (07 참조) | `default + 0.5 · a`, `clip_actions 6.0` |
| 스캔 중심 | torso | **torso_link** (G1도 스캐너가 torso_link에 붙어 있음 — 단 lin/ang/gravity는 **pelvis(root)** 기준) |
| 발 링크 | `*_ankle_link` | `*_ankle_roll_link` |
| 서 있는 root 높이 | 1.06 m | **0.72 m** (`base_height` 목표), 발바닥-골반 0.754 m (21번) |

**행동 차원이 두 배가 되는 것의 의미**: G1은 손가락까지 37 DOF다. 탈출 과제에
손 관절은 무의미하고 탐색 공간만 키운다. 팔은 **균형·지지에 실제로 쓰이므로**
반드시 남긴다(네발로 기어 나가는 탈출을 허용하는 것이 과제 정의다). 손가락
(`*_zero/one/two/three/four/five/six_link` 구동 관절)만 낮은 게인으로 기본자세에
고정하는 것을 권장한다 — 단 **관측·행동 차원은 310/37로 유지**해야 보행 정책
웜스타트가 성립한다.

---

## 3. 단계별 절차

### Step 0 — 선행 차단 요인 해소 (2~3일)

1. **콜라이더 복구** (§2.1). `g1_escape.usd` 생성.
2. **무동작 낙하 테스트** — §11의 진단을 그대로 재현한다. 액션 0으로 400스텝
   돌려 **최저 링크 z**를 기록.
   - 판정: 지면 아래 **−300 mm 이내**면 통과 (H1 수정 후 −275 mm).
   - −1000 mm를 넘으면 콜라이더가 여전히 안 붙은 것이다. 여기서 멈추고 고친다.
3. **최신 정책 export** — Mars 중력 런은 `exported/policy.pt`가 없다.
   ```powershell
   .\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py `
     --task Isaac-Velocity-Mars-G1-Play-v0 --num_envs 4 physics=newton_mjwarp --viz none `
     --checkpoint C:\IsaacLab-3.0.0-beta2\logs\rsl_rl\g1_mars\2026-07-31_14-14-14_mars_gait_v2\model_4999.pt
   ```
   > PhysX GPU에서는 관절 구동이 무시되므로 반드시 `physics=newton_mjwarp`.
4. **관절 순서 고정** — `params/env.yaml`(또는 학습 로그의
   `Resolved joint names for the action term JointPositionAction`)에서 37개 순서를
   그대로 파이썬 리스트로 박제해 `G1_JOINT_ORDER`로 저장. **Newton `add_usd` 순서와
   다를 수 있으므로 런타임에서 인덱스 리맵을 만든다** (07의 H1 리맵과 동일 패턴).

### Step 1 — G1 MPM 런타임 포팅 (3~5일)

`07_h1_mars_newton.py` → `30_g1_mars_newton.py` (CAPSTONE `terrain/scripts/`에 배치).
`mpm_learning.md` §2.2의 (a)~(e)가 그대로 이 단계다. 바꿀 곳:

| # | 항목 | 내용 |
|---|---|---|
| a | 로봇 에셋 | `download_asset("unitree_h1")` → `g1_escape.usd` 로드, root `/g1/pelvis` |
| b | 관측 310 | 순서 엄수. 노이즈 없음(play와 동일 조건) |
| c | height scan | 17×11, `root_z − hit_z − 0.5`, clip(−1,1). 07의 DTM 쌍선형 샘플 재사용 + **모래 상면 오프셋 +0.30** |
| d | 행동→PD | `default + 0.5·a`, 게인은 `G1_CFG` 액추에이터 값 (legs 150–200/5, ankles 20/2, arms 40/10) |
| e | 양방향 결합 | wrench 적용 링크 → `*_ankle_roll_link`. 클램프 **400 N/80 N·m → 250 N/50 N·m** (체중비 120/191 = 0.63) |

> **정정 (2026-08-06 측정).** 이 문서의 초판은 "포화 빈도 < 2%"를 게이트로 제시했다.
> **틀렸다.** 같은 조건에서 H1을 재보니(`26_escape_env_vec.py` + 릴리스 탈출 정책 +
> 탐색 노이즈 0.3, 8 env × 300스텝) **H1도 발 포화 17.5%**다. 즉 잘 학습되는 설정조차
> 이 게이트를 통과하지 못한다. PROJECT_PLAN §5의 "<2%"는 준정적 측정 프로토콜에서의
> 재보정 작업을 가리킨 것이지, 정책이 구동하는 런타임의 기준이 아니다.
>
> 실측 대조:
>
> | | 클램프 | 발 포화 | 비고 |
> |---|---|---:|---|
> | H1 (26, 학습 성공한 설정) | 400 N / 80 N·m | **17.5%** | 기준선 |
> | G1 (32) | 250 N / 50 N·m | **21.2%** | 유의한 차이 없음 |
>
> **따라서 포화 빈도는 G1과 H1을 가르는 지표가 아니다.** 클램프 값은 여전히 기록해야
> 하지만, 이것을 진행 게이트로 쓰지 말 것. 쓸 수 있는 게이트는 아래 Step 4의
> "학습 25회차까지 `soil_bad_per_step` = 0"이다.

**검증 게이트**: 균질 gusev_center 30 m USD 위에서 60 s 완주(넘어짐 없음),
정지 침하가 09번 판 압입 곡선의 G1 발 면적 예측과 일치.

### Step 2 — G1 발빠짐 성립 재측정 (3~5일) ★ 가장 중요

이 단계의 산출물이 **캡스톤의 새 기여**다. H1에서 이미 한 측정을 G1 접촉 형상으로
반복해 "G1의 탈출 가능/불가능 경계"를 새로 그린다.

**2-1. Troy 허용 대역 스윕 (G1 접촉)**

29번에 `g1_foot` 프리셋이 이미 있다. 다만 우리 자산의 실측값으로 갱신한다:

```python
# 29_troy_parameter_sweep.py CONTACTS 에 추가
"g1_foot_ours": {"load_n": 120.0, "half_length": 0.1016, "half_width": 0.0327},
```

```powershell
python terrain\scripts\29_troy_parameter_sweep.py --contact g1_foot_ours `
    --crust-depths 0.01 0.02 0.03 --crust-scales 2.0 3.0 4.0 `
    --pocket-scales 0.1 0.2 0.3 --output-dir terrain\output\troy_band_g1
```

**읽는 법**: `peak_fraction_of_load < 1.0`이면 그 조합에서 G1 발이 껍질을 뚫는다.
27조합 전체가 이미 Spirit 고착과 모순 없음이 확인돼 있으므로, **뚫리는 조합이
하나라도 있으면 물리적 타당성을 유지한 채 시나리오가 성립한다.** 하나도 없으면
§2.2의 2번(동적 진입) 또는 3번(깨진 껍질)으로 간다.

**2-2. G1 앵커링 수지 (탈출 가능/불가 경계)**

09/11번을 G1 발 치수로 다시 돌려 §1.2 표의 G1 열을 채운다.

```powershell
# 지지발 최대 지지력 (균질 gusev_center)
python terrain\scripts\09_mars_mpm_plate_sinkage.py `
    --plate-half-length 0.1016 --plate-half-width 0.0327 `
    --patch-length 0.8 --bin-walls --mpm-depth 0.30 --mpm-spacing 0.0125 `
    --mpm-substeps 8 --mpm-iterations 200 `
    --press-mode incremental --press-increment 0.002 --press-relax-frames 40

# 매장 깊이별 인발력
python terrain\scripts\11_mars_mpm_foot_extraction.py ...(동일 프로토콜 + G1 발 치수)
```

> **확정 측정 프로토콜을 그대로 쓴다** (PROJECT_PLAN §4.5).
> 금지 옵션: `--settle-tolerance`, `--reference-settled-surface`,
> `--kinematic-floor-layers`. 지표는 **고정 깊이 이완 반력** (교차 깊이는 비단조라 폐기).

채워야 할 표:

| 항목 | H1 | **G1 (측정할 것)** |
|---|---:|---:|
| 화성 체중 W | 191 N | 120 N |
| 지지발 최대 지지력 S | 278 N | ? |
| 매장 깊이 d에서 인발력 F(d) | 93 N @ 226 mm | ? |
| **수지 S − (W + F)** | **−6 N → 불가** | ? |
| 탈출 불가가 되는 매장 깊이 | ≈ 220 mm | ? |
| 그에 대응하는 포켓 바닥 깊이 | 270 mm | ? (침하 ≈ 0.85~0.91 × 포켓 바닥) |

**이 표가 커리큘럼을 자동으로 준다** — 수지가 넉넉한 얕은 포켓에서 시작해
수지가 음수가 되는 깊이까지 포켓 바닥을 옮기면 난이도가 단조 증가하고,
각 단계의 물리적 의미(여유 몇 N)가 명확하다.

> **외삽 금지**. 이 프로젝트에서 외삽이 틀린 것이 두 번이다(√g 스케일링,
> 깊이^3.68 인발 법칙 → 5.3배 과대). H1 값을 질량비로 스케일하지 말고 **직접 측정**한다.

### Step 3 — 발빠짐 스냅샷 생성기 (1~2일)

22번의 `make_snapshot()`을 G1용으로 이식한다(`31_g1_escape_env.py` 또는 30번에 통합).

- 사전학습 G1 보행 정책으로 포켓을 향해 걷게 하고, 매 프레임 양발 침하를 재서
  가장 깊은 프레임의 전체 상태(joint_q/qd, body_q/qd, 입자 상태 전부)를 pickle.
- `--snapshot-target-mm`은 **Step 2에서 구한 "탈출 불가 직전" 깊이**로 설정한다
  (H1은 220 mm). 이 값이 커리큘럼의 상한이다.
- `trapped_index`(어느 발이 빠졌는지)는 침하 argmax로 자동 결정.

**검증 게이트**: 스냅샷 복원 후 첫 스텝 관측이 저장 시점과 **차원 전체에서 diff 0.0**.
(§7.5ad에서 벡터 env 패리티를 잡을 때 쓴 바로 그 기준이다.)

### Step 4 — 벡터 탈출 env 포팅 (4~7일)

`26_escape_env_vec.py` → `32_g1_escape_env_vec.py`. 구조는 그대로 두고 상수와
링크 이름만 바꾸는 것이 원칙이다. **판정·보상 로직은 손대지 않는다** — H1에서
20라운드에 걸쳐 조정된 것이고, 각 항의 주석에 왜 그 값인지가 적혀 있다.

바꿔야 할 상수:

| 상수 | H1 | G1 제안 | 근거 |
|---|---:|---:|---|
| `ACTION_DIM` | 19 | **37** | |
| `OBS_DIM` | 256 | **310** | |
| `STANDING_ROOT_HEIGHT_M` | 1.06 | **0.72** | `base_height` 목표 |
| `NONFOOT_CLEARANCE_M` | 0.12 | **0.08** ← 재측정 | "무릎이 가장 낮은 비-발 링크"라는 전제가 키에 의존 |
| `WALK_SPEED_MIN_MPS` | 0.15 | **0.25** | 명령의 절반. G1 정책은 `lin_vel_x ∈ (0.5,1.5)`로 학습됨 |
| `--command-x` | 0.3 | **0.5** | 0.3은 G1 정책의 학습 분포 밖 |
| `REACTION_FORCE_LIMIT_N` | 400 | **250** ← Step 1에서 확정 | 체중비 0.63, 포화 <2% 기준 |
| `REACTION_TORQUE_LIMIT_NM` | 80 | **50** | 〃 |
| `EXIT_RADIUS_M` | 1.0 | **1.0 유지** | 과제 정의. 바꾸면 H1 결과와 비교 불가 |
| `TRAPPED_BURIAL_M` | 0.10 | Step 2 결과에 맞춰 | walk-in이 "빠지지 않고 통과"를 탈출로 오판하는 것을 막는 관문 |
| `HOLD_STEPS` | 1.5 s | **1.5 s 유지** | 확정 기준 |
| 발 링크 필터 | `*_ankle_link` | `*_ankle_roll_link` | |
| 접촉 복구 링크 | pelvis/torso/knee/elbow | 동일 6개 (G1 이름) | §11 |

**`NONFOOT_CLEARANCE_M`은 반드시 실측으로 정한다.** 이 값 하나가 `on_feet`을
결정하고, `on_feet`은 생존 보상·탈출 판정·넘어짐 종료 **셋 모두의 관문**이다.
정상 보행 300스텝 동안의 "최저 비-발 링크 높이" 분포를 찍고, **깊은 웅크림은
통과시키되 정강이로 버티는 자세는 배제**하는 값을 고른다.

**그대로 두어야 하는 것** (H1에서 비싸게 배운 것들):

```python
--mpm-iterations 200          # 120으로 낮추면 토양이 조용히 물러진다 (50: 0.96 kPa vs 200: 3.27 kPa)
--particle-speed-cap 6.0      # 기본 1.5는 250스텝 중 246스텝에서 binding — 안전망이 아니라 왜곡
--down-seconds 2.0            # 6.0으로 올리면 발산이 즉시 재현된다 (§11)
표층 8 cm loose_surface 층화   # 07/스냅샷과 재질 구성이 다르면 패리티가 깨진다 (§7.5af-4)
재활용 입자 순정 재질 복원      # 안 하면 연약 포켓이 로봇을 따라다닌다 (§7.5af-3)
sim_time 증가                  # 안 하면 USD 녹화가 전부 타임코드 0에 덮어써진다 (§7.5af-1)
```

**검증 게이트 (스모크로는 부족)** — §7.5ad의 교훈:

1. 스모크: `--num-envs 4 --smoke` 통과, 발 높이 편차 0.00 mm (env 간 물리 독립)
2. **관측 패리티**: 리셋 직후 310차원 **전부** 단일 env 대비 diff 0.0
3. **폐루프 궤적 대조**: 동일 정책·동일 시드로 단일 env와 벡터 env가 같은
   실패 프로파일을 보이는가
4. 무동작 400스텝 최저 링크 z > −300 mm (Step 0의 재확인)

### Step 5 — 학습 (1~2주)

27번은 로봇 비의존이므로 **수정 없이 그대로 쓴다**(모듈 로드 경로만 32번으로).

```powershell
python terrain\scripts\33_train_g1_escape_vec.py --num-envs 16 --iterations 800 `
    --init-checkpoint C:\IsaacLab-3.0.0-beta2\logs\rsl_rl\g1_mars\2026-07-31_14-14-14_mars_gait_v2\model_4999.pt `
    --snapshot-path terrain\output\escape_env\g1_entrapment_snapshot.pkl `
    --hold-seconds 0.5 --start-mode trapped --particle-speed-cap 6.0
```

**웜스타트의 전제**: 27번의 `--init-checkpoint`는 RSL-RL `runner.load()`를 쓰므로
**네트워크 구조가 같아야 한다.** 우리 G1 보행 정책도 CAPSTONE 탈출 트레이너도
모두 MLP [512, 256, 128] + elu이므로 구조는 맞는다. 다만 확인할 것:

- `distribution_cfg`: 탈출 트레이너는 `GaussianDistribution / std_type="scalar"`.
  우리 rigid 런의 분포 설정이 다르면 state_dict 키가 안 맞는다 → 맞춰준다.
- `obs_groups`, `empirical_normalization=False` 일치 (양쪽 다 False — OK).
- `--max-action-std 0.3` (기본값)이 **반드시 걸려야 한다.** 보행 체크포인트는
  발목을 1.15 rad로 탐색하는데, 발목이 바로 토양에 묻힌 관절이고 20 ms마다
  반 라디안씩 때리는 것이 솔브를 깨뜨린다.

**커리큘럼** (두 축):

| 축 | 값 | 비고 |
|---|---|---|
| `--hold-seconds` | 0.5 → 1.0 → **1.5** | 확정 기준은 1.5. H1은 0.5에서 95%, 1.0에서 2%로 절벽 |
| 포켓 바닥 깊이 | Step 2의 "여유 +N" 조합에서 시작 → 음수 수지까지 | `--pocket-scale`/포켓 깊이로 조절 |
| `--start-mode` | `trapped` (탈출 밀도↑) → `mixed --approach-m 1.5` (전체 아크) | 최종 평가는 mixed |

**처리량 기준선**: H1은 16 env 106 steps/s, 64 env 127. G1은 관절이 두 배라
MuJoCo 쪽이 조금 무겁지만 병목은 MPM 격자다. **16 env가 스위트스폿**
(iteration당 벽시계 ~10 s, 업데이트 빈도 우선).

**학습이 붕괴하면** — 보상부터 만지지 말고 §11의 순서를 따른다:
1. `on_feet` 비율, `diverged_per_step`, `soil_bad_per_step` 로그를 먼저 본다
   (27번이 이미 전부 로깅한다)
2. 무동작 낙하 테스트를 다시 돌린다 (환경이 깨진 것인지 정책이 깨진 것인지)
3. **MPM은 원자적 연산을 쓰므로 동일 조건 재실행이 비트 단위로 재현되지 않는다.**
   단발 A/B로 원인을 특정하려 하지 말고 학습 곡선 수십 회차를 지표로 삼는다.

### Step 6 — 평가·베이스라인·녹화 (3~5일)

28번을 그대로 쓴다.

```powershell
python terrain\scripts\34_record_g1_successes.py `
    --checkpoint <학습된 G1 탈출 정책> `
    --snapshot-path terrain\output\escape_env\g1_entrapment_snapshot.pkl `
    --wanted 3 --start-mode mixed --approach-m 1.5 `
    --down-seconds 6.0 --particle-speed-cap 6.0
```

**베이스라인 3자 비교** (제안서 §3.2의 "검증 기준 부재" 대응이자 H1과 동일한 틀):

| 비교군 | 내용 | H1 참고값 |
|---|---|---|
| ① 무수정 보행 정책 | `model_4999.pt` 그대로 탈출 과제 투입 | 15.6% (5/32) |
| ② 휴리스틱 스크립트 | 단순 수직 인발 | — (앵커링 수지상 실패 예상) |
| ③ 학습된 탈출 정책 | Step 5 산출물 | 37.5% (12/32) |

**보고 규칙** (H1에서 정한 것을 그대로 따른다):
- 결정론(탐색 노이즈 제거) 32 에피소드 기준으로 잰다.
- **전체 에피소드 기준**과 **실제로 빠진 에피소드 기준** 두 수치를 **구분해서** 쓴다
  (walk-in에서는 포켓을 지나쳐 버리는 개체가 나온다. H1: 12/32 = 37.5% vs 12/25 = 48%).
- 커리큘럼 중간 단계(`--hold-seconds 0.5` 등)의 성공률을 **확정 기준 수치로
  인용하지 않는다.**

---

## 4. 파일 배치 제안

CAPSTONE 번호 체계를 이어간다(29까지 사용 중). Mars_Terrain 쪽에는 문서만 둔다.

| 새 파일 | 원본 | 위치 |
|---|---|---|
| `g1_escape.usd` | `g1.usd` + 콜라이더 6개 | `Mars_Terrain/usd(completed)/` |
| `30_g1_mars_newton.py` | `07_h1_mars_newton.py` | `CAPSTONE/terrain/scripts/` |
| `31_g1_escape_env.py` | `22_escape_env.py` | 〃 |
| `32_g1_escape_env_vec.py` | `26_escape_env_vec.py` | 〃 |
| `33_train_g1_escape_vec.py` | `27_train_escape_vec.py` (거의 그대로) | 〃 |
| `34_record_g1_successes.py` | `28_record_successes.py` | 〃 |
| `G1_escape.md` (본 문서) | — | `Mars_Terrain/` |

> 29번 `CONTACTS`에 `g1_foot_ours` 추가는 기존 파일 수정이다. H1 결과 재현성을
> 위해 기존 프리셋은 지우지 말고 **추가만** 한다.

---

## 5. 반드시 지킬 것 — H1에서 비싸게 배운 체크리스트

각 항목은 CAPSTONE이 실제로 며칠씩 잃고 얻은 것이다.

- [ ] **보상을 고치기 전에 그 행동이 물리적으로 가능한지 먼저 확인한다.**
      판정 기준을 바꿀 때 환경이 그 기준을 지원하는지 확인하지 않은 것이 §11의
      근본 실수였다. "기어 나가면 성공"인데 기어 다닐 바닥이 없었다.
- [ ] **종료 조건을 완화할 때는 새로 도달 가능해지는 물리 상태를 먼저 열거**하고
      각각에 수치 안전장치가 있는지 확인한다 (§10).
- [ ] **관문이 되는 양은 반드시 로그에 남긴다.** `on_feet`은 생존 보상과 탈출
      판정 양쪽의 관문인데 로그에 없어서 "발로 서 있는가"와 "페널티가 큰가"를
      구분할 수 없었다.
- [ ] **측정 범위를 2배 넘겨 외삽하지 않는다.** 이 프로젝트에서 두 번 틀렸다.
- [ ] **MPM은 재현되지 않는다.** 단발 A/B로 원인을 특정하지 말 것 (동일 설정에서
      발산 201건 대 0건이 관측됐다).
- [ ] **벡터화 검증은 스모크·격리로 부족하다.** 학습된 정책의 관측 분포와 보상
      지형까지 단일 env와 수치 대조한다 (§7.5ad).
- [ ] **반력 클램프의 포화 빈도를 잰다.** 상한에 계속 걸려 있으면 그것은 안전망이
      아니라 물리 왜곡이다 (입자 속도 캡 1.5가 250스텝 중 246스텝 binding).
- [ ] **버그 아래에서 잰 값은 전부 폐기한다.** 미련을 두면 논문에 잘못된 수치가 남는다.

---

## 6. 검증 게이트 요약

각 Step은 아래를 통과해야 다음으로 간다. 실패 시 **다음 단계로 진행하지 않는다.**

| Step | 게이트 | 통과 기준 |
|---|---|---|
| 0 | 무동작 400스텝 낙하 | 최저 링크 z > **−300 mm** |
| 0 | 정책 export | `exported/policy.pt` 생성, 310→37 형상 확인 |
| 1 | 균질 지형 보행 | gusev_center 60 s 완주, 넘어짐 0 |
| 1 | 반력 클램프 | ~~포화 < 2%~~ **철회** — H1도 17.5%다. 값은 기록만 하고 게이트로 쓰지 않는다 |
| 1 | 관절 armature | **0이 아닐 것.** `G1_CFG`의 0.01(사지)/0.001(손가락). 포팅에서 빠뜨리면 14 g 손가락 링크가 로터 관성 없이 40 N·m/rad로 구동된다 |
| 5 | 학습 안정성 | **25회차까지 `soil_bad_per_step` = 0** ← 실제로 유효한 게이트 |
| 2 | Troy 대역 스윕 | G1 발이 뚫는 조합이 27조합 중 **1개 이상** |
| 2 | 앵커링 수지 | S − (W + F) 부호가 포켓 깊이로 **뒤집힘** (과제 성립) |
| 3 | 스냅샷 복원 | 관측 310차원 전부 **diff 0.0** |
| 4 | env 격리 | 16 env 발 높이 편차 **0.00 mm** |
| 4 | 패리티 | 단일 env와 폐루프 궤적 **동일 실패 프로파일** |
| 5 | 학습 건전성 | `diverged_per_step ≈ 0`, 에피소드 길이가 상한 근처로 증가 |
| 6 | 최종 | 결정론 32 에피소드, 베이스라인 ①보다 **유의하게 높은 탈출률** |

---

## 7. 리스크

| 리스크 | 징후 | 대응 |
|---|---|---|
| **G1이 어떤 허용 조합에서도 안 빠진다** | Step 2-1에서 27조합 전부 `peak ≥ load` | 동적 진입 측정 → 그래도 안 되면 "깨진 껍질" 시나리오로 전환하고 그 가정을 논문에 명시 |
| **탈출이 자명하게 쉽다** | Step 2-2 수지가 어느 깊이에서도 양수 | 체중이 작아 앵커링 여유가 큰 것. 포켓을 더 깊게/무르게 (단, Spirit 정합성 유지) |
| 콜라이더 추가로 처리량 급감 | 16 env가 50 steps/s 이하 | 6개 링크만 켰는지 확인. H1은 오히려 빨라졌다(86→106) |
| 37 DOF 탐색 공간 | 학습 정체, action std 발산 | 손가락 관절 고정, `--max-action-std` 하향(0.2) |
| 보행 정책 웜스타트 실패 | `runner.load()` state_dict 키 불일치 | `distribution_cfg`·`obs_groups`를 rigid 런 설정과 맞춤 |
| 명령 분포 밖 운용 | 정책이 즉시 넘어짐 | `--command-x`를 0.5(학습 하한)로. 필요하면 0.3~1.5 커리큘럼으로 rigid 재학습 |
| RAM/VRAM | `tasklist` 감시 | 단일 env도 24 GB+ 사례 있음 (`mpm_learning.md` §2.5) |
| MPM 프로세스가 안 죽음 | 재실행 시 CUDA 점유 | `tasklist \| findstr python` → `taskkill /F /PID <pid>` |

---

## 8. 빠른 참조

```powershell
# --- 환경 ---
$env:OMNI_KIT_ACCEPT_EULA = "YES"
.\isaaclab.bat -p -m pip install fast_simplification   # 없으면 지형 콜라이더가 bbox로 퇴화

# --- Step 0: 정책 export ---
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task Isaac-Velocity-Mars-G1-Play-v0 `
  --num_envs 4 physics=newton_mjwarp --viz none `
  --checkpoint C:\IsaacLab-3.0.0-beta2\logs\rsl_rl\g1_mars\2026-07-31_14-14-14_mars_gait_v2\model_4999.pt

# --- Step 2: Troy 허용 대역 (G1 접촉) ---
python terrain\scripts\29_troy_parameter_sweep.py --contact g1_foot_ours --output-dir terrain\output\troy_band_g1

# --- Step 4: 벡터 env 스모크 ---
python terrain\scripts\32_g1_escape_env_vec.py --num-envs 4 --smoke

# --- Step 5: 학습 ---
python terrain\scripts\33_train_g1_escape_vec.py --num-envs 16 --iterations 800 `
  --init-checkpoint <rigid 보행 model_4999.pt> --hold-seconds 0.5 --particle-speed-cap 6.0

# --- Step 6: 성공 녹화 + 보기 ---
python terrain\scripts\34_record_g1_successes.py --checkpoint <탈출 정책> --wanted 3 `
  --start-mode mixed --approach-m 1.5 --down-seconds 6.0 --particle-speed-cap 6.0
python terrain\scripts\25_open_usd.py <녹화.usd>      # Isaac Sim GUI, 하단 ▶ 눌러야 움직임

# --- 참고: H1 원본 데모 ---
python terrain\scripts\26_escape_env_vec.py --num-envs 4 --smoke
```

---

## 9. 산출물 체크리스트

- [ ] `g1_escape.usd` (콜라이더 6개 복구) + 무동작 낙하 테스트 로그
- [ ] 최신 Mars 중력 정책의 `exported/policy.pt` + `G1_JOINT_ORDER` 박제
- [ ] `30_g1_mars_newton.py` + 반력 클램프 포화 빈도 측정값
- [ ] **G1 Troy 허용 대역 스윕 CSV** (본 단계의 새 기여)
- [ ] **G1 앵커링 수지표** (S, F(d), 탈출 불가 임계 깊이, 포켓 바닥 대응값)
- [ ] `g1_entrapment_snapshot.pkl` + 관측 패리티 diff 0.0 로그
- [ ] `32_g1_escape_env_vec.py` + 패리티 검증 4종 통과 기록
- [ ] 학습 체크포인트 + `--hold-seconds` 커리큘럼별 성적표
- [ ] 베이스라인 3자 비교표 (전체 기준 / 실제로 빠진 에피소드 기준 병기)
- [ ] 성공 녹화 USD 3건 + 탈출 시각
- [ ] **H1 vs G1 비교 분석** — 같은 지형·같은 판정에서 두 로봇의 탈출 경계가
      어떻게 다른가. 이것이 이 확장의 최종 논지다.
