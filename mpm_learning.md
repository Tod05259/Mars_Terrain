# MPM 지형 보행학습 가이드 (G1 × Newton Implicit-MPM 화성 토양)

> **목표**: 우리가 rigid 지형(Gale)에서 학습시킨 G1 보행 정책을, CAPSTONE의
> Newton implicit-MPM 동적 토양 환경에서 **평가 → 적응 → (소수-env) 미세조정**하는
> 구체적 절차.
>
> **전제가 되는 결론** (2026-07-31 검토): MPM 위에서 512-env 병렬로 *처음부터*
> 학습하는 것은 현재 불가하다 (배치드 MPM 미지원, 단일 env도 24 GB+ RAM 실증,
> 처리량 ~1000배 격차 — 낙관적으로 실시간이어도 61M 스텝 ≈ 14일). 올바른 설계는
> **"학습은 rigid에서, 적응·검증은 MPM에서"** 하는 sim-to-sim 전이이며, CAPSTONE
> 파이프라인(pretrained H1 + MPM 런타임)이 이미 이 구조로 동작 중이다.
>
> **참조 문서**
> - 우리 쪽: `Mars_Terrain/scripts/G1_MARS_RL_GUIDE.md` (G1 rigid 학습 구현),
>   `Mars_Terrain/scripts/ISAACLAB_CUSTOM_ROBOT_RL_PLAYBOOK.md` (일반 방법론)
> - CAPSTONE 쪽: `CAPSTONE/terrain/mars_terrain/mars_real_terrain_env/MARS_TERRAIN_PIPELINE_SUMMARY.md`
>   (MPM 물성·결합 구조), `SESSION_HANDOFF.md` (버그·RAM 이슈 이력)

---

## 0. 두 파이프라인의 현재 상태 (출발점)

| | 우리 (IsaacLab-3.0.0-beta2) | CAPSTONE (MPM 런타임) |
|---|---|---|
| 로봇 | Unitree G1 (37관절, `g1.usd`) | Unitree H1 (19관절, pretrained) |
| 지형 | Gale 100 m rigid (winding 수정 완료) | HiRISE DTM 7지역 30/100 m + MPM 모래 창 |
| 물리 | Newton MJWarp, 512 envs, 17.6k steps/s | Newton MuJoCo(강체) + SolverImplicitMPM(모래), 단일 로봇 |
| 학습 | rsl_rl PPO 5000 iters ≈ 66분, 보행 성공 | 학습 없음 — pretrained 정책 배포/평가 전용 |
| 결합 | — | explicit two-way: MPM impulse → 발목 wrench (400 N/80 N·m clamp, 1스텝 지연) |
| 중력 | 지구 −9.81 (Mars 옵션 주석 처리 상태) | **Mars −3.721** |
| MPM 창 | — | 로봇 중심 1×1×0.08 m, 4,800 입자, 50 Hz 제어, substep 2, implicit iter 200 |

핵심 파일:
- 우리 태스크: `IsaacLab-3.0.0-beta2/source/isaaclab_tasks/.../velocity/config/g1_mars/`
- 우리 체크포인트: `IsaacLab-3.0.0-beta2/logs/rsl_rl/g1_mars/<run>/model_*.pt`
  (예: `2026-07-24_14-10-37/model_3850.pt` = 보행 확보 버전)
- CAPSTONE 런타임: `CAPSTONE/terrain/scripts/07_h1_mars_newton.py` (1,362줄, `H1MarsNewton` 단일 로봇 루프)
- 지역별 MPM 물성: `04_build_env_usd.py`의 `TERRAIN_CONFIGS` + `REGIONAL_MPM_CALIBRATION.md`

---

## 1. 전체 로드맵

```
Phase 0  G1 정책 zero-shot MPM 평가            ~1일     학습 없음, 이식만
   │     (07_h1_mars_newton.py를 G1용으로 포팅)
Phase 1  soft-soil 도메인 랜덤화 rigid 재학습    1~2일    우리 파이프라인에서 1시간 학습
   │     (+ Mars 중력 전환) → MPM 재평가
Phase 2  소수-env(4~8) MPM 미세조정             1~2주    rsl_rl VecEnv 래퍼 구축
   │     (rigid 정책 warm-start, 3~6M 스텝)
Phase 3  배치드 MPM 병렬 학습                   장기     Newton 스택 연구 과제
```

각 Phase는 이전 Phase의 산출물을 소비한다. **Phase 0의 평가 지표가 개선 여부의
기준선**이므로 반드시 먼저 만든다.

---

## 2. Phase 0 — G1 정책 zero-shot MPM 평가 (~1일)

### 2.1 정책 내보내기 (TorchScript)

rsl_rl `play.py`를 한 번 실행하면 액터가 TorchScript로 자동 내보내진다
(`logs/rsl_rl/g1_mars/<run>/exported/policy.pt`). raw Newton 스크립트에서
`torch.jit.load`로 불러 쓰면 액터 클래스를 재구현할 필요가 없다.

```powershell
# IsaacLab 루트에서 — 재생 겸 policy.pt 내보내기
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py `
  --task Isaac-Velocity-Mars-G1-Play-v0 --num_envs 4 physics=newton_mjwarp --viz none `
  --checkpoint C:\IsaacLab-3.0.0-beta2\logs\rsl_rl\g1_mars\<run>\model_XXXX.pt
```

우리 액터는 `obs_normalization=False`(러닝 스탯 없음)라 이식이 단순하다:
**입력 310 → MLP [512,256,128] elu → 출력 37** (평가 시 mean 사용, 샘플링 없음).

### 2.2 평가 스크립트 작성: `11_g1_mars_mpm_eval.py`

`CAPSTONE/terrain/scripts/07_h1_mars_newton.py`를 복제해 G1용으로 수정한다.
바꿀 곳은 5군데다.

**(a) 로봇 에셋** — H1 대신 우리 `g1.usd`(`Mars_Terrain/usd(completed)/g1.usd`)를
`builder.add_usd`로 로드. ArticulationRoot는 `/g1/pelvis`. 발 충돌체는 이미
convexHull이라 그대로 동작한다.

**(b) 관측 벡터 (310차원, 순서 엄수)** — 학습 때와 완전히 같은 순서로 구성:

| 순서 | 항목 | 차원 | 비고 |
|---|---|---:|---|
| 1 | base_lin_vel (body frame) | 3 | pelvis 선속도를 body frame으로 회전 |
| 2 | base_ang_vel (body frame) | 3 | |
| 3 | projected_gravity | 3 | 중력 단위벡터의 body frame 사영 |
| 4 | velocity_commands | 3 | [vx, vy, wz] — 평가 시 고정값 (예: 0.8, 0, 0) |
| 5 | joint_pos − default | 37 | **관절 순서 = 학습 런의 Newton 순서** |
| 6 | joint_vel | 37 | 〃 |
| 7 | last_actions | 37 | 직전 정책 출력 (스케일 전 raw) |
| 8 | height_scan | 187 | 아래 (c) |

- 관절 순서는 학습 로그의 `Resolved joint names for the action term JointPositionAction`
  줄(또는 `logs/.../params/env.yaml`)에서 그대로 복사한다. **PhysX 순서와 다르므로
  반드시 학습 런 기록에서 가져올 것.**
- default joint pos는 `isaaclab_assets/robots/unitree.py`의 `G1_CFG.init_state.joint_pos`
  (hip_pitch −0.20, knee 0.42, ankle_pitch −0.23, elbow 0.87, …).
- 관측 노이즈는 넣지 않는다 (play와 동일 조건).

**(c) height scan** — GridPattern 1.6×1.0 m, 해상도 0.1 m(17×11=187), yaw 정렬,
torso 기준. 값 = `root_z − hit_z − 0.5`, **clip(−1, 1)**. 07 스크립트가 이미
DTM 대상 height scan을 구현하므로 패턴/포맷만 우리 것으로 맞춘다.
레이는 rigid DTM만 본다(MPM 침하는 미반영) — 깊이 8 cm 창이라 허용.

**(d) 행동 → PD 타겟** — `target = default_joint_pos + 0.5 * action`
(action_scale 0.5, use_default_offset). PD 게인·effort limit은 `G1_CFG` 값을
Newton MuJoCo 관절 타겟 게인으로 설정 (legs 150–200/5, ankles 20/2, arms 40/10,
effort 300/20 N·m — 우리 진단 로그의 관절 테이블 참조). 50 Hz 제어(20 ms)로
07 스크립트의 CONTROL_DT와 일치한다.

**(e) 양방향 결합 재타게팅** — impulse→wrench 적용 링크를 H1 발목 → G1
`left/right_ankle_roll_link`로 변경. 힘 클램프는 질량비(G1 ≈ 35 kg vs H1 ≈ 47 kg)로
축소: **300 N / 60 N·m** 시작.

### 2.3 중력 불일치 처리 (중요)

우리 정책은 지구 중력(−9.81)에서 학습됐고 MPM 런타임·물성 보정은 Mars(−3.721)다.
zero-shot 첫 실행은 **두 조건 모두** 돌려 기준선을 잡는다:

1. **Earth run**: 런타임 중력을 −9.81로 바꿔 정책 조건과 일치시켜 "결합 자체"를 검증
   (단, MPM 물성은 Mars 보정값이라 침하가 과대평가될 수 있음을 기록)
2. **Mars run**: −3.721 그대로 — 정책이 가벼워진 동역학에서 얼마나 버티는지 확인

→ Phase 1에서 Mars 중력으로 재학습하면 이 불일치가 사라진다.

### 2.4 평가 프로토콜과 지표

7개 지역 × 30 m USD(`crop30m/*.usd` — terrain에 collision 있음. **add_stone_version
USD는 CollisionAPI가 없으므로 사용 금지**)에서 각 60 s:

- **MPM 스탠딩 테스트** (rigid에서 쓰던 방법의 MPM판): 액션 0 → 침하 깊이가
  해당 지역 plate 보정 곡선과 일치하는지 (결합 sanity check)
- 보행: 전진 0.5 / 0.8 m/s 명령 → 속도 추종 오차, 넘어짐률(전도 시각),
  침하 깊이 통계, 발목 반력 클램프 도달 빈도, MPM 입자 recycle 통계
- 산출물: 지역(물성) × 중력 × 명령속도 매트릭스 CSV

**판정**: gusev_center(중간 물성)에서 60 s 완주하면 Phase 1로. mawrth_center
(무른 clay)에서 실패하는 건 정상 범위 — Phase 1·2의 개선 대상이다.

### 2.5 운영 주의사항 (CAPSTONE 이력에서)

- `fast_simplification` 미설치 시 지형 콜라이더가 bounding box로 퇴화해 **모래가
  지형을 관통**한다: `isaaclab.bat -p -m pip install fast_simplification`
- MPM 프로세스는 TaskStop으로 안 죽는 경우가 있음 → `tasklist` 확인 후
  `taskkill /F /PID <pid>`
- RAM 감시: 단일 env도 코드 경로에 따라 24 GB+ 사례 있음 (raw Newton 루프는 가벼움,
  IsaacLab ManagerBasedRLEnv와 MPM 동시 구동이 무거웠음)

---

## 3. Phase 1 — soft-soil 도메인 랜덤화 rigid 재학습 (1~2일)

MPM 없이 MPM과의 격차를 줄이는 정공법. 우리 `g1_mars` 태스크 cfg만 수정해
1시간 재학습한다.

### 3.1 `mars_env_cfg.py` 수정 항목

```python
# (1) Mars 중력 활성화 — 주석 해제 (MPM 런타임/물성 보정과 일치시킴)
self.sim.gravity = (0.0, 0.0, -3.71)

# (2) 마찰 랜덤화: CAPSTONE 7지역 보정 friction 0.44~0.84를 포괄하는 범위
self.events.physics_material.params["static_friction_range"] = (0.4, 0.9)
self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.75)

# (3) 외란 강화 — 무른 지반의 예측 불가 반력을 흉내
#     push_robot 주기 단축(10~15s → 6~10s) 또는 속도 범위 확대
self.events.push_robot.interval_range_s = (6.0, 10.0)
```

주의:
- Mars 중력 전환은 **처음부터 재학습**을 권장 (중력은 보행 동역학의 근간이라
  fine-tune보다 fresh가 깔끔). 기존 지구-중력 정책은 비교군으로 보존.
- Mars 중력에서는 `base_height` 목표(0.72)나 PD 게인은 그대로 둬도 되지만,
  착지가 "떠 보이는" 걸음이 나오면 `feet_air_time` threshold를 낮추는(0.4) 역조정 여지.
- (고급, 선택) 발 접촉 강성 랜덤화: CAPSTONE ke 20k~120k N/m 지역차를 흉내내려면
  Newton shape 접촉 파라미터 랜덤화가 필요한데 현재 이벤트로는 미지원 — Phase 2에서
  MPM이 직접 제공하므로 생략 가능.

### 3.2 실행과 판정

```powershell
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py `
  --task Isaac-Velocity-Mars-G1-v0 --num_envs 512 physics=newton_mjwarp --headless
```

학습 후 Phase 0 평가를 **동일 프로토콜로 재실행** → 지역별 매트릭스에서
넘어짐률/추종 오차가 개선됐는지 정량 비교. 이 비교표가 "도메인 랜덤화만으로
충분한가, Phase 2가 필요한가"를 결정한다.

---

## 4. Phase 2 — 소수-env MPM 미세조정 (1~2주, 필요시)

Phase 1로도 무른 지역(mawrth, jezero)에서 격차가 남을 때만 진행한다.
**처음부터 학습이 아니라 rigid 정책 warm-start 미세조정**이므로 3~6M 스텝이면 되고,
4~8 env × ~50 steps/s ≈ 200~400 steps/s로 **2~8시간** 규모다.

### 4.1 아키텍처: `G1MarsMPMVecEnv`

`11_g1_mars_mpm_eval.py`(Phase 0)를 rsl_rl `VecEnv` 인터페이스로 감싼다:

```
G1MarsMPMVecEnv (num_envs=4~8)
├─ Newton Model: G1 × N (월드 복제) + rigid DTM
├─ MPM: env별 독립 창 (1×1×0.08 m, 4,800 입자 × N)
├─ step(actions[N,37]):
│    1) PD 타겟 설정 → rigid solver step
│    2) env별 MPM substep ×2 (현재는 파이썬 순회 — 4.3 참고)
│    3) impulse 수집 → env별 발목 wrench (클램프)
│    4) obs[N,310], reward[N], done[N] 반환
├─ reset(env_ids):
│    로봇 pose/joint 리셋 + 해당 env의 MPM 창 재생성
│    (입자 재샘플 + 30×5 ms settling) — "reset isolation"
└─ 보상/종료: g1_mars 것 이식 + NaN 안전망
```

이는 CAPSTONE 파이프라인 문서 9장 TODO **5번(multi-env MPM ownership, reset
isolation, batched pose transfer)과 6번(보행 reward/reset lifecycle)**을 구현하는
것과 동일하다 — 완료 시 CAPSTONE 쪽 기여로도 반영 가능.

### 4.2 학습 설정 (기존 정책 보존이 최우선)

```python
# agents cfg (미세조정 전용)
resume = True                      # Phase 1 체크포인트에서 warm-start
learning_rate = 1.0e-4             # 낮게 고정 (adaptive 대신) — 걸음새 파괴 방지
desired_kl = 0.008                 # KL 제약 강화
entropy_coef = 0.003               # 탐색 축소 (이미 걷는 정책의 국소 적응)
num_steps_per_env = 48             # env 수가 적으니 rollout 길이로 배치 확보
max_iterations = 300~800
clip_actions = 6.0                 # 유지 (필수)
```

보상은 g1_mars 그대로 + 선택적으로 침하 페널티(`-c · sinkage_depth`)나 에너지
페널티를 추가해 "모래에서 발을 빼는" 행동을 유도할 수 있다.

### 4.3 알려진 난점 (미리 계획할 것)

| 난점 | 대응 |
|---|---|
| MPM solver의 멀티월드 지원 여부 불명 | 우선 env별 독립 solver를 파이썬 순회(단순·확실) → 처리량 부족하면 Newton `SolverImplicitMPM`의 다중 창 배치 가능성 조사 |
| 파이썬 결합 루프가 env 수에 비례해 느려짐 | impulse 수집·wrench 계산을 warp 커널/텐서 배치로 벡터화 |
| RL 탐색 액션의 impulse 스파이크 → 반력 발산 | wrench 클램프 유지 + 관측/보상 `torch.isfinite` 마스크(우리 레이-inf 사례와 동일 패턴) + 발산 env 즉시 리셋 |
| 리셋 시 입자 잔류 오염 | env별 입자 풀 완전 재샘플 + settling, 창 이탈 입자 recycle은 기존 로직 재사용 |
| CUDA graph 미적용으로 스텝 오버헤드 | 미세조정 규모(수 시간)에선 감수. graph 캡처는 Phase 3 과제 |
| RAM | env 수를 4부터 늘리며 `tasklist`로 감시, 24 GB 상한 경험 유념 |

### 4.4 검증

- 미세조정 전/후를 Phase 0 프로토콜로 재평가 (특히 무른 지역 넘어짐률)
- **회귀 확인**: rigid Gale 환경에서도 재평가해 단단한 지형 성능이 망가지지
  않았는지 확인 (망가졌으면 LR·KL을 더 조이고 재시도)

---

## 5. Phase 3 — 배치드 MPM 병렬 학습 (장기 과제)

수백 env 동시 학습은 다음이 갖춰져야 한다: 다중 월드 입자를 한 번에 푸는 배치드
implicit-MPM, 결합 경로의 완전 커널화, CUDA graph 캡처, env당 입자 수 축소
(창 축소·해상도 완화 + 12.5→25 mm 수렴성 재검증). 현 Newton 스택에 없는 기능이
포함되므로 캡스톤 범위에서는 **Phase 0~2 결과(전이 + 소수-env 적응)로 마무리하고,
Phase 3는 향후 연구로 명시**하는 것을 권장한다.

---

## 6. 산출물 체크리스트

- [ ] Phase 0: `CAPSTONE/terrain/scripts/11_g1_mars_mpm_eval.py` (07 포팅)
- [ ] Phase 0: `exported/policy.pt` (play.py로 생성), 지역×중력×속도 평가 CSV
- [ ] Phase 1: `mars_env_cfg.py` 수정 (Mars 중력 + 마찰 DR + push 강화), 재학습 런
- [ ] Phase 1: Phase 0 대비 개선 비교표
- [ ] Phase 2(필요시): `G1MarsMPMVecEnv` + 미세조정 agents cfg + 전/후 평가
- [ ] 각 단계의 발견을 `ISAACLAB_CUSTOM_ROBOT_RL_PLAYBOOK.md` 트러블슈팅에 역반영

## 7. 빠른 참조 — 자주 쓸 명령

```powershell
# 정책 내보내기(play 1회 실행으로 exported/policy.pt 생성)
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task Isaac-Velocity-Mars-G1-Play-v0 --num_envs 4 physics=newton_mjwarp --viz none --checkpoint <model.pt 전체경로>

# MPM 의존성
.\isaaclab.bat -p -m pip install fast_simplification

# CAPSTONE MPM 데모(참고용 — 결합·물성 동작 확인)
isaaclab.bat -p terrain/scripts/05c_mpm_drop_object_demo.py --terrain gusev_center --object sphere --num-steps 90 --headless

# Phase 1 재학습
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Velocity-Mars-G1-v0 --num_envs 512 physics=newton_mjwarp --headless

# 죽지 않는 MPM 프로세스 정리
tasklist | findstr python
taskkill /F /PID <pid>
```
