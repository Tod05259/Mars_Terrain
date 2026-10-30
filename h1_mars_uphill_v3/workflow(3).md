# H1 화성 MPM 오르막 학습 — Workflow (3)

작성일: 2026-10-11
상태: **승인됨 (2026-10-11, 사용자 "확인했어. 진행해줘.")** — 진행 결과는 `results/status.md`에 기록. 결정 D1~D4는 같은 날 사용자 답변으로 확정(§7).
기준 문서: CAPSTONE `terrain/TEAM_GUIDE.md` (커밋 `fbaf458`, 마지막 항목 2026-10-11 20:04) 우선, 보조로 `terrain/TEAM_SETTINGS_SHARE.md`.

---

## 0. 요약

- **목표**: Unitree H1이 MPM 모래(팀 표준 불규칙 베드) 위의 실제 화성 경사를 두 발로 올라가는 정책을 만든다. 팀 FSM의 ASCEND 모드(TEAM_GUIDE §6, 팀원 2)에 해당한다.
- **방식(D1 확정)**: 팀이 일어서기에서 검증한 경로를 그대로 따른다.
  Isaac Lab(4096 env, 강체 지형 + **모래 반력 대리 모델**, 경사 지형 생성기)에서 **처음부터 학습** →
  **MPM 모래 환경(`26_escape_env_vec.py --task slope_ascend`)에서 판정·시연**.
- **근거**: MPM 안에서 PPO를 돌린 시도는 전부 실패했다(이전 상승 시도 2건의 최신 평가 0/6·0/3, 팀의 일어서기 MPM 이어 학습 A1·A2·S5). 같은 팀이 랩 + 반력 모델 백지 학습(V7)으로 MPM 체인 183/185 = 98.9 %를 냈고, 반력 모델만 뺀 대조(v7r)는 5 %였다(TEAM_GUIDE 2026-10-11).
- **산출물**: 팀원이 폴더를 받아 README 순서대로 실행하면 (1) 랩에서 정책 재생, (2) MPM 경사에서 정책 재생, (3) 채점 결과·영상을 바로 볼 수 있게 한다.

---

## 1. 목표와 성공 기준

| 항목 | 내용 |
|---|---|
| 로봇 | Unitree H1, 관측 256 / 행동 19 / 액터 [512, 256, 128] ELU — **팀 인터페이스 변경 금지**(TEAM_SETTINGS A2) |
| 명령 | `[v_x, v_y, yaw_rate]`, 헤딩은 외부 PI(A3). 명령 0 = STAND |
| 판정 환경 | MPM: `26 --task slope_ascend`, 화성 중력, 50 Hz, 사지 접촉(`limbs`, 체인 판정과 맞출 경우 `all`), 팀 표준 모래 0.25 ± 0.12 m seed 11, `--soil-material auto` |
| 판정 규칙 | TEAM_GUIDE §7.4 / A7: `--episode-seconds` ≥ 실행 길이, `env._steps = 0`, 전도는 `info["fallen"]`로만, 51의 물리 유효성(반력 포화·공중·침하) 통과분만 인용, 최종 평가는 `--scan-source surface` |
| 성공 1회 | 26 `slope_ascend` 정의: 시작점에서 경사 방향으로 `--descend-target-m`(기본 3.0 m) 등반 + 발 지지, 최고점 대비 되밀림 < `--arrest-runway-m`(1.0 m), 전도 없음 |
| 추가 확인 | 등반 뒤 STAND(워커 명령 0, 3 s)로 넘겨 재전도 없음(B1·B2 핸드오프) |

**합격선(2026-10-12 사용자 확정, 기준선 it25: 16.8° 6/10, 20° 1/10, 22.5° 0/10 — `results/status.md`)** — 같은 체크포인트를 40회 평가, Wilson 95 % 구간 함께 보고. 성공 1회 = 도달 + 명령 0으로 3 s 유지 + 물리 유효(판정 정의는 `scripts/eval_mpm.py`):

| 베드 | GO | 계속 | NO-GO |
|---|---|---|---|
| 주 경사 16.8° (워커가 1.2 m 후 전도하는 경사) | ≥ 32/40 (80 %) | 20~31 | ≤ 19 |
| 20.0° | ≥ 24/40 (60 %) | 보고 | 보고 |
| 22.5° (안식각 30.8°의 73 %) | 보고만 | | |
| 2차 흙 조건 S2 "전형적 흙" (16.8°) | ≥ 28/40 | 보고 | 보고 |

기준선: 기본 워커 `walk_irregular_it25.pt`를 같은 조건으로 먼저 잰다(단계 2). 합격선은 학습 전에 고정하고, 결과를 본 뒤 바꾸지 않는다.

---

## 2. 참고 자료와 이번 작업에서의 역할

경로는 `C:\Users\2hj05\repos\CAPSTONE\terrain` 기준. CAPSTONE은 **읽기 전용**으로만 사용한다.

| 자료 | 역할 |
|---|---|
| `TEAM_GUIDE.md` §0·1·6 | 공용 환경 26, 팀원 2 = `--task slope_ascend`, uphill 스냅샷, `limbs` 접촉 |
| `TEAM_GUIDE.md` §3 (2026-10-02~10-11) | 실패 원인 감사(학습률·성공 혼입·리셋 동기화·포켓), 랩 + 반력 모델 전환 근거(V7·v7r·V8), 랩 보행기 `walk1`의 물리 무효 걸음새 |
| `TEAM_GUIDE.md` §7·§8, `TEAM_SETTINGS_SHARE.md` A1~A7·B1~B3 | 관측·명령·모래·평가 규칙, FSM 진입/종료 상태, 제출물 목록 |
| `docs/walk_lab_design_2026-10-11.md` | 랩 보행 과제 설계 — 이번 랩 경사 과제의 직접 출발점 |
| `docs/soil_surrogate_v7_plan_2026-10-10.md`, `docs/getup_lab_audit_2026-10-08.md` | 반력 모델 설계·보정, 랩↔MPM 규격 차이 점검 목록 |
| `getup_lab/` (`walk_env_cfg.py`, `walk_mdp.py`, `soil*.py`, `getup_env_cfg.py`, `agent_cfg.py`) | 재사용할 Isaac Lab 과제 코드(로봇·DC 모터·스캔·반력 모델·지반 혼합) |
| `scripts/26`·`27` | MPM 판정 환경 / (필요 시) MPM 학습기 |
| `scripts/22`, `33`, `build_site_snapshots.sh` | 경사 사이트 조사, 팀 표준 모래 스냅샷 생성 |
| `scripts/35`, `51`, `55`, `63`, `28`·`38` | 경사 평가, 물리 유효성, FSM 하네스, 체인, 녹화·렌더 |
| `scripts/82`, `84` | 랩 안정성 관문(G4), 랩 보행 채점(경사판 채점의 원형) |
| `release/walk/` (`walk_irregular_it25.pt`, `flat_snap_irregular_s11.pkl`), `release/fsm/stand_pose_orig_it25_3s.npz` | 기준선 워커, 팀 표준 모래, STAND 자세 |
| `output/escape_env/uphill_snap_{16.8,20.0,22.5}deg.pkl` | 기존 오르막 스냅샷(2026-09-17, 균일 모래·사이트 정보 없음) → **팀 표준 모래로 다시 만든다** |
| `Mars_Terrain/mpm_learning.md`, `scripts/ISAACLAB_CUSTOM_ROBOT_RL_PLAYBOOK.md` | 이 PC의 Newton/MPM 운용 메모, 자산 진단 우선 원칙 |
| `C:\IsaacLab-3.0.0-beta2\h1_mars_mpm_uphill`, `h1_mars_team_guide_uphill` | 이전 상승 시도 1·2 — 교훈만 가져오고 코드·결과는 재사용하지 않음 |

---

## 3. 이전 시도와 팀 기록에서 가져온 교훈

| 출처 | 결과 | 이번에 반영 |
|---|---|---|
| 시도 1 (`h1_mars_mpm_uphill`, 09-19~21) | 자체 단일 env MPM + PPO 약 2만 스텝 → 등반 0/6 | MPM 안에서 학습하지 않는다 |
| 시도 2 (`h1_mars_team_guide_uphill`, 09-25~30) | 26/27 이식, 2 env 보행 적응 → 유효 보행 0/3, 등반 정책 없음 | 〃. 26은 판정에만 쓴다 |
| 팀 일어서기 이어 학습 6건 (MPM: A1·A2·S5, 랩: V6·V6C 등) | 전부 기준선(32/74) 미만, 2/74까지 붕괴 | 이어 학습 금지, **백지 학습** + 큰 탐색 잡음(엔트로피 0.01, init std 1.0) 유지 |
| 팀 V7 / v7r | 반력 모델로 학습 99 % vs 반력 모델 없이 5 % | 반력 모델 + 깊이 커리큘럼 필수 |
| 팀 `walk1` (랩 보행기) | 넘어지진 않으나 공중 16~23 %·주저앉음·디딤발 활주 → 물리 무효 0/8 | 공중 구간·디딤발 미끄러짐·몸통 높이 항을 처음부터 넣고, **51 물리 유효성을 합격 조건에 포함** |
| 팀 감사 (10-02~03) | warm start 뒤 lr 1e-5 고착, 서서 시작한 에피소드가 성공으로 섞임 | 첫 iteration lr·std 로그 확인, 성공 집계는 시작 조건과 분리 |
| 팀 측정 | 같은 체크포인트 재평가에서 상태의 1/3이 성패 뒤바뀜 | 판정은 반복 평가(× 2 이상) + Wilson 구간 |
| 팀 운영 | LF 줄바꿈 배치 파일이 단계를 건너뜀 | 무인 배치는 CRLF, 로그에서 단계 시작·종료 대조 |
| 이 PC 메모리 | PhysX GPU에서 관절 구동이 무시됨(2026-07-14), conda `env_isaaclab`은 구버전을 가리킴 | 단계 0에서 재검증, 실행은 `isaaclab.bat`(Kit Python) |

---

## 4. 작업 폴더와 이식성 (D2 확정)

위치: `C:\Users\2hj05\repos\Mars_Terrain\h1_mars_uphill_v3` (본인 GitHub 레포 `Tod05259/Mars_Terrain` 안 — 바로 커밋·업로드 가능).
IsaacLab 레포의 `origin`은 공개 업스트림이라 그 안의 폴더는 어차피 따로 옮겨야 한다.

```text
h1_mars_uphill_v3/
  workflow(3).md          # 이 문서
  README.md               # 팀원용: 준비 확인 → 랩 재생 → MPM 재생 → 결과 보기
  run.ps1                 # isaaclab.bat 래퍼 (CONDA_PREFIX/VIRTUAL_ENV 비우기, 캐시를 .work로)
  configs/
    paths.example.json    # CAPSTONE·IsaacLab 경로 (팀원 PC마다 복사해 수정)
    sites.json            # 경사 사이트 좌표·경사각·경계 여유
    judge_flags.txt       # MPM 판정 플래그 한 줄 (고정)
  uphill_lab/             # Isaac Lab external_callback 패키지 (Mars-Ascend-H1-v1, -Eval-v1)
  scripts/                # preflight, build_snapshots, train, eval_lab, eval_mpm, play_lab, play_mpm, record
  tests/                  # CPU 단위시험
  checkpoints/            # 채택한 정책 .pt (개당 약 7 MB)
  results/                # 채점 json/csv, status.md, GIF/MP4 (소형)
  references/             # CAPSTONE 커밋 해시, 참조 파일 SHA-256 목록
  .gitignore              # .work/, runs/, logs/, *.pkl, *.usd 녹화본
```

- **CAPSTONE 코드 참조 방식**: `getup_lab`과 26 등은 CAPSTONE 커밋 `fbaf458`을 경로로 참조한다(PYTHONPATH). 받는 팀원도 같은 커밋을 체크아웃하면 된다. `references/`에 파일별 해시를 남겨 버전이 어긋나면 preflight가 경고한다.
- **MPM 스냅샷**(46~55 MB/개)은 git에 넣지 않는다. 팀 관행(`build_site_snapshots.sh`, LFS 쿼터)대로 `scripts/build_snapshots`로 재생성(사이트당 약 5~10 분).
- **기존 파일 보호**: CAPSTONE, IsaacLab의 미커밋 수정 4개 파일, 이전 시도 폴더 2개의 해시를 작업 전후로 비교해 변경 0을 확인한다.
- 커밋·push는 사용자가 요청할 때만 한다.

---

## 5. 단계별 진행

각 단계는 "할 일 → 통과 조건"이다. ⏸ 표시는 결과를 보고한 뒤 사용자 확인을 받고 넘어가는 지점이다.

### 단계 0 — 이 PC 환경 점검 (약 1 h)

1. 버전 기록: Kit Python, Newton 1.2.1, Warp 1.13.0, Torch 2.10.0+cu128, RSL-RL 5.0.1, isaaclab 6.1.11, RTX 5090 32 GB (확인됨).
   팀은 Newton 1.5/1.6 기준으로 26을 고쳐 왔다(A1). 26·51·35·63이 이 버전에서 import·실행되는지, 필요한 플래그(정식 물리, 정격 토크·속도, `--mpm-reset public`)가 동작하는지 목록화한다.
2. TEAM_SETTINGS A1 확인 명령: 51로 `walk_irregular_it25.pt` 평지 2회 × 8 s → 전도 0.
3. 랩 경로 확인: `getup_lab`을 `--external_callback getup_lab.register`로 등록, `84_eval_walk_lab.py`로 `walk1`(또는 it25)을 짧게 채점해 TEAM_GUIDE 값과 비교.
   **PhysX GPU 관절 구동을 다시 확인**(최대 행동 20스텝 → 관절 속도). 여전히 고장이면 Newton(MJWarp) 프리셋으로 반력 모델(wrench composer 경로)이 동작하는지 확인한다.
- 통과: 2·3이 팀 기록과 같은 범위. ⏸ 실패 항목이 있으면 우회 방법(호환 패치 / 백엔드 변경 / 버전 업그레이드)을 정리해 확인받는다.

### 단계 1 — 경사 판정 베드 준비 (MPM, 약 0.5일)

1. `33_slope_survey.py`로 30 m 크롭 7지형에서 16.8°·20°·22.5° 근처의 평면성 좋은 오르막 패치를 찾는다. 등반 3 m + 되밀림 여유 1 m + 출발 여유가 `--env-pitch` 패치 안에 들어가는지 확인한다.
2. 팀 표준 모래로 스냅샷 생성: `22_escape_env.py --make-snapshot --capture-mode settled --mpm-settle-steps 0 --mpm-spacing 0.0125 --mpm-depth 0.25 --mpm-depth-noise-amp 0.12 --mpm-depth-noise-seed 11`, 파일명 `uphill_<site>_<deg>_irregular_s11.pkl`.
3. 검증: 입자 수 고정, 정착 후 모래 흐름(경사 아래로 빠져나간 양), 리셋 직후 입자 튐 없음, 오르막 방향 벡터(26 `_compute_downhill`)가 사이트 조사와 일치.
- 통과: 3개 경사 베드 + `configs/sites.json`. 정착 중 모래가 크게 흘러내리면 그 경사는 제외하고 보고.

### 단계 2 — 기준선과 판정 절차 확정 (MPM, 약 0.5일)

1. 워커 it25(헤딩 PI)로 각 베드 20회 × 1: 등반 거리, 전도 시각, 물리 유효성. TEAM_GUIDE의 "16.8°에서 1.2 m 후 전도"가 재현되는지 본다.
2. 같은 체크포인트를 한 번 더 돌려 평가 잡음(상태별 성패 뒤바뀜)을 잰다.
3. 판정 플래그 한 줄을 `configs/judge_flags.txt`에 고정하고, 판정 스크립트(`scripts/eval_mpm`: 35 + 51 지표 + `info["fallen"]` + 결과 json)를 만든다.
- 통과: 기준선 표와 판정 절차. ⏸ §1 합격선을 이 기준선을 보고 확정한다(이후 변경 금지).

### 단계 3 — 랩 경사 과제 구현 `Mars-Ascend-H1-v1` (약 1일)

`getup_lab.walk_env_cfg`의 랩 보행 설정을 상속하고(로봇·DC 모터 포화비 1.5·관절 목표 자르기·높이 스캔·반력 모델·지반 혼합 재사용), 지형·명령·보상·종료만 바꾼다. CAPSTONE 파일은 수정하지 않고 `uphill_lab/`에서 상속한다.

| 구성 | 내용 |
|---|---|
| 지형 | Isaac Lab 생성기: 역피라미드 경사(가운데서 출발해 올라감) 0°~약 25°, 경사 + 요철 혼합, 평지 약 20 %. **지형 레벨 커리큘럼**(잘 오르면 더 가파른 칸으로) |
| 모래 | `walk1`의 지반 혼합(강체 20 %, 두께 ≤ 0.36 m, 강도 0.5~4배) + 깊이 커리큘럼 |
| 명령 | 전진 v_x 0~0.5 m/s, 헤딩 명령, 정지 명령 25 %(STAND) |
| 보상 | Isaac Lab H1 rough 기본값 + 모래 지지를 포함한 발 접촉 판정, `walk1` 실패 대응(공중 구간·디딤발 미끄러짐·몸통 높이) |
| 종료 | 20 s 시간 초과, 기울기 0.65 rad, 모래면 기준 몸통 < 0.50 m |
| 학습기 | 백지, 4096 env, 24 steps/env, adaptive lr 1e-3, 엔트로피 0.01, init std 1.0 |

시험:
- CPU 단위시험: 설정 생성, 보상 항 부호·크기, **경사면에서 반력 모델의 모래 깊이가 국소 지면 기준으로 계산되는지**(평지에서만 검증된 부분).
- 안정성 관문 `82 --gates G4`(4096 env, NaN 없음, 링크 최고 속도, PPO 30회 유한).
- 인터페이스 확인: it25를 랩 경사 채점에 넣어 평지 0.3 m/s 근처로 걷고 가파른 칸에서 넘어지는지(MPM 기준선과 같은 방향인지).
- ⏸ 구현 요약과 관문 결과 보고 후 학습 시작.

### 단계 4 — 랩 학습 (3~5 h)

- 3,000~5,000회. 시작 직후 lr·std·반력 포화율·첫 지형 레벨 로그 확인.
- 1,500회·최종 체크포인트를 랩 채점(경사 레벨별 등반 성공·전도)으로 본 뒤 MPM 짧은 판정(16.8°, 12회)으로 방향을 확인한다. 랩 지표는 참고, 결정은 MPM으로 한다.
- 선택: 모래 깊이 0 대조 학습 `-v1r` — 향상이 반력 모델 덕인지 가르기 위함(GPU 3~5 h 추가, 단계 5 결과를 보고 실행 여부를 묻는다).

### 단계 5 — MPM 판정 (약 0.5일)

1. §1 합격선 평가: 3개 경사 × 20 시작 × 2회, 팀 모래 S1. 16.8°는 S2 "전형적 흙"(`--soil-yield-scale 2 --soil-young-scale 2 --soil-cohesion-pa 1000`)도.
2. 진입 검증(B3-3): 워커 STAND 상태에서 넘겨받아 10회.
3. 센서 열화(A6·B3-2): `--scan-dropout 0 / 0.3 / 0.5`.
4. 체인: STAND(3 s) → ASCEND → 도착 → STAND → 워커 보행. 재전도 여부.
- GO면 단계 7. 계속이면 단계 6. NO-GO 두 번이면 멈추고 원인 분석을 보고한다. ⏸

### 단계 6 — 개선 (필요할 때만, 최대 3회)

- 한 번에 **하나의 레버**만 바꾸고 판정은 MPM으로 한다. 후보: 경사 상한·커리큘럼 속도, 미끄러짐/공중 보상 가중치, 지반 혼합 범위, 명령 분포.
- 경사면에서 모래가 흘러내리는 현상(MPM의 체적 흐름)은 반력 모델에 없다 → 판정에서 격차가 나면 먼저 이 차이를 측정(발 밑 모래의 경사 방향 이동량)하고 보고한다.

### 단계 7 — 패키징과 팀원 확인 경로 (약 0.5일)

1. README 3단계: `run.ps1 scripts\preflight.py` → `play_lab`(Isaac Lab, `--viz newton`, 수 초 안에 확인) → `play_mpm`(스냅샷 자동 생성 후 Newton GL 뷰어). 미리보기 GIF/MP4는 `results/`.
2. TEAM_SETTINGS B3 제출물: 체크포인트 + 전체 플래그 한 줄 + 스냅샷 생성법, 스코어카드(surface 스캔, 드롭아웃 0/0.3/0.5), 진입 검증, 종료 상태 npz, 트리거/해제 조건(경사 > 8° & 진행 방향 오르막 / < 5°).
3. 깨끗한 복제 시험: 레포를 다른 폴더에 clone → README대로 실행해 재생까지 되는지 확인.
4. 업로드(commit/push)는 사용자 승인 후.

---

## 6. 운영 규칙

- 실행은 `run.ps1`을 통해 `C:\IsaacLab-3.0.0-beta2\isaaclab.bat -p`(Kit Python)로 한다. 시스템 설치를 바꾸지 않는다.
- GPU 동시 실행: **랩 학습 1개 + MPM 평가 1개까지**(TEAM_GUIDE §7.6). MPM 작업 2개 동시 금지.
- 모든 실행은 고유 이름, 덮어쓰기 금지, 설정·CAPSTONE 커밋·seed·체크포인트 해시를 결과 json에 기록.
- `results/status.md`에 단계별 결과를 성공·실패 구분 없이 남기고, 측정한 것과 추정한 것을 구분해 쓴다.
- 팀 결정 대기 항목(핸드오프 유지 0.5 s vs 2.0 s, 기립 기울기 0.20 vs 0.35 rad)은 현재 표준으로 판정하고 다른 값도 함께 보고한다.

---

## 7. TEAM_GUIDE와 다른 결정 (2026-10-11 확정)

네 항목 모두 사용자가 "이 문서의 결정" 열대로 확정했다.

| # | 항목 | TEAM_GUIDE | 이 문서의 결정 | 이유 |
|---|---|---|---|---|
| D1 | 학습 장소 | §6: `26 --task slope_ascend`(MPM에서 PPO) | 랩(Isaac Lab + 반력 모델) 백지 학습 → MPM 판정 | §3 최신 기록(10-08~10-11)에서 팀 스스로 MPM 학습을 접고 이 경로로 99 %. "MPM 적용"은 반력 모델(MPM으로 보정) + MPM 최종 판정으로 충족 |
| D2 | 작업 폴더·코드 위치 | 26에 task를 늘리고 `getup_lab`에 과제를 추가(모두 CAPSTONE 안), 산출물은 `terrain/output` | 별도 폴더 `Mars_Terrain\h1_mars_uphill_v3`, CAPSTONE은 커밋 고정 참조 | 사용자 지침(별도 폴더·업로드) 우선. CAPSTONE 원본을 바꾸지 않음 |
| D3 | Python 환경 | §5·A1: conda `env_isaaclab`, Newton 1.5/1.6 | `isaaclab.bat`(Kit Python, Newton 1.2.1) + 필요 시 폴더 내 호환 처리 | 이 PC의 `env_isaaclab`은 구버전 IsaacLab을 가리켜 import 실패(기록됨) |
| D4 | 목표 경사·지형·합격선 | §6: uphill 스냅샷(16.8/20/22.5°) | 16.8°(주)·20°·22.5°를 팀 표준 모래로 재생성, §1 합격선 | 이전 시도의 Gusev highland 경로(시도 1 평균 8.3°, 시도 2 약 12.6°)는 워커 유효 범위(~12°) 근처라 ASCEND 모드의 시험이 되기 어려움 |

---

## 8. 예상 일정 (GPU 단독 사용 기준)

| 단계 | 예상 |
|---|---|
| 0 환경 점검 | 1 h |
| 1 경사 베드 | 0.5일 |
| 2 기준선·판정 | 0.5일 |
| 3 랩 과제 구현·관문 | 1일 |
| 4 랩 학습 | 3~5 h |
| 5 MPM 판정 | 0.5일 |
| 6 개선 | 회당 0.5~1일 (필요 시) |
| 7 패키징 | 0.5일 |

합계 약 4~6일. 단계 0에서 Newton 버전 문제가 크면 늘어난다.
