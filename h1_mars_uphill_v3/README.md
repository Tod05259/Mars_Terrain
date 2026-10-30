# H1 화성 오르막 (MPM 판정) — 시도 3

Unitree H1이 화성 모래(MPM) 위의 실제 경사(Gusev Columbia Hills, 16.8° / 20° / 22.5°)를 오르는 정책.
팀 FSM의 ASCEND 모드(CAPSTONE `terrain/TEAM_GUIDE.md` §6, 팀원 2)에 해당한다.

- **학습**: Isaac Lab, 4096 env, 강체 경사 지형 + CAPSTONE의 모래 반력 대리 모델(`getup_lab`), 처음부터 학습 (`uphill_lab/`)
- **판정·시연**: CAPSTONE의 MPM 모래 환경 `26_escape_env_vec.py --task slope_ascend`, 팀 판정 플래그 (`scripts/eval_mpm.py`)
- 계획: [`workflow(3).md`](workflow(3).md) · 진행과 결과: [`results/status.md`](results/status.md)

> 이 문서는 작업 중이다. 최종 정책과 결과 요약은 학습·판정이 끝나면 채운다.

## 필요한 것

| | 버전 / 위치 |
|---|---|
| GPU | NVIDIA RTX (개발: RTX 5090 32 GB) |
| Isaac Lab | 3.0.0-beta2 (Kit Python 포함 설치본) |
| CAPSTONE | `https://github.com/conner9403/CAPSTONE` 커밋 `fbaf458` |
| MPM 쪽 패키지 | Newton 1.6.0 / warp 1.17.0 / mujoco_warp 3.12.0 — `scripts/setup_mpm_env.ps1`가 이 폴더의 `.work/`에만 설치 |

## 처음 한 번

PowerShell, 이 폴더에서:

```powershell
# 1) 경로 설정: 예시를 복사해 자기 PC 경로로 고친다
Copy-Item configs\paths.example.json configs\paths.json
notepad configs\paths.json

# 2) MPM 판정용 패키지 + NVIDIA H1 체크포인트 (인터넷 필요, Isaac Lab 설치는 건드리지 않음)
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_mpm_env.ps1

# 3) 판정 경사 베드 스냅샷 생성 (경사당 약 10 초~5 분, snapshots\ 에 저장)
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 scripts\build_snapshots.py
```

모든 Python 실행은 `run.ps1`을 거친다(`isaaclab.bat -p`, 경로·캐시 설정).

## 결과 보기

```powershell
# MPM 판정 (16.8° 베드, 10회, 결과는 runs\eval\<label>\result.json)
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 scripts\eval_mpm.py --checkpoint checkpoints\<정책>.pt --site slope16 --episodes 10

# 랩 채점 (역피라미드 0/16.8/20/22.5°, 512대)
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 scripts\eval_lab.py --checkpoint checkpoints\<정책>.pt --headless

# 창으로 보기 — MPM 판정 환경 (Newton GL, 창을 닫으면 종료)
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 scripts\eval_mpm.py --checkpoint checkpoints\h1_ascend_v3_steep_it2000.pt --site slope20 --episodes 3 --view

# 창으로 보기 — Isaac Lab 랩 환경 (역피라미드 경사 + 모래 반력 모델)
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 C:\IsaacLab-3.0.0-beta2\scripts\reinforcement_learning\rsl_rl\play.py `
  --task Mars-Ascend-H1-Eval-v1 --external_callback uphill_lab.register `
  --checkpoint checkpoints\h1_ascend_v3_steep_it2000.pt --num_envs 16 --viz kit
```

## 다시 학습하기

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 C:\IsaacLab-3.0.0-beta2\scripts\reinforcement_learning\rsl_rl\train.py `
  --task Mars-Ascend-H1-v1 --external_callback uphill_lab.register --num_envs 4096 --seed 42 --run_name ascend_v1 --headless
```

체크포인트는 `logs\rsl_rl\h1_ascend_mars_soil\<시각>_<run_name>\model_<N>.pt`.

## 폴더

| 경로 | 내용 |
|---|---|
| `uphill_lab/` | Isaac Lab 과제 `Mars-Ascend-H1-v1`(학습), `-v1r`(모래 없는 대조), `-Eval-v1`(랩 채점) |
| `uphill/` | 경로(`paths.py`), 판정 사이트(`sites.py`), MPM 모듈 로더(`mpm.py`), Newton 1.2.1 호환 처리(`compat.py`, 비교용) |
| `scripts/` | `eval_mpm.py` 판정, `eval_lab.py` 랩 채점, `build_snapshots.py`, `survey_slopes.py`, `parity_plate.py`(모래 재료 동등성), `lab_gates.py`(CAPSTONE 82 관문), `capstone.py`(CAPSTONE 스크립트를 수정 없이 실행), `audit_workspace.py` |
| `configs/` | `sites.json`(판정 사이트), `sites_survey.json`(조사 원자료), `paths.example.json` |
| `results/` | `status.md` 진행 기록, 채택한 결과 요약 |
| `checkpoints/` | 채택한 정책 |
| `references/` | 보호 대상 기준 상태 |
| `.work/`, `runs/`, `logs/`, `snapshots/` | 캐시·실행 기록·스냅샷 (git 제외, 재생성 가능) |

CAPSTONE과 Isaac Lab의 파일은 수정하지 않는다(`scripts/audit_workspace.py`로 확인).
