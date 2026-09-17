# mars_escape_env — 화성 레골리스 발빠짐 탈출 학습 환경

실제 화성 지형(HiRISE DTM)과 문헌 보정된 MPM 레골리스 위에서 휴머노이드 H1을
학습시키는 환경이다. 현재 학습에 쓰고 있는 구성 그대로다.

## 실행

`env_isaaclab` 환경(IsaacLab 6.x, newton, warp-lang, rsl-rl-lib 5.x, CUDA torch)에서
저장소 루트를 기준으로 실행한다.

```powershell
# 1) 매몰 스냅샷 만들기 (모든 학습·평가의 시작 상태)
python mars_escape_env\scripts\30_build_buried_state.py --settle-seconds 3.0

# 2) 학습
python mars_escape_env\scripts\27_train_escape_vec.py `
    --num-envs 16 --iterations 400 --start-mode trapped `
    --hold-seconds 1.5 --down-seconds 6.0 `
    --max-action-std 0.3 --particle-speed-cap 6.0 --env-pitch 12.0

# 3) 성공 케이스 녹화
python mars_escape_env\scripts\28_record_successes.py --checkpoint <ckpt> --wanted 3

# 4) 환경만 점검
python mars_escape_env\scripts\26_escape_env_vec.py --num-envs 4 --smoke
```

## 구성

| 파일 | 역할 |
|---|---|
| `07_h1_mars_newton.py` | 런타임. H1 강체 + 실제 DTM 충돌 + MPM 토양, 양방향 결합 |
| `22_escape_env.py` | 단일 env, 스냅샷 생성(걸어 들어가기 방식) |
| `26_escape_env_vec.py` | 벡터 env. 학습에 쓰는 것 |
| `27_train_escape_vec.py` | RSL-RL PPO 학습기 |
| `28_record_successes.py` | 성공 에피소드만 USD로 녹화 |
| `30_build_buried_state.py` | 매몰 상태 직접 구성 + 검증 |
| `23`, `24`, `25` | 단일 env 학습, 재생, USD 뷰어 |
| `mars_dtm_tools.py` | DTM 로딩·샘플링 유틸 |
| `terrain/*.usd` | Gusev(Spirit 착륙지) 30 m 크롭. Troy 시험지 |

## 물리 설정

```
지형      실제 HiRISE DTM, Gusev 분화구 Columbia Hills, 1.01 m/px
중력      3.721 m/s²
토양      MPM Drucker-Prager, gusev_center 보정
          물질점 간격 12.5 mm, ppc 3, 200 iterations, 2 substeps
결합      발 ↔ 토양 양방향(명시적 반력), 링크당 400 N / 80 N·m 상한
로봇      H1 51.4 kg, 화성 중량 191 N, 토크 300/100 N·m(실제 사양)
제어      50 Hz, 목표 = 기본자세 + 0.5 × action, 관절범위 클램프
관측      256차원 (IMU·엔코더·명령·이전액션·높이스캔 187점)
```

물질점 간격 12.5 mm는 보정·수렴 검증이 이뤄진 해상도다. 6.25 mm 대비 3.1 % 이내로
수렴하며, 이전에 쓰던 25 mm는 27.7 % 어긋난 미수렴 조건이었다. MPM 물질점은 알갱이가
아니라 연속체의 적분점이므로 입도 물리는 보정된 구성방정식이 담당한다.

## 성공 판정

함정 중심에서 몸통 1 m 이상 벗어난 뒤, 발바닥으로 지지하며 다시 움직이는 상태를
1.5 초 유지. 자세와 걸음걸이는 판정하지 않는다 — 걸어 나가든 뛰어 나가든 네발로 기어
나가든 마지막에 일어서면 성공이다. 각도 임계값을 쓰지 않고 발바닥 지지(발 이외 링크가
지면 위 120 mm 초과)와 보행 회복(평면속도 0.15 m/s 이상)만 본다.

## 현재 상태 — 반드시 읽을 것

**발빠짐 시나리오가 아직 성립하지 않는다.** 보정된 레골리스 물성에서 H1의 발은
잡히지 않는다. 세 가지 독립적인 방식이 같은 결론을 냈다.

| 방식 | 결과 |
|---|---|
| 걸어 들어가기 (껍질·포켓 20여 조합) | 갇힘 미발생. 발이 들어갔다 3초 안에 빠져나옴 |
| 깊고 넓고 무른 포켓 | 지반이 자기 무게로 먼저 붕괴 |
| 매몰 상태 직접 구성 + 무구동 정착 | 발 위 269 mm의 흙이 3초 만에 흘러내림 |

물리적 이유는 인발 저항이다. 정상 레골리스의 수직 인발 저항은 체중의 8 %이고
보정 격자에서는 0.1 % 수준인데, H1은 무릎·고관절에 300 N·m를 쓸 수 있다. 바퀴는
들어 올릴 수 없어 갇히지만 다리는 그냥 든다. Spirit이 Troy에서 고착한 것과 이족
로봇이 같은 함정을 만나는 것은 다른 문제다.

따라서 **아래 수치는 전부 폐기 대상이다** — 이 저장소에 이전에 기록된 모든 탈출률
(90 %, 38 %, 46.9 %, 50 % 등). 발이 실제로 묻히지 않은 환경, 또는 미수렴 격자에서
측정된 값이다. 자세한 경위는 `terrain/mars_terrain/mars_real_terrain_env/
EXTRACTION_HANDOFF.md` §10~§12에 있다.

## 유효한 것

물리·지형·물성 계열은 이 문제와 무관하게 유효하다.

- 7개 화성 지역 30 m/100 m 지형 USD (원본 HiRISE DTM, 1.01 m/px 무손실)
- MPM 토양 물성 보정 및 공간·시간 수렴 검증
- 외부 실측 3중 검증: Viking Lander 1 발판(5.51 kPa → 165 mm), MER 정상 주행,
  Spirit Troy 고착 재현
- Troy 파라미터 허용 대역 스윕 (껍질 두께 × 강도 × 포켓 강성 27조합 × 2접촉)
- 앵커링 수지: 226 mm 매장 인발 93 N, 지지발 실측 최대 278 N, 여유 −6 N
