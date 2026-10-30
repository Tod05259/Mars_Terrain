"""Run the MPM judge conditions of workflow(3) section 1 for one checkpoint and score them.

    run.ps1 scripts\\judge_suite.py --checkpoint <model.pt> --name <policy name> [--episodes 40]
        [--only slope16 slope20] [--quick]

Conditions, one after another (MPM jobs run one at a time, TEAM_GUIDE 7.6):

| key            | site    | soil    | scan dropout | pass line (decided 2026-10-12)  |
|----------------|---------|---------|--------------|---------------------------------|
| slope16        | 16.8    | team    | 0            | >= 80 % (32/40)                 |
| slope20        | 20.0    | team    | 0            | >= 60 % (24/40)                 |
| slope22        | 22.5    | team    | 0            | report only                     |
| slope16_typical| 16.8    | typical | 0            | >= 70 % (28/40)                 |
| slope16_drop30 | 16.8    | team    | 0.3 (block)  | report (TEAM_SETTINGS B3-2)     |
| slope16_drop50 | 16.8    | team    | 0.5 (block)  | report (TEAM_SETTINGS B3-2)     |
| slope16_entry  | 16.8    | team    | 0            | report: the walker it25 holds STAND 3 s, then hands over (B3-3) |

Each condition writes ``runs/eval/<name>_<key>/result.json`` (``scripts/eval_mpm.py``); a finished
condition is not run again. The summary goes to ``results/judge_<name>.json`` and ``.md``.
``--quick`` runs 12 episodes of slope16 and slope20 only (the mid-training check of workflow step 4).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from uphill import paths

CONDITIONS = {
    "slope16": {"site": "slope16", "soil": "team", "dropout": 0.0, "pass": 0.80},
    "slope20": {"site": "slope20", "soil": "team", "dropout": 0.0, "pass": 0.60},
    "slope22": {"site": "slope22", "soil": "team", "dropout": 0.0, "pass": None},
    "slope16_typical": {"site": "slope16", "soil": "typical", "dropout": 0.0, "pass": 0.70},
    "slope16_drop30": {"site": "slope16", "soil": "team", "dropout": 0.3, "pass": None, "episodes": 10},
    "slope16_drop50": {"site": "slope16", "soil": "team", "dropout": 0.5, "pass": None, "episodes": 10},
    "slope16_entry": {"site": "slope16", "soil": "team", "dropout": 0.0, "pass": None, "episodes": 10, "entry": True},
}
BASELINE = {"slope16": "base_it25_slope16_team", "slope20": "base_it25_slope20_team",
            "slope22": "base_it25_slope22_team", "slope16_typical": "base_it25_slope16_typical"}


def keep_awake() -> None:
    """Ask Windows not to sleep while this process runs (``SetThreadExecutionState``; released on exit).

    The PC went to sleep twice during unattended training (2026-10-12). This changes no power setting:
    it is the request a media player makes, and it ends with the process. The display may still turn off.
    """
    if sys.platform == "win32":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def load(label: str) -> dict | None:
    f = paths.RUNS / "eval" / label / "result.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--only", nargs="+", default=None, choices=list(CONDITIONS))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--allow-sleep", action="store_true", help="do not hold off Windows sleep while running")
    args = parser.parse_args()
    if not args.allow_sleep:
        keep_awake()

    keys = args.only or list(CONDITIONS)
    episodes = args.episodes
    if args.quick:
        keys, episodes = args.only or ["slope16", "slope20"], 12
    for key in keys:
        cond = CONDITIONS[key]
        label = f"{args.name}_{key}"
        done = load(label)
        n = cond.get("episodes", episodes)
        if done is not None and done["summary"]["episodes"] >= n:
            print(f"[suite] {key}: done ({label})")
            continue
        cmd = [sys.executable, str(paths.ROOT / "scripts" / "eval_mpm.py"), "--checkpoint", str(args.checkpoint),
               "--site", cond["site"], "--soil", cond["soil"], "--episodes", str(n), "--label", label]
        if cond["dropout"] > 0:
            cmd += ["--scan-dropout", str(cond["dropout"])]
        if cond.get("entry"):
            cmd += ["--entry-walker", str(paths.IT25)]
        log = paths.RUNS / "eval" / f"{label}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"[suite] {key}: {n} episodes ...", flush=True)
        started = time.time()
        with log.open("w", encoding="utf-8") as f:
            code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        print(f"[suite] {key}: exit {code} [{(time.time() - started) / 60:.0f} min]", flush=True)

    rows, lines = {}, []
    lines.append(f"# MPM 판정 — {args.name}\n")
    lines.append(f"체크포인트 `{args.checkpoint}`. 판정 `scripts/eval_mpm.py`, 성공 = 3 m 도달 + 명령 0으로 3 s 유지 + 물리 유효.\n")
    lines.append("| 조건 | 성공 | Wilson 95 % | 합격선 | 판정 | 도달 / 유지 / 유효 | 넘어짐 | 되밀림 | 패치 이탈 | 평균 등반 | 도달 시각 중앙 | 기준선 it25 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, cond in CONDITIONS.items():
        res = load(f"{args.name}_{key}")
        if res is None:
            continue
        s = res["summary"]
        lo, hi = s["success_wilson95"]
        verdict = "보고" if cond["pass"] is None else ("통과" if s["success_rate"] >= cond["pass"] else "미달")
        base = load(BASELINE[key]) if key in BASELINE else None
        b = f"{base['summary']['success']}/{base['summary']['episodes']}" if base else "-"
        t = "-" if s["reached_at_median_s"] is None else f"{s['reached_at_median_s']:.1f} s"
        rows[key] = {**s, "pass_line": cond["pass"], "verdict": verdict, "baseline": b}
        pl = "-" if cond["pass"] is None else f"{cond['pass']:.0%}"
        lines.append(
            f"| {key} | {s['success']}/{s['episodes']} | {lo:.0%}~{hi:.0%} | {pl} | {verdict} | "
            f"{s['reached']} / {s['held']} / {s['valid']} | {s['fallen']} | {s['slid_back']} | {s['off_patch']} | "
            f"{s['climbed_mean_m']:.2f} m | {t} | {b} |"
        )
    out = paths.RESULTS / f"judge_{args.name}"
    out.with_suffix(".json").write_text(json.dumps({"checkpoint": str(args.checkpoint), "conditions": rows}, indent=2),
                                        encoding="utf-8")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
