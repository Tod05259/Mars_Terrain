"""Short MPM judge of several checkpoints of one training run, to pick the one to judge in full.

    run.ps1 scripts\\sweep_checkpoints.py --run logs\\rsl_rl\\h1_ascend_mars_soil\\<run> --iters 1500 2000 2500
        [--sites slope16 slope20] [--episodes 12] [--tag name]

On the uphill task the sand result does not follow the lab curves (ascend_v2: 16.8 deg 3/12 at 1500
iterations, 7/12 at 2500, about 1/12 at 4000), so the candidate is chosen on the sand. The episodes of
this sweep only choose; the chosen checkpoint is then judged afresh by ``judge_suite.py`` (40 episodes),
so the reported numbers are not the ones the choice was made on.

Runs ``eval_mpm.py`` one condition after another (labels ``sweep_<tag>_<iter>_<site>``; finished ones are
skipped), holds off Windows sleep while running, and writes ``results/sweep_<tag>.md``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from uphill import paths

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_suite import keep_awake, load  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--iters", type=int, nargs="+", required=True)
    parser.add_argument("--sites", nargs="+", default=["slope16"])
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    keep_awake()
    run = args.run if args.run.is_absolute() else paths.ROOT / args.run
    tag = args.tag or run.name.split("_", 2)[-1]
    for it in args.iters:
        ckpt = run / f"model_{it}.pt"
        if not ckpt.is_file():
            print(f"[sweep] missing {ckpt}")
            continue
        for site in args.sites:
            label = f"sweep_{tag}_{it}_{site}"
            done = load(label)
            if done is not None and done["summary"]["episodes"] >= args.episodes:
                continue
            log = paths.RUNS / "eval" / f"{label}.log"
            print(f"[sweep] {it} {site} ...", flush=True)
            started = time.time()
            with log.open("w", encoding="utf-8") as f:
                code = subprocess.run(
                    [sys.executable, str(paths.ROOT / "scripts" / "eval_mpm.py"), "--checkpoint", str(ckpt),
                     "--site", site, "--episodes", str(args.episodes), "--label", label],
                    stdout=f, stderr=subprocess.STDOUT,
                ).returncode
            print(f"[sweep] {it} {site}: exit {code} [{(time.time() - started) / 60:.0f} min]", flush=True)

    lines = [f"# 체크포인트 비교 — {tag}\n", f"`{run}`, 사이트별 {args.episodes}회, `scripts/eval_mpm.py`. 선택용이며 보고 수치가 아니다.\n",
             "| 회 | 사이트 | 성공 | 도달 / 유지 / 유효 | 넘어짐 | 되밀림 | 패치 이탈 | 평균 등반 | 헤딩 오차 평균 |", "|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for it in args.iters:
        for site in args.sites:
            res = load(f"sweep_{tag}_{it}_{site}")
            if res is None:
                continue
            s = res["summary"]
            hdg = sum(e["heading_err_mean_deg"] for e in res["episodes"]) / max(1, len(res["episodes"]))
            rows.append({"iter": it, "site": site, **s, "heading_err_mean_deg": hdg})
            lines.append(f"| {it} | {site} | {s['success']}/{s['episodes']} | {s['reached']} / {s['held']} / {s['valid']} | "
                         f"{s['fallen']} | {s['slid_back']} | {s['off_patch']} | {s['climbed_mean_m']:.2f} m | {hdg:.1f}° |")
    out = paths.RESULTS / f"sweep_{tag}"
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
