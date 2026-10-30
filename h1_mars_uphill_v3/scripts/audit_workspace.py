"""Record or check that the files this project must not touch are unchanged.

    run.ps1 scripts\\audit_workspace.py --save     # write references/protected_baseline.json
    run.ps1 scripts\\audit_workspace.py            # compare against it

Covers the CAPSTONE checkout (HEAD and tracked-file status), the uncommitted edits in the
Isaac Lab tree, and the two earlier uphill attempts under the Isaac Lab folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from uphill import paths

BASELINE = paths.ROOT / "references" / "protected_baseline.json"
EARLIER = ("h1_mars_mpm_uphill", "h1_mars_team_guide_uphill")
# Folders the earlier attempts' own tools keep writing to (viewer layouts, scratch); not results.
SKIP_PARTS = {".work", "__pycache__"}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _tree_digest(root: Path) -> dict:
    """Size and mtime of every file under ``root`` (hashing tens of GB of runs is too slow)."""
    h = hashlib.sha256()
    count = 0
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        if SKIP_PARTS & set(f.relative_to(root).parts):
            continue
        st = f.stat()
        h.update(f"{f.relative_to(root).as_posix()}|{st.st_size}|{st.st_mtime_ns}\n".encode())
        count += 1
    return {"files": count, "digest": h.hexdigest()}


def collect() -> dict:
    lab = paths.ISAACLAB
    lab_changed = [
        line[3:].strip() for line in _git(lab, "status", "--porcelain").splitlines() if line[:2].strip()
    ]
    lab_files = {
        rel: _sha(lab / rel)
        for rel in lab_changed
        if (lab / rel).is_file() and not rel.startswith(EARLIER)
    }
    return {
        "capstone": {
            "head": _git(paths.CAPSTONE, "rev-parse", "HEAD").strip(),
            "tracked_changes": _git(paths.CAPSTONE, "status", "--porcelain", "--untracked-files=no"),
        },
        "isaaclab": {"head": _git(lab, "rev-parse", "HEAD").strip(), "changed_files": lab_files},
        "earlier_attempts": {name: _tree_digest(lab / name) for name in EARLIER if (lab / name).is_dir()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="write the baseline instead of checking")
    args = parser.parse_args()
    now = collect()
    if args.save:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(now, indent=2), encoding="utf-8")
        print(f"[audit] baseline written: {BASELINE}")
        return 0
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    problems = []
    if now["capstone"] != base["capstone"]:
        problems.append(f"CAPSTONE changed: {base['capstone']} -> {now['capstone']}")
    for rel, sha in base["isaaclab"]["changed_files"].items():
        if now["isaaclab"]["changed_files"].get(rel) != sha:
            problems.append(f"Isaac Lab file changed: {rel}")
    for name, d in base["earlier_attempts"].items():
        if now["earlier_attempts"].get(name) != d:
            problems.append(f"earlier attempt changed: {name}")
    for p in problems:
        print(f"[audit] FAIL {p}")
    print(f"[audit] {'OK — protected files unchanged' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
