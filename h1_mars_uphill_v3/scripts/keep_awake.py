"""Hold off Windows sleep while this project's training or judging processes run, then exit.

    run.ps1 scripts\\keep_awake.py

Uses ``SetThreadExecutionState`` (the request a media player makes; no power setting changes, released
when this process exits). Checks every minute for python processes running ``train.py``,
``eval_mpm.py``, ``judge_suite.py`` or ``eval_lab.py``; exits after 15 checks in a row that find none
(queued jobs start a little after the previous one ends).
"""

import ctypes
import subprocess
import sys
import time

PATTERNS = ("train.py", "eval_mpm.py", "judge_suite.py", "eval_lab.py")


def busy() -> bool:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { $_.CommandLine }"],
        capture_output=True, text=True,
    ).stdout
    return any(p in out for p in PATTERNS)


def main() -> int:
    if sys.platform != "win32":
        return 0
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    print("[awake] holding off sleep while training/judging runs", flush=True)
    idle = 0
    # queued jobs start a little after the previous one ends: leave only after 15 idle minutes
    while idle < 15:
        time.sleep(60)
        idle = 0 if busy() else idle + 1
    print("[awake] nothing running, released", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
