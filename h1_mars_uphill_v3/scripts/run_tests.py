"""Run this project's CPU tests (no Isaac Sim app, no GPU).

    run.ps1 scripts\\run_tests.py
"""

import os
import sys

import pytest

from uphill import paths

if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    sys.exit(pytest.main([str(paths.ROOT / "tests"), "-q", "-p", "no:cacheprovider"] + sys.argv[1:]))
