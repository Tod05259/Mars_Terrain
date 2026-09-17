# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open a USD file in the Isaac Sim GUI.

``isaacsim.exe <path>`` treats its first argument as a .kit experience file,
not a stage, so recorded animations cannot be opened that way. This launches
the Isaac Lab GUI experience and opens the stage directly — press play on the
timeline to run the recorded animation smoothly.

Usage
-----
    python terrain/scripts/25_open_usd.py terrain/output/escape_env/replay.usd
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", type=Path, help="USD file to open.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False
if not getattr(args_cli, "visualizer", None):
    args_cli.visualizer = ["kit"]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd  # noqa: E402


def main() -> None:
    stage_path = args_cli.stage.expanduser().resolve()
    if not stage_path.is_file():
        raise FileNotFoundError(stage_path)
    context = omni.usd.get_context()
    opened = context.open_stage(str(stage_path))
    print(f"\n  opened: {stage_path}  (ok={opened})")
    print("  press the timeline Play button to run the recording.\n")
    while simulation_app.is_running():
        simulation_app.update()
    simulation_app.close()


if __name__ == "__main__":
    main()
