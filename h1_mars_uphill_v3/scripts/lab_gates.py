"""Run CAPSTONE's lab gates (``82_soil_lab_gates.py``) unmodified, with the uphill tasks registered.

    run.ps1 scripts\\lab_gates.py --gates G4 --train_task Mars-Ascend-H1-v1 --output runs\\gates\\g4.json --headless

82 registers only CAPSTONE's tasks; this registers ``uphill_lab``'s first (the registration imports
nothing from Isaac Lab, so 82 still starts the app itself) and then runs 82 as ``__main__``.
"""

import runpy
import sys

import uphill_lab
from uphill import paths


def main() -> None:
    uphill_lab.register()
    script = paths.SCRIPTS / "82_soil_lab_gates.py"
    sys.argv = [str(script)] + sys.argv[1:]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
