"""H1 uphill task on the Mars soil surrogate, for Isaac Lab (external callback).

    isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task Mars-Ascend-H1-v1 \\
        --external_callback uphill_lab.register --headless

with this project's folder and ``CAPSTONE/terrain`` (for ``getup_lab``) on ``PYTHONPATH``
(``run.ps1`` sets both). Isaac Lab and CAPSTONE files are not modified.
"""

import gymnasium as gym

_TASKS = {
    "Mars-Ascend-H1-v1": "H1AscendSoilEnvCfg",
    "Mars-Ascend-H1-v1r": "H1AscendRigidEnvCfg",
    "Mars-Ascend-H1-v2": "H1AscendSoilEnvCfgV2",
    "Mars-Ascend-H1-v3": "H1AscendSoilEnvCfgV3",
    "Mars-Ascend-H1-Eval-v1": "H1AscendSoilEnvCfg_EVAL",
}


def register() -> None:
    """Register the uphill environments (also registers CAPSTONE's get-up and walking tasks)."""
    import getup_lab

    getup_lab.register()
    for task_id, cfg in _TASKS.items():
        if task_id in gym.registry:
            continue
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"uphill_lab.ascend_env_cfg:{cfg}",
                "rsl_rl_cfg_entry_point": "uphill_lab.agent_cfg:H1AscendSoilPPORunnerCfg",
            },
        )
