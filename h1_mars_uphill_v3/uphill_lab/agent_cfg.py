"""rsl_rl PPO settings of the uphill task: CAPSTONE's lab walker settings, from scratch."""

from isaaclab.utils.configclass import configclass

from getup_lab.agent_cfg import H1WalkSoilPPORunnerCfg


@configclass
class H1AscendSoilPPORunnerCfg(H1WalkSoilPPORunnerCfg):
    """:class:`getup_lab.agent_cfg.H1WalkSoilPPORunnerCfg` (24 steps per env, adaptive learning rate
    from 1e-3, entropy bonus 0.01, [512, 256, 128] ELU actor and critic without observation
    normalisation, initial action std 1.0) for 4000 iterations, a checkpoint every 250.

    Trained from scratch, not continued from a walker: on the team's get-up task every continuation
    lost the transfer to the sand (V6C 2/74 against V4 32/74, TEAM_GUIDE 2026-10-10)."""

    max_iterations = 4000
    save_interval = 250
    experiment_name = "h1_ascend_mars_soil"
