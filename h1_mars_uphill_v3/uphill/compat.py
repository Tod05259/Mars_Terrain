"""Run the CAPSTONE MPM scripts (written for Newton 1.5/1.6) on the installed Newton.

The CAPSTONE files are not modified. :func:`apply` patches the installed Newton classes so they
behave the way the scripts expect, before the scripts are imported. Each shim checks for the
behaviour it replaces and is skipped when the installed Newton already has it.
"""

from __future__ import annotations

import inspect

import newton

APPLIED: list[str] = []


def _skip_soft_contacts_when_unbudgeted() -> str | None:
    """``CollisionPipeline(soft_contact_max=0)`` must skip the particle-shape pass.

    26 builds its pipeline with ``soft_contact_max=0`` and passes the real state (with ~250 k MPM
    particles per env) to ``collide()``, relying on Newton 1.6, which gates the soft-contact pass on
    ``soft_contact_max > 0``. Newton 1.2 launches ``create_soft_contacts`` over every
    particle x shape pair whenever the state has particles; the kernel writes its per-thread slot
    unconditionally, so a zero-size buffer is written out of bounds (CUDA error 700 on the first
    step). The soft contacts are unused here (the soil is coupled through the MPM solver's own
    colliders, and MuJoCo reads only rigid contacts), so the pass is skipped by hiding the
    particles from ``collide()`` for the call — which is what 26 itself did before Newton 1.6.
    """
    from newton._src.sim import collide as collide_mod

    pipeline_cls = collide_mod.CollisionPipeline
    if "soft_contact_max > 0" in inspect.getsource(pipeline_cls.collide):
        return None
    original = pipeline_cls.collide

    def collide(self, state, contacts, *args, **kwargs):
        if self.soft_contact_max > 0 or getattr(state, "particle_q", None) is None:
            return original(self, state, contacts, *args, **kwargs)
        particle_q = state.particle_q
        state.particle_q = None
        try:
            return original(self, state, contacts, *args, **kwargs)
        finally:
            state.particle_q = particle_q

    collide.__doc__ = original.__doc__
    pipeline_cls.collide = collide
    return "collide: no soft-contact pass when soft_contact_max == 0"


def _mujoco_solver_reset() -> str | None:
    """``SolverMuJoCo.reset(state, world_mask=None, flags=0)`` (added after Newton 1.2).

    26 calls it after restoring an env so a NaN left in MuJoCo's acceleration warm start cannot
    poison every later solve; ``flags=0`` keeps the restored joint state. Newton 1.2 copies the
    Newton state into MuJoCo's data at every step, so only the solver-owned arrays need clearing:
    warm start, applied generalized and Cartesian forces, activations and controls.
    """
    from newton.solvers import SolverMuJoCo

    if hasattr(SolverMuJoCo, "reset"):
        return None

    def reset(self, state, world_mask=None, flags=None):
        if flags != 0:
            # Only the "keep the joint state" form is used by the CAPSTONE scripts.
            raise NotImplementedError("compat SolverMuJoCo.reset supports flags=0 only")
        data = getattr(self, "mjw_data", None)
        if data is None:
            return
        import torch
        import warp as wp

        mask = None
        if world_mask is not None:
            mask = world_mask if isinstance(world_mask, torch.Tensor) else wp.to_torch(world_mask)
            mask = mask.to(dtype=torch.bool)
        for name in ("qacc_warmstart", "qfrc_applied", "xfrc_applied", "act", "ctrl"):
            arr = getattr(data, name, None)
            if arr is None or arr.size == 0:
                continue
            if mask is None:
                arr.zero_()
            else:
                t = wp.to_torch(arr)
                t[mask.to(t.device)] = 0

    SolverMuJoCo.reset = reset
    return "SolverMuJoCo.reset(flags=0): clear warm start, applied forces, act, ctrl"


def _drive_pd_alias() -> str | None:
    """``newton.actuators.DrivePD`` (26 ``--velocity-limits urdf``).

    Newton 1.2 ships the same per-DOF PD drive as ``ControllerPD``: effort = const + feedforward +
    kp (target_pos - q) + kd (target_vel - qd), written to ``Control.joint_f``, with the same
    ``add_actuator(cls, index=, pos_index=, clamping=[(ClampingDCMotor, {...})], kp=, kd=)`` call.
    """
    import newton.actuators as actuators

    if hasattr(actuators, "DrivePD"):
        return None
    actuators.DrivePD = actuators.ControllerPD
    return "actuators.DrivePD = ControllerPD"


def apply() -> list[str]:
    """Install the shims needed by the installed Newton version; returns what was installed."""
    if APPLIED:
        return APPLIED
    APPLIED.append(f"newton {getattr(newton, '__version__', '?')}")
    for shim in (_skip_soft_contacts_when_unbudgeted, _mujoco_solver_reset, _drive_pd_alias):
        done = shim()
        if done:
            APPLIED.append(done)
    return APPLIED
