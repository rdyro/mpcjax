from __future__ import annotations

from typing import Any, Callable, Optional
from copy import copy

import jfi
from jax import Array
from jfi import jaxm

from .jax_solver import _build_problems, _jax_sanitize, _augment_cost
from .solver_definitions import _default_obj_fn
from .dynamics_definitions import rollout_scp
from .function_cache import DYNAMICS_FUNCTION_STORE, OBJECTIVE_FUNCTION_STORE


def generate_optimality_fn(
    f_fx_fu_fn: Callable,
    Q: Array,
    R: Array,
    x0: Array,
    X_ref: Optional[Array] = None,
    U_ref: Optional[Array] = None,
    X_prev: Optional[Array] = None,
    U_prev: Optional[Array] = None,
    x_l: Optional[Array] = None,
    x_u: Optional[Array] = None,
    u_l: Optional[Array] = None,
    u_u: Optional[Array] = None,
    reg_x: float = 1e0,
    reg_u: float = 1e-2,
    slew_rate: Optional[float] = None,
    u0_slew: Optional[Array] = None,
    lin_cost_fn: Optional[Callable] = None,
    diff_cost_fn: Optional[Callable] = None,
    solver_settings: Optional[dict[str, Any]] = None,
    solver_state: Optional[Any] = None,
    dtype: Any | None = None,
    device: Any | None = None,
    direct_solve: bool = True,
    **extra_kw,
) -> tuple[Array, Array, dict[str, Any]]:
    dtype = Q.dtype if dtype is None else jfi.default_dtype_for_device(device)
    device = (Q.device() if hasattr(Q, "device") else "cpu") if device is None else device

    problems = _build_problems(
        f_fx_fu_fn=f_fx_fu_fn,
        Q=Q,
        R=R,
        x0=x0,
        X_ref=X_ref,
        U_ref=U_ref,
        X_prev=X_prev,
        U_prev=U_prev,
        x_l=x_l,
        x_u=x_u,
        u_l=u_l,
        u_u=u_u,
        reg_x=reg_x,
        reg_u=reg_u,
        slew_rate=slew_rate,
        u0_slew=u0_slew,
        solver_settings=solver_settings,
        solver_state=solver_state,
        dtype=dtype,
        device=device,
        **extra_kw,
    )

    problems = copy(problems)
    f_fx_fu_fn = problems["f_fx_fu_fn"]
    rollout_fn = DYNAMICS_FUNCTION_STORE.get_rollout_fn(
        DYNAMICS_FUNCTION_STORE.get_dyn_function(f_fx_fu_fn)
    )
    problems["f_fx_fu_fn"] = None
    problems["reg_x"], problems["reg_u"] = 0.0, 0.0
    problems["smooth_alpha"] = problems["solver_settings"]["smooth_alpha"]

    # define the objective function
    if direct_solve:
        obj_fn = OBJECTIVE_FUNCTION_STORE.get_obj_fn(rollout_fn, _default_obj_fn, diff_cost_fn)
    else:
        obj_fn = OBJECTIVE_FUNCTION_STORE.get_obj_fn(rollout_scp, _default_obj_fn, diff_cost_fn)
    k_fn = jaxm.jit(jaxm.grad(obj_fn, argnums=0))

    problems_model = problems

    def optimality_fn(U, problems_input):
        problems_input = _jax_sanitize(problems_input)
        problems = copy(problems_model)
        for k, v in problems_input.items():
            if k in problems:
                problems[k] = v
        smooth_alpha = problems_input.get("solver_settings", dict()).get("smooth_alpha")
        if smooth_alpha is not None:
            problems["smooth_alpha"] = smooth_alpha
        problems["reg_x"], problems["reg_u"] = 0.0, 0.0

        if not direct_solve:
            X = rollout_fn(U, problems)
            f, fx, fu = f_fx_fu_fn(X[..., :-1, :], U, problems["P"])
            problems["f"], problems["fx"], problems["fu"] = f, fx, fu
            problems["X_prev"], problems["U_prev"] = X[..., 1:, :], U

        X_ref_, U_ref_ = _augment_cost(
            lin_cost_fn,
            problems["X_prev"],
            problems["U_prev"],
            problems["Q"],
            problems["R"],
            problems["X_ref"],
            problems["U_ref"],
            dict(problems, solver_settings=_jax_sanitize(problems["solver_settings"])),
            force_dtype=False,
        )
        problems["X_ref"], problems["U_ref"] = X_ref_, U_ref_
        return k_fn(U, problems)

    return optimality_fn
