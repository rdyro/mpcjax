from __future__ import annotations

from typing import Any, Callable
from copy import copy
from warnings import warn

from jfi import jaxm
from jax import Array


from .function_cache import DYNAMICS_FUNCTION_STORE, OBJECTIVE_FUNCTION_STORE
from .dynamics_definitions import rollout_scp, Ft_ft_fn_scp
from .solver_definitions import SOLVER_BFGS, SOLVER_CVX, SOLVER_LBFGS, SOLVER_SQP
from .solver_definitions import _default_obj_fn, SOLVERS_STORE
from .utils import vec, bmv


@jaxm.jit
def _U2X(U, U_prev, Ft, ft):
    bshape = U.shape[:-2]
    xdim = ft.shape[-1] // U.shape[-2]
    X = (bmv(Ft, vec(U - U_prev, 2)) + ft).reshape(bshape + (U.shape[-2], xdim))
    return X


# main affine solve for a single iteration of SCP ##################################################
def scp_affine_solve(
    problems: dict[str, Array], diff_cost_fn: Callable | None = None
) -> tuple[Array, Array, Any]:
    """Solve a single instance of a linearized MPC problem.

    Args:
        problems (Dict[str, Array]): A dictionary of stacked (batched) problem arrays.
        reg_x (Array): State deviation penalty (SCP regularization).
        reg_u (Array): Control deviation penalty (SCP regularization).
        solver_settings (Optional[Dict[str, Any]], optional): Solver settings. Defaults to None.
        diff_cost_fn (Optional[Callable], optional): Extra obj_fn to add to the default objective
                                                     function. Defaults to None.
        differentiate_rollout (bool, optional): Whether to differentiate rollout or assume linear
                                                per-state approximation to the dynamics. Requires
                                                differentiable dynamics function. Defaults to False.
    Returns:
        Tuple[Array, Array, Any]: X, U, solver_data
    """
    # solver_settings = copy(solver_settings) if solver_settings is not None else dict()
    problems = copy(problems)
    solver_settings = dict()
    if "solver_settings" in problems:
        solver_settings = copy(problems["solver_settings"])
        del problems["solver_settings"]
    alpha = solver_settings["smooth_alpha"]
    U_prev = problems["U_prev"]

    # use the linearized dynamics or linearize dynamics ourselves if `differentiate_rollout` is True
    problems["f_fx_fu_fn"] = None
    # Ft_ft_fn = get_rollout_and_linearization("default")[1]
    # Ft_ft_fn = DYNAMICS_FUNCTION_STORE.get_linearization_fn("scp")
    # Ft, ft = Ft_ft_fn(x0, U_prev, f, fx, fu, X_prev, U_prev)
    # problems["Ft"], problems["ft"] = Ft, ft
    problems["smooth_alpha"] = alpha

    # retrive the solver and the correct settings
    solver = solver_settings.get("solver", "SQP").upper()
    solver_map = dict(
        BFGS=(SOLVER_BFGS, 100),
        LBFGS=(SOLVER_LBFGS, 100),
        CVX=(SOLVER_CVX, 30),
        SQP=(SOLVER_SQP, 30),
    )
    # assert solver in solver_map, f"Solver {solver_settings.get('solver')} not supported."
    if solver not in solver_map:
        warn(f"Solver {solver_settings.get('solver')} not supported. Defaulting to SQP solver.")
        solver = "SQP"
    solver, max_inner_it = solver_map[solver]
    max_inner_it = solver_settings.get("max_it", max_inner_it)

    # define the objective function
    obj_fn = OBJECTIVE_FUNCTION_STORE.get_obj_fn(rollout_scp, _default_obj_fn, diff_cost_fn)

    # retrieve the (cached) optimization routines based on the objective and solver settings
    init_state_fn, run_with_state_fn = SOLVERS_STORE.get_routines(obj_fn, solver_settings)
    state = solver_settings.get("solver_state", None)

    if state is None or solver in {SOLVER_CVX, SOLVER_SQP}:
        state = init_state_fn(solver, U_prev, problems)

    # solve
    U, state = run_with_state_fn(solver, U_prev, problems, state, max_it=max_inner_it)

    # remove the nans with previous solution (if any)
    mask = jaxm.tile(
        jaxm.isfinite(state.value)[..., None, None], (1,) * state.value.ndim + U.shape[-2:]
    )
    U = jaxm.where(mask, U, U_prev)
    X = rollout_scp(U, problems)
    ret = X, U, dict(solver_state=state, obj=state.value)
    return ret