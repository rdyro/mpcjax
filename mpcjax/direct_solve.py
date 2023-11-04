from __future__ import annotations

from copy import copy
from functools import partial
from typing import Any, Callable, Optional, Tuple, Union
from warnings import warn

from jax import Array
from jfi import jaxm

from .function_cache import DYNAMICS_FUNCTION_STORE, OBJECTIVE_FUNCTION_STORE
from .solver_definitions import SOLVER_BFGS, SOLVER_CVX, SOLVER_LBFGS, SOLVER_SQP
from .solver_definitions import _default_obj_fn, SOLVERS_STORE


# main affine solve for a single iteration of SCP ##################################################
def _direct_affine_solve(
    problems: dict[str, Array], diff_cost_fn: Callable | None = None
) -> Tuple[Array, Array, Any]:
    """Solve a single instance of a linearized MPC problem.

    Args:
        problems (Dict[str, Array]): A dictionary of stacked (batched) problem arrays.
        reg_x (Array): State deviation penalty (SCP regularization).
        reg_u (Array): Control deviation penalty (SCP regularization).
        solver_settings (Optional[Dict[str, Any]], optional): Solver settings. Defaults to None.
        diff_cost_fn (Optional[Callable], optional): Extra obj_fn to add to the default objective
                                                     function. Defaults to None.
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
    f_fx_fu_fn = problems["f_fx_fu_fn"]
    rollout_fn = DYNAMICS_FUNCTION_STORE.get_rollout_fn(
        DYNAMICS_FUNCTION_STORE.get_dyn_function(f_fx_fu_fn)
    )
    problems["f_fx_fu_fn"] = None
    problems["smooth_alpha"] = alpha

    # retrive the solver and the correct settings
    solver = solver_settings.get("solver", "SQP").upper()
    solver_map = dict(BFGS=(SOLVER_BFGS, 100), LBFGS=(SOLVER_LBFGS, 100), SQP=(SOLVER_SQP, 30))
    # assert solver in solver_map, f"Solver {solver_settings.get('solver')} not supported."
    if solver not in solver_map:
        warn(f"Solver {solver_settings.get('solver')} not supported. Defaulting to SQP solver.")
        solver = "SQP"
    solver, max_inner_it = solver_map[solver]
    max_inner_it = solver_settings.get("max_it", max_inner_it)

    # define the objective function
    obj_fn = OBJECTIVE_FUNCTION_STORE.get_obj_fn(rollout_fn, _default_obj_fn, diff_cost_fn)

    # retrive the (cached) optimization routines based on the objective and solver settings
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
    X = rollout_fn(U, problems)
    ret = X, U, dict(solver_state=state, obj=state.value)
    return ret


def direct_affine_solve(*args, **kw):
    #from line_profiler import LineProfiler

    #LP = LineProfiler()
    #LP.add_function(_direct_affine_solve)
    #ret = LP.wrap_function(_direct_affine_solve)(*args, **kw)
    #LP.print_stats(output_unit=1e-3)
    #return ret
    return _direct_affine_solve(*args, **kw)
