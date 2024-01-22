from __future__ import annotations

import math
from copy import copy
from functools import partial
from typing import Any, Callable, Optional, Tuple, Union
from warnings import warn

from jaxfi import jaxm
from jax import Array
import jax

from .function_cache import DynamicsFunctionStore, ObjectiveFunctionStore
from .function_cache import DYNAMICS_FUNCTION_STORE, OBJECTIVE_FUNCTION_STORE
from .solver_definitions import generate_routines_for_obj_fn
from .solver_definitions import SOLVER_MAP
from .solver_definitions import _default_obj_fn, SOLVERS_STORE
from .utils import _jax_sanitize
from .solver_settings import SolverSettings


# jit compatible interface #########################################################################
@partial(
    jax.jit,
    static_argnames=("diff_cost_fn", "f_fx_fu_fn", "solver_settings"),
)
def direct_affine_solve(
    diff_cost_fn: Callable | None = None,
    **problems: dict[str, Array],
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
    U_prev = problems["U_prev"]
    solver_settings: SolverSettings = problems["solver_settings"]

    # use the linearized dynamics or linearize dynamics ourselves if `differentiate_rollout` is True
    f_fx_fu_fn = problems["f_fx_fu_fn"]
    rollout_fn = DynamicsFunctionStore._generate_rollout(
        DynamicsFunctionStore._generate_dyn_fn(f_fx_fu_fn)
    )
    problems["f_fx_fu_fn"] = None

    # retrive the solver and the correct settings
    solver = SOLVER_MAP[solver_settings.solver.lower()]
    max_inner_it = solver_settings.max_it

    # define the objective function
    obj_fn = ObjectiveFunctionStore._generate_objective_function(
        rollout_fn, _default_obj_fn, diff_cost_fn
    )

    # retrive the (cached) optimization routines based on the objective and solver settings
    routines = generate_routines_for_obj_fn(obj_fn, solver_settings)
    init_state_fn, run_with_state_fn = routines["pinit_state"], routines["prun_with_state"]

    problems = _jax_sanitize(problems)
    state = init_state_fn(solver, U_prev, problems)

    # solve
    U, state = run_with_state_fn(solver, U_prev, problems, state, max_it=max_inner_it)
    U = jaxm.where(jaxm.isfinite(state.value), 1.0, math.nan)[..., None, None] * U
    X = rollout_fn(U, problems)
    ret = X, U, dict(solver_state=state, obj=state.value)
    return ret
