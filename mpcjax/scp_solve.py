from __future__ import annotations

import math
from typing import Any, Callable, Tuple
from copy import copy
from warnings import warn
from functools import partial

from jaxfi import jaxm
import jax
from jax import Array


from .function_cache import DynamicsFunctionStore, ObjectiveFunctionStore
from .function_cache import DYNAMICS_FUNCTION_STORE, OBJECTIVE_FUNCTION_STORE
from .dynamics_definitions import rollout_scp, Ft_ft_fn_scp
from .solver_definitions import generate_routines_for_obj_fn
from .solver_definitions import SOLVER_MAP
from .solver_definitions import _default_obj_fn, SOLVERS_STORE
from .utils import vec, bmv, _jax_sanitize
from .solver_settings import SolverSettings


@jaxm.jit
def _U2X(U, U_prev, Ft, ft):
    bshape = U.shape[:-2]
    xdim = ft.shape[-1] // U.shape[-2]
    X = (bmv(Ft, vec(U - U_prev, 2)) + ft).reshape(bshape + (U.shape[-2], xdim))
    return X


# main affine solve for a single iteration of SCP ##################################################
@partial(
    jax.jit,
    static_argnames=("diff_cost_fn", "f_fx_fu_fn", "solver_settings"),
)
def scp_affine_solve(
    diff_cost_fn: Callable | None = None,
    **problems: dict[str, Array],
) -> Tuple[Array, Array, Any]:
    """Solve a single instance of a linearized MPC problem."""

    # use the linearized dynamics or linearize dynamics ourselves if `differentiate_rollout` is True
    U_prev = problems["U_prev"]
    solver_settings: SolverSettings = problems["solver_settings"]
    problems["f_fx_fu_fn"] = None

    # retrieve the solver and the correct settings
    solver = SOLVER_MAP[solver_settings.solver.lower()]
    max_inner_it = solver_settings.max_it

    # define the objective function
    obj_fn = ObjectiveFunctionStore._generate_objective_function(
        rollout_scp, _default_obj_fn, diff_cost_fn
    )

    # retrieve the (cached) optimization routines based on the objective and solver settings
    routines = generate_routines_for_obj_fn(obj_fn, solver_settings)
    init_state_fn, run_with_state_fn = routines["pinit_state"], routines["prun_with_state"]

    problems = _jax_sanitize(problems)
    state = init_state_fn(solver, U_prev, problems)

    # solve
    U, state = run_with_state_fn(solver, U_prev, problems, state, max_it=max_inner_it)
    U = jaxm.where(jaxm.isfinite(state.value), 1.0, math.nan)[..., None, None] * U
    X = rollout_scp(U, problems)
    ret = X, U, dict(solver_state=state, obj=state.value)
    return ret


# def scp_affine_solve(
#    problems: dict[str, Array], diff_cost_fn: Callable | None = None
# ) -> tuple[Array, Array, Any]:
#    # import line_profiler
#    # LP = line_profiler.LineProfiler()
#    # LP.add_function(_scp_affine_solve)
#    # ret = LP.wrap_function(_scp_affine_solve)(problems, diff_cost_fn=diff_cost_fn)
#    # LP.print_stats(output_unit=1e-3)
#    # return ret
#    return _scp_affine_solve(problems, diff_cost_fn=diff_cost_fn)
