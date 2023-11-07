from __future__ import annotations

from typing import Callable, Any, Dict, Optional, List
import inspect
import traceback
from functools import partial

import cloudpickle as cp
from jfi import jaxm
import jaxopt

from jax import Array

# from .second_order_solvers_old import ConvexSolver, SQPSolver
from .second_order_solvers import ConvexSolver, SQPSolver
from .utils import vec, bmv

####################################################################################################

SOLVER_BFGS = 0
SOLVER_LBFGS = 1
SOLVER_CVX = 2
SOLVER_SQP = 3

SOLVERS_STORE = dict()
STATIC_ARGUMENTS = ["jit", "linesearch", "maxls", "reg0", "tol", "device"]

####################################################################################################


def _default_obj_fn(X: Array, U: Array, problem: Dict[str, List[Array]]) -> Array:
    NUMINF = 1e20
    X_prev, U_prev = problem["X_prev"], problem["U_prev"]
    Q, R, X_ref, U_ref = problem["Q"], problem["R"], problem["X_ref"], problem["U_ref"]
    reg_x, reg_u = problem["reg_x"], problem["reg_u"]
    slew_rate, slew0_rate, u0_slew = problem["slew_rate"], problem["slew0_rate"], problem["u0_slew"]
    x_l, x_u, u_l, u_u = problem["x_l"], problem["x_u"], problem["u_l"], problem["u_u"]
    alpha = problem["smooth_alpha"]

    dX, dU = X - X_ref, U - U_ref
    J = 0.5 * jaxm.mean(jaxm.sum(dX * bmv(Q, dX), axis=-1))
    J = J + 0.5 * jaxm.mean(jaxm.sum(dU * bmv(R, dU), axis=-1))
    J = J + 0.5 * reg_x * jaxm.mean(jaxm.sum((X - X_prev) ** 2, -1))
    J = J + 0.5 * reg_u * jaxm.mean(jaxm.sum((U - U_prev) ** 2, -1))

    # box constraints
    x_l_ = jaxm.where(jaxm.isfinite(x_l), x_l, -NUMINF)
    x_u_ = jaxm.where(jaxm.isfinite(x_u), x_u, NUMINF)
    u_l_ = jaxm.where(jaxm.isfinite(u_l), u_l, -NUMINF)
    u_u_ = jaxm.where(jaxm.isfinite(u_u), u_u, NUMINF)
    J = J + jaxm.mean(
        jaxm.sum(jaxm.where(jaxm.isfinite(x_l), -jaxm.log(-alpha * (-X + x_l_)) / alpha, 0.0), -1)
    )
    J = J + jaxm.mean(
        jaxm.sum(jaxm.where(jaxm.isfinite(x_u), -jaxm.log(-alpha * (X - x_u_)) / alpha, 0.0), -1)
    )
    J = J + jaxm.mean(
        jaxm.sum(jaxm.where(jaxm.isfinite(u_l), -jaxm.log(-alpha * (-U + u_l_)) / alpha, 0.0), -1)
    )
    J = J + jaxm.mean(
        jaxm.sum(jaxm.where(jaxm.isfinite(u_u), -jaxm.log(-alpha * (U - u_u_)) / alpha, 0.0), -1)
    )

    # slew rate
    J_slew = 0.5 * slew_rate * jaxm.mean(jaxm.sum((U[..., :-1, :] - U[..., 1:, :]) ** 2, -1))
    u0_slew = jaxm.where(jaxm.isfinite(u0_slew), u0_slew, 0.0)
    J_slew = J_slew + jaxm.mean(0.5 * slew0_rate * jaxm.sum((U[..., 0, :] - u0_slew) ** 2, -1))
    J = J + jaxm.where(jaxm.isfinite(J_slew), J_slew, 0.0)
    return jaxm.where(jaxm.isfinite(J), J, jaxm.inf)


def default_obj_fn(U: Array, problem: Dict[str, List[Array]]) -> Array:
    Ft, ft, U_prev = problem["Ft"], problem["ft"], problem["U_prev"]
    X = (bmv(Ft, vec(U - U_prev, 2)) + ft).reshape(U.shape[:-1] + (-1,))
    return _default_obj_fn(X, U, problem)


####################################################################################################



def filter_kws(method, d):
    spec = inspect.getfullargspec(method)
    return {k: d[k] for k in d.keys() if k in (list(spec.args) + list(spec.kwonlyargs))}


def generate_routines_for_obj_fn(
    obj_fn: Callable,
    solver_settings: Optional[Dict[str, Any]] = None,
):
    opts = solver_settings if solver_settings is not None else dict()
    try:
        jaxm.jax.devices("gpu")
        device = opts.get("device", "cuda")
    except RuntimeError:
        device = opts.get("device", "cpu")

    nonlinear_opts = dict(
        maxiter=100,
        verbose=False,
        jit=True,
        tol=1e-9,
        linesearch="backtracking",
        min_stepsize=1e-7,
        max_stepsize=1e2,
    )
    cvx_opts = dict(nonlinear_opts, maxls=25, reg0=1e-6, linesearch="binary_search", device=device)
    sqp_opts = dict(cvx_opts, linesearch="scan", maxls=50)
    # nonlinear_opts = dict(nonlinear_opts, **opts)
    cvx_opts = dict(cvx_opts, **opts)
    sqp_opts = dict(sqp_opts, **opts)

    # create solvers with the provided config
    solvers = dict()
    try:
        solvers[SOLVER_BFGS] = jaxopt.BFGS(obj_fn, **filter_kws(jaxopt.BFGS, nonlinear_opts))
    except (AssertionError, ValueError) as e:
        print(f"Could not create BFGS solver: {e}")
        traceback.print_exc()
    try:
        solvers[SOLVER_LBFGS] = jaxopt.LBFGS(obj_fn, **filter_kws(jaxopt.LBFGS, nonlinear_opts))
    except (AssertionError, ValueError) as e:
        print(f"Could not create LBFGS solver: {e}")
        traceback.print_exc()
    try:
        solvers[SOLVER_CVX] = ConvexSolver(obj_fn, **filter_kws(ConvexSolver, cvx_opts))
    except (AssertionError, ValueError) as e:
        print(f"Could not create CVX solver: {e}")
        traceback.print_exc()
    try:
        solvers[SOLVER_SQP] = SQPSolver(obj_fn, **filter_kws(SQPSolver, sqp_opts))
    except (AssertionError, ValueError) as e:
        print(f"Could not create SQP solver: {e}")
        traceback.print_exc()
    run_methods = {k: jaxm.jit(solver.run) for k, solver in solvers.items()}
    update_methods = {k: jaxm.jit(solver.update) for k, solver in solvers.items()}
    # run_methods = {k: solver.run for k, solver in solvers.items()}
    # update_methods = {k: solver.update for k, solver in solvers.items()}

    @partial(jaxm.jit, static_argnums=(0,))
    def run_with_state(
        solver: int, z: Array, args: Dict[str, List[Array]], state, max_it: int = 100
    ):
        update_method = update_methods[solver]

        def body_fn(i, z_state):
            return update_method(*z_state, args)

        z_state = body_fn(0, (z, state))
        return jaxm.jax.lax.fori_loop(1, max_it, body_fn, z_state)

    @partial(jaxm.jit, static_argnums=(0,))
    def init_state(solver: int, U_prev: Array, args: Dict[str, List[Array]]):
        return solvers[solver].init_state(U_prev, args)

    @partial(jaxm.jit, static_argnums=(0,))
    def prun_with_state(
        solver: int, z: Array, args: Dict[str, List[Array]], state, max_it: int = 100
    ):
        in_axes = jaxm.jax.tree_util.tree_map(
            lambda x: 0
            if (hasattr(x, "shape") and x.ndim > 0 and x.shape[0] == z.shape[0])
            else None,
            (solver, z, args, state, max_it),
        )
        return jaxm.jax.vmap(run_with_state, in_axes=in_axes)(solver, z, args, state, max_it)

    @partial(jaxm.jit, static_argnums=(0,))
    def pinit_state(solver: int, U_prev: Array, args: Dict[str, List[Array]]):
        in_axes = jaxm.jax.tree_util.tree_map(
            lambda x: 0
            if (hasattr(x, "shape") and x.ndim > 0 and x.shape[0] == U_prev.shape[0])
            else None,
            (U_prev, args),
        )
        return jaxm.jax.vmap(solvers[solver].init_state, in_axes=in_axes)(U_prev, args)

    return dict(
        solvers=solvers,
        run_methods=run_methods,
        update_methods=update_methods,
        init_state=init_state,
        run_with_state=run_with_state,
        prun_with_state=prun_with_state,
        pinit_state=pinit_state,
    )

####################################################################################################
####################################################################################################
####################################################################################################

class SolverStore:
    def __init__(self):
        self.store = dict()

    def get_routines(self, obj_fn: Callable, solver_settings=None):
        ss = solver_settings if solver_settings is not None else dict()
        obj_fn_key = cp.dumps(
            (hash(obj_fn), tuple((k, ss[k]) for k in STATIC_ARGUMENTS if k in ss.keys()))
        )
        if obj_fn_key not in self.store:
            #print("Generating a new solver")
            self.store[obj_fn_key] = generate_routines_for_obj_fn(obj_fn, ss)
        return (self.store[obj_fn_key]["pinit_state"], self.store[obj_fn_key]["prun_with_state"])

    
SOLVERS_STORE = SolverStore()

####################################################################################################