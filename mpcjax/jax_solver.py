from __future__ import annotations

import time
import math
from typing import Any, Callable, Optional
from functools import partial

from jaxfi import jaxm
import jax
from jax import Array

from .utils import TablePrinter  # noqa: E402
from .utils import _jax_sanitize, _to_dtype_device, atleast_nd
from .direct_solve import direct_affine_solve
from .scp_solve import scp_affine_solve
from .solver_settings import SolverSettings

# utility routines #################################################################################

print_fn = print

# cost augmentation ################################################################################
_get_new_ref = jaxm.jit(lambda ref, A, c: ref - jaxm.linalg.solve(A, c[..., None])[..., 0])


@partial(
    jax.jit,
    static_argnames=(
        "lin_cost_fn",
        "force_dtype",
    ),
)
def _augment_cost(
    lin_cost_fn: Callable,
    X_prev: Array,
    U_prev: Array,
    Q: Array,
    R: Array,
    X_ref: Array,
    U_ref: Array,
    problems: dict[str, Array] | None = None,
    force_dtype: bool = True,
) -> tuple[Array, Array]:
    """Modify the linear reference trajectory to account for the linearized non-linear cost term."""
    if lin_cost_fn is None:
        return X_ref, U_ref

    topts = dict(dtype=X_prev.dtype) if force_dtype else dict()

    if lin_cost_fn is not None:
        cx, cu = lin_cost_fn(X_prev, U_prev, problems)

        # augment the state cost #############
        if cx is not None:
            cx = jaxm.to(cx, **topts) if force_dtype else cx
            X_ref = _get_new_ref(X_ref, Q, cx)

        # augment the control cost ###########
        if cu is not None:
            cu = jaxm.to(cu, **topts) if force_dtype else cu
            U_ref = _get_new_ref(U_ref, R, cu)
    return X_ref, U_ref


# SCP MPC main routine #############################################################################


def _debug_callback(reg_x, reg_u, total_it, objs, resids, smooth_alpha):
    field_names = ["it", "elaps", "obj", "resid", "reg_x", "reg_u", "alpha"]
    fmts = ["%04d", "%8.3e", "%8.3e", "%8.3e", "%.1e", "%.1e", "%.1e"]
    tp = TablePrinter(field_names, fmts=fmts)
    print_fn(tp.make_header())
    for it in range(total_it):
        vals = (
            it + 1,
            0.0,
            objs[it],
            resids[it],
            reg_x,
            reg_u,
            smooth_alpha,
        )
        print_fn(tp.make_values(vals))
    print_fn(tp.make_footer())


def _debug_get_time():
    return jax.pure_callback(
        lambda: jaxm.array(time.time(), dtype=jaxm.float64), jax.ShapeDtypeStruct((), jaxm.float64)
    )


def _build_problems(
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
    slew0_rate: Optional[float] = None,
    u0_slew: Optional[Array] = None,
    cost_fn: Optional[Callable] = None,  # deprecated, do not use
    solver_settings: SolverSettings | None = None,
    dtype: Any | None = None,
    smooth_alpha: float = 1e2,
    **extra_kw,
) -> dict[str, Any]:
    if cost_fn is not None:
        raise ValueError("cost_fn is deprecated, use lin_cost_fn instead.")

    dtype = Q.dtype if dtype is None else dtype
    topts = dict(dtype=dtype)

    # create variables and reference trajectories ##############################
    x0 = jaxm.to(jaxm.array(x0), **topts)
    reg_x, reg_u = jaxm.to(jaxm.array(reg_x), **topts), jaxm.to(jaxm.array(reg_u), **topts)
    Q, R = jaxm.to(jaxm.copy(Q), **topts), jaxm.to(jaxm.copy(R), **topts)
    if x0.ndim == 1:  # single particle case
        assert x0.ndim == 1 and R.ndim == 3 and Q.ndim == 3
        args = Q, R, x0, X_ref, U_ref, X_prev, U_prev, x_l, x_u, u_l, u_u
        dims = [4, 4, 2, 3, 3, 3, 3, 3, 3, 3, 3]
        args = [atleast_nd(z, dim) for (z, dim) in zip(args, dims)]
        Q, R, x0, X_ref, U_ref, X_prev, U_prev, x_l, x_u, u_l, u_u = args
    else:  # multiple particle cases
        assert x0.ndim == 2 and R.ndim == 4 and Q.ndim == 4
    M, N, xdim, udim = Q.shape[:3] + R.shape[-1:]

    X_ref = (
        jaxm.zeros((M, N, xdim), **topts) if X_ref is None else jaxm.to(jaxm.array(X_ref), **topts)
    )
    U_ref = (
        jaxm.zeros((M, N, udim), **topts) if U_ref is None else jaxm.to(jaxm.array(U_ref), **topts)
    )
    X_prev = jaxm.to(jaxm.array(X_prev), **topts) if X_prev is not None else X_ref
    U_prev = jaxm.to(jaxm.array(U_prev), **topts) if U_prev is not None else U_ref
    X_prev, U_prev = X_prev.reshape((M, N, xdim)), U_prev.reshape((M, N, udim))
    X_ref, U_ref = X_ref.reshape((M, N, xdim)), U_ref.reshape((M, N, udim))
    x_l = (
        jaxm.to(jaxm.array(x_l), **topts)
        if x_l is not None
        else jaxm.nan * jaxm.ones(X_prev.shape, **topts)
    )
    x_u = (
        jaxm.to(jaxm.array(x_u), **topts)
        if x_u is not None
        else jaxm.nan * jaxm.ones(X_prev.shape, **topts)
    )
    u_l = (
        jaxm.to(jaxm.array(u_l), **topts)
        if u_l is not None
        else jaxm.nan * jaxm.ones(U_prev.shape, **topts)
    )
    u_u = (
        jaxm.to(jaxm.array(u_u), **topts)
        if u_u is not None
        else jaxm.nan * jaxm.ones(U_prev.shape, **topts)
    )
    u0_slew = (
        jaxm.to(jaxm.array(u0_slew), **topts)
        if u0_slew is not None
        else jaxm.nan * jaxm.ones(x0.shape[:-1] + (U_prev.shape[-1],), **topts)
    )
    slew_rate = slew_rate if slew_rate is not None else 0.0
    slew0_rate = slew0_rate if slew0_rate is not None else 0.0
    solver_settings = solver_settings if solver_settings is not None else SolverSettings()
    P = extra_kw.get("P", None)

    # create variables and reference trajectories ##############################

    problems = dict(M=M, N=N, xdim=xdim, udim=udim, reg_x=reg_x, reg_u=reg_u, f_fx_fu_fn=f_fx_fu_fn)
    problems = dict(problems, x0=x0, X_prev=X_prev, U_prev=U_prev)
    problems = dict(problems, slew_rate=slew_rate, slew0_rate=slew0_rate, u0_slew=u0_slew)
    problems = dict(problems, x_l=x_l, x_u=x_u, u_l=u_l, u_u=u_u)
    problems = dict(problems, Q=Q, R=R, X_ref=X_ref, U_ref=U_ref, P=P, smooth_alpha=smooth_alpha)
    problems = dict(_to_dtype_device(extra_kw, **topts), **problems)
    problems = dict(problems, solver_settings=solver_settings)
    return problems


static_argnames = (
    "f_fx_fu_fn",
    "max_it",
    "verbose",
    "diff_cost_fn",
    "lin_cost_fn",
    "dtype",
    "direct_solve",
    "solver_settings",
)


@partial(jax.jit, static_argnames=static_argnames)
def solve(
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
    verbose: bool = False,
    max_it: int = 100,
    res_tol: float = 1e-5,
    reg_x: float = 1e0,
    reg_u: float = 1e-2,
    slew_rate: Optional[float] = None,
    u0_slew: Optional[Array] = None,
    lin_cost_fn: Optional[Callable] = None,
    diff_cost_fn: Optional[Callable] = None,
    cost_fn: Optional[Callable] = None,  # deprecated, do not use
    solver_settings: SolverSettings | None = None,
    dtype: Any | None = None,
    direct_solve: bool = False,
    smooth_alpha: float = 1e2,
    **extra_kw,
) -> tuple[Array, Array, dict[str, Any]]:
    """Compute the SCP solution to a non-linear dynamics, quadratic cost, control problem with
    optional non-linear cost term.

    Args:
        f_fx_fu_fn (Callable): Dynamics with linearization callable.
        Q (Array): The quadratic state cost.
        R (Array): The quadratic control cost.
        x0 (Array): Initial state.
        X_ref (Optional[Array], optional): Reference state trajectory. Defaults to zeros.
        U_ref (Optional[Array], optional): Reference control trajectory. Defaults to zeros.
        X_prev (Optional[Array], optional): Previous state solution. Defaults to x0.
        U_prev (Optional[Array], optional): Previous control solution. Defaults to zeros.
        x_l (Optional[Array], optional): Lower bound state constraint. Defaults to no cstrs.
        x_u (Optional[Array], optional): Upper bound state constraint. Defaults to no cstrs.
        u_l (Optional[Array], optional): Lower bound control constraint.. Defaults to no cstrs.
        u_u (Optional[Array], optional): Upper bound control constraint.. Defaults to no cstrs.
        verbose (bool, optional): Whether to print output. Defaults to False.
        max_it (int, optional): Max number of SCP iterations. Defaults to 100.
        res_tol (float, optional): Residual tolerance. Defaults to 1e-5.
        reg_x (float, optional): State improvement regularization. Defaults to 1e0.
        reg_u (float, optional): Control improvement regularization. Defaults to 1e-2.
        slew_rate (float, optional): Slew rate regularization. Defaults to 0.0.
        u0_slew (Optional[Array], optional): Slew control to regularize to. Defaults to None.
        lin_cost_fn (Optional[Callable], optional): Linearization of an extra non-linear cost
                                                    function. Defaults to None.
        diff_obj_fn (Optional[Callable], optional): Extra additive obj function. Defaults to None.
        solver_settings (Optional[SolverSettings], optional): Solver settings. Defaults to None.
        dtype: data type to use in the solver
        differentiate_rollout: bool: Whether to differentiate the rollout function. Defaults to
                                     False.
        **extra_kw: extra keyword arguments to pass to the objective function.
    Returns:
        tuple[Array, Array, dict[str, Any]]: X, U, data
    """
    two_dim_output = x0.ndim == 1

    dtype = Q.dtype if dtype is None else dtype
    topts = dict(dtype=dtype)

    data = dict(solver_data=[], hist=[], sol_hist=[])
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
        cost_fn=cost_fn,
        solver_settings=solver_settings,
        dtype=dtype,
        smooth_alpha=smooth_alpha,
        **extra_kw,
    )

    # solve sequentially, linearizing ##############################################################
    X_prev, U_prev = problems["X_prev"], problems["U_prev"]
    x0 = problems["x0"]

    def loop_fn(arg):
        it, X_prev, U_prev, objs, resids = arg
        X_ = jaxm.cat([x0[..., None, :], X_prev[..., :-1, :]], -2)
        f, fx, fu = f_fx_fu_fn(X_, U_prev, problems["P"])
        if fx is not None:
            fx = jaxm.to(jaxm.array(fx), **topts)
            fx = fx.reshape((problems["M"], problems["N"], problems["xdim"], problems["xdim"]))
        if fu is not None:
            fu = jaxm.to(jaxm.array(fu), **topts)
            fu = fu.reshape((problems["M"], problems["N"], problems["xdim"], problems["udim"]))

        # augment the cost or add extra constraints ################################################
        if "extra_cstrs_fns" in extra_kw:
            msg = "The GPU version does not support custom convex constraints. "
            msg += "Please provide an `diff_obj_fn: Callable[[X, U], dict[str, Array]]` instead.\n"
            msg += "i.e., The function signature should be `diff_obj_fn(X, U, problem)` where "
            msg += "`problem` contains problem data."
            raise ValueError(msg)
        problems_with_f_fx_fu = dict(problems, f=f, fx=fx, fu=fu, X_prev=X_prev, U_prev=U_prev)
        X_ref_, U_ref_ = _augment_cost(
           lin_cost_fn,
           X_prev,
           U_prev,
           problems["Q"],
           problems["R"],
           problems["X_ref"],
           problems["U_ref"],
           _jax_sanitize(problems),
        )
        X_ref_, U_ref_ = problems["X_ref"], problems["U_ref"]
        problems_to_solve = dict(problems_with_f_fx_fu, X_ref=X_ref_, U_ref=U_ref_)
        # augment the cost or add extra constraints ################################################

        # call the main affine problem solver ######################################################
        if not direct_solve:
            X, U, solver_data = scp_affine_solve(diff_cost_fn=diff_cost_fn, **problems_to_solve)
        else:
            X, U, solver_data = direct_affine_solve(**problems_to_solve, diff_cost_fn=diff_cost_fn)

        X = X.reshape((problems["M"], problems["N"] + 1, problems["xdim"]))
        U = U.reshape((problems["M"], problems["N"], problems["udim"]))
        # call the main affine problem solver ######################################################

        # compute residuals ########################################################################
        X_ = X[..., 1:, :]
        dX, dU = X_ - X_prev, U - U_prev
        resid = jaxm.maximum(
            jaxm.max(jaxm.linalg.norm(dX, 2, -1)), jaxm.max(jaxm.linalg.norm(dU, 2, -1))
        )
        obj = jaxm.mean(solver_data.get("obj", 0.0))
        X_prev, U_prev = X_, U  # non-jit optimization
        objs = objs.at[it].set(obj)
        resids = resids.at[it].set(resid)
        return it + 1, X_prev, U_prev, objs, resids

    def cond_fn(it_X_prev_U_prev):
        it, X_prev, U_prev, objs, resids = it_X_prev_U_prev
        resid_it = jaxm.maximum(0, it - 1)
        resid_cond = jaxm.logical_or(resids[resid_it] > res_tol, it < 1)
        nan_cond = jaxm.logical_and(jaxm.all(jaxm.isnan(X_prev)), jaxm.all(jaxm.isnan(U_prev)))
        return jaxm.logical_and(
            it < max_it, jaxm.logical_and(resid_cond, jaxm.logical_not(nan_cond))
        )

    objs, resids = math.nan * jaxm.ones((max_it,)), math.nan * jaxm.ones((max_it,))
    total_it, X_prev, U_prev, objs, resids = jaxm.while_loop(
        cond_fn, loop_fn, (0, X_prev, U_prev, objs, resids)
    )

    X, U = jaxm.cat([x0[..., None, :], X_prev], -2), U_prev
    # data["objs"] = objs
    # data["resids"] = resids

    # solve sequentially, linearizing ##############################################################
    # return the solution ##########################################################################
    if verbose:
        jax.debug.callback(_debug_callback, reg_x, reg_u, total_it, objs, resids, smooth_alpha)
        # print_fn(tp.make_footer())
    # if verbose and max_res > 1e-2:
    #    msg = "Bad solution found, the solution is approximate to a residual:"
    #    print_fn("#" * 80)
    #    print_fn(msg, "%9.4e" % max_res)
    #    print_fn("#" * 80)
    if two_dim_output:
        X = X.reshape((problems["N"] + 1, problems["xdim"]))
        U = U.reshape((problems["N"], problems["udim"]))
        return X, U, data
    else:
        X = X.reshape((problems["M"], problems["N"] + 1, problems["xdim"]))
        U = U.reshape((problems["M"], problems["N"], problems["udim"]))
        return X, U, data
    # return the solution ##########################################################################


####################################################################################################
