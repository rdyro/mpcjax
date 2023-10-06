import time
from typing import Optional, Tuple, Dict, Callable, Any
from copy import copy

import jfi
from jfi import jaxm

from jax import Array

from .utils import TablePrinter  # noqa: E402
from .solver_definitions import get_pinit_state, get_prun_with_state, default_obj_fn
from .solver_definitions import SOLVER_BFGS, SOLVER_LBFGS, SOLVER_CVX, SOLVER_SQP
from .dynamics_definitions import get_rollout_and_linearization
from .utils import _jax_sanitize, _to_dtype_device

# utility routines #################################################################################

print_fn = print


def bmv(A, x):
    return (A @ x[..., None])[..., 0]


def vec(x, n=2):
    return x.reshape(x.shape[:-n] + (-1,))


def atleast_nd(x: Optional[Array], n: int):
    if x is None:
        return None
    else:
        return x.reshape((1,) * max(n - x.ndim, 0) + x.shape)


@jaxm.jit
def _U2X(U, U_prev, Ft, ft):
    bshape = U.shape[:-2]
    xdim = ft.shape[-1] // U.shape[-2]
    X = (bmv(Ft, vec(U - U_prev, 2)) + ft).reshape(bshape + (U.shape[-2], xdim))
    return X


# main affine solve for a single iteration of SCP ##################################################
def affine_solve(
    problems: Dict[str, Array],
    reg_x: Array,
    reg_u: Array,
    solver_settings: Optional[Dict[str, Any]] = None,
    diff_cost_fn: Optional[Callable] = None,
    differentiate_rollout: bool = False,
) -> Tuple[Array, Array, Any]:
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
    solver_settings = copy(solver_settings) if solver_settings is not None else dict()
    alpha = solver_settings["smooth_alpha"]
    x0, f, fx, fu = problems["x0"], problems["f"], problems["fx"], problems["fu"]
    X_prev, U_prev = problems["X_prev"], problems["U_prev"]

    # use the linearized dynamics or linearize dynamics ourselves if `differentiate_rollout` is True
    f_fx_fu_fn = problems["f_fx_fu_fn"]
    problems["f_fx_fu_fn"] = None
    if differentiate_rollout:
        Ft_ft_fn = get_rollout_and_linearization(lambda x, u: f_fx_fu_fn(x, u)[0])[1]
        Ft, ft = Ft_ft_fn(x0, U_prev)
    else:
        Ft_ft_fn = get_rollout_and_linearization("default")[1]
        Ft, ft = Ft_ft_fn(x0, U_prev, f, fx, fu, X_prev, U_prev)
    problems["Ft"], problems["ft"] = Ft, ft
    problems["reg_x"], problems["reg_u"] = reg_x, reg_u
    problems["smooth_alpha"] = alpha

    # retrive the solver and the correct settings
    solver = solver_settings.get("solver", "SQP").upper()
    solver_map = dict(
        BFGS=(SOLVER_BFGS, 100),
        LBFGS=(SOLVER_LBFGS, 100),
        CVX=(SOLVER_CVX, 30),
        SQP=(SOLVER_SQP, 30),
    )
    assert solver in solver_map, f"Solver {solver_settings.get('solver')} not supported."
    solver, max_inner_it = solver_map[solver]
    max_inner_it = solver_settings.get("max_it", max_inner_it)

    # define the objective function
    if diff_cost_fn is None:
        obj_fn = default_obj_fn
    else:

        def obj_fn(U, problems):
            return default_obj_fn(U, problems) + diff_cost_fn(
                _U2X(U, problems["U_prev"], problems["Ft"], problems["ft"]), U, problems
            )

    # retrive the (cached) optimization routines based on the objective and solver settings
    pinit_state = get_pinit_state(obj_fn, solver_settings)
    prun_with_state = get_prun_with_state(obj_fn, solver_settings)
    state = solver_settings.get("solver_state", None)

    if state is None or solver in {SOLVER_CVX, SOLVER_SQP}:
        state = pinit_state(solver, U_prev, problems)

    # solve
    U, state = prun_with_state(solver, U_prev, problems, state, max_it=max_inner_it)

    # remove the nans with previous solution (if any)
    mask = jaxm.tile(
        jaxm.isfinite(state.value)[..., None, None], (1,) * state.value.ndim + U.shape[-2:]
    )
    U = jaxm.where(mask, U, U_prev)
    X = _U2X(U, U_prev, Ft, ft)
    ret = jaxm.cat([x0[..., None, :], X], -2), U, dict(solver_state=state, obj=state.value)
    #ret = tree_map(lambda x: x.astype(dtype_org), ret)
    return ret


# cost augmentation ################################################################################
_get_new_ref = jaxm.jit(lambda ref, A, c: ref - jaxm.linalg.solve(A, c[..., None])[..., 0])


def _augment_cost(
    lin_cost_fn: Callable,
    X_prev: Array,
    U_prev: Array,
    Q: Array,
    R: Array,
    X_ref: Array,
    U_ref: Array,
    problems: Optional[Dict[str, Array]] = None,
) -> Tuple[Array, Array]:
    """Modify the linear reference trajectory to account for the linearized non-linear cost term."""
    topts = dict(dtype=X_prev.dtype, device=X_prev.device())
    if lin_cost_fn is not None:
        cx, cu = lin_cost_fn(X_prev, U_prev, problems)

        # augment the state cost #############
        if cx is not None:
            X_ref = _get_new_ref(X_ref, Q, jaxm.to(jaxm.array(cx), **topts))

        # augment the control cost ###########
        if cu is not None:
            U_ref = _get_new_ref(U_ref, R, jaxm.to(jaxm.array(cu), **topts))
    return X_ref, U_ref


# SCP MPC main routine #############################################################################


def scp_solve(
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
    time_limit: float = 1000.0,
    res_tol: float = 1e-5,
    reg_x: float = 1e0,
    reg_u: float = 1e-2,
    slew_rate: Optional[float] = None,
    u0_slew: Optional[Array] = None,
    lin_cost_fn: Optional[Callable] = None,
    diff_cost_fn: Optional[Callable] = None,
    cost_fn: Optional[Callable] = None,  # deprecated, do not use
    solver_settings: Optional[Dict[str, Any]] = None,
    solver_state: Optional[Any] = None,
    return_min_viol: bool = False,
    min_viol_it0: int = -1,
    dtype: Any = None,
    device: Any = "cuda",
    differentiate_rollout: bool = False,
    **extra_kw,
) -> Tuple[Array, Array, Dict[str, Any]]:
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
        time_limit (float, optional): Time limit in seconds. Defaults to 1000.0.
        res_tol (float, optional): Residual tolerance. Defaults to 1e-5.
        reg_x (float, optional): State improvement regularization. Defaults to 1e0.
        reg_u (float, optional): Control improvement regularization. Defaults to 1e-2.
        slew_rate (float, optional): Slew rate regularization. Defaults to 0.0.
        u0_slew (Optional[Array], optional): Slew control to regularize to. Defaults to None.
        lin_cost_fn (Optional[Callable], optional): Linearization of an extra non-linear cost
                                                    function. Defaults to None.
        diff_obj_fn (Optional[Callable], optional): Extra additive obj function. Defaults to None.
        solver_settings (Optional[Dict[str, Any]], optional): Solver settings. Defaults to None.
        return_min_viol (bool, optional): Whether to return minimum violation solution as well.
                                          Defaults to False.
        min_viol_it0 (int, optional): First iteration to store minimum violation solutions.
                                      Defaults to -1, which means immediately.
        dtype: data type to use in the solver
        device: device to use in the solver (e.g., "cpu", "cuda" / "gpu")
        differentiate_rollout: bool: Whether to differentiate the rollout function. Defaults to
                                     False.
        **extra_kw: extra keyword arguments to pass to the objective function.
    Returns:
        Tuple[Array, Array, Dict[str, Any]]: X, U, data
    """
    if cost_fn is not None:
        raise ValueError("cost_fn is deprecated, use lin_cost_fn instead.")

    t_elaps = time.time()
    topts = dict(device=device)
    topts["dtype"] = dtype if dtype is not None else jfi.default_dtype_for_device(device)

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
        single_particle_problem_flag = True
    else:  # multiple particle cases
        assert x0.ndim == 2 and R.ndim == 4 and Q.ndim == 4
        single_particle_problem_flag = False
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
    data = dict(solver_data=[], hist=[], sol_hist=[])

    field_names = ["it", "elaps", "obj", "resid", "reg_x", "reg_u", "alpha"]
    fmts = ["%04d", "%8.3e", "%8.3e", "%8.3e", "%.1e", "%.1e", "%.1e"]
    tp = TablePrinter(field_names, fmts=fmts)
    solver_settings = solver_settings if solver_settings is not None else dict()
    min_viol = jaxm.inf
    # create variables and reference trajectories ##############################

    # solve sequentially, linearizing ##############################################################
    if verbose:
        print_fn(tp.make_header())
    it = 0
    X, U, solver_data = None, None, None
    while it < max_it:
        X_ = jaxm.cat([x0[..., None, :], X_prev[..., :-1, :]], -2)
        f, fx, fu = f_fx_fu_fn(X_, U_prev)
        if fx is not None:
            fx = jaxm.to(jaxm.array(fx), **topts).reshape((M, N, xdim, xdim))
        if fu is not None:
            fu = jaxm.to(jaxm.array(fu), **topts).reshape((M, N, xdim, udim))

        # augment the cost or add extra constraints ################################################
        if "extra_cstrs_fns" in extra_kw:
            msg = "The GPU version does not support custom convex constraints. "
            msg += "Please provide an `diff_obj_fn: Callable[[X, U], Dict[str, Array]]` instead.\n"
            msg += "i.e., The function signature should be `diff_obj_fn(X, U, problem)` where "
            msg += "`problem` contains problem data."
            raise ValueError(msg)
        problems = dict(f_fx_fu_fn=f_fx_fu_fn)
        problems = dict(problems, f=f, fx=fx, fu=fu, x0=x0, X_prev=X_prev, U_prev=U_prev)
        problems = dict(problems, slew_rate=slew_rate, u0_slew=u0_slew)
        problems = dict(problems, x_l=x_l, x_u=x_u, u_l=u_l, u_u=u_u)
        problems = dict(problems, Q=Q, R=R, X_ref=X_ref, U_ref=U_ref)
        problems = dict(_to_dtype_device(extra_kw, **topts), **problems)
        # add user-provided extra arguments
        solver_settings = solver_settings if solver_settings is not None else dict()
        solver_settings["solver_state"] = solver_state
        smooth_alpha = solver_settings.get("smooth_alpha", 1e2)
        if it == 0 and "device" in solver_settings:
            msg = "Warning: `device` option is not supported in `solver_settings`, "
            msg += "specify it via a keyword to the `solve` function directly instead."
            raise ValueError(msg)
        solver_settings = dict(solver_settings, smooth_alpha=smooth_alpha, device=device)
        X_ref_, U_ref_ = _augment_cost(
            lin_cost_fn,
            X_prev,
            U_prev,
            Q,
            R,
            X_ref,
            U_ref,
            dict(problems, solver_settings=_jax_sanitize(solver_settings)),
        )
        problems = dict(problems, X_ref=X_ref_, U_ref=U_ref_)
        # augment the cost or add extra constraints ################################################

        # call the main affine problem solver ######################################################
        t_aff_solve = time.time()
        X, U, solver_data = affine_solve(
            problems,
            reg_x,
            reg_u,
            solver_settings=solver_settings,
            diff_cost_fn=diff_cost_fn,
            differentiate_rollout=differentiate_rollout,
        )
        t_aff_solve = time.time() - t_aff_solve

        solver_state = solver_data.get("solver_state", None)
        X, U = X.reshape((M, N + 1, xdim)), U.reshape((M, N, udim))
        # call the main affine problem solver ######################################################

        # return if the solver failed ##############################################################
        if jaxm.any(jaxm.isnan(X)) or jaxm.any(jaxm.isnan(U)):
            if verbose:
                print_fn("Solver failed...")
            return None, None, None
        # return if the solver failed ##############################################################

        # compute residuals ########################################################################
        X_ = X[..., 1:, :]
        dX, dU = X_ - X_prev, U - U_prev
        max_res = max(jaxm.max(jaxm.linalg.norm(dX, 2, -1)), jaxm.max(jaxm.linalg.norm(dU, 2, -1)))
        dX, dU = X_ - X_ref, U - U_ref
        obj = jaxm.mean(solver_data.get("obj", 0.0))
        X_prev, U_prev = X[..., 1:, :], U
        if extra_kw.get("return_solhist", False):
            data.setdefault("solhist", [])
            data["solhist"].append((X_prev, U_prev))

        t_run = time.time() - t_elaps
        vals = (
            it + 1,
            t_run,
            obj,
            max_res,
            jaxm.mean(reg_x),
            jaxm.mean(reg_u),
            jaxm.mean(smooth_alpha),
        )
        if verbose:
            print_fn(tp.make_values(vals))
        data["solver_data"].append(solver_data)
        data["hist"].append({k: val for (k, val) in zip(field_names, vals)})
        data.setdefault("t_aff_solve", [])
        data["t_aff_solve"].append(t_aff_solve)
        # compute residuals ########################################################################

        # store the minimum violation solution #####################################################
        if return_min_viol and (it >= min_viol_it0 or min_viol_it0 < 0):
            if min_viol > max_res:
                data["min_viol_sol"], min_viol = (X, U), max_res
        # store the minimum violation solution #####################################################

        if max_res < res_tol:
            break
        it += 1
        if (time.time() - t_elaps) * (it + 1) / it > time_limit:
            break
    # solve sequentially, linearizing ##############################################################

    # return the solution ##########################################################################
    if verbose:
        print_fn(tp.make_footer())
    if verbose and max_res > 1e-2:
        msg = "Bad solution found, the solution is approximate to a residual:"
        print_fn("#" * 80)
        print_fn(msg, "%9.4e" % max_res)
        print_fn("#" * 80)
    if not single_particle_problem_flag:
        return X.reshape((M, N + 1, xdim)), U.reshape((M, N, udim)), data
    else:
        return X.reshape((N + 1, xdim)), U.reshape((N, udim)), data
    # return the solution ##########################################################################


solve = scp_solve
