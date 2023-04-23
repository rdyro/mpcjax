import math, os
from typing import Callable, Tuple, Union
from jfi import jaxm
import cloudpickle as cp

bmv = lambda A, x: (A @ x[..., None])[..., 0]

DYNAMICS_STORE = dict()

# linear rollouts based on f, fx, fu ###############################################################


def rollout_step_fx(x, u_f_fx_fu_x_prev_u_prev):
    u, f, fx, fu, x_prev, u_prev = u_f_fx_fu_x_prev_u_prev
    xp = f + bmv(fx, x - x_prev) + bmv(fu, u - u_prev)
    return xp, xp


@jaxm.jit
def rollout_fx(x0, U, f, fx, fu, X_prev, U_prev):
    """Rolls out dynamics into the future based on an initial state x0"""
    xs = [x0[..., None, :]]
    if X_prev.shape[-2] == U_prev.shape[-2]:
        X_prev = jaxm.cat([x0[..., None, :], X_prev[..., :-1, :]], -2)
    else:
        X_prev = X_prev[..., :-1, :]
    U, f, X_prev, U_prev = [jaxm.moveaxis(z, -2, 0) for z in [U, f, X_prev, U_prev]]
    fx, fu = [jaxm.moveaxis(z, -3, 0) for z in [fx, fu]]
    xs = jaxm.moveaxis(jaxm.lax.scan(rollout_step_fx, x0, (U, f, fx, fu, X_prev, U_prev))[1], 0, -2)
    return jaxm.cat([x0[..., None, :], xs], -2)


@jaxm.jit
def Ft_ft_fn(x0, U, f, fx, fu, X_prev, U_prev):
    bshape, N, xdim, udim = U.shape[:-2], U.shape[-2], X_prev.shape[-1], U.shape[-1]

    ft_ = rollout_fx(x0, U, f, fx, fu, X_prev, U_prev)[..., 1:, :]
    sum_axes = tuple(range(0, ft_.ndim - 2))
    Ft_ = jaxm.jacobian(
        lambda U: jaxm.sum(rollout_fx(x0, U, f, fx, fu, X_prev, U_prev)[..., 1:, :], sum_axes)
    )(U)
    Ft_ = jaxm.moveaxis(Ft_, -3, 0)
    Ft, ft = Ft_, ft_

    Ft, ft = Ft.reshape(bshape + (N * xdim, N * udim)), ft.reshape(bshape + (N * xdim,))
    return Ft, ft


# dynamically generated rollouts based on a dynamics function ######################################


def generate_rollout_step(dyn_fn):
    def rollout_step(x, u):
        xp = dyn_fn(x, u)
        return xp, xp

    return rollout_step


def generate_rollout_and_linearization(dyn_fn):
    rollout_step = generate_rollout_step(dyn_fn)

    @jaxm.jit
    def rollout(x0, U):
        """Rolls out dynamics into the future based on an initial state x0"""
        xs = [x0[..., None, :]]
        U = jaxm.moveaxis(U, -2, 0)
        xs = jaxm.moveaxis(jaxm.lax.scan(rollout_step, x0, U)[1], 0, -2)
        return jaxm.cat([x0[..., None, :], xs], -2)

    @jaxm.jit
    def Ft_ft_fn(x0, U):
        bshape, N, xdim, udim = U.shape[:-2], U.shape[-2], x0.shape[-1], U.shape[-1]

        ft_ = rollout(x0, U)[..., 1:, :]
        sum_axes = tuple(range(0, ft_.ndim - 2))
        Ft_ = jaxm.jacobian(lambda U: jaxm.sum(rollout(x0, U)[..., 1:, :], sum_axes))(U)
        Ft_ = jaxm.moveaxis(Ft_, -3, 0)
        Ft, ft = Ft_, ft_

        Ft, ft = Ft.reshape(bshape + (N * xdim, N * udim)), ft.reshape(bshape + (N * xdim,))
        return Ft, ft

    return rollout, Ft_ft_fn


####################################################################################################


def get_rollout_and_linearization(dyn_fn: Union[Callable, str]) -> Tuple[Callable, Callable]:
    if isinstance(dyn_fn, str):
        if dyn_fn.lower() == "default":
            return rollout_fx, Ft_ft_fn
        else:
            raise ValueError(
                "Only `default` key for dynamics is supported by string, pass a function instead."
            )

    dyn_fn_key = cp.dumps(dyn_fn)
    if dyn_fn_key not in DYNAMICS_STORE:
        DYNAMICS_STORE[dyn_fn_key] = generate_rollout_and_linearization(dyn_fn)
    return DYNAMICS_STORE[dyn_fn_key]
