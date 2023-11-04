from __future__ import annotations

from typing import Callable, Tuple, Union, Any

import cloudpickle as cp
from jfi import jaxm
from jax import Array


def bmv(A, x):
    return (A @ x[..., None])[..., 0]


DYN_FN_STORE = dict()
DYNAMICS_FN_STORE = dict()

# linear rollouts based on f, fx, fu ###############################################################


def rollout_step_fx(x, u_f_fx_fu_x_prev_u_prev):
    u, f, fx, fu, x_prev, u_prev = u_f_fx_fu_x_prev_u_prev
    xp = f + bmv(fx, x - x_prev) + bmv(fu, u - u_prev)
    return xp, xp


# def rollout_fx(x0, U, f, fx, fu, X_prev, U_prev):
@jaxm.jit
def rollout_scp(U, problems):
    """Rolls out dynamics into the future based on an initial state x0"""
    x0, f, fx, fu, X_prev, U_prev = [
        problems[z] for z in ["x0", "f", "fx", "fu", "X_prev", "U_prev"]
    ]
    xs = [x0[..., None, :]]
    if X_prev.shape[-2] == U_prev.shape[-2]:
        X_prev = jaxm.cat([x0[..., None, :], X_prev[..., :-1, :]], -2)
    else:
        X_prev = X_prev[..., :-1, :]
    U, f, X_prev, U_prev = [jaxm.moveaxis(z, -2, 0) for z in [U, f, X_prev, U_prev]]
    fx, fu = [jaxm.moveaxis(z, -3, 0) for z in [fx, fu]]
    xs = jaxm.moveaxis(jaxm.lax.scan(rollout_step_fx, x0, (U, f, fx, fu, X_prev, U_prev))[1], 0, -2)
    return jaxm.cat([x0[..., None, :], xs], -2)


# def Ft_ft_fn(x0, U, f, fx, fu, X_prev, U_prev):
@jaxm.jit
def Ft_ft_fn_scp(U, problems):
    X_prev = problems["X_prev"]
    bshape, N, xdim, udim = U.shape[:-2], U.shape[-2], X_prev.shape[-1], U.shape[-1]

    ft_ = rollout_scp(U, problems)[..., 1:, :]
    sum_axes = tuple(range(0, ft_.ndim - 2))
    Ft_ = jaxm.jacobian(lambda U: jaxm.sum(rollout_scp(U, problems)[..., 1:, :], sum_axes))(U)
    Ft_ = jaxm.moveaxis(Ft_, -3, 0)
    Ft, ft = Ft_, ft_

    Ft, ft = Ft.reshape(bshape + (N * xdim, N * udim)), ft.reshape(bshape + (N * xdim,))
    return Ft, ft
