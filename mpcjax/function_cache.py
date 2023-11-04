from __future__ import annotations

from typing import Any, Callable

from jfi import jaxm
import jax
from jax import Array


####################################################################################################


class ObjectiveFunctionStore:
    def __init__(self):
        self.store = dict()

    def _generate_objective_function(
        self, rollout_fn: Callable, obj_fn: Callable, custom_obj_fn: Callable | None = None
    ) -> Callable:
        @jax.jit
        def _obj_fn(U: Array, problems: dict[str, Any]) -> Array:
            X = rollout_fn(U, problems)[..., 1:, :]
            J = obj_fn(X, U, problems)
            if custom_obj_fn is not None:
                J = J + custom_obj_fn(X, U, problems)
            return J

        return _obj_fn

    def get_obj_fn(
        self, rollout_fn: Callable, obj_fn: Callable, custom_obj_fn: Callable | None = None
    ) -> Callable:
        fn_key = hash((rollout_fn, obj_fn, custom_obj_fn))
        if fn_key not in self.store:
            self.store[fn_key] = self._generate_objective_function(
                rollout_fn, obj_fn, custom_obj_fn
            )
        return self.store[fn_key]


####################################################################################################
####################################################################################################
####################################################################################################


class DynamicsFunctionStore:
    def __init__(self):
        self.rollout_store = dict()
        self.dyn_store = dict()
        self.Ft_ft_store = dict()

    def _generate_rollout(self, dyn_fn: Callable) -> Callable:
        @jaxm.jit
        def rollout(U, problems):
            """Rolls out dynamics into the future based on an initial state x0"""

            def rollout_step(x, u_p):
                u, p = u_p
                xp = dyn_fn(x, u, p)
                return xp, xp

            x0, P = problems["x0"], problems["P"]
            xs = [x0[..., None, :]]
            U = jaxm.moveaxis(U, -2, 0)
            P = jaxm.broadcast_to(P, U.shape[:-1] + P.shape[-1:])
            xs = jaxm.moveaxis(jaxm.lax.scan(rollout_step, x0, (U, P))[1], 0, -2)
            return jaxm.cat([x0[..., None, :], xs], -2)

        return rollout

    def _generate_linearization(self, dyn_fn: Callable) -> Callable:
        rollout_fn = self.get_rollout_fn(dyn_fn)

        @jaxm.jit
        def Ft_ft_fn(x0, U, P):
            bshape, N, xdim, udim = U.shape[:-2], U.shape[-2], x0.shape[-1], U.shape[-1]

            ft_ = rollout_fn(x0, U, P)[..., 1:, :]
            sum_axes = tuple(range(0, ft_.ndim - 2))
            Ft_ = jaxm.jacobian(lambda U: jaxm.sum(rollout_fn(x0, U, P)[..., 1:, :], sum_axes))(U)
            Ft_ = jaxm.moveaxis(Ft_, -3, 0)
            Ft, ft = Ft_, ft_

            Ft, ft = Ft.reshape(bshape + (N * xdim, N * udim)), ft.reshape(bshape + (N * xdim,))
            return Ft, ft

        return Ft_ft_fn

    def get_dyn_function(self, f_fx_fu_fn: Callable) -> Callable:
        fn_key = hash(f_fx_fu_fn)
        if fn_key not in self.dyn_store:
            self.dyn_store[fn_key] = jaxm.jit(lambda x, u, p: f_fx_fu_fn(x, u, p)[0])
        return self.dyn_store[fn_key]

    def get_rollout_fn(self, dyn_fn: Callable | str) -> Callable:
        dyn_fn_key = hash(dyn_fn)
        if dyn_fn_key not in self.rollout_store:
            self.rollout_store[dyn_fn_key] = self._generate_rollout(dyn_fn)
        return self.rollout_store[dyn_fn_key]

    def get_linearization_fn(self, dyn_fn: Callable | str) -> tuple[Callable, Callable]:
        dyn_fn_key = hash(dyn_fn)
        if dyn_fn_key not in self.Ft_ft_store:
            self.Ft_ft_store[dyn_fn_key] = self._generate_linearization(dyn_fn)
        return self.Ft_ft_store[dyn_fn_key]


####################################################################################################
####################################################################################################
####################################################################################################

OBJECTIVE_FUNCTION_STORE = ObjectiveFunctionStore()
DYNAMICS_FUNCTION_STORE = DynamicsFunctionStore()
