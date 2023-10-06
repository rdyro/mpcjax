from __future__ import annotations

from enum import Enum
from typing import NamedTuple, Callable, Any
from functools import partial

from jfi import jaxm
from jaxopt import base  # noqa: E402
import jax
from jax import numpy as jnp

#from .utils import balanced_solve, f64_solve


class LineSearch(Enum):
    backtracking = 0
    scan = 1
    binary_search = 2


class ConvexState(NamedTuple):
    best_params: jaxm.jax.Array
    best_loss: jaxm.jax.Array
    value: jaxm.jax.Array
    aux: Any | None = None


####################################################################################################


class ConvexSolver(base.IterativeSolver):
    def __init__(
        self,
        fun,
        maxiter=100,
        tol=1e-9,
        verbose=False,
        jit=True,
        maxls=20,
        min_stepsize=1e-7,
        max_stepsize=1e1,
        reg0=1e-6,
        linesearch="scan",
        force_step: bool = False,
        g_fn: Callable | None = None,
        h_fn: Callable | None = None,
        has_aux: bool = False,
    ):
        """A jaxopt compatible iterative solver for convex problems only.

        Args:
            fun (Callable): Convex objective function to minimize.
            maxiter (int, optional): Number of iterations in the `run` method. Defaults to 100.
            tol (float, optional): Absolute tolerance (gradient norm). Defaults to 1e-9.
            verbose (bool, optional): Whether to print debugging output. Defaults to False.
            jit (bool, optional): Whether to jit objective and derivatives. Defaults to True.
            maxls (int, optional): Max num of objective evaluations in lineasearch. Defaults to 20.
            min_stepsize (int, optional): Minimum stepsize to take. Defaults to 1e-7.
            max_stepsize (int, optional): Maximum stepsie to take. Defaults to 1e1.
            reg0 (float, optional): Hessian l2 regularization. Defaults to 1e-6.
            linesearch (str, optional): Which linesearch to use from
                                    ["scan", "backtracking", "binary_search"]. Defaults to "scan".
        """
        self.maxiter, self.maxls = maxiter, maxls
        self.min_stepsize, self.max_stepsize = min_stepsize, max_stepsize
        self.tol, self.reg0 = tol, reg0
        self.verbose = verbose
        self.force_step = force_step
        self.has_aux = has_aux
        assert linesearch.lower() in ["scan", "backtracking", "binary_search"]
        if linesearch.lower() == "scan":
            self.linesearch = LineSearch.scan
        elif linesearch.lower() == "backtracking":
            self.linesearch = LineSearch.backtracking
        elif linesearch.lower() == "binary_search":
            self.linesearch = LineSearch.binary_search
        else:
            raise ValueError(f"Unknown linesearch: {linesearch}")

        def g_fn_constructed(params, *args):
            if g_fn is not None:
                g = g_fn(params, *args)
            else:
                if self.has_aux:
                    g = jaxm.grad(lambda params, *args: fun(params, *args)[0])(params, *args)
                else:
                    g = jaxm.grad(fun)(params, *args)
            return g.reshape((params.size,))

        def h_fn_constructed(params, *args):
            if h_fn is not None:
                H = h_fn(params, *args)
            else:
                if self.has_aux:
                    H = jaxm.hessian(lambda params, *args: fun(params, *args)[0])(params, *args)
                else:
                    H = jaxm.hessian(fun)(params, *args)
            return H.reshape((params.size, params.size))

        if jit:
            self.f_fn = jaxm.jit(fun)
            self.g_fn = jaxm.jit(g_fn_constructed)
            self.h_fn = jaxm.jit(h_fn_constructed)
        else:
            self.f_fn = fun
            self.g_fn = g_fn_constructed
            self.h_fn = h_fn_constructed

    # @partial(jax.jit, static_argnums=(0,), static_argnames=("maxls", "force_step", "linesearch"))
    def init_state(self, params, *args, **config):
        if self.has_aux:
            best_loss, aux = self.f_fn(params, *args)
            return ConvexState(params, best_loss, best_loss, aux)
        else:
            best_loss = self.f_fn(params, *args)
            return ConvexState(params, best_loss, best_loss)

    # @partial(jax.jit, static_argnums=(0,), static_argnames=("maxls", "force_step", "linesearch"))
    def update(self, params, state, *args, **config):
        g, H = self.g_fn(params, *args), self.h_fn(params, *args)
        cond = jaxm.norm(g) > self.tol
        return jaxm.jax.lax.cond(
            cond,
            lambda: self._update(g, H, params, state, *args, **config),
            lambda: (params, state),
        )

    def _update(self, g, H, params, state, *args, **config):
        """Update routine based on the values of the gradient and Hessian."""
        dtype = H.dtype
        linesearch = config.get("linesearch", self.linesearch)
        maxls = config.get("maxls", self.maxls)
        force_step = config.get("force_step", self.force_step)

        # compute descent direction
        dp = -jaxm.scipy.linalg.cho_solve(
            jaxm.scipy.linalg.cho_factor(H + self.reg0 * jnp.eye(H.shape[-1], dtype=dtype)), g
        ).reshape(params.shape)
        # dp = -jaxm.linalg.solve(H + self.reg0 * jnp.eye(H.shape[-1], dtype=dtype), g).reshape(
        #    params.shape
        # )
        # dp = -balanced_solve(H + self.reg0 * jnp.eye(H.shape[-1], dtype=dtype), g[..., None])[
        #    ..., 0
        # ].reshape(params.shape)
        # dp = -f64_solve(H + self.reg0 * jnp.eye(H.shape[-1], dtype=dtype), g[..., None])[
        #    ..., 0
        # ].reshape(params.shape)
        if linesearch == LineSearch.scan:
            lower_ls = max(round(float(0.7 * maxls)), 1)
            bets_low = jnp.logspace(jaxm.log10(self.min_stepsize), 0.0, lower_ls, dtype=dtype)
            bets_up = jnp.logspace(
                0.0, jaxm.log10(self.max_stepsize), maxls - lower_ls, dtype=dtype
            )[1:]
            bets = jaxm.cat([bets_low, bets_up], -1)

            if self.has_aux:
                losses = jaxm.jax.vmap(lambda bet: self.f_fn(params + bet * dp, *args)[0])(bets)
            else:
                losses = jaxm.jax.vmap(lambda bet: self.f_fn(params + bet * dp, *args))(bets)
            losses = jaxm.where(jaxm.isnan(losses), jaxm.inf, losses)
            idx = jaxm.argmin(losses)
            bet, new_loss = bets[idx], losses[idx]
            new_params = params + bet * dp
        elif linesearch == LineSearch.backtracking:

            def cond_fn(step):
                step_not_too_small = step >= self.min_stepsize
                not_better_loss = self.f_fn(params + step * dp, *args) > state.best_loss
                return jaxm.logical_and(step_not_too_small, not_better_loss)

            def body_fn(step):
                return step * 0.7

            step_size = jaxm.jax.lax.while_loop(cond_fn, body_fn, 1.0)
            new_params = params + step_size * dp
            new_loss = self.f_fn(new_params, *args)
        elif linesearch == LineSearch.binary_search:

            def body_fn(i, val):
                betl, betr, fl, fr = val
                # betm = 10.0 ** ((jaxm.log10(betl) + jaxm.log10(betr)) / 2.0)
                betm = (betl + betr) / 2.0
                fm = self.f_fn(params + betm * dp, *args)
                cond = fl < fr
                betr, betl = jaxm.where(cond, betm, betr), jaxm.where(cond, betl, betm)
                fr, fl = jaxm.where(cond, fm, fr), jaxm.where(cond, fl, fm)
                return (betl, betr, fl, fr)

            betl, betr = self.min_stepsize, 1e1
            fl = self.f_fn(params + betl * dp, *args)
            fr = self.f_fn(params + betr * dp, *args)
            betl, betr, fl, fr = jaxm.jax.lax.fori_loop(
                0, jaxm.maximum(maxls - 2, 1), body_fn, (betl, betr, fl, fr)
            )
            cond = fl < fr
            new_params = jaxm.where(cond, params + betl * dp, params + betr * dp)
            new_loss = jaxm.where(cond, fl, fr)
        else:
            raise ValueError(f"Unknown line search method {linesearch}")

        best_params = jaxm.where(new_loss <= state.best_loss, new_params, state.best_params)
        best_loss = jaxm.where(new_loss <= state.best_loss, new_loss, state.best_loss)
        if self.has_aux:
            _, aux = self.f_fn(best_params, *args)
            state = ConvexState(best_params, best_loss, best_loss, aux)
        else:
            state = ConvexState(best_params, best_loss, best_loss)
        if not force_step:
            new_params = state.best_params
        return new_params, state


####################################################################################################
cho_factor = jaxm.scipy.linalg.cho_factor


@partial(jaxm.jit, static_argnames=("max_iter",))
def _find_cholesky_factorization(H, Fl, laml, max_iter):
    # find the right lambda ###########################
    def cond_fn(lam_F):
        _, _, F = lam_F
        return jaxm.logical_not(jaxm.isfinite(F[0][0, 0]))

    def body_fn(lam_F):
        _, lam, _ = lam_F
        lam, lam_prev = lam * 5, lam
        F = cho_factor(H + lam * jnp.eye(H.shape[-1], dtype=H.dtype))
        return (lam_prev, lam, F)

    F = cho_factor(H + laml * jnp.eye(H.shape[-1], dtype=H.dtype))
    laml, lamr, Fr = jaxm.jax.lax.while_loop(cond_fn, body_fn, (laml, laml, F))

    # find the middle lambda via bisection ############
    def body_fn(i, val):
        laml, lamr, Fl, Fr = val
        lamm = (laml + lamr) / 2.0
        Fm = cho_factor(H + lamm * jnp.eye(H.shape[-1], dtype=H.dtype))
        cond = jaxm.isfinite(Fm[0][0, 0])
        laml, lamr = jaxm.where(cond, laml, lamm), jaxm.where(cond, lamm, lamr)
        Fl = (jaxm.where(cond, Fl[0], Fm[0]), jaxm.where(cond, Fl[1], Fm[1]))
        Fr = (jaxm.where(cond, Fm[0], Fr[0]), jaxm.where(cond, Fm[1], Fr[1]))
        return (laml, lamr, Fl, Fr)

    laml, lamr, Fl, Fr = jaxm.jax.lax.fori_loop(0, max_iter, body_fn, (laml, lamr, Fl, Fr))
    # print(f"diff = ", lamr - laml)
    return lamr, Fr


@partial(jaxm.jit, static_argnames=("max_iter",))
def positive_cholesky_factorization(H, reg0=0.0, max_iter=10):
    F = cho_factor(H + reg0 * jnp.eye(H.shape[-1], dtype=H.dtype))
    ret = jaxm.jax.lax.cond(
        jaxm.isfinite(F[0][0, 0]),
        lambda: (reg0, F),
        lambda: _find_cholesky_factorization(H, F, reg0, max_iter),
    )
    return ret


####################################################################################################


class SQPSolver(ConvexSolver):
    def __init__(
        self,
        fun,
        maxiter=100,
        tol=1e-9,
        verbose=False,
        jit=True,
        maxls=20,
        min_stepsize=1e-7,
        max_stepsize=1e1,
        reg0=1e-6,
        linesearch="scan",
        force_step: bool = False,
        g_fn: Callable | None = None,
        h_fn: Callable | None = None,
        has_aux: bool = False,
    ):
        """A jaxopt compatible iterative solver using sequential quadratic programming.

        Args:
            fun (Callable): Convex objective function to minimize.
            maxiter (int, optional): Number of iterations in the `run` method. Defaults to 100.
            tol (float, optional): Absolute tolerance (gradient norm). Defaults to 1e-9.
            verbose (bool, optional): Whether to print debugging output. Defaults to False.
            jit (bool, optional): Whether to jit objective and derivatives. Defaults to True.
            maxls (int, optional): Max num of objective evaluations in lineasearch. Defaults to 20.
            min_stepsize (int, optional): Minimum stepsize to take. Defaults to 1e-7.
            max_stepsize (int, optional): Maximum stepsie to take. Defaults to 1e1.
            reg0 (float, optional): Hessian l2 regularization. Defaults to 1e-6.
            linesearch (str, optional): Which linesearch to use from
                                    ["scan", "backtracking", "binary_search"]. Defaults to "scan".
        """
        assert linesearch != "binary_search", "binary_search not supported for non-convex objetives"
        super().__init__(
            fun,
            maxiter=maxiter,
            tol=tol,
            verbose=verbose,
            jit=jit,
            maxls=maxls,
            min_stepsize=min_stepsize,
            max_stepsize=max_stepsize,
            reg0=reg0,
            linesearch=linesearch,
            force_step=force_step,
            g_fn=g_fn,
            h_fn=h_fn,
            has_aux=has_aux,
        )

    @partial(jax.jit, static_argnums=(0,), static_argnames=("maxls", "force_step", "linesearch"))
    def update(self, params, state, *args, **config):
        g, H = self.g_fn(params, *args), self.h_fn(params, *args)
        dtype = H.dtype
        cond = jaxm.norm(g) > self.tol

        def fix_H_and_update(H):
            # reg = jaxm.minimum(-jaxm.min(jaxm.linalg.eigvalsh(H)), 0.0)
            reg = positive_cholesky_factorization(H, reg0=self.reg0)[0]
            H = H + reg * jnp.eye(H.shape[-1], dtype=dtype)
            return self._update(g, H, params, state, *args, **config)

        return jaxm.jax.lax.cond(cond, lambda H: fix_H_and_update(H), lambda H: (params, state), H)
