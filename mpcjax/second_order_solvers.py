from enum import Enum
from typing import NamedTuple
from functools import partial
from jfi import jaxm
from jaxopt import base  # noqa: E402


class LineSearch(Enum):
    backtracking = 0
    scan = 1
    binary_search = 2


class ConvexState(NamedTuple):
    best_params: jaxm.jax.Array
    best_loss: jaxm.jax.Array
    value: jaxm.jax.Array


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
        device="cpu",
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
        self.device = device
        assert linesearch.lower() in ["scan", "backtracking", "binary_search"]
        if linesearch.lower() == "scan":
            self.linesearch = LineSearch.scan
        elif linesearch.lower() == "backtracking":
            self.linesearch = LineSearch.backtracking
        elif linesearch.lower() == "binary_search":
            self.linesearch = LineSearch.binary_search
        else:
            raise ValueError(f"Unknown linesearch: {linesearch}")

        self.f_fn = fun

        def g_fn(params, *args, **kw):
            g = jaxm.grad(self.f_fn)(params, *args, **kw)
            return g.reshape((params.size,))

        def h_fn(params, *args, **kw):
            H = jaxm.hessian(self.f_fn)(params, *args, **kw)
            return H.reshape((params.size, params.size))

        self.g_fn, self.h_fn = g_fn, h_fn

        if jit:
            self.f_fn = jaxm.jit(self.f_fn)
            self.g_fn = jaxm.jit(self.g_fn)
            self.h_fn = jaxm.jit(self.h_fn)

    def init_state(self, params, *args, **kw):
        best_loss = self.f_fn(params, *args, **kw)
        return ConvexState(params, best_loss, best_loss)

    def update(self, params, state, *args, **kw):
        g, H = self.g_fn(params, *args, **kw), self.h_fn(params, *args, **kw)
        cond = jaxm.norm(g) > self.tol
        return jaxm.jax.lax.cond(
            cond, lambda: self._update(g, H, params, state, *args, **kw), lambda: (params, state)
        )

    def _update(self, g, H, params, state, *args, **kw):
        """Update routine based on the values of the gradient and Hessian."""
        dtype = H.dtype

        # compute descent direction
        dp = -jaxm.scipy.linalg.cho_solve(
            jaxm.scipy.linalg.cho_factor(
                H + self.reg0 * jaxm.eye(H.shape[-1], dtype=dtype, device=self.device)
            ),
            g,
        ).reshape(params.shape)
        if self.linesearch == LineSearch.scan:
            lower_ls = max(round(float(0.7 * self.maxls)), 1)
            bets_low = jaxm.to(
                jaxm.logspace(jaxm.log10(self.min_stepsize), 0.0, lower_ls, dtype=dtype),
                device=self.device,
            )
            bets_up = jaxm.to(
                jaxm.logspace(
                    0.0,
                    jaxm.log10(self.max_stepsize),
                    self.maxls - lower_ls,
                    dtype=dtype,
                ),
                device=self.device,
            )[1:]
            bets = jaxm.cat([bets_low, bets_up], -1)
            losses = jaxm.jax.vmap(lambda bet: self.f_fn(params + bet * dp, *args, **kw))(bets)
            losses = jaxm.where(jaxm.isnan(losses), jaxm.inf, losses)
            idx = jaxm.argmin(losses)
            bet, new_loss = bets[idx], losses[idx]
            new_params = params + bet * dp
        elif self.linesearch == LineSearch.backtracking:

            def cond_fn(step):
                step_not_too_small = step >= self.min_stepsize
                not_better_loss = self.f_fn(params + step * dp, *args, **kw) > state.best_loss
                return jaxm.logical_and(step_not_too_small, not_better_loss)

            def body_fn(step):
                return step * 0.7

            step_size = jaxm.jax.lax.while_loop(cond_fn, body_fn, 1.0)
            new_params = params + step_size * dp
            new_loss = self.f_fn(new_params, *args, **kw)
        elif self.linesearch == LineSearch.binary_search:

            def body_fn(i, val):
                betl, betr, fl, fr = val
                # betm = 10.0 ** ((jaxm.log10(betl) + jaxm.log10(betr)) / 2.0)
                betm = (betl + betr) / 2.0
                fm = self.f_fn(params + betm * dp, *args, **kw)
                cond = fl < fr
                betr, betl = jaxm.where(cond, betm, betr), jaxm.where(cond, betl, betm)
                fr, fl = jaxm.where(cond, fm, fr), jaxm.where(cond, fl, fm)
                return (betl, betr, fl, fr)

            betl, betr = self.min_stepsize, 1e1
            fl = self.f_fn(params + betl * dp, *args, **kw)
            fr = self.f_fn(params + betr * dp, *args, **kw)
            betl, betr, fl, fr = jaxm.jax.lax.fori_loop(
                0, jaxm.maximum(self.maxls - 2, 1), body_fn, (betl, betr, fl, fr)
            )
            cond = fl < fr
            new_params = jaxm.where(cond, params + betl * dp, params + betr * dp)
            new_loss = jaxm.where(cond, fl, fr)
        else:
            raise ValueError(f"Unknown line search method {self.linesearch}")

        best_params = jaxm.where(new_loss <= state.best_loss, new_params, state.best_params)
        best_loss = jaxm.where(new_loss <= state.best_loss, new_loss, state.best_loss)
        state = ConvexState(best_params, best_loss, best_loss)
        new_params = state.best_params
        return new_params, state


####################################################################################################
cho_factor = jaxm.scipy.linalg.cho_factor
device = jaxm.jax.devices("cpu")[0]


@partial(jaxm.jit, static_argnames=("device", "max_iter"))
def _find_cholesky_factorization(H, Fl, laml, lamr=1e3, device=device, max_iter=10):
    # find the right lambda ###########################
    def cond_fn(lam_F):
        _, F = lam_F
        return jaxm.logical_not(jaxm.isfinite(F[0][0, 0]))

    def body_fn(lam_F):
        lam, _ = lam_F
        lam = lam * 5
        F = cho_factor(H + lam * jaxm.eye(H.shape[-1], dtype=H.dtype, device=device))
        return (lam, F)

    Fr = cho_factor(H + lamr * jaxm.eye(H.shape[-1], dtype=H.dtype, device=device))
    lamr, Fr = jaxm.jax.lax.while_loop(cond_fn, body_fn, (lamr, Fr))

    # find the middle lambda via bisection ############
    def body_fn(i, val):
        laml, lamr, Fl, Fr = val
        lamm = (laml + lamr) / 2.0
        Fm = cho_factor(H + lamm * jaxm.eye(H.shape[-1], dtype=H.dtype, device=device))
        cond = jaxm.isfinite(Fm[0][0, 0])
        laml, lamr = jaxm.where(cond, laml, lamm), jaxm.where(cond, lamm, lamr)
        Fl = (jaxm.where(cond, Fl[0], Fm[0]), jaxm.where(cond, Fl[1], Fm[1]))
        Fr = (jaxm.where(cond, Fm[0], Fr[0]), jaxm.where(cond, Fm[1], Fr[1]))
        return (laml, lamr, Fl, Fr)

    laml, lamr, Fl, Fr = jaxm.jax.lax.fori_loop(0, max_iter, body_fn, (laml, lamr, Fl, Fr))
    return lamr, Fr


@partial(jaxm.jit, static_argnames=("device", "max_iter"))
def positive_cholesky_factorization(H, reg0=0.0, device=device, max_iter=10):
    F = cho_factor(H + reg0 * jaxm.eye(H.shape[-1], dtype=H.dtype, device=device))
    return jaxm.jax.lax.cond(
        jaxm.isfinite(F[0][0, 0]),
        lambda: (reg0, F),
        lambda: _find_cholesky_factorization(H, F, reg0, device=device, max_iter=max_iter),
    )


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
        device="cpu",
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
            device=device,
        )

    def update(self, params, state, *args, **kw):
        g, H = self.g_fn(params, *args, **kw), self.h_fn(params, *args, **kw)
        dtype = H.dtype
        cond = jaxm.norm(g) > self.tol

        def find_H_factorization(H):
            # reg = jaxm.minimum(-jaxm.min(jaxm.linalg.eigvalsh(H)), 0.0)
            reg = positive_cholesky_factorization(H, reg0=self.reg0, device=self.device)[0]
            H = H + reg * jaxm.eye(H.shape[-1], dtype=dtype, device=self.device)
            return self._update(g, H, params, state, *args, **kw)

        return jaxm.jax.lax.cond(
            cond, lambda H: find_H_factorization(H), lambda H: (params, state), H
        )
