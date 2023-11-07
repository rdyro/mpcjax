import sys
from pathlib import Path

import numpy as np
from jfi import jaxm
import jax
from jax import numpy as jnp

root_path = Path("").absolute().parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from mpcjax import solve
from tests.dynamics import f_fx_fu_fn


def test_simple_solve():
    N, xdim, udim = 20, 4, 2

    Q = np.tile(np.eye(xdim), (N, 1, 1))
    R = np.tile(1e-2 * np.eye(udim), (N, 1, 1))
    x0 = np.tile(np.ones(xdim), (1,))
    X_ref, U_ref = np.zeros((N, xdim)), np.zeros((N, udim))
    X_prev, U_prev = np.zeros((N, xdim)), np.zeros((N, udim))
    u_lim = 1e0
    u_l, u_u = -u_lim * np.ones((N, udim)), u_lim * np.ones((N, udim))

    try:
        jax.devices("cuda")
        devices = ["cpu", "cuda"]
    except:
        devices = ["cpu"]

    for device in devices:
        for dtype in [jnp.float32, jnp.float64]:
            problem = dict(
                f_fx_fu_fn=f_fx_fu_fn,
                Q=Q,
                R=R,
                x0=x0,
                X_ref=X_ref,
                U_ref=U_ref,
                X_prev=X_prev,
                U_prev=U_prev,
                u_l=u_l,
                u_u=u_u,
                solver_settings=dict(smooth_alpha=1e2, solver="sqp", linesearch="scan", maxls=50),
                reg_x=1e0,
                reg_u=1e-1,
                max_it=100,
                res_tol=1e-12,
                verbose=True,
                slew_rate=1e-2,
                P=1.0 * jaxm.ones((N,)),
                dtype=dtype,
                device=device,
            )

            X1, U1, _ = solve(**problem, direct_solve=True)
            X2, U2, _ = solve(**problem, direct_solve=False)

            assert jaxm.norm((X1 - X2).reshape(-1)) < 5e-2
            assert jaxm.norm((U1 - U2).reshape(-1)) < 5e-2
            assert jaxm.max(jaxm.abs(U1)) < u_lim
            assert jaxm.max(jaxm.abs(U2)) < u_lim


def test_compare_to_pmpc():
    try:
        import pmpc
        from pmpc import solve as pmpc_solve
    except ImportError:
        print("pmpc not installed, skipping test")
        return

    N, xdim, udim = 20, 4, 2

    Q = np.tile(np.eye(xdim), (N, 1, 1))
    R = np.tile(1e-2 * np.eye(udim), (N, 1, 1))
    x0 = np.tile(np.ones(xdim), (1,))
    X_ref, U_ref = np.zeros((N, xdim)), np.zeros((N, udim))
    X_prev, U_prev = np.zeros((N, xdim)), np.zeros((N, udim))
    u_lim = 1e0
    u_l, u_u = -u_lim * np.ones((N, udim)), u_lim * np.ones((N, udim))

    try:
        jax.devices("cuda")
        devices = ["cpu", "cuda"]
    except:
        devices = ["cpu"]

    for device in devices:
        for dtype in [jnp.float32, jnp.float64]:
            problem = dict(
                f_fx_fu_fn=f_fx_fu_fn,
                Q=Q,
                R=R,
                x0=x0,
                X_ref=X_ref,
                U_ref=U_ref,
                X_prev=X_prev,
                U_prev=U_prev,
                u_l=u_l,
                u_u=u_u,
                solver_settings=dict(smooth_alpha=1e2, solver="sqp", linesearch="scan", maxls=50),
                reg_x=1e0,
                reg_u=1e-1,
                max_it=100,
                res_tol=1e-12,
                verbose=True,
                slew_rate=1e-2,
                P=1.0 * jaxm.ones((N,)),
                dtype=dtype,
                device=device,
            )

            X1, U1, _ = solve(**problem, direct_solve=True)
            X2, U2, _ = solve(**problem, direct_solve=False)
            X_pmpc, U_pmpc, _ = pmpc_solve(
                **dict(problem, solver_settings=dict(problem["solver_settings"], solver="ecos"))
            )

            accuracy = 3e-2

            err_1 = jaxm.norm((X1 - X_pmpc).reshape(-1))
            assert err_1 < accuracy, f"err_1 = {err_1}"
            err_2 = jaxm.norm((U1 - U_pmpc).reshape(-1))
            assert err_2 < accuracy, f"err_2 = {err_2}"
            err_3 = jaxm.norm((X2 - X_pmpc).reshape(-1))
            assert err_3 < accuracy, f"err_3 = {err_3}"
            err_4 = jaxm.norm((U2 - U_pmpc).reshape(-1))
            assert err_4 < accuracy, f"err_4 = {err_4}"

            assert jaxm.max(jaxm.abs(U1)) < u_lim
            assert jaxm.max(jaxm.abs(U2)) < u_lim
            assert jaxm.max(jaxm.abs(U_pmpc)) < u_lim


if __name__ == "__main__":
    test_simple_solve()
    test_compare_to_pmpc()