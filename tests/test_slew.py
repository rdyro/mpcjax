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


def test_slew():
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
                slew_rate=1e5,
                P=1.0 * jaxm.ones((N,)),
                dtype=dtype,
                device=device,
            )

            X1, U1, _ = solve(**problem, direct_solve=True)
            X2, U2, _ = solve(**problem, direct_solve=False)
            traj_err = jaxm.norm((X1 - X2).reshape(-1)) 
            assert traj_err < 1e-1, f"max traj error: {traj_err:.4e}"
            ctrl_err = jaxm.norm((U1 - U2).reshape(-1))
            assert ctrl_err < 5e-2, f"max ctrl error: {ctrl_err:.4e}"

            slew_err = jaxm.max(jaxm.norm(U1[1:, :] - U1[:-1, :], axis=-1))
            assert slew_err < 1e-2, f"max slew error: {slew_err:.4e}"

def test_slew0():
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
                slew0_rate=1e5,
                P=1.0 * jaxm.ones((N,)),
                dtype=dtype,
                device=device,
            )

            X1, U1, _ = solve(**problem, direct_solve=True)
            X2, U2, _ = solve(**problem, direct_solve=False)
            traj_err = jaxm.norm((X1 - X2).reshape(-1)) 
            assert traj_err < 1e-1, f"max traj error: {traj_err:.4e}"
            ctrl_err = jaxm.norm((U1 - U2).reshape(-1))
            assert ctrl_err < 5e-2, f"max ctrl error: {ctrl_err:.4e}"

            # slew difference of first control to 0
            slew0_err = jaxm.norm(U1[0, :])
            assert slew0_err < 1e-4, f"max slew error: {slew0_err:.4e}"
            other_ctrls_norm = jaxm.norm(U1[1:, :])
            assert other_ctrls_norm > 1e-1, f"max other ctrls norm: {other_ctrls_norm:.4e}"

            # slew difference of first control to 0
            u0_slew = 0.6 * jaxm.ones((udim,), dtype=dtype, device=device)
            X1, U1, _ = solve(**dict(problem, u0_slew=u0_slew), direct_solve=False)
            slew0_err = jaxm.norm(U1[0, :] - u0_slew)
            assert slew0_err < 1e-4, f"max slew error: {slew0_err:.4e}"
            other_ctrls_norm = jaxm.norm(U1[1:, :])
            assert other_ctrls_norm > 1e-1, f"max other ctrls norm: {other_ctrls_norm:.4e}"


if __name__ == "__main__":
    test_slew()
    test_slew0()