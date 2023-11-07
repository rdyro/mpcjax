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


def test_simple_diff_cost():
    N, xdim, udim = 20, 4, 2

    Q = np.tile(np.eye(xdim), (N, 1, 1))
    R = np.tile(1e-2 * np.eye(udim), (N, 1, 1))
    x0 = np.tile(np.ones(xdim), (1,))
    X_ref, U_ref = np.zeros((N, xdim)), np.zeros((N, udim))
    X_prev, U_prev = np.zeros((N, xdim)), np.zeros((N, udim))
    u_lim = 1e0
    u_l, u_u = -u_lim * np.ones((N, udim)), u_lim * np.ones((N, udim))

    x1_target = 2.0

    def diff_cost_fn(X, U, problem):
        return 1e2 * jaxm.mean((X[..., :, 1] - x1_target) ** 2)

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
                solver_settings=dict(smooth_alpha=1e0, solver="sqp", linesearch="scan", maxls=100),
                reg_x=1e0,
                reg_u=1e-1,
                max_it=100,
                res_tol=1e-12,
                verbose=True,
                slew_rate=1e-2,
                P=1.0 * jaxm.ones((N,)),
                diff_cost_fn=diff_cost_fn,
                dtype=dtype,
                device=device,
            )

            X1, U1, _ = solve(**problem, direct_solve=True)
            X2, U2, _ = solve(**problem, direct_solve=False)
            print(jaxm.abs(U1 - U2))
            print(jaxm.abs(X1 - X2))

            traj_err = jaxm.max(jaxm.abs((X1 - X2)))
            print(f"max traj error: {traj_err:.4e}")
            assert traj_err < 1e-1, f"max traj error: {traj_err:.4e}"
            ctrl_err = jaxm.max(jaxm.abs((U1 - U2)))
            print(f"max ctrl error: {ctrl_err:.4e}")
            assert ctrl_err < 1e-1, f"max ctrl error: {ctrl_err:.4e}"
            assert jaxm.max(jaxm.abs(U1)) < u_lim
            assert jaxm.max(jaxm.abs(U2)) < u_lim

            target_err = jaxm.norm(X1[..., -1, 1] - x1_target)
            assert target_err < 1e-1


if __name__ == "__main__":
    test_simple_diff_cost()
