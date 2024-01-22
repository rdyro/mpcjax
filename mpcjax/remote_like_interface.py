from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from jaxfi import jaxm

from .jax_solver import solve
from .utils import _is_numeric
#from .optimality import generate_optimality_fn
from .optimality import optimality_fn
from .sanitize import sanitize_keywords

Array = jaxm.jax.Array
tree_flatten = jaxm.jax.tree_util.tree_flatten
tree_structure = jaxm.jax.tree_util.tree_structure
tree_unflatten = jaxm.jax.tree_util.tree_unflatten

# handling solution in a list-of-problems format ###################################################


def _stack_problems(problems: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack a list of problems into the same argument structure, but batched data."""
    assert len(problems) > 0
    problems = list(problems)
    prob_structure = tree_structure(problems[0])
    problems = [tree_flatten(problem)[0] for problem in problems]
    numeric_mask = [_is_numeric(val) for val in problems[0]]
    prob_vals = [
        np.stack([np.array(problem[i]) for problem in problems], 0)
        if is_numeric
        else problems[0][i]
        for (i, is_numeric) in enumerate(numeric_mask)
    ]
    problems = tree_unflatten(prob_structure, prob_vals)
    return problems


def solve_problems(
    problems: List[Dict[str, Any]],
    split: bool = True,
    **kw,
) -> Union[Tuple[Array, Array, Dict[str, Any]], List[Tuple[Array, Array, Dict[str, Any]]]]:
    """Utility routine to apply `solve` to a list of problems. Returns stacked solutions by
    default.

    Args:
        problems (List[Dict[str, Any]]): List of problem definitions.
        split (bool, optional): Whether to split the final solution into a list. Defaults to False.

    Returns:
        A list of solutions, List[[X, U, data]] or a stack solution: X, U, data.
    """
    solver_settings = problems[0].get("solver_settings", dict())

    # solve the rest of this problem ###############################################################
    problems = _stack_problems(problems)
    problems["solver_settings"] = solver_settings

    # extract positional arguments #################################################################
    f_fx_fu_fn = problems["f_fx_fu_fn"]
    Q, R, x0 = problems["Q"], problems["R"], problems["x0"]
    solve_kws = {k: problems[k] for k in problems.keys() if k not in {"f_fx_fu_fn", "Q", "R", "x0"}}
    solve_kws = sanitize_keywords(solve, solve_kws, warn=True)

    # reinsert non-jax-batchable options ###########################################################
    static_argnames = (
        ("max_it", int),
        ("verbose", bool),
        ("direct_solve", bool),
        ("res_tol", float),
        ("reg_x", float),
        ("reg_u", float),
        ("dtype", lambda x: x),
    )
    for k, t in static_argnames:
        if k in solve_kws:
            try:
                solve_kws[k] = t(solve_kws[k][0])
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Failed to convert {k} to {t} with error {e}")
                print(end="", flush=True)
    try:
        X, U, data = solve(f_fx_fu_fn, Q, R, x0, **solve_kws, **kw)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print({k: type(v) for k, v in solve_kws.items()})
        print(end="", flush=True)

    # split the problem into separate solutions if requested #######################################
    if split:
        data_struct = tree_structure(data)
        data = tree_flatten(data)[0]
        mask = [hasattr(x, "shape") and x.ndim > 0 and x.shape[0] == X.shape[0] for x in data]
        data = [np.array(x) if m else x for (m, x) in zip(mask, data)]
        data_list = [[x[i] if m else x for (m, x) in zip(mask, data)] for i in range(X.shape[0])]
        data_list = [tree_unflatten(data_struct, data) for data in data_list]
        X, U = np.array(X), np.array(U)
        return [(X[i, ...], U[i, ...], data_list[i]) for i in range(X.shape[0])]
    else:
        return X, U, data


def generate_optimality_fns(
    problems: List[Dict[str, Any]],
    **kw,
) -> Callable:
    """Utility routine to apply `solve` to a list of problems. Returns stacked solutions by
    default.

    Args:
        problems (List[Dict[str, Any]]): List of problem definitions.
        split (bool, optional): Whether to split the final solution into a list. Defaults to False.

    Returns:
        A list of solutions, List[[X, U, data]] or a stack solution: X, U, data.
    """
    solver_settings = problems[0].get("solver_settings", dict())

    # solve the rest of this problem ###############################################################
    problems = _stack_problems(problems)
    problems["solver_settings"] = solver_settings

    # extract positional arguments #################################################################
    f_fx_fu_fn = problems["f_fx_fu_fn"]
    Q, R, x0 = problems["Q"], problems["R"], problems["x0"]
    solve_kws = {k: problems[k] for k in problems.keys() if k not in {"f_fx_fu_fn", "Q", "R", "x0"}}

    # reinsert non-jax-batchable options ###########################################################
    for k in ["verbose", "max_it", "res_tol", "time_limit"]:
        if k in solve_kws:
            solve_kws[k] = float(solve_kws[k][0])

    return generate_optimality_fn(f_fx_fu_fn, Q, R, x0, **solve_kws, **kw)
