from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from jfi import jaxm

from .utils import _is_numeric
from .jax_solver import scp_solve

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
    #lin_cost_fn: Optional[Callable] = None,
    #diff_cost_fn: Optional[Callable] = None,
    **kw,
) -> Union[Tuple[Array, Array, Dict[str, Any]], List[Tuple[Array, Array, Dict[str, Any]]]]:
    """Utility routine to apply `scp_solve` to a list of problems. Returns stacked solutions by
    default.

    Args:
        problems (List[Dict[str, Any]]): List of problem definitions.
        split (bool, optional): Whether to split the final solution into a list. Defaults to False.

    Returns:
        A list of solutions, List[[X, U, data]] or a stack solution: X, U, data.
    """
    solver_settings = problems[0].get("solver_settings", dict())

    # handle the lin_cost_fn #######################################################################
    #if "lin_cost_fn" in problems[0] and lin_cost_fn is None:
    #    msg = """
    #    WARNING: specifying `lin_cost_fn` in `problems` without providing a batched version is not 
    #    currently supported. Please provide a batched version of `lin_cost_fn` as an argument to
    #    this function.""".strip()
    #    raise ValueError(msg)

    ## handle the diff_cost_fn ######################################################################
    #if "diff_cost_fn" in problems[0] and diff_cost_fn is None:
    #    msg = """
    #    WARNING: specifying `diff_cost_fn` in `problems` without providing a batched version is not 
    #    currently supported. Please provide a batched version of `diff_cost_fn` as an argument to
    #    this function.""".strip()
    #    raise ValueError(msg)

    # solve the rest of this problem ###############################################################
    problems_list = problems
    problems = _stack_problems(problems)
    problems["solver_settings"] = solver_settings

    # extract positional arguments #################################################################
    f_fx_fu_fn = problems["f_fx_fu_fn"]
    #lin_cost_fn = problems.get("lin_cost_fn", None)
    #diff_cost_fn = problems.get("diff_cost_fn", None)
    Q, R, x0 = problems["Q"], problems["R"], problems["x0"]
    scp_solve_kws = {
        k: problems[k] for k in problems.keys() if k not in {"f_fx_fu_fn", "Q", "R", "x0"}
    }

    # reinsert non-jax-batchable options ###########################################################
    for k in ["verbose", "max_it", "res_tol", "time_limit"]:
        if k in scp_solve_kws:
            scp_solve_kws[k] = float(scp_solve_kws[k][0])
    #if lin_cost_fn is not None:
    #    scp_solve_kws["lin_cost_fn"] = lin_cost_fn
    #if diff_cost_fn is not None:
    #    scp_solve_kws["diff_cost_fn"] = diff_cost_fn

    X, U, data = scp_solve(f_fx_fu_fn, Q, R, x0, **scp_solve_kws, **kw)

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
