"""A module whose job is to filter out unknown keywords inputs if they are not numeric - for JAX."""

from __future__ import annotations

from inspect import signature
from typing import Callable
from warnings import warn as warn_fn

from jax.tree_util import tree_flatten

from .utils import _is_numeric, _jax_sanitize

SIGNATURE_KEYWORDS_CACHE = dict()


def _all_numeric(v):
    return all(_is_numeric(x) for x in tree_flatten(v)[0])


def _sanitize_keywords(fn: Callable, kw: dict | None = None, warn: bool = False):
    """Sanitize a function call by removing keyword arguments that JAX cannot
    handle and that are not in function signature."""
    if fn not in SIGNATURE_KEYWORDS_CACHE:
        SIGNATURE_KEYWORDS_CACHE[fn] = set(signature(fn).parameters.keys())
    keys = SIGNATURE_KEYWORDS_CACHE[fn]
    mask = [k in keys or _all_numeric(v) for k, v in kw.items()]
    if warn and any(not m for m in mask):
        msg = "JAX cannot handle the following keyword arguments at least partially:\n"
        msg = msg + "\n".join([f"  {k}" for i, k in enumerate(kw.keys()) if not mask[i]])
        warn_fn(msg)
    kw_sanitized = {k: v if mask[i] else _jax_sanitize(v) for i, (k, v) in enumerate(kw.items())}
    return kw_sanitized


def sanitize_keywords(fn: Callable, kw: dict | None = None, warn: bool = False):
    # from line_profiler import LineProfiler
    # LP = LineProfiler()
    # LP.add_function(_sanitize_keywords)
    # ret = LP.wrap_function(_sanitize_keywords)(fn, kw, warn)
    # LP.print_stats(output_unit=1e-6)
    # return ret
    return _sanitize_keywords(fn, kw, warn)
