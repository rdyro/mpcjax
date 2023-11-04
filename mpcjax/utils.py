from __future__ import annotations

from typing import Any, TypeVar

from jfi import jaxm
from jax.tree_util import tree_map
from jax import Array, Device

Dtype = TypeVar("Dtype")



def bmv(A, x):
    return (A @ x[..., None])[..., 0]


def vec(x, n=2):
    return x.reshape(x.shape[:-n] + (-1,))


def atleast_nd(x: Array | None, n: int):
    if x is None:
        return None
    else:
        return x.reshape((1,) * max(n - x.ndim, 0) + x.shape)



def _is_numeric(x: Any) -> bool:
    """Check whether and object can be represented as a JAX array."""
    try:
        jaxm.array(x)
        return True
    except (ValueError, TypeError):
        return False


def _to_dtype_device(d: Any, device=None, dtype=None) -> Device | Dtype:
    """Convert an arbitrary nested python object to specified dtype and device."""
    return tree_map(
        lambda x: jaxm.to(jaxm.array(x), dtype=dtype, device=device) if _is_numeric(x) else None, d
    )


def _jax_sanitize(x: Any) -> Array | None:
    """Replace all data that cannot be expressed as a JAX array (e.g., str) with None"""
    return tree_map(lambda x: x if _is_numeric(x) else None, x)


def balanced_solve(A: Array, y: Array) -> Array:
    assert A.ndim == y.ndim
    Pl = 1.0 / jaxm.linalg.norm(A, axis=-1)
    Pr = 1.0 / jaxm.linalg.norm(A, axis=-2)
    # Pr = jaxm.ones(A.shape[:-1])
    Pl = (Pl + Pr) / 2.0
    Pr = Pl
    A_norm = Pl[..., None] * A * Pr[..., None, :]
    # solve_fn = jaxm.linalg.solve
    solve_fn = lambda M, h: jaxm.scipy.linalg.cho_solve(  # noqa: E731
        jaxm.scipy.linalg.cho_factor(M), h
    )
    return Pr[..., None, :] * solve_fn(A_norm, Pl[..., None] * y)


def f64_solve(A: Array, y: Array) -> Array:
    org_dtype = A.dtype
    A, y = A.astype(jaxm.float64), y.astype(jaxm.float64)
    return jaxm.linalg.solve(A, y).astype(org_dtype)


####################################################################################################


class TablePrinter:
    def __init__(self, names, fmts=None, prefix=""):
        self.names = names
        self.fmts = fmts if fmts is not None else ["%9.4e" for _ in names]
        self.widths = [max(self.calc_width(fmt), len(name)) + 2 for (fmt, name) in zip(fmts, names)]
        self.prefix = prefix

    def calc_width(self, fmt):
        f = fmt[-1]
        width = None
        if f == "f" or f == "e" or f == "d" or f == "i":
            width = max(len(fmt % 1), len(fmt % (-1)))
        elif f == "s":
            width = len(fmt % "")
        else:
            raise ValueError("I can't recognized the [%s] print format" % fmt)
        return width

    def pad_field(self, s, width, lj=True):
        # lj -> left justify
        assert len(s) <= width
        rem = width - len(s)
        if lj:
            return (" " * (rem // 2)) + s + (" " * ((rem // 2) + (rem % 2)))
        else:
            return (" " * ((rem // 2) + (rem % 2))) + s + (" " * (rem // 2))

    def make_row_sep(self):
        return "+" + "".join([("-" * width) + "+" for width in self.widths])

    def make_header(self):
        s = self.prefix + self.make_row_sep() + "\n"
        s += self.prefix
        for name, width in zip(self.names, self.widths):
            s += "|" + self.pad_field("%s" % name, width, lj=True)
        s += "|\n"
        return s + self.prefix + self.make_row_sep()

    def make_footer(self):
        return self.prefix + self.make_row_sep()

    def make_values(self, vals):
        assert len(vals) == len(self.fmts)
        s = self.prefix + ""
        for val, fmt, width in zip(vals, self.fmts, self.widths):
            s += "|" + self.pad_field(fmt % val, width, lj=False)
        s += "|"
        return s

    def print_header(self):
        print(self.make_header())

    def print_footer(self):
        print(self.make_footer())

    def print_values(self, vals):
        print(self.make_values(vals))
