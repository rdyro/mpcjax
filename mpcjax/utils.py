from typing import Any

from jfi import jaxm
from jax.tree_util import tree_map


def _is_numeric(x):
    """Check whether and object can be represented as a JAX array."""
    try:
        jaxm.array(x)
        return True
    except (ValueError, TypeError):
        return False


def _to_dtype_device(d: Any, device=None, dtype=None):
    """Convert an arbitrary nested python object to specified dtype and device."""
    return tree_map(
        lambda x: jaxm.to(jaxm.array(x), dtype=dtype, device=device) if _is_numeric(x) else None, d
    )


def _jax_sanitize(x: Any) -> Any:
    """Replace all data that cannot be expressed as a JAX array (e.g., str) with None"""
    return tree_map(lambda x: x if _is_numeric(x) else None, x)

####################################################################################################


class TablePrinter:
    def __init__(self, names, fmts=None, prefix=""):
        self.names = names
        self.fmts = fmts if fmts is not None else ["%9.4e" for _ in names]
        self.widths = [
            max(self.calc_width(fmt), len(name)) + 2
            for (fmt, name) in zip(fmts, names)
        ]
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
        for (name, width) in zip(self.names, self.widths):
            s += "|" + self.pad_field("%s" % name, width, lj=True)
        s += "|\n"
        return s + self.prefix + self.make_row_sep()

    def make_footer(self):
        return self.prefix + self.make_row_sep()

    def make_values(self, vals):
        assert len(vals) == len(self.fmts)
        s = self.prefix + ""
        for (val, fmt, width) in zip(vals, self.fmts, self.widths):
            s += "|" + self.pad_field(fmt % val, width, lj=False)
        s += "|"
        return s

    def print_header(self):
        print(self.make_header())

    def print_footer(self):
        print(self.make_footer())

    def print_values(self, vals):
        print(self.make_values(vals))
