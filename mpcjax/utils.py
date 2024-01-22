from __future__ import annotations

from copy import copy
from typing import Any, TypeVar

from jaxfi import jaxm
from jax.tree_util import tree_map, tree_flatten, tree_unflatten
from jax import Array, Device
from jax import api_util
import numpy as np

Dtype = TypeVar("Dtype")

####################################################################################################


def logbarrier_cstr_with_lagrange_fallback(
    cstr: Array, alpha: float | Array, lam: float | Array
) -> Array:
    """Evaluate the constraint as its logbarrier reformulation and switch to a
    lagrange penalty if infeasible."""
    #    ┌─────────────────────────────────────────────────┐
    #  3 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀│
    #    │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡼⠀│
    #    │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠇⠀│
    #    │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡾⠀⠀│
    #    │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⠃⠀⠀│
    #    │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡼⡟⠀⠀⠀│
    #    │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠞⢱⠃⠀⠀⠀│
    #    │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡴⠋⠀⡞⠀⠀⠀⠀│
    #    │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠖⠋⠀⠀⢸⠁⠀⠀⠀⠀│
    #    │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⠖⠋⠁⠀⠀⠀⠀⡏⠀⠀⠀⠀⠀│
    #    │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡤⠴⠚⠉⠀⠀⠀⠀⠀⠀⠀⢸⠁⠀⠀⠀⠀⠀│
    #    │⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⢒⣒⡶⠶⠒⠛⠓⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⢒⡟⠒⠒⠒⠒⠒⠒│
    #    │⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⡤⠤⠖⠚⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⠀⠀⠀⠀⠀⠀⠀│
    #    │⣀⡤⠤⠴⠒⠚⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡇⠀⠀⠀⠀⠀⠀⠀│
    # -1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⠀⠀⠀⠀⠀⠀⠀⠀│
    #    └─────────────────────────────────────────────────┘
    #    ⠀-2⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀0⠀
    # we find a lagrange penalty, linear function that intersects tangentially to the logbarrier
    # we use a specified lambda - the slope of the linear function
    # we solve for b, such that the intersection is at a point and tangential
    # functions equal each other:   -log(-alpha * x) / alpha = lam * x + b
    # and, the slope matches:       -1 / (alpha * x) = lam
    # this gives us, the intersection point: x0 = -1 / (alpha * lam)
    # and lagrange penalty bias/offset:      b =  -log(-alpha * x0) / alpha - lam * x0
    x0 = -1 / (alpha * lam)
    b = -jaxm.log(-alpha * x0) / alpha - lam * x0
    return jaxm.where(cstr < 0.0, -jaxm.log(-alpha * cstr) / alpha, lam * cstr + b)


####################################################################################################


def bmv(A, x):
    return (A @ x[..., None])[..., 0]


def vec(x, n=2):
    return x.reshape(x.shape[:-n] + (-1,))


def atleast_nd(x: Array | None, n: int):
    if x is None:
        return None
    else:
        return x.reshape((1,) * max(n - x.ndim, 0) + x.shape)


# vendored directly from Equinox: https://github.com/patrick-kidger/equinox
def is_array_like(element: Any) -> bool:
    """Returns `True` if `element` is a JAX array, a NumPy array, or a Python
    `float`/`complex`/`bool`/`int`.
    """
    return isinstance(
        element, (jax.Array, np.ndarray, np.generic, float, complex, bool, int)
    ) or hasattr(element, "__jax_array__")


def _is_numeric(x: Any) -> bool:
    """Check whether and object can be represented as a JAX array."""
    return is_array_like(x)
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


def combine_dicts(primary: dict, secondary: dict) -> dict[str, Any]:
    result = copy(primary)
    for key, value in primary.items():
        if value is None and key in secondary:
            result[key] = secondary[key]
        if key in secondary and isinstance(value, dict):
            result[key] = combine_dicts(value, secondary[key])
    remaining_keys = set(secondary.keys()) - set(primary.keys())
    for key in remaining_keys:
        result[key] = secondary[key]
    return result


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


####################################################################################################


def auto_pmap(fn, device="cpu", in_axes=0, out_axes=0):
    assert device.lower() in {"cpu"}
    devices = jaxm.jax.devices(device)
    n_devices = len(devices)

    def _pmap_fn(*args):
        args_flat, args_struct = tree_flatten(args)
        # in_axes_flat = tree_flatten(in_axes)[0]
        in_axes_flat = api_util.flatten_axes("hello", args_struct, in_axes)
        assert all(
            axes in (0, None) for axes in in_axes_flat
        ), "We only support 0 or None in in_axes"
        assert len(args_flat) == len(in_axes_flat)

        batch_sizes = [x.shape[0] for x, axes in zip(args_flat, in_axes_flat) if axes == 0]
        batch_size = batch_sizes[0]
        assert all(x == batch_size for x in batch_sizes), "All batch sizes must be equal"
        n_devices_ = min(n_devices, batch_size)

        floor_size = (batch_size // n_devices_) * n_devices_
        print(f"floor_size: {floor_size}, batch_size: {batch_size}")

        # pmap #####################################################################################
        args_flat_pmap = [
            x if axes is None else x[:floor_size, ...].reshape((n_devices_, -1) + x.shape[1:])
            for x, axes in zip(args_flat, in_axes_flat)
        ]
        out_pmap = jaxm.jax.pmap(
            jaxm.jax.vmap(fn, in_axes=in_axes),
            devices=devices[:n_devices_],
            in_axes=in_axes,
        )(*tree_unflatten(args_struct, args_flat_pmap))
        out_pmap_flat, out_struct = tree_flatten(out_pmap)
        out_pmap_flat = [x.reshape((floor_size,) + x.shape[2:]) for x in out_pmap_flat]
        # pmap #####################################################################################

        # pmap2 ####################################################################################
        if floor_size < batch_size:
            remain_size = batch_size - floor_size
            args_flat_pmap2 = [
                x if axes is None else x[floor_size:, ...].reshape((remain_size, 1) + x.shape[1:])
                for x, axes in zip(args_flat, in_axes_flat)
            ]
            out_pmap2 = jaxm.jax.pmap(
                jaxm.jax.vmap(fn, in_axes=in_axes),
                devices=devices[:remain_size],
                in_axes=in_axes,
            )(*tree_unflatten(args_struct, args_flat_pmap2))
            out_pmap_flat2 = tree_flatten(out_pmap2)[0]
            out_pmap_flat2 = [x.reshape((remain_size,) + x.shape[2:]) for x in out_pmap_flat2]
            out_all_flat = [
                jaxm.concatenate((x, y), axis=0) for x, y in zip(out_pmap_flat, out_pmap_flat2)
            ]
            return tree_unflatten(out_struct, out_all_flat)
        else:
            return tree_unflatten(out_struct, out_pmap_flat)
        # pmap2 ####################################################################################

        # vmap #####################################################################################
        if floor_size < batch_size:
            args_flat_vmap = [
                x if axes is None else x[floor_size:] for x, axes in zip(args_flat, in_axes_flat)
            ]
            out_vmap = jaxm.jax.vmap(fn, in_axes=in_axes, out_axes=out_axes)(
                *tree_unflatten(args_struct, args_flat_vmap)
            )
            out_vmap_flat = tree_flatten(out_vmap)[0]
            out_all_flat = [
                jaxm.concatenate((x, y), axis=0) for x, y in zip(out_pmap_flat, out_vmap_flat)
            ]
            return tree_unflatten(out_struct, out_all_flat)
        else:
            return tree_unflatten(out_struct, out_pmap_flat)
        # vmap #####################################################################################

    return _pmap_fn


# from jax.sharding import Mesh, PartitionSpec
# from jax.experimental import mesh_utils
# from jax.experimental.shard_map import shard_map

####################################################################################################

import jax
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding


def auto_sharding(fn, device="cpu", in_axes=0, out_axes=0):
    assert device.lower() in {"cpu"}
    devices = jaxm.jax.devices(device)
    n_devices = len(devices)

    def _pmap_fn(*args):
        args_flat, args_struct = tree_flatten(args)
        # in_axes_flat = tree_flatten(in_axes)[0]
        in_axes_flat = api_util.flatten_axes("hello", args_struct, in_axes)
        assert all(
            axes in (0, None) for axes in in_axes_flat
        ), "We only support 0 or None in in_axes"
        assert len(args_flat) == len(in_axes_flat)

        batch_sizes = [x.shape[0] for x, axes in zip(args_flat, in_axes_flat) if axes == 0]
        batch_size = batch_sizes[0]
        assert all(x == batch_size for x in batch_sizes), "All batch sizes must be equal"
        n_devices_ = min(n_devices, batch_size)

        devices = mesh_utils.create_device_mesh(
            (n_devices_,), devices=jax.devices("cpu")[:n_devices_]
        )
        sharding = PositionalSharding(devices)

        floor_size = (batch_size // n_devices_) * n_devices_
        print(f"floor_size: {floor_size}, batch_size: {batch_size}")

        # pmap #####################################################################################
        args_flat_pmap = [
            x if axes is None else x[:floor_size].reshape((n_devices_, -1) + x.shape[1:])
            for x, axes in zip(args_flat, in_axes_flat)
        ]

        args_flat_pmap = [
            jax.device_put(x, sharding.replicate())
            if axes is None
            else jax.device_put(x, sharding.reshape((-1,) + (1,) * (x.ndim - 1)))
            for x, axes in zip(args_flat, in_axes_flat)
        ]
        args_pmap = tree_unflatten(args_struct, args_flat_pmap)
        out_pmap = jaxm.jax.vmap(fn, in_axes=in_axes)(*args_pmap)
        # out_pmap = jaxm.jax.pmap(
        #    jaxm.jax.vmap(fn, in_axes=in_axes),  # out_axes=out_axes),
        #    # fn, #out_axes=out_axes),
        #    devices=devices[:n_devices_],
        #    in_axes=in_axes,
        #    # out_axes=out_axes,
        # )(*tree_unflatten(args_struct, args_flat_pmap))
        print("Ran once successfully")

        out_pmap_flat, out_struct = tree_flatten(out_pmap)
        out_pmap_flat = [x.reshape((floor_size,) + x.shape[2:]) for x in out_pmap_flat]
        # out_pmap_flat = [x.reshape((floor_size,) + x.shape[1:]) for x in out_pmap_flat]
        # pmap #####################################################################################

        # vmap #####################################################################################
        if floor_size < batch_size:
            args_flat_vmap = [
                x if axes is None else x[floor_size:] for x, axes in zip(args_flat, in_axes_flat)
            ]
            out_vmap = jaxm.jax.vmap(fn, in_axes=in_axes, out_axes=out_axes)(
                *tree_unflatten(args_struct, args_flat_vmap)
            )
            out_vmap_flat = tree_flatten(out_vmap)[0]
            out_all_flat = [
                jaxm.concatenate((x, y), axis=0) for x, y in zip(out_pmap_flat, out_vmap_flat)
            ]
            return tree_unflatten(out_struct, out_all_flat)
        else:
            return tree_unflatten(out_struct, out_pmap_flat)
        # vmap #####################################################################################

    return _pmap_fn
