from typing import Mapping
from copy import copy
from functools import partial
from warnings import warn

from jfi import jaxm


class Problem(Mapping):
    """A way of initializing an optimal control problem with a majority of
    arguments initialized to defaults."""

    dim_map = {
        "Q": ("N", "xdim", "xdim"),
        "R": ("N", "udim", "udim"),
        "X_ref": ("N", "xdim"),
        "U_ref": ("N", "udim"),
        "X_prev": ("N", "xdim"),
        "U_prev": ("N", "udim"),
        "u_l": ("N", "udim"),
        "u_u": ("N", "udim"),
        "x_l": ("N", "udim"),
        "x_u": ("N", "udim"),
        "x0": ("xdim",),
    }

    def _figure_out_dims(self, **kw):
        dims = dict({k: v for k, v in kw.items() if k in ["N", "xdim", "udim"]})
        for k, v in Problem.dim_map.items():
            if k in kw:
                for i in range(0, -len(v) - 1, -1):
                    dims[Problem.dim_map[k][i]] = kw[k].shape[i]
        for k in ["N", "xdim", "udim"]:
            if k not in dims:
                raise ValueError(f"Missing dimension {k}")
        return dims

    def __init__(self, **kw):
        self._dims = self._figure_out_dims(**kw)
        self._set_defaults()
        self._move_all_to_device_dtype()
        for k in Problem.dim_map.keys():
            self._generate_property(k)
        for k in self._dims.keys():
            setattr(Problem, k, property(lambda self, k=k: self._dims[k]))

    def __repr__(self):
        return f"Problem({self._dims}, id={id(self)})"

    ################################################################################################

    def _move_all_to_device_dtype(self):
        for k in Problem.dim_map.keys():
            setattr(
                self,
                f"_{k}",
                jaxm.to(getattr(self, f"_{k}"), dtype=self._dtype, device=self._device),
            )

    ################################################################################################

    @property
    def dims(self):
        return copy(self._dims)

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype
        self._move_all_to_device_dtype()

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device
        self._move_all_to_device_dtype()

    def _generate_property(self, k):
        def _check_dims_and_tile_and_set(k, self, v):
            correct_shape = tuple(self._dims[k_] for k_ in Problem.dim_map[k])
            if v is not None:
                msg = (
                    f"v does not have the correct shape, v.shape = {v.shape}, "
                    + f"correct_shape = {correct_shape[-v.ndim:]}"
                )
                assert v.shape == correct_shape[-v.ndim :], msg
                v = jaxm.to(jaxm.array(v), dtype=self._dtype, device=self._device)
                v = jaxm.tile(v, correct_shape[: -v.ndim] + ((1,) * v.ndim))
            setattr(self, f"_{k}", v)

        getter = lambda self: getattr(self, f"_{k}") # noqa: E731
        setter = partial(_check_dims_and_tile_and_set, k)
        setattr(Problem, k, property(getter, setter))

    ################################################################################################

    def _set_defaults(self, **kw):
        self._Q = jaxm.tile(jaxm.diag(jaxm.ones((self._dims["xdim"],))), (self._dims["N"], 1, 1))
        self._R = jaxm.tile(
            jaxm.diag(1e-1 * jaxm.ones((self._dims["udim"],))), (self._dims["N"], 1, 1)
        )
        self._x0 = jaxm.zeros((self._dims["xdim"],))
        self._X_ref = jaxm.zeros((self._dims["N"], self._dims["xdim"]))
        self._U_ref = jaxm.zeros((self._dims["N"], self._dims["udim"]))
        self._X_prev = jaxm.tile(self._x0, (self._dims["N"], 1))
        self._U_prev = jaxm.zeros((self._dims["N"], self._dims["udim"]))
        self._u_l = None
        self._u_u = None
        self._x_l = None
        self._x_u = None
        self.solver_settings = dict(smooth_alpha=1e2, solver="cvx", linesearch="scan", maxls=100)
        self.reg_x = 1e0
        self.reg_u = 1e0
        self.max_it = 30
        self.res_tol = 1e-6
        self.verbose = True
        self.slew_rate = 0.0
        self.P = None
        self._dtype = jaxm.float64
        self._device = "cpu"
        for k, v in kw.items():
            setattr(self, f"_{k}", v)

    def to_dict(self):
        # most normal keys
        keys = list(Problem.dim_map.keys()) + [
            "solver_settings",
            "reg_x",
            "reg_u",
            "max_it",
            "res_tol",
            "verbose",
            "slew_rate",
            "P",
            "dtype",
            "device",
        ]
        problem = {k: getattr(self, k) for k in keys}

        # dynamics
        if hasattr(self, "f_fx_fu_fn"):
            problem["f_fx_fu_fn"] = self.f_fx_fu_fn
        else:
            warn("No dynamics function specified, please set `prob.f_fx_fu_fn`")

        # optional keys
        optional_keys = ["lin_cost_fn", "diff_cost_fn"]
        problem = dict(problem, **{k: getattr(self, k) for k in optional_keys if hasattr(self, k)})

        return problem

    ################################################################################################

    def __iter__(self):
        return iter(self.to_dict().keys())

    def __getitem__(self, k):
        return self.to_dict()[k]

    def __len__(self):
        return len(self.to_dict())

    ################################################################################################
