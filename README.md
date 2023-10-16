# `mpcjax`

Pure JAX implementation of Model Predictive Control (SCP MPC).

This is non-linear dynamics finite horizon MPC solver and support for arbitrary
cost. It supports batching on the CPU and GPU.

# Installation

Install by issuing
```bash
$ pip install mpcjax
```
Alternatively, installing from source
```bash
$ git clone https://github.com/rdyro/mpcjax.git
$ cd mpcjax
$ pip install .
```

# Basic Usage

The solver is capable of MPC consensus optimization for several system instantiations. For the basic usage, we'll focus on a single system MPC.

A basic MPC problem is defined using the dynamics and a quadratic cost

## Defining dynamics

- `x0` the initial state of shape

where
```python
np.shape(x0) == (xdim,)
```

- `f, fx, fu = f_fx_fu_fn(xt, ut)` an affine dynamics linearization, such that
$$x^{(i+1)} \approx f^{(i)} + f_x^{(i)} (x^{(i)} - \tilde{x}^{(i)}) + f_u^{(i)} (u^{(i)} - \tilde{u}^{(i)}) $$
where 
```python
np.shape(xt) == (N, xdim)
np.shape(ut) == (N, udim)
np.shape(f) == (N, xdim)
np.shape(fx) == (N, xdim, xdim)
np.shape(fu) == (N, xdim, udim)
```

## Defining Cost

- `X_ref, Q` a reference and quadratic weight matrix for state cost
- `U_ref, R` a reference and quadratic weight matrix for control cost

The cost is given as 

$$J = \sum_{i=0}^N 
\frac{1}{2} (x^{(i+1)} - x_\text{ref}^{(i+1)}) Q^{(i)} (x^{(i+1)} - x_\text{ref}^{(i+1)}) + 
\frac{1}{2} (u^{(i)} - u_\text{ref}^{(i)}) R^{(i)} (u^{(i)} - u_\text{ref}^{(i)})$$

*Note: Initial state, x0, is assumed constant and thus does not feature in the cost.*

*Note: When handling controls, we'll always have `np.shape(U) == (N, udim)`*

*Note: When handling states, we'll have either `np.shape(X) == (N + 1, xdim)` with `x0` included at the beginning or `np.shape(X) == (N, xdim)` with `x0` NOT INCLUDED. `X[:, -1]` always refers to the state N, whereas `U[:, -1]` always refers to control N - 1.*

Thus, an example call would be

```python
>>> import mpcjax
>>> X, U, debug = mpcjax.solve(f_fx_fu_fn, Q, R, x0, X_ref, U_ref)
>>> help(mpcjax.solve)
```

Take a look at
- `tests/simple.py` for simple usage
- `tests/dubins_car.py` for defining dynamics

# `solve` Method Arguments Glossary

## Solver Hyperparameters

The solver has two scalar hyperparamters, the dynamics linearization deviation penalty for states and controls

$$
J_\text{deviation} = \sum_{i=0}^N \frac{1}{2} 
\rho_x (x^{(i+1)} - x_\text{prev}^{(i+1)})^T (x^{(i+1)} - x_\text{prev}^{(i+1)})
+ \rho_u (u^{(i)} - u_\text{prev}^{(i)})^T (u^{(i)} - u_\text{prev}^{(i)})
$$

- `reg_x` - state deviation in-between SCP iterations regularization
- `reg_u` - control deviation in-between SCP iterations regularization

Higher values will slow evolution between SCP iterations and will require more
SCP iterations to converge to a solution, but will avoid instability in the
solution if the dynamics are not sufficiently smooth.

## Solver Settings

- `verbose` - whether to print iteration status (user-facing)
- `debug` - whether to print very low-level debugging information (developer-facing)
- `max_it` - maximum number of SCP iterations to perform (can be fewer if tolerance met earlier)
- `time_limit` - the time limit in seconds for SCP iteration
- `res_tol` - deviation tolerance past which solution is accepted (measure of convergence)
- `slew_rate` - the quadratic penalty between time-consecutive controls (encourages smooth controls)
- `u_slew` - the previous action taken to align the first plan action with (useful for smooth receding horizon control)

## Additional Dynamics Settings

- `X_prev` - previous state solution (guess), $x^{(i)} ~~ \forall i \in [1, \dots, N]$, `shape = (N, xdim)`
- `U_prev` - previous control solution (guess),  $u^{(i)} ~~ \forall i \in [0, \dots, N - 1]$, `shape = (N, udim)`
- `x_l` - state lower box constraints, $x^{(i)} ~~ \forall i \in [1, \dots, N]$, `shape = (N, xdim)`
- `x_u` - state upper box constraints, $x^{(i)} ~~ \forall i \in [1, \dots, N]$, `shape = (N, xdim)`
- `u_l` - control lower box constraints, $u^{(i)} ~~ \forall i \in [0, \dots, N - 1]$, `shape = (N, udim)`
- `u_u` - control upper box constraints, $u^{(i)} ~~ \forall i \in [0, \dots, N - 1]$, `shape = (N, udim)`

## Nonlinear Cost and Constraints

The solver supports custom arbitrary cost via each-SCP-iteration cost linearization and custom constraints via each-SCP-iteration constraint reformulation into any convex-cone constraint.


- `lin_cost_fn` is an optional callable which allows specifying a custom cost, it
should take arbitrary `X`, `U` and return a tuple
    - `cx`, the linearization of the cost with respect to the state, `np.shape(cx) == (N, xdim) or cx is None`
    - `cu`, the linearization of the cost with respect to the controls, `np.shape(cu) == (N, udim) or cu is None`

I highly recommend using an auto-diff library to produce the linearizations to avoid unnecessary bugs.

- `diff_cost_fn` is an optional callable for specifying arbitrary differentiable cost
  - this includes custom penalized constraints (e.g., log-barrier or augmented-Lagrangian penalties)
  - the function can return NaNs for infeasible arguments, but a feasible guess must be provided in `X_prev`, `U_prev`

The functions must accept the call `lin_cost_fn(X, U, problem)` and `diff_cost_fn(X, U, problem)` where `problem` is a Python dictionary with problem data.

## Misc Settings

- `solver_settings` - a dictionary of settings to pass to the lower-level Julia solver
- `solver_state` - a previous solver state to pass to the lower-level Julia solver

# Advanced Usage

## Non-convex Cost Example

```python
N, xdim, udim = 10, 3, 2
X_ref = np.random.rand(N, xdim)

def lin_cost_fn(X, U):
    # cost is np.sum((X - X_ref) ** 2)
    cx = 2 * (X - X_ref)
    cu = None
    return cx, cu
```

# Warm-start support

*Warm-start* in SCP MPC can refer to either
- warm-starting the SCP procedure through a good `X_prev, U_prev` - this is supported
- warm-starting the underlying convex solver - not supported

Warm-starting of the SCP procedure by providing a good `X_prev, U_prev` guess is supported and very much encouraged for good SCP performance!

Warm-starting of the underlying convex solver is currently not supported, as it does not lead to a noticeable
performance improvement on problems we tested the solver on.
