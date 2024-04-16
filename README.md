# Integrators

This library includes bare-bones implementations of some first-order ODE integrators.
The integrators are small functions with a setup accepting a right-hand side function `rhs`, a time step `dt`, and possibly extra arguments depending on the integrator.
Time-stepping is done by calling the returned function, `integrate`, accepting a new time-point `t` and the current value `y`.

## Solvers
The ever-relevant Runge-Kutta 4 integrator.

The Dormand-Prince integrator.

The bare-bones Crank-Nicolson solver using conjugate-gradient iterations (by default) to solve the implicit equations with JAX.


## Installation
Via pip for CPU:
```bash
pip install "integrators[cpu] @ git+ssh://git@github.com/Schoyen/integrators"
```

Via pip for Nvidia GPU using Cuda12:
```bash
pip install "integrators[cuda12_pip] @ git+ssh://git@github.com/Schoyen/integrators" --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

The extra flags are required by JAX. See options [here](https://github.com/google/jax#installation).
