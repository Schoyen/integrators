# Integrators

This library includes bare-bones implementations of some first-order ODE integrators.
The integrators are small classes with a setup accepting a right-hand side function `rhs`, an initial value `y0`, an initial time-step `t0`, and possibly extra arguments depending on the integrator.
Time-stepping is done via the class-method `integrate` accepting a new time-point `t`.

## Solvers
The ever-relevant Runge-Kutta 4 integrator.

The bare-bones Crank-Nicolson solver using conjugate-gradient iterations to solve the implicit equations with JAX.
This means that the method only supports symmetric/hermitian operators for the right-hand side.
The solver takes in an extra argument `tol` adjusting the convergence threshold for the conjugate-gradient iterations.


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
