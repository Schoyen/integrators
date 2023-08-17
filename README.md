# Crank-Nicolson

The bare-bones Crank-Nicolson solver using conjugate-gradient iterations to
solve the implicit equations with JAX.
This means that the method only supports symmetric/hermitian operators for the
right-hand side.


## Installation
Via pip for CPU:
```bash
pip install "crank-nicolson[cpu] @ git+ssh://git@github.com/Schoyen/crank-nicolson"
```

Via pip for Nvidia GPU using Cuda12:
```bash
pip install "crank-nicolson[cuda12_pip] @ git+ssh://git@github.com/Schoyen/crank-nicolson" --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

The extra flags are required by JAX. See options [here](https://github.com/google/jax#installation).
