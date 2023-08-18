import unittest
import jax
import jax.numpy as jnp

from integrators import RungeKutta4


@jax.jit
def rhs(t, y):
    return -y


class ExpODETest(unittest.TestCase):
    def test_exp_ode(self):
        dt = 0.01
        t0 = 0
        y0 = jnp.array(1)
        analytical_y = lambda t: jnp.exp(-t)

        rk4 = RungeKutta4(rhs, y0, t0)

        y = [y0]
        t = [t0]

        for i in range(10):
            rk4.integrate(rk4.t + dt)
            y.append(rk4.y)
            t.append(rk4.t)

        assert jnp.allclose(jnp.array(y), analytical_y(jnp.array(t)))
