import unittest
import jax
import jax.numpy as jnp

from integrators import CrankNicolson


@jax.jit
def rhs(t, y):
    return -y


class ExpODETest(unittest.TestCase):
    def test_exp_ode(self):
        dt = 0.01
        t0 = 0
        y0 = jnp.array(1)
        tol = 1e-7
        analytical_y = lambda t: jnp.exp(-t)

        cn = CrankNicolson(rhs, y0, t0, tol)

        y = [y0]
        t = [t0]

        for i in range(10):
            cn.integrate(cn.t + dt)
            y.append(cn.y)
            t.append(cn.t)

        assert jnp.allclose(jnp.array(y), analytical_y(jnp.array(t)))
