import unittest
import jax
import jax.numpy as jnp

from integrators import get_dormand_prince_solver


@jax.jit
def rhs(t, y):
    return -y


class ExpODETest(unittest.TestCase):
    def test_exp_ode(self):
        dt = 0.01
        t = 0
        y = jnp.array(1)
        analytical_y = lambda t: jnp.exp(-t)

        rk4 = get_dormand_prince_solver(rhs, dt)

        y_list = [y]
        t_list = [t]

        for i in range(10):
            t_prev = t

            t, y = rk4(t, y)

            assert abs(t - t_prev - dt) < dt * 0.001

            y_list.append(y)
            t_list.append(t)

        assert jnp.allclose(jnp.array(y_list), analytical_y(jnp.array(t_list)))
