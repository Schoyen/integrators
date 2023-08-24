import jax


def get_runge_kutta_4_solver(rhs, dt):
    @jax.jit
    def integrate(t, y, dt=dt, rhs=rhs):
        k1 = rhs(t, y)
        k2 = rhs(t + dt / 2, y + dt * k1 / 2)
        k3 = rhs(t + dt / 2, y + dt * k2 / 2)
        k4 = rhs(t + dt, y + dt * k3)

        y = y + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * dt

        return t + dt, y

    return integrate


def get_crank_nicolson_solver(rhs, dt, solver=None):
    if solver is None:
        solver = jax.jit(jax.scipy.sparse.linalg.cg, static_argnums=(0,))

    @jax.jit
    def integrate(t, y, dt=dt, rhs=rhs, solver=solver):
        A = lambda c, t=t, dt=dt, rhs=rhs: c - dt / 2 * rhs(t + dt / 2, c)
        b = y + dt / 2 * rhs(t + dt / 2, y)

        return t + dt, solver(A, b)[0]

    return integrate
