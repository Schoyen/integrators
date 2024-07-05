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


def get_dormand_prince_solver(rhs, dt):
    @jax.jit
    def integrate(t, y, dt=dt, rhs=rhs):
        k1 = rhs(t, y)
        k2 = rhs(t + dt / 5, y + dt * k1 / 5)
        k3 = rhs(t + dt * 3 / 10, y + dt * (3 * k1 + 9 * k2) / 40)
        k4 = rhs(t + dt * 4 / 5, y + dt * (k1 * 44 / 45 - k2 * 56 / 15 + k3 * 32 / 9))
        k5 = rhs(
            t + dt * 8 / 9,
            y
            + dt
            * (
                k1 * 19372 / 6561
                - k2 * 25360 / 2187
                + k3 * 64448 / 6561
                - k4 * 212 / 729
            ),
        )
        k6 = rhs(
            t + dt,
            y
            + dt
            * (
                k1 * 9017 / 3168
                - k2 * 355 / 33
                + k3 * 46732 / 5247
                + k4 * 49 / 176
                - k5 * 5103 / 18656
            ),
        )
        k7 = rhs(
            t + dt,
            y
            + dt
            * (
                k1 * 35 / 384
                + k3 * 500 / 1113
                + k4 * 125 / 192
                - k5 * 2187 / 6784
                + k6 * 11 / 84
            ),
        )

        y = y + dt * (
            k1 * 5179 / 57600
            + k3 * 7571 / 16695
            + k4 * 393 / 640
            - k5 * 92097 / 339200
            + k6 * 187 / 2100
            + k7 / 40
        )

        return t + dt, y

    return integrate


def get_crank_nicolson_solver(rhs, dt, solver=None):
    if solver is None:
        solver = lambda A, b: jax.jit(jax.scipy.sparse.linalg.cg, static_argnums=(0,))(
            A, b
        )[0]

    @jax.jit
    def integrate(t, y, dt=dt, rhs=rhs, solver=solver):
        A = lambda c, t=t, dt=dt, rhs=rhs: c - dt / 2 * rhs(t + dt / 2, c)
        b = y + dt / 2 * rhs(t + dt / 2, y)

        return t + dt, solver(A, b)

    return integrate
