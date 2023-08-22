import functools
import jax


class RungeKutta4:
    def __init__(self, rhs, y0, t0):
        self.rhs = rhs
        self.y = y0
        self.t = t0

    def integrate(self, t):
        dt = t - self.t

        k1 = self.rhs(self.t, self.y)
        k2 = self.rhs(self.t + dt / 2, self.y + dt * k1 / 2)
        k3 = self.rhs(self.t + dt / 2, self.y + dt * k2 / 2)
        k4 = self.rhs(self.t + dt, self.y + dt * k3)

        self.y = self.y + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * dt
        self.t += dt

        return self.t, self.y


class CrankNicolson:
    def __init__(self, rhs, y0, t0, solver=None):
        self.rhs = rhs
        self.y = y0
        self.t = t0

        if solver is None:
            solver = jax.scipy.sparse.linalg.cg

        self.solver = solver

    def integrate(self, t):
        dt = t - self.t

        A = lambda c, t=t, dt=dt, rhs=self.rhs: c - dt / 2 * rhs(t + dt / 2, c)
        b = self.y + dt / 2 * self.rhs(t + dt / 2, self.y)

        self.y, _ = self.solver(A, b)
        self.t += dt

        return self.t, self.y
