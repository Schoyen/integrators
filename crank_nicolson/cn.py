import jax


class CrankNicolson:
    def __init__(self, rhs, y0, t0, tol):
        self.rhs = rhs
        self.y = y0
        self.t = t0
        self.tol = tol

    def integrate(self, t):
        dt = t - self.t

        A = lambda c, t=t, dt=dt, rhs=self.rhs: c - dt / 2 * rhs(t + dt / 2, c)
        b = self.y + dt / 2 * self.rhs(t + dt / 2, self.y)

        self.y, _ = jax.scipy.sparse.linalg.cg(A, b, x0=self.y, tol=self.tol)
        self.t += dt

        return self.t, self.y
