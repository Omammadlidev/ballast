"""Wind field: steady component plus exponentially correlated gusts.

Gusts are an Ornstein-Uhlenbeck process: Gaussian, zero-mean, with an
exponentially decaying autocorrelation of time constant ``tau``.  Its power
spectrum is

.. math:: S(\omega) = \frac{2\sigma^2\tau}{1 + (\omega\tau)^2},

which has the same single-pole form as the *longitudinal* Dryden gust spectrum
with :math:`\tau = L_u / V`.  The lateral and vertical Dryden components carry
an extra zero in the numerator and are **not** reproduced here -- this is a
first-order turbulence model, not a full Dryden implementation.

The practical advantage is exact discretisation: the update below is the exact
solution of the OU stochastic differential equation over a step, so the gust
statistics do not drift with the integrator step size.  A test checks the
stationary standard deviation and the correlation time against ``sigma`` and
``tau`` over 400 s of samples.
"""
from __future__ import annotations

import numpy as np

__all__ = ["Wind", "NO_WIND"]


class Wind:
    """Steady wind with optional Ornstein-Uhlenbeck gusts.

    Parameters
    ----------
    steady:
        Mean wind vector in world coordinates [m/s].
    sigma:
        Standard deviation of the gust component, per axis [m/s].
    tau:
        Gust correlation time [s].  Short tau gives choppy air.
    """

    def __init__(self, steady=(0.0, 0.0, 0.0), sigma: float = 0.0,
                 tau: float = 1.0, seed: int = 0):
        self.steady = np.asarray(steady, dtype=float)
        self.sigma = float(sigma)
        self.tau = float(tau)
        self._gust = np.zeros(3)
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self._gust = np.zeros(3)

    def step(self, dt: float) -> np.ndarray:
        """Advance the gust state by ``dt`` and return the wind vector."""
        if self.sigma > 0.0:
            a = np.exp(-dt / self.tau)
            self._gust = (a * self._gust
                          + self.sigma * np.sqrt(1.0 - a * a)
                          * self._rng.standard_normal(3))
        return self.steady + self._gust

    @property
    def value(self) -> np.ndarray:
        return self.steady + self._gust

    def __repr__(self) -> str:
        return (f"Wind(steady={np.round(self.steady, 2).tolist()}, "
                f"sigma={self.sigma}, tau={self.tau})")


NO_WIND = Wind()
