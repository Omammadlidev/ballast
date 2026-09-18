"""Reference trajectories.

Each returns ``(position, velocity, acceleration)`` at time ``t``.  Feeding the
outer loop true velocity and acceleration feedforward matters here: the vehicle
is slow enough that purely reactive tracking lags visibly.
"""
from __future__ import annotations

import numpy as np

__all__ = ["hover", "step", "circle", "figure_eight", "helix", "waypoints", "TRAJECTORIES"]

_Z = np.zeros(3)


def hover(position=(0.0, 0.0, 0.0)):
    p = np.asarray(position, dtype=float)
    return lambda t: (p, _Z, _Z)


def step(target=(1.5, -1.0, 2.0), at: float = 1.0):
    tgt = np.asarray(target, dtype=float)
    return lambda t: (tgt if t >= at else _Z, _Z, _Z)


def circle(radius: float = 2.0, rate: float = 0.35, altitude: float = 1.5,
           spin_up: float = 4.0, start: float = 2.0):
    def f(t):
        s = max(t - start, 0.0)
        z = altitude * min(s / spin_up, 1.0) if spin_up > 0 else altitude
        vz = altitude / spin_up if 0.0 < s < spin_up else 0.0
        w = rate
        return (np.array([radius * np.cos(w * s) - radius, radius * np.sin(w * s), z]),
                np.array([-radius * w * np.sin(w * s), radius * w * np.cos(w * s), vz]),
                np.array([-radius * w * w * np.cos(w * s),
                          -radius * w * w * np.sin(w * s), 0.0]))
    return f


def figure_eight(amplitude: float = 2.0, rate: float = 0.30, altitude: float = 1.5,
                 spin_up: float = 4.0, start: float = 2.0):
    """Lemniscate of Gerono: x = A sin(wt), y = (A/1.6) sin(2wt)."""
    ay = amplitude / 1.6

    def f(t):
        s = max(t - start, 0.0)
        w = rate
        z = altitude * min(s / spin_up, 1.0) if spin_up > 0 else altitude
        vz = altitude / spin_up if 0.0 < s < spin_up else 0.0
        return (np.array([amplitude * np.sin(w * s), ay * np.sin(2 * w * s), z]),
                np.array([amplitude * w * np.cos(w * s),
                          2 * ay * w * np.cos(2 * w * s), vz]),
                np.array([-amplitude * w * w * np.sin(w * s),
                          -4 * ay * w * w * np.sin(2 * w * s), 0.0]))
    return f


def helix(radius: float = 1.5, rate: float = 0.5, climb: float = 0.3,
          start: float = 2.0):
    def f(t):
        s = max(t - start, 0.0)
        w = rate
        return (np.array([radius * np.cos(w * s) - radius,
                          radius * np.sin(w * s), climb * s]),
                np.array([-radius * w * np.sin(w * s),
                          radius * w * np.cos(w * s), climb]),
                np.array([-radius * w * w * np.cos(w * s),
                          -radius * w * w * np.sin(w * s), 0.0]))
    return f


def waypoints(points, hold: float = 6.0, start: float = 1.0):
    """Hold each waypoint for ``hold`` seconds, then jump to the next."""
    pts = [np.asarray(p, dtype=float) for p in points]

    def f(t):
        if t < start:
            return (pts[0], _Z, _Z)
        i = min(int((t - start) // hold), len(pts) - 1)
        return (pts[i], _Z, _Z)
    return f


TRAJECTORIES = {
    "hover": hover,
    "step": step,
    "circle": circle,
    "figure8": figure_eight,
    "helix": helix,
}
