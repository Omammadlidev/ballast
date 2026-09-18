"""Fixed-step RK4 simulator with telemetry capture."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import math3d as m3
from .dynamics import (I_Q, I_S, I_SD, Control, derivative, initial_state,
                       origin_position, origin_velocity, unpack)
from .params import Vehicle
from .wind import Wind

__all__ = ["Telemetry", "Simulator", "simulate"]


@dataclass
class Telemetry:
    """Recorded flight history.  All arrays share a leading time axis."""

    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    position: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    reference: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    quat: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    omega: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    tilt: np.ndarray = field(default_factory=lambda: np.zeros(0))
    heading: np.ndarray = field(default_factory=lambda: np.zeros(0))
    thrust_cmd: np.ndarray = field(default_factory=lambda: np.zeros(0))
    slider: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    slider_cmd: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    wind: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    diverged_at: float | None = None
    saturation: float = 0.0
    slider_travel: float = 0.060

    # ---- derived metrics ----------------------------------------------------
    @property
    def error(self) -> np.ndarray:
        return self.position - self.reference

    def rms_error(self, last_seconds: float | None = None) -> float:
        e = self.error
        if last_seconds is not None and len(self.t) > 1:
            dt = float(self.t[1] - self.t[0])
            e = e[-max(int(last_seconds / dt), 1):]
        return float(np.sqrt((e ** 2).sum(axis=1).mean())) if len(e) else float("nan")

    def max_error(self, last_seconds: float | None = None) -> float:
        e = self.error
        if last_seconds is not None and len(self.t) > 1:
            dt = float(self.t[1] - self.t[0])
            e = e[-max(int(last_seconds / dt), 1):]
        return float(np.sqrt((e ** 2).sum(axis=1)).max()) if len(e) else float("nan")

    @property
    def max_tilt_deg(self) -> float:
        return float(np.degrees(self.tilt).max()) if len(self.tilt) else float("nan")

    @property
    def yaw_drift_deg(self) -> float:
        if len(self.heading) < 2:
            return float("nan")
        return float(np.degrees(np.unwrap(self.heading)[-1] - self.heading[0]))

    @property
    def slider_usage(self) -> float:
        """Peak slider excursion as a fraction of available travel.

        Values slightly above 1.0 are the position servo overshooting its
        command; the per-axis end-stops still bound the physical travel.
        """
        if not len(self.slider):
            return float("nan")
        return float(np.linalg.norm(self.slider, axis=1).max() / self.slider_travel)

    def summary(self, name: str = "run", tail: float = 5.0) -> str:
        flag = "" if self.diverged_at is None else f"  ** DIVERGED @ {self.diverged_at:.2f}s"
        return (f"{name:<26s} RMS {self.rms_error(tail):7.4f} m | "
                f"max {self.max_error(tail):7.4f} m | "
                f"tilt {self.max_tilt_deg:5.1f} deg | "
                f"slider {100 * self.slider_usage:5.1f}% | "
                f"yaw {self.yaw_drift_deg:+7.1f} deg{flag}")


class Simulator:
    """Integrate the vehicle forward under a controller and a wind field."""

    def __init__(self, vehicle: Vehicle, dt: float = 1e-3,
                 wind: Wind | None = None, log_every: int = 20):
        self.veh = vehicle
        self.dt = float(dt)
        self.wind = wind
        self.log_every = int(log_every)

    # ---- integration --------------------------------------------------------
    def _apply_endstops(self, x: np.ndarray) -> np.ndarray:
        """Inelastic hard stops at the end of the slider rails."""
        travel = self.veh.slider.travel
        for i in range(2):
            pos, rate = x[I_S.start + i], x[I_SD.start + i]
            if abs(pos) > travel:
                x[I_S.start + i] = np.clip(pos, -travel, travel)
                if pos * rate > 0.0:
                    x[I_SD.start + i] = 0.0
        return x

    def step(self, x: np.ndarray, u: Control, wind_vec: np.ndarray) -> np.ndarray:
        dt = self.dt
        f = lambda y: derivative(self.veh, y, u, wind_vec)
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        y = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        y[I_Q] = m3.quat_normalize(y[I_Q])
        return self._apply_endstops(y)

    # ---- driver -------------------------------------------------------------
    def run(self, controller, duration: float, x0: np.ndarray | None = None,
            trajectory=None, altitude_floor: float = -1.0,
            divergence_limit: float = 60.0) -> Telemetry:
        veh, dt = self.veh, self.dt
        x = (initial_state(veh) if x0 is None else np.array(x0, dtype=float)).copy()
        if self.wind is not None:
            self.wind.reset()

        n = int(round(duration / dt))
        rows: list[tuple] = []
        saturated = 0
        diverged = None

        for k in range(n):
            t = k * dt
            if trajectory is None:
                ref = (np.zeros(3), np.zeros(3), np.zeros(3))
            else:
                ref = trajectory(t)
            u = controller(veh, x, t, dt, *ref)
            travel = veh.slider.travel
            if np.hypot(u.slider_x, u.slider_y) >= travel - 1e-9:
                saturated += 1
            wind_vec = self.wind.step(dt) if self.wind is not None else np.zeros(3)

            if k % self.log_every == 0:
                st = unpack(x)
                rows.append((
                    t, origin_position(veh, x), origin_velocity(veh, x), ref[0],
                    st.quat.copy(), st.omega.copy(), m3.tilt_angle(st.R),
                    m3.heading_angle(st.R), u.thrust, st.slider.copy(),
                    np.array([u.slider_x, u.slider_y]), wind_vec.copy(),
                ))

            x = self.step(x, u, wind_vec)
            if not np.all(np.isfinite(x)) or abs(x[2]) > divergence_limit:
                diverged = t
                break

        tel = Telemetry(
            t=np.array([r[0] for r in rows]),
            position=np.array([r[1] for r in rows]),
            velocity=np.array([r[2] for r in rows]),
            reference=np.array([r[3] for r in rows]),
            quat=np.array([r[4] for r in rows]),
            omega=np.array([r[5] for r in rows]),
            tilt=np.array([r[6] for r in rows]),
            heading=np.array([r[7] for r in rows]),
            thrust_cmd=np.array([r[8] for r in rows]),
            slider=np.array([r[9] for r in rows]),
            slider_cmd=np.array([r[10] for r in rows]),
            wind=np.array([r[11] for r in rows]),
            diverged_at=diverged,
            saturation=saturated / max(n, 1),
            slider_travel=veh.slider.travel,
        )
        return tel


def simulate(vehicle: Vehicle, controller, duration: float, **kwargs) -> Telemetry:
    """Convenience wrapper around :class:`Simulator`."""
    sim_kwargs = {k: kwargs.pop(k) for k in ("dt", "wind", "log_every") if k in kwargs}
    return Simulator(vehicle, **sim_kwargs).run(controller, duration, **kwargs)
