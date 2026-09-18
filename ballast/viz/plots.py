"""Static figures: telemetry dashboards, mode maps, envelope charts."""
from __future__ import annotations

import numpy as np

from . import style

__all__ = ["dashboard", "plot_trajectory_3d", "plot_modes", "plot_wind_envelope"]


def _plt():
    import matplotlib.pyplot as plt
    style.apply(plt)
    return plt


def _equalize_3d(ax, pts: np.ndarray) -> None:
    """Matplotlib has no true 3-D aspect lock, so impose one by hand."""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    mid = 0.5 * (lo + hi)
    span = max(float((hi - lo).max()), 0.5) * 0.6
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)


def plot_trajectory_3d(tel, ax=None, title: str = "flight path"):
    plt = _plt()
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
    p, r = tel.position, tel.reference
    ax.plot(r[:, 0], r[:, 1], r[:, 2], color=style.REFERENCE, lw=1.2,
            ls="--", label="reference")
    ax.plot(p[:, 0], p[:, 1], p[:, 2], color=style.ACCENT, lw=1.8, label="flown")
    ax.scatter(*p[0], color=style.OK, s=22, label="start")
    ax.scatter(*p[-1], color=style.LIMIT, s=22, label="end")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(title)
    ax.legend(loc="upper left")
    _equalize_3d(ax, np.vstack([p, r]))
    return ax


def dashboard(tel, vehicle=None, title: str = "flight telemetry", save: str | None = None):
    """Six-panel summary of one flight."""
    plt = _plt()
    fig = plt.figure(figsize=(12.5, 7.4))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3,
                          left=0.06, right=0.98, top=0.90, bottom=0.08)

    ax3d = fig.add_subplot(gs[0:2, 0], projection="3d")
    plot_trajectory_3d(tel, ax=ax3d, title="flight path")

    ax = fig.add_subplot(gs[0, 1])
    for i, lbl in enumerate("xyz"):
        ax.plot(tel.t, tel.reference[:, i], color=style.SERIES[i], ls="--", lw=1.0, alpha=0.7)
        ax.plot(tel.t, tel.position[:, i], color=style.SERIES[i], label=lbl)
    ax.set_title("position vs reference"); ax.set_ylabel("[m]"); ax.legend(ncol=3)

    ax = fig.add_subplot(gs[1, 1])
    err = np.linalg.norm(tel.error, axis=1)
    tail = min(0.25 * float(tel.t[-1]), 20.0) if len(tel.t) else 0.0
    settled = tel.rms_error(last_seconds=tail)
    ax.plot(tel.t, err, color=style.ACCENT)
    ax.axhline(settled, color=style.OK, ls=":", lw=1.1)
    if tail > 0:
        ax.axvspan(tel.t[-1] - tail, tel.t[-1], color=style.OK, alpha=0.08)
    ax.set_title(f"tracking error   whole run {tel.rms_error():.3f} m  |  "
                 f"settled {settled:.3f} m")
    ax.set_ylabel("[m]")

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(tel.t, np.degrees(tel.tilt), color=style.ACCENT, label="tilt")
    ax.plot(tel.t, np.degrees(np.unwrap(tel.heading)), color=style.SLIDER,
            lw=1.1, label="heading (free)")
    ax.set_title("attitude"); ax.set_ylabel("[deg]"); ax.set_xlabel("t [s]")
    ax.legend(ncol=2)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(tel.t, tel.thrust_cmd, color=style.ACCENT, label="commanded")
    if vehicle is not None:
        ax.axhline(vehicle.rotor.thrust_max, color=style.LIMIT, ls="--", lw=1.0,
                   label="rotor limit")
        ax.axhline(vehicle.hover_thrust, color=style.OK, ls=":", lw=1.0, label="hover")
    ax.set_title("rotor thrust"); ax.set_ylabel("[N]"); ax.set_xlabel("t [s]")
    ax.legend(ncol=3)

    ax = fig.add_subplot(gs[0:2, 2])
    travel = tel.slider_travel
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(travel * np.cos(th) * 1e3, travel * np.sin(th) * 1e3,
            color=style.LIMIT, ls="--", lw=1.0, label="travel limit")
    ax.plot(tel.slider[:, 0] * 1e3, tel.slider[:, 1] * 1e3,
            color=style.SLIDER, lw=1.0, label="slider path")
    ax.set_aspect("equal")
    ax.set_title(f"moving mass  (peak {100 * tel.slider_usage:.0f}% of travel)")
    ax.set_xlabel("s_x [mm]"); ax.set_ylabel("s_y [mm]"); ax.legend()

    ax = fig.add_subplot(gs[2, 2])
    if tel.wind.size and np.abs(tel.wind).max() > 1e-9:
        for i, lbl in enumerate("xyz"):
            ax.plot(tel.t, tel.wind[:, i], color=style.SERIES[i], lw=1.0, label=lbl)
        ax.set_title("wind"); ax.set_ylabel("[m/s]"); ax.legend(ncol=3)
    else:
        ax.plot(tel.t, np.linalg.norm(tel.slider, axis=1) / travel * 100,
                color=style.SLIDER)
        ax.axhline(100, color=style.LIMIT, ls="--", lw=1.0)
        ax.set_title("slider travel used"); ax.set_ylabel("[%]")
    ax.set_xlabel("t [s]")

    fig.suptitle(title, fontsize=12, fontweight="bold", color=style.FG, x=0.06,
                 ha="left")
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


def plot_modes(eigenvalues, title: str = "open-loop modes", save: str | None = None):
    """Pole map of the linearised vehicle."""
    plt = _plt()
    ev = np.asarray(eigenvalues)
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.axvline(0, color=style.LIMIT, lw=1.0, ls="--")
    ax.axhline(0, color=style.GRID, lw=0.8)
    ax.scatter(ev.real, ev.imag, s=46, marker="x", color=style.ACCENT, lw=1.8)
    ax.set_xlabel("Re  [1/s]"); ax.set_ylabel("Im  [rad/s]")
    ax.set_title(title)
    ax.set_xlim(min(ev.real.min() * 1.15, -1.0), max(1.0, ev.real.max() + 1.0))
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


def plot_wind_envelope(rows, title: str = "wind envelope", save: str | None = None):
    """Trim tilt and remaining steering margin against steady wind speed."""
    plt = _plt()
    v = np.array([r[0] for r in rows])
    tilt = np.array([r[1] for r in rows])
    margin = np.array([r[3] for r in rows])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True)
    a1.plot(v, tilt, color=style.ACCENT)
    a1.set_ylabel("trim tilt [deg]"); a1.set_title(title)
    a2.plot(v, margin * 1e3, color=style.OK)
    a2.axhline(0, color=style.LIMIT, ls="--", lw=1.0)
    a2.fill_between(v, 0, margin * 1e3, where=margin > 0, color=style.OK, alpha=0.15)
    a2.set_ylabel("steering margin [mN.m]"); a2.set_xlabel("steady wind [m/s]")
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig
