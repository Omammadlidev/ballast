"""3-D animation of the vehicle, drawn from its actual parameters.

The airframe is not a generic drone glyph: the rotor sits where
``rotor.position`` says it does, the slider rails span ``slider.travel``, and
the moving mass and the resulting centre of mass are drawn where the simulation
put them.  Watching the mass lead the tilt is the clearest explanation of how
the vehicle steers.
"""
from __future__ import annotations

import numpy as np

from .. import math3d as m3
from . import style

__all__ = ["vehicle_geometry", "animate", "save_animation"]


def vehicle_geometry(veh) -> dict:
    """Body-frame line segments describing the airframe."""
    rt = float(veh.rotor.position[2])
    travel = veh.slider.travel
    zs = veh.slider.plane_z
    bottom = zs - 0.09
    r_rotor = 0.145
    r_body = 0.032

    th = np.linspace(0, 2 * np.pi, 49)
    rotor_ring = np.stack([r_rotor * np.cos(th), r_rotor * np.sin(th),
                           np.full_like(th, rt)], axis=1)

    # fuselage: a slim hexagonal shell between the rotor mast and the legs
    hex_th = np.linspace(0, 2 * np.pi, 7)
    top = np.stack([r_body * np.cos(hex_th), r_body * np.sin(hex_th),
                    np.full_like(hex_th, zs + 0.055)], axis=1)
    bot = np.stack([r_body * np.cos(hex_th), r_body * np.sin(hex_th),
                    np.full_like(hex_th, zs - 0.045)], axis=1)
    shell = [top, bot] + [np.stack([top[i], bot[i]]) for i in range(6)]

    mast = np.array([[0.0, 0.0, zs + 0.055], [0.0, 0.0, rt]])

    # four fixed anti-torque vanes in the slipstream
    vanes = []
    for a in np.arange(4) * (np.pi / 2) + np.pi / 4:
        d = np.array([np.cos(a), np.sin(a), 0.0])
        vanes.append(np.stack([d * 0.035 + [0, 0, rt - 0.055],
                               d * 0.105 + [0, 0, rt - 0.055],
                               d * 0.105 + [0, 0, rt - 0.020],
                               d * 0.035 + [0, 0, rt - 0.020],
                               d * 0.035 + [0, 0, rt - 0.055]]))

    rails = [np.array([[-travel, 0.0, zs], [travel, 0.0, zs]]),
             np.array([[0.0, -travel, zs], [0.0, travel, zs]])]

    legs = [np.array([[0.0, 0.0, zs - 0.045],
                      [0.07 * np.cos(a), 0.07 * np.sin(a), bottom]])
            for a in np.arange(3) * (2 * np.pi / 3)]

    return {"rotor_ring": rotor_ring, "shell": shell, "mast": mast,
            "vanes": vanes, "rails": rails, "legs": legs,
            "rotor_z": rt, "r_rotor": r_rotor}


def _blades(r_rotor: float, rotor_z: float, phase: float) -> list[np.ndarray]:
    out = []
    for k in range(2):
        a = phase + k * np.pi
        out.append(np.array([[0.0, 0.0, rotor_z],
                             [r_rotor * np.cos(a), r_rotor * np.sin(a), rotor_z]]))
    return out


def animate(tel, veh, stride: int = 4, trail: int = 600, fps: int = 25,
            title: str = "ballast", follow: bool = False, span: float = 0.75,
            scale: float = 3.5, elev: float = 20.0, azim: float = -60.0):
    """Build a :class:`matplotlib.animation.FuncAnimation` of a flight.

    ``stride`` decimates the telemetry.  ``scale`` magnifies the drawn airframe:
    at true scale a 0.3 m vehicle is a speck inside a 4 m figure-eight, so the
    default exaggerates it enough to read its attitude while keeping the whole
    flight path in frame.  Set ``follow=True`` with a small ``span`` to chase
    the vehicle instead.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    style.apply(plt)

    geo = vehicle_geometry(veh)
    idx = np.arange(0, len(tel.t), stride)

    fig = plt.figure(figsize=(6.6, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor(style.BG)
        pane.pane.set_edgecolor(style.GRID)
        pane.pane.set_alpha(1.0)
    ax.grid(True, color=style.GRID, linewidth=0.4)

    ref_line, = ax.plot([], [], [], color=style.REFERENCE, ls="--", lw=1.1)
    trail_line, = ax.plot([], [], [], color=style.ACCENT, lw=2.0, alpha=0.9)
    body_lines = [ax.plot([], [], [], color=style.FG, lw=1.3)[0]
                  for _ in range(len(geo["shell"]) + len(geo["legs"]) + 1)]
    vane_lines = [ax.plot([], [], [], color=style.MUTED, lw=1.0)[0]
                  for _ in geo["vanes"]]
    rail_lines = [ax.plot([], [], [], color=style.GRID, lw=1.0)[0]
                  for _ in geo["rails"]]
    ring_line, = ax.plot([], [], [], color=style.MUTED, lw=1.0)
    blade_lines = [ax.plot([], [], [], color=style.FG, lw=2.0)[0] for _ in range(2)]
    slider_pt, = ax.plot([], [], [], "o", color=style.SLIDER, ms=9.0,
                         mec=style.BG, mew=0.8)
    com_pt, = ax.plot([], [], [], "o", color=style.LIMIT, ms=5.0)
    thrust_line, = ax.plot([], [], [], color=style.ACCENT, lw=1.6, alpha=0.5)

    label = ax.text2D(0.015, 0.90, "", transform=ax.transAxes, fontsize=8.5,
                      color=style.MUTED, family="monospace", va="top")
    ax.text2D(0.015, 0.99, title, transform=ax.transAxes, fontsize=11,
              color=style.FG, fontweight="bold", va="top")
    ax.set_xlabel("x [m]", labelpad=-4); ax.set_ylabel("y [m]", labelpad=-4)
    ax.set_zlabel("z [m]", labelpad=-4)
    ax.tick_params(pad=-2)

    # Fit the box to the flight, with room for the (magnified) airframe, and
    # give the axes a box aspect matching the real extents so nothing is wasted.
    all_pts = np.vstack([tel.position, tel.reference])
    lo, hi = all_pts.min(0), all_pts.max(0)
    mid = 0.5 * (lo + hi)
    pad = 0.12 * scale
    half_xyz = np.maximum(0.5 * (hi - lo) + pad, span)
    limits = np.stack([mid - half_xyz, mid + half_xyz], axis=1)
    if not follow:
        ax.set_box_aspect(tuple(np.maximum(half_xyz / half_xyz.max(), 0.32)))

    def draw(frame: int):
        i = idx[frame]
        R = m3.quat_to_rot(tel.quat[i])
        p = tel.position[i]                      # airframe origin
        s_xy = tel.slider[i]
        s_body = veh.slider.position(s_xy[0], s_xy[1])
        com_body = veh.com_body(s_body)

        def W(pts):                              # body -> world, drawn at `scale`
            return scale * (R @ np.atleast_2d(pts).T).T + p

        segs = [W(g) for g in geo["shell"]] + [W(geo["mast"])] + \
               [W(g) for g in geo["legs"]]
        for ln, seg in zip(body_lines, segs):
            ln.set_data(seg[:, 0], seg[:, 1]); ln.set_3d_properties(seg[:, 2])
        for ln, g in zip(vane_lines, geo["vanes"]):
            seg = W(g)
            ln.set_data(seg[:, 0], seg[:, 1]); ln.set_3d_properties(seg[:, 2])
        for ln, g in zip(rail_lines, geo["rails"]):
            seg = W(g)
            ln.set_data(seg[:, 0], seg[:, 1]); ln.set_3d_properties(seg[:, 2])

        ring = W(geo["rotor_ring"])
        ring_line.set_data(ring[:, 0], ring[:, 1]); ring_line.set_3d_properties(ring[:, 2])
        phase = 42.0 * tel.t[i]
        for ln, g in zip(blade_lines, _blades(geo["r_rotor"], geo["rotor_z"], phase)):
            seg = W(g)
            ln.set_data(seg[:, 0], seg[:, 1]); ln.set_3d_properties(seg[:, 2])

        sp = W(s_body)[0]
        slider_pt.set_data([sp[0]], [sp[1]]); slider_pt.set_3d_properties([sp[2]])
        cp = W(com_body)[0]
        com_pt.set_data([cp[0]], [cp[1]]); com_pt.set_3d_properties([cp[2]])

        tip = W(np.array([0.0, 0.0, geo["rotor_z"] + 0.10]))[0]
        base = W(np.array([0.0, 0.0, geo["rotor_z"]]))[0]
        thrust_line.set_data([base[0], tip[0]], [base[1], tip[1]])
        thrust_line.set_3d_properties([base[2], tip[2]])

        a = max(i - trail, 0)
        tr = tel.position[a:i + 1]
        trail_line.set_data(tr[:, 0], tr[:, 1]); trail_line.set_3d_properties(tr[:, 2])
        rf = tel.reference[:i + 1]
        ref_line.set_data(rf[:, 0], rf[:, 1]); ref_line.set_3d_properties(rf[:, 2])

        if follow:
            ax.set_box_aspect((1, 1, 1))
            ax.set_xlim(p[0] - span, p[0] + span)
            ax.set_ylim(p[1] - span, p[1] + span)
            ax.set_zlim(p[2] - span, p[2] + span)
        else:
            ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_zlim(*limits[2])

        label.set_text(
            f"t       {tel.t[i]:6.2f} s\n"
            f"tilt    {np.degrees(tel.tilt[i]):5.1f} deg\n"
            f"slider  {1e3 * s_xy[0]:+5.1f}, {1e3 * s_xy[1]:+5.1f} mm\n"
            f"CoM     {1e3 * com_body[0]:+5.1f}, {1e3 * com_body[1]:+5.1f} mm\n"
            f"error   {np.linalg.norm(tel.error[i]):5.3f} m")
        return body_lines + vane_lines + rail_lines + blade_lines + [
            ring_line, slider_pt, com_pt, trail_line, ref_line, thrust_line, label]

    anim = FuncAnimation(fig, draw, frames=len(idx), interval=1000 / fps, blit=False)
    anim._ballast_fig = fig
    return anim


def save_animation(anim, path: str, fps: int = 30, dpi: int = 100) -> str:
    """Write the animation to ``.gif`` or ``.mp4``."""
    from matplotlib.animation import FFMpegWriter, PillowWriter
    writer = PillowWriter(fps=fps) if str(path).endswith(".gif") else FFMpegWriter(fps=fps)
    anim.save(path, writer=writer, dpi=dpi)
    return path
