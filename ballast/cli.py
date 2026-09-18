"""Command line entry point: ``python -m ballast ...``"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import analysis, trajectories
from .control import FlightController
from .params import Vehicle
from .sim import Simulator
from .trim import describe_modes, hover_trim, linearize, modes
from .wind import Wind


def _vehicle(args) -> Vehicle:
    veh = Vehicle()
    if getattr(args, "slider_mass", None):
        veh.slider.mass = args.slider_mass
    if getattr(args, "travel", None):
        veh.slider.travel = args.travel
    return veh


def _wind(args) -> Wind | None:
    if not getattr(args, "wind", 0.0):
        return None
    return Wind((args.wind, 0.4 * args.wind, 0.0),
                sigma=args.gust if args.gust is not None else 0.25 * args.wind,
                tau=1.0, seed=args.seed)


def cmd_info(args) -> int:
    veh = _vehicle(args)
    print(veh.summary())
    print(f"max steady tilt     {analysis.max_steady_tilt(veh):.1f} deg")
    bw = analysis.slider_bandwidth_limit(veh)
    print(f"slider bandwidth    {bw['servo_bandwidth_hz']:.2f} Hz servo, "
          f"{bw['rate_limited_hz']:.2f} Hz rate-limited ({bw['limiting']}-limited)")
    tp = hover_trim(veh)
    print(f"hover trim residual {tp.residual:.4f} rad/s^2 "
          f"(yaw only -- no actuator can null it)")
    return 0


def cmd_modes(args) -> int:
    veh = _vehicle(args)
    A, B = linearize(veh)
    print(describe_modes(modes(A)))
    if args.save:
        from .viz import plot_modes
        plot_modes(np.linalg.eigvals(A), save=args.save)
        print(f"\nwrote {args.save}")
    return 0


def cmd_envelope(args) -> int:
    veh = _vehicle(args)
    rows = analysis.wind_envelope(veh)
    print(f"{'wind [m/s]':>11s} {'trim tilt [deg]':>16s} "
          f"{'aero torque [mN.m]':>19s} {'margin [mN.m]':>15s}")
    print("-" * 66)
    for v, tilt, tq, margin in rows[::4]:
        flag = "  <-- exhausted" if margin <= 0 else ""
        print(f"{v:11.1f} {tilt:16.2f} {1e3 * tq:19.2f} {1e3 * margin:15.2f}{flag}")
    ok = [r[0] for r in rows if r[3] > 0]
    print(f"\nsteering margin holds up to {max(ok):.1f} m/s of steady wind")
    if args.save:
        from .viz import plot_wind_envelope
        plot_wind_envelope(rows, save=args.save)
        print(f"wrote {args.save}")
    return 0


def cmd_fly(args) -> int:
    veh = _vehicle(args)
    factory = trajectories.TRAJECTORIES[args.trajectory]
    traj = factory()
    tel = Simulator(veh, dt=args.dt, wind=_wind(args)).run(
        FlightController(), args.duration, trajectory=traj)
    print(tel.summary(args.trajectory))
    if args.dashboard:
        from .viz import dashboard
        dashboard(tel, veh, title=f"{args.trajectory} tracking", save=args.dashboard)
        print(f"wrote {args.dashboard}")
    if args.animate:
        from .viz import animate, save_animation
        anim = animate(tel, veh, stride=args.stride)
        save_animation(anim, args.animate, fps=22, dpi=74)
        print(f"wrote {args.animate}")
    return 1 if tel.diverged_at is not None else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ballast",
        description="single-rotor UAV steered by an internal moving mass")
    p.add_argument("--slider-mass", type=float, help="override slider mass [kg]")
    p.add_argument("--travel", type=float, help="override slider half-travel [m]")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("info", help="print the vehicle's sizing and limits")
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("modes", help="linearise about hover and list the modes")
    s.add_argument("--save", help="write a pole map to this path")
    s.set_defaults(func=cmd_modes)

    s = sub.add_parser("envelope", help="steering margin against steady wind")
    s.add_argument("--save", help="write the envelope chart to this path")
    s.set_defaults(func=cmd_envelope)

    s = sub.add_parser("fly", help="simulate a flight")
    s.add_argument("trajectory", choices=sorted(trajectories.TRAJECTORIES))
    s.add_argument("--duration", type=float, default=30.0)
    s.add_argument("--dt", type=float, default=2e-3)
    s.add_argument("--wind", type=float, default=0.0, help="steady wind [m/s]")
    s.add_argument("--gust", type=float, default=None, help="gust sigma [m/s]")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--dashboard", help="write a telemetry dashboard here")
    s.add_argument("--animate", help="write a .gif or .mp4 here")
    s.add_argument("--stride", type=int, default=5)
    s.set_defaults(func=cmd_fly)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
