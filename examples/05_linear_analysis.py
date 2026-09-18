"""Linearise the vehicle about hover and check every mode against theory.

The point of this example is that nothing in the mode list is mysterious: each
eigenvalue can be predicted from a single parameter, which is a strong check
that the nonlinear model and the linearisation agree.
"""
import sys

import numpy as np

from ballast import Vehicle, describe_modes, linearize, modes

veh = Vehicle()
A, B = linearize(veh)
print(describe_modes(modes(A)))
print()

J = veh.inertia_about_com(veh.slider.position(0.0, 0.0))
sp, rp, ap = veh.slider, veh.rotor, veh.aero
predictions = {
    "slider servo": complex(-sp.zeta * sp.omega_n,
                            sp.omega_n * np.sqrt(1 - sp.zeta ** 2)),
    "rotor spool": complex(-1.0 / rp.tau_spool, 0.0),
    "yaw rate damping": complex(-ap.rate_damp_lin[2] / J[2, 2], 0.0),
    "roll rate damping": complex(-ap.rate_damp_lin[0] / J[0, 0], 0.0),
}
ev = np.linalg.eigvals(A)
print(f"{'mode':<20s} {'predicted':>22s} {'nearest eigenvalue':>24s} {'error':>10s}")
print("-" * 80)
for name, want in predictions.items():
    got = ev[np.argmin(np.abs(ev - want))]
    print(f"{name:<20s} {want.real:+10.4f}{want.imag:+9.4f}j "
          f"{got.real:+12.4f}{got.imag:+9.4f}j {abs(got - want):10.2e}")
print()
print("The remaining eigenvalues are structural zeros: three position")
print("integrators, the horizontal x <- v <- theta <- omega chains, and the")
print("free heading.  A hovering thrust-vectored vehicle is never self-stable;")
print("it is stabilised entirely by feedback.")
print()
print("Input directions")
print("-" * 80)
print(f"  d(rotor spool)/d(thrust cmd)   {B[12, 0]:9.3f}")
print(f"  d(slider acc)/d(slider cmd)    {B[15, 1]:9.3f}  (= omega_n^2)")
print(f"  d(yaw acc)/d(slider cmd)       {B[11, 1]:9.3e}  <- structurally zero")

if "--plot" in sys.argv:
    from ballast.viz import plot_modes
    plot_modes(ev, save="modes.png")
    print("\nwrote modes.png")
