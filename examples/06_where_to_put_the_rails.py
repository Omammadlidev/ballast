r"""Why the slider rails belong *below* the centre of mass.

A moving-mass actuator does not only shift the centre of gravity.  Accelerating
the mass also pushes back on the airframe, and that reaction torque

.. math:: -\dot h_{\mathrm{rel}} = -\mu\,(s \times \ddot s),
   \qquad \mu = \frac{m_b m_s}{M},

arrives *immediately*, while the steering torque it is supposed to produce only
appears once the mass has actually travelled.  Its sign is set by one design
choice: the height of the rails relative to the shell centre of mass.

* rails **below** -- the reaction tilts the vehicle the way the CoM shift
  eventually will.  The actuator leads.
* rails **above** -- the reaction tilts it the wrong way first.  The plant is
  non-minimum phase, and every attitude gain that would otherwise be fine now
  drives an overshoot.

A model that freezes the inertia tensor and drops the reaction terms cannot see
any of this: it gives the same answer for both layouts.
"""
import numpy as np

import ballast as B
from ballast.dynamics import Control, derivative, initial_state
from ballast.multibody import angular_acceleration
from ballast import inertia_terms

print(__doc__)

print("1.  Sign of the instantaneous reaction, before the mass has moved")
print("=" * 76)
print(f"  {'rail height':>12s} {'slider accel':>14s} {'pitch accel':>14s}   effect")
for zs in (-0.030, 0.0, +0.030):
    veh = B.Vehicle()
    veh.slider.plane_z = zs
    dx = derivative(veh, initial_state(veh), Control(veh.hover_thrust, 0.04, 0.0))
    effect = ("reinforces steering" if dx[11] > 1e-9 else
              "none" if abs(dx[11]) < 1e-9 else "OPPOSES  (non-minimum phase)")
    print(f"  {zs:+12.3f} m {dx[16]:12.2f} m/s^2 {dx[11]:+12.4f} rad/s^2   {effect}")
print()

print("2.  What it costs in closed loop: step to (1.5, -1.0, 2.0) m")
print("=" * 76)
print(f"  {'rail height':>12s} {'settled RMS':>13s} {'peak tilt':>11s}")
for zs in (-0.045, -0.030, -0.015, 0.0, +0.015, +0.030, +0.045):
    veh = B.Vehicle()
    veh.slider.plane_z = zs
    tel = B.Simulator(veh, dt=2e-3).run(
        B.FlightController(), 25.0,
        trajectory=B.trajectories.step((1.5, -1.0, 2.0)))
    print(f"  {zs:+12.3f} m {tel.rms_error(5.0):11.4f} m {tel.max_tilt_deg:9.1f} deg")
print()
print("  Same mass, same travel, same gains, same rotor.  Moving the rails from")
print("  30 mm below the centre of mass to 30 mm above costs two orders of")
print("  magnitude in tracking accuracy and six times the peak tilt.")
print()

print("3.  How big are the terms a simplified model would drop?")
print("=" * 76)
veh = B.Vehicle()
print(f"  {'slider speed':>13s} {'|dJ.w|':>12s} {'|w x h_rel|':>13s} "
      f"{'|h_rel_dot|':>13s} {'total vs authority':>20s}")
for rate in (0.0, 0.2, 0.4, 0.8):
    s = veh.slider.position(0.045, -0.02)
    s_dot = np.array([rate, -0.4 * rate, 0.0])
    s_ddot = np.array([0.0, veh.slider.omega_n ** 2 * 0.02, 0.0])
    omega = np.array([0.4, -0.3, 1.2])
    _, J_dot, h_rel = inertia_terms(veh, s, s_dot)
    a = np.linalg.norm(J_dot @ omega)
    b = np.linalg.norm(np.cross(omega, h_rel))
    c = veh.reduced_mass * np.linalg.norm(np.cross(s, s_ddot))
    print(f"  {rate:10.2f} m/s {1e6 * a:9.1f} uN.m {1e6 * b:10.1f} uN.m "
          f"{1e6 * c:10.1f} uN.m {100 * (a + b + c) / veh.control_torque_max:17.1f} %")
print()
print("  The reaction term alone is the same order as the entire steering")
print("  authority during a fast slider move.  These are not small corrections;")
print("  they are the dominant transient, and they are perfectly correlated with")
print("  the control input -- the worst kind of disturbance to leave unmodelled.")
print()

print("4.  Sanity: hold the slider still and the model must reduce to Euler")
print("=" * 76)
s = veh.slider.position(0.03, -0.01)
omega = np.array([0.3, -0.5, 0.9])
tau = np.array([0.01, -0.02, 0.005])
z = np.zeros(3)
exact = angular_acceleration(veh, omega, s, z, z, tau).omega_dot
J, _, _ = inertia_terms(veh, s, z)
euler = np.linalg.solve(J, tau - np.cross(omega, J @ omega))
print(f"  exact       {np.round(exact, 12)}")
print(f"  rigid body  {np.round(euler, 12)}")
print(f"  difference  {np.linalg.norm(exact - euler):.2e}")
