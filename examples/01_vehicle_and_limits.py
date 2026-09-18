"""What the airframe can and cannot do, before any control is involved.

Everything printed here follows from four numbers -- slider mass, slider travel,
total mass, and thrust -- and together they explain every later result.
"""
import numpy as np

from ballast import Vehicle, analysis, hover_trim

veh = Vehicle()
print(veh.summary())
print()

print("Where the steering torque comes from")
print("-" * 52)
print(f"  slider mass ratio   m_s/M      = {veh.slider.mass / veh.total_mass:.3f}")
print(f"  slider half-travel  s_max      = {veh.slider.travel * 1e3:.0f} mm")
print(f"  CoM offset          (m_s/M)s   = {veh.com_offset_max * 1e3:.1f} mm")
print(f"  torque at hover     f * offset = {veh.control_torque_max * 1e3:.1f} mN.m")
J = veh.inertia_about_com(veh.slider.position(0.0, 0.0))[0, 0]
print(f"  roll inertia        J_xx       = {J * 1e3:.2f} g.m^2")
print(f"  peak ang. accel                = {veh.angular_accel_max:.2f} rad/s^2")
print("  for comparison, a 250 g racing quadrotor reaches ~200 rad/s^2.")
print()

print("Actuator bandwidth")
print("-" * 52)
for k, v in analysis.slider_bandwidth_limit(veh).items():
    print(f"  {k:<26s} {v if isinstance(v, str) else round(v, 3)}")
print()

print("Trim")
print("-" * 52)
tp = hover_trim(veh)
print(f"  residual                       = {tp.residual:.4f} rad/s^2")
print("  Level hover balances forces, roll and pitch exactly.  The leftover is")
print("  pure yaw: the fraction of rotor reaction torque the fixed vanes miss.")
print("  No slider position can cancel it, because the slider generates torque")
print("  only perpendicular to the thrust axis.")
print()

print("Steering authority vs thrust")
print("-" * 52)
print(f"  {'thrust [N]':>11s} {'torque [mN.m]':>14s} {'ang.acc [rad/s^2]':>19s}")
for pt in analysis.authority_envelope(veh, np.linspace(3.0, veh.rotor.thrust_max, 6)):
    print(f"  {pt.thrust:11.2f} {1e3 * pt.torque_max:14.2f} {pt.angular_accel:19.2f}")
print()
print("  Authority is proportional to thrust, so the vehicle is least")
print("  controllable exactly when it is descending fastest.")
