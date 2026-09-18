"""Where the wind envelope actually ends.

Two different questions, with two different answers:

* *Static* -- at what steady wind speed does the aerodynamic moment about the
  centre of mass eat the whole steering margin?
* *Dynamic* -- at what wind speed does the closed loop stop holding station in
  gusty air?

The second number is much smaller than the first, and it is the one that
matters.
"""
import sys

from ballast import FlightController, Simulator, Vehicle, Wind, analysis

veh = Vehicle()

print("Static trim margin")
print("-" * 70)
rows = analysis.wind_envelope(veh)
for v, tilt, torque, margin in rows[::6]:
    print(f"  {v:5.1f} m/s   tilt {tilt:5.2f} deg   aero {1e3 * torque:6.2f} mN.m"
          f"   margin {1e3 * margin:6.2f} mN.m")
ok = [r[0] for r in rows if r[3] > 0]
print(f"  -> margin survives to {max(ok):.1f} m/s of steady wind\n")

print("Closed-loop station keeping in gusty air")
print("-" * 70)
for speed in (2.0, 4.0, 6.0, 8.0, 10.0, 12.0):
    tel = Simulator(veh, dt=2e-3,
                    wind=Wind((speed, 0.5 * speed, 0.0),
                              sigma=0.3 * speed, tau=1.0, seed=7)
                    ).run(FlightController(), 45.0)
    print(" ", tel.summary(f"{speed:4.1f} m/s + gusts", tail=15.0))

if "--plot" in sys.argv:
    from ballast.viz import plot_wind_envelope
    plot_wind_envelope(rows, save="wind_envelope.png")
    print("\nwrote wind_envelope.png")
