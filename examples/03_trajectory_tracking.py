"""Track a figure-eight and a climbing helix.

Feedforward matters here.  The vehicle's attitude loop is slow, so a purely
reactive outer loop lags the reference visibly; passing true reference velocity
and acceleration removes most of that lag.
"""
import sys

from ballast import FlightController, Simulator, Vehicle, trajectories

veh = Vehicle()

for name, traj, duration in [
    ("figure-8 (2 m)", trajectories.figure_eight(), 60.0),
    ("helix (1.5 m, 0.3 m/s climb)", trajectories.helix(), 30.0),
    ("circle (2 m)", trajectories.circle(), 45.0),
]:
    tel = Simulator(veh, dt=2e-3).run(FlightController(), duration, trajectory=traj)
    print(tel.summary(name, tail=15.0))

if "--plot" in sys.argv:
    from ballast.viz import dashboard
    tel = Simulator(veh, dt=2e-3).run(FlightController(), 60.0,
                                      trajectory=trajectories.figure_eight())
    dashboard(tel, veh, title="figure-eight tracking", save="figure8.png")
    print("wrote figure8.png")
