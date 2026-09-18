"""Hold station, then move somewhere else.

Run with ``--plot`` to write a telemetry dashboard (needs matplotlib).
"""
import sys

from ballast import (FlightController, Simulator, Vehicle, initial_state,
                     trajectories)

veh = Vehicle()

print("Recovering from a 0.54 m displacement")
tel = Simulator(veh, dt=2e-3).run(
    FlightController(), 40.0,
    x0=initial_state(veh, position=(0.4, -0.3, 0.2)))
print(" ", tel.summary("hover"))

print("\nStep to (1.5, -1.0, 2.0) m")
step = Simulator(veh, dt=2e-3).run(
    FlightController(), 25.0, trajectory=trajectories.step((1.5, -1.0, 2.0)))
print(" ", step.summary("step"))
print(f"  peak slider excursion {100 * step.slider_usage:.0f}% of travel")
print(f"  peak tilt             {step.max_tilt_deg:.1f} deg")

if "--plot" in sys.argv:
    from ballast.viz import dashboard
    dashboard(step, veh, title="step response", save="step_response.png")
    print("\nwrote step_response.png")
