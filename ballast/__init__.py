"""ballast -- a single-rotor UAV steered by an internal moving mass.

One motor. No swashplate, no tail rotor, no second propeller. The vehicle points
itself by sliding a mass inside its own fuselage, which moves the centre of mass
off the thrust line and lets the rotor's own thrust do the steering.

Quick start
-----------
>>> from ballast import Vehicle, FlightController, Simulator, trajectories
>>> veh = Vehicle()
>>> tel = Simulator(veh).run(FlightController(), 20.0,
...                          trajectory=trajectories.step((1.5, -1.0, 2.0)))
>>> round(tel.rms_error(last_seconds=5.0), 4) < 0.05
True
"""
from . import analysis, math3d, trajectories
from .control import (AttitudeController, AttitudeGains, ControllerGains,
                      FlightController, PositionController, PositionGains,
                      allocate)
from .dynamics import Control, derivative, initial_state, origin_position, unpack
from .multibody import angular_acceleration, inertia_terms
from .params import AeroParams, RotorParams, SliderParams, Vehicle
from .sim import Simulator, Telemetry, simulate
from .trim import describe_modes, hover_trim, linearize, modes
from .wind import Wind

__version__ = "0.1.0"

__all__ = [
    "Vehicle", "RotorParams", "SliderParams", "AeroParams",
    "Control", "derivative", "initial_state", "origin_position", "unpack",
    "inertia_terms", "angular_acceleration",
    "Simulator", "Telemetry", "simulate", "Wind",
    "FlightController", "ControllerGains", "PositionController", "PositionGains",
    "AttitudeController", "AttitudeGains", "allocate",
    "hover_trim", "linearize", "modes", "describe_modes",
    "trajectories", "analysis", "math3d", "__version__",
]
