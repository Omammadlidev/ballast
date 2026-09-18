"""End-to-end closed-loop flight.

These are the assertions that would actually catch a regression in the vehicle:
not 'the code runs', but 'the thing still flies, to this accuracy, inside these
actuator limits'.
"""
import numpy as np
import pytest

from ballast import (FlightController, Simulator, Vehicle, Wind, initial_state,
                     math3d as m3, trajectories)

DT = 2e-3


def fly(duration, vehicle=None, **kwargs):
    veh = vehicle or Vehicle()
    wind = kwargs.pop("wind", None)
    return Simulator(veh, dt=DT, wind=wind).run(FlightController(), duration, **kwargs)


def test_hover_holds_station():
    tel = fly(30.0, x0=initial_state(Vehicle(), position=(0.4, -0.3, 0.2)))
    assert tel.diverged_at is None
    assert tel.rms_error(last_seconds=5.0) < 0.02


def test_step_response_settles():
    tel = fly(25.0, trajectory=trajectories.step((1.5, -1.0, 2.0)))
    assert tel.diverged_at is None
    assert tel.rms_error(last_seconds=5.0) < 0.05
    assert tel.max_tilt_deg < 26.0                # inside the commanded cone


def test_figure_eight_tracks():
    tel = fly(60.0, trajectory=trajectories.figure_eight())
    assert tel.diverged_at is None
    assert tel.rms_error(last_seconds=20.0) < 0.10


def test_helix_climb_tracks():
    tel = fly(30.0, trajectory=trajectories.helix())
    assert tel.diverged_at is None
    assert tel.rms_error(last_seconds=10.0) < 0.10


def test_recovers_from_a_large_upset():
    veh = Vehicle()
    x0 = initial_state(
        veh, position=(1.2, -0.8, 0.6), velocity=(1.0, 0.6, -0.4),
        quat=m3.quat_from_axis_angle(np.array([1.0, 0.4, 0.0]), np.radians(25)))
    tel = fly(30.0, vehicle=veh, x0=x0)
    assert tel.diverged_at is None
    assert tel.rms_error(last_seconds=5.0) < 0.02


@pytest.mark.parametrize("speed,limit", [(2.0, 0.06), (4.0, 0.20), (7.0, 0.55)])
def test_holds_station_in_wind(speed, limit):
    tel = fly(45.0, wind=Wind((speed, 0.5 * speed, 0.0), sigma=0.3 * speed,
                              tau=1.0, seed=7))
    assert tel.diverged_at is None
    assert tel.rms_error(last_seconds=15.0) < limit


def test_survives_the_edge_of_the_envelope():
    """At 10 m/s the slider is saturating; it must degrade, not depart."""
    tel = fly(45.0, wind=Wind((10.0, 5.0, 0.0), sigma=3.0, tau=1.0, seed=3))
    assert tel.diverged_at is None
    assert tel.max_error(last_seconds=20.0) < 3.0


def test_slider_never_leaves_its_rails():
    veh = Vehicle()
    tel = fly(30.0, vehicle=veh, trajectory=trajectories.figure_eight())
    assert np.all(np.abs(tel.slider) <= veh.slider.travel + 1e-9)


def test_thrust_command_stays_inside_the_rotor_limits():
    veh = Vehicle()
    tel = fly(25.0, vehicle=veh, trajectory=trajectories.step((2.0, 2.0, 3.0)))
    assert tel.thrust_cmd.min() >= 0.0
    assert tel.thrust_cmd.max() <= veh.rotor.thrust_max + 1e-9


def test_heading_is_uncontrolled_but_slow():
    """No yaw actuator exists, so heading drifts -- it just must not run away."""
    tel = fly(60.0)
    drift_rate = abs(tel.yaw_drift_deg) / 60.0
    assert 0.1 < drift_rate < 10.0


def test_heading_drift_does_not_disturb_position():
    """The controller frames its attitude command around the current heading,
    so a rotating airframe should cost nothing in position accuracy."""
    tel = fly(60.0, x0=initial_state(Vehicle(), position=(0.3, 0.3, 0.3)))
    assert abs(tel.yaw_drift_deg) > 20.0          # it really did rotate
    assert tel.rms_error(last_seconds=20.0) < 0.02


def test_perfectly_trimmed_vanes_hold_heading():
    veh = Vehicle()
    veh.rotor.vane_ratio = 1.0
    tel = fly(30.0, vehicle=veh)
    assert abs(tel.yaw_drift_deg) < 1.0


def test_a_weaker_slider_degrades_gracefully():
    veh = Vehicle()
    veh.slider.travel = 0.025                     # less than half the authority
    tel = fly(30.0, vehicle=veh, trajectory=trajectories.step((1.0, 0.0, 1.0)))
    assert tel.diverged_at is None
    assert tel.rms_error(last_seconds=5.0) < 0.20


# --- the rail-height result, protected against regression -------------------

def _rails_at(z):
    veh = Vehicle()
    veh.slider.plane_z = z
    return veh


def test_rails_below_com_make_the_reaction_reinforce_steering():
    from ballast.dynamics import Control, derivative, initial_state
    veh = _rails_at(-0.030)
    dx = derivative(veh, initial_state(veh), Control(veh.hover_thrust, 0.04, 0.0))
    assert dx[11] > 0                        # pitches the way the CoM shift will


def test_rails_above_com_make_the_plant_non_minimum_phase():
    from ballast.dynamics import Control, derivative, initial_state
    veh = _rails_at(+0.030)
    dx = derivative(veh, initial_state(veh), Control(veh.hover_thrust, 0.04, 0.0))
    assert dx[11] < 0                        # pitches the wrong way first


def test_rails_at_com_height_produce_no_reaction():
    from ballast.dynamics import Control, derivative, initial_state
    veh = _rails_at(0.0)
    dx = derivative(veh, initial_state(veh), Control(veh.hover_thrust, 0.04, 0.0))
    assert abs(dx[11]) < 1e-12


def test_rail_height_dominates_closed_loop_accuracy():
    """The headline design result: same vehicle, rails flipped, 50x worse."""
    traj = trajectories.step((1.5, -1.0, 2.0))
    below = fly(25.0, vehicle=_rails_at(-0.030), trajectory=traj)
    above = fly(25.0, vehicle=_rails_at(+0.030), trajectory=traj)
    assert below.rms_error(5.0) < 0.01
    assert above.rms_error(5.0) > 0.20
    assert above.max_tilt_deg > 3 * below.max_tilt_deg
