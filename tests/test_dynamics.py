import numpy as np
import pytest

from ballast import Vehicle, math3d as m3
from ballast.dynamics import (Control, derivative, initial_state,
                              origin_position, origin_velocity, unpack)
from ballast.wind import Wind


@pytest.fixture
def veh():
    return Vehicle()


def test_hover_has_zero_translational_acceleration(veh):
    x = initial_state(veh)
    dx = derivative(veh, x, Control(veh.hover_thrust, 0.0, 0.0))
    assert np.allclose(dx[3:6], 0.0, atol=1e-12)


def test_hover_has_zero_roll_and_pitch_acceleration(veh):
    x = initial_state(veh)
    dx = derivative(veh, x, Control(veh.hover_thrust, 0.0, 0.0))
    assert np.allclose(dx[10:12], 0.0, atol=1e-12)


def test_uncancelled_rotor_torque_shows_up_only_in_yaw(veh):
    """The architecture's defining limitation, asserted rather than assumed."""
    x = initial_state(veh)
    dx = derivative(veh, x, Control(veh.hover_thrust, 0.0, 0.0))
    assert abs(dx[12]) > 1e-3
    assert np.sign(dx[12]) == -np.sign(veh.rotor.spin_dir)


def test_slider_offset_produces_the_expected_torque_sign(veh):
    """+x slider shifts the CoM +x, which pitches the nose up (+y torque)."""
    x = initial_state(veh, slider=(0.05, 0.0))
    _, d = derivative(veh, x, Control(veh.hover_thrust, 0.05, 0.0),
                      with_diagnostics=True)
    assert d.torque_body[1] > 0
    assert abs(d.torque_body[0]) < 1e-9

    y = initial_state(veh, slider=(0.0, 0.05))
    _, e = derivative(veh, y, Control(veh.hover_thrust, 0.0, 0.05),
                      with_diagnostics=True)
    assert e.torque_body[0] < 0
    assert abs(e.torque_body[1]) < 1e-9


def test_torque_magnitude_matches_the_analytic_lever(veh):
    s_x = 0.05
    x = initial_state(veh, slider=(s_x, 0.0))
    _, d = derivative(veh, x, Control(veh.hover_thrust, s_x, 0.0),
                      with_diagnostics=True)
    expected = d.thrust * (veh.slider.mass / veh.total_mass) * s_x
    assert d.torque_body[1] == pytest.approx(expected, rel=1e-12)


def test_rotor_position_height_does_not_change_torque(veh):
    """The vertical lever multiplies a force parallel to it, so it cancels."""
    a = Vehicle()
    b = Vehicle()
    b.rotor.position = np.array([0.0, 0.0, 0.40])
    xa = initial_state(a, slider=(0.04, 0.0))
    xb = initial_state(b, slider=(0.04, 0.0))
    _, da = derivative(a, xa, Control(a.hover_thrust, 0.04, 0.0), with_diagnostics=True)
    _, db = derivative(b, xb, Control(b.hover_thrust, 0.04, 0.0), with_diagnostics=True)
    assert np.allclose(da.torque_body[:2], db.torque_body[:2], atol=1e-14)


def test_origin_offset_tracks_the_slider(veh):
    x = initial_state(veh, slider=(0.06, 0.0))
    p_origin = origin_position(veh, x)
    expected = -(veh.slider.mass / veh.total_mass) * 0.06
    assert p_origin[0] == pytest.approx(expected, abs=1e-12)


def test_origin_velocity_includes_slider_coupling(veh):
    x = initial_state(veh)
    x[16] = 0.5                                   # slider moving in +x
    v = origin_velocity(veh, x)
    assert v[0] == pytest.approx(-(veh.slider.mass / veh.total_mass) * 0.5, abs=1e-12)


def test_drag_opposes_relative_airflow(veh):
    x = initial_state(veh, velocity=(3.0, 0.0, 0.0))
    _, d = derivative(veh, x, Control(veh.hover_thrust, 0.0, 0.0),
                      with_diagnostics=True)
    assert d.drag_body[0] < 0
    x2 = initial_state(veh, velocity=(3.0, 0.0, 0.0))
    _, d2 = derivative(veh, x2, Control(veh.hover_thrust, 0.0, 0.0),
                       wind=np.array([3.0, 0.0, 0.0]), with_diagnostics=True)
    assert np.allclose(d2.drag_body, 0.0, atol=1e-12)


def test_free_fall_with_no_thrust(veh):
    x = initial_state(veh, trimmed=False)
    dx = derivative(veh, x, Control(0.0, 0.0, 0.0))
    assert dx[5] == pytest.approx(-veh.gravity, abs=1e-12)


def test_slider_respects_commanded_travel_limit(veh):
    x = initial_state(veh)
    dx = derivative(veh, x, Control(veh.hover_thrust, 10.0, 0.0))
    # commanded far outside the rails; acceleration is finite and bounded
    assert np.isfinite(dx[16])
    assert dx[16] <= veh.slider.omega_n ** 2 * veh.slider.travel + 1e-9


def test_derivative_is_finite_over_random_states(veh):
    rng = np.random.default_rng(7)
    for _ in range(300):
        x = initial_state(
            veh,
            position=rng.uniform(-5, 5, 3), velocity=rng.uniform(-8, 8, 3),
            quat=m3.quat_normalize(rng.standard_normal(4)),
            omega=rng.uniform(-6, 6, 3),
            slider=rng.uniform(-0.06, 0.06, 2))
        u = Control(rng.uniform(0, 14), rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1))
        dx = derivative(veh, x, u, wind=rng.uniform(-10, 10, 3))
        assert np.all(np.isfinite(dx))


def test_wind_is_reproducible_from_a_seed():
    a = Wind((1.0, 0.0, 0.0), sigma=2.0, tau=0.5, seed=42)
    b = Wind((1.0, 0.0, 0.0), sigma=2.0, tau=0.5, seed=42)
    sa = np.array([a.step(1e-3) for _ in range(5000)])
    sb = np.array([b.step(1e-3) for _ in range(5000)])
    assert np.allclose(sa, sb)


def test_wind_reset_clears_the_gust_state():
    w = Wind((1.0, 0.0, 0.0), sigma=3.0, tau=0.5, seed=1)
    for _ in range(1000):
        w.step(1e-3)
    w.reset()
    assert np.allclose(w.value, [1.0, 0.0, 0.0])


def test_gusts_match_the_ou_stationary_statistics():
    """Long enough to average out sampling noise: N_eff = T/tau = 800."""
    w = Wind((1.0, 0.0, 0.0), sigma=2.0, tau=0.5, seed=42)
    s = np.array([w.step(1e-3) for _ in range(400_000)])
    assert abs(s[:, 1].std() - 2.0) < 0.2             # stationary std -> sigma
    assert abs(s[:, 2].std() - 2.0) < 0.2
    assert abs(s[:, 0].mean() - 1.0) < 0.2            # steady component preserved


def test_gust_correlation_time_matches_tau():
    w = Wind(sigma=1.0, tau=0.5, seed=1)
    s = np.array([w.step(1e-3) for _ in range(400_000)])[:, 0]
    s = s - s.mean()
    ac = np.correlate(s, s, "full")[len(s) - 1:]
    ac /= ac[0]
    tau_measured = np.argmax(ac < np.exp(-1.0)) * 1e-3
    assert 0.35 < tau_measured < 0.65


def test_still_air_produces_no_gusts():
    w = Wind((2.0, 1.0, 0.0))
    assert np.allclose(w.step(1e-3), [2.0, 1.0, 0.0])
