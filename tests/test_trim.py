"""The linearisation is checked against modes whose values are known in closed
form, which is a much stronger test than 'the eigenvalues look plausible'."""
import numpy as np
import pytest

from ballast import Vehicle
from ballast.dynamics import derivative
from ballast.trim import NE, describe_modes, hover_trim, linearize, modes


@pytest.fixture
def veh():
    return Vehicle()


def test_hover_trim_balances_forces(veh):
    tp = hover_trim(veh)
    dx = derivative(veh, tp.state, tp.control)
    assert np.allclose(dx[0:6], 0.0, atol=1e-12)      # position and velocity
    assert np.allclose(dx[10:12], 0.0, atol=1e-12)    # roll and pitch


def test_hover_trim_residual_is_pure_yaw(veh):
    tp = hover_trim(veh)
    dx = derivative(veh, tp.state, tp.control)
    assert tp.residual == pytest.approx(abs(dx[12]), rel=1e-9)


def test_perfect_vanes_give_an_exact_trim():
    veh = Vehicle()
    veh.rotor.vane_ratio = 1.0
    assert hover_trim(veh).residual < 1e-12


def test_jacobian_shapes(veh):
    A, B = linearize(veh)
    assert A.shape == (NE, NE)
    assert B.shape == (NE, 3)
    assert np.all(np.isfinite(A)) and np.all(np.isfinite(B))


def test_slider_servo_mode_matches_its_design(veh):
    """Should appear at -zeta*wn +- j*wn*sqrt(1-zeta^2)."""
    A, _ = linearize(veh)
    sp = veh.slider
    want = complex(-sp.zeta * sp.omega_n, sp.omega_n * np.sqrt(1 - sp.zeta ** 2))
    evs = np.linalg.eigvals(A)
    assert np.min(np.abs(evs - want)) < 1e-4


def test_rotor_spool_mode_matches_its_time_constant(veh):
    A, _ = linearize(veh)
    want = -1.0 / veh.rotor.tau_spool
    evs = np.linalg.eigvals(A)
    assert np.min(np.abs(evs - want)) < 1e-4


def test_rate_damping_modes_match_k_over_J(veh):
    A, _ = linearize(veh)
    J = veh.inertia_about_com(veh.slider.position(0.0, 0.0))
    evs = np.linalg.eigvals(A)
    for axis in (0, 2):
        want = -veh.aero.rate_damp_lin[axis] / J[axis, axis]
        assert np.min(np.abs(evs - want)) < 1e-3


def test_open_loop_has_no_genuinely_divergent_mode(veh):
    """Marginally stable, as every thrust-vectored hovering vehicle is."""
    A, _ = linearize(veh)
    tol = 1e-4 * max(float(np.linalg.norm(A, 2)), 1.0)
    assert np.max(np.linalg.eigvals(A).real) < tol


def test_rigid_body_integrators_are_present(veh):
    A, _ = linearize(veh)
    tol = 1e-4 * max(float(np.linalg.norm(A, 2)), 1.0)
    n_zero = int(np.sum(np.abs(np.linalg.eigvals(A)) < tol))
    # 3 position + 2 horizontal velocity + 2 horizontal attitude + yaw + yaw rate
    assert n_zero >= 8


def test_thrust_command_acts_only_through_the_rotor_state(veh):
    """There is no direct feed-through: the spool lag stands in the way."""
    _, B = linearize(veh)
    assert abs(B[5, 0]) < 1e-9                        # no instant a_z response


def test_thrust_command_drives_the_spool_at_the_analytic_rate(veh):
    r = veh.rotor
    _, B = linearize(veh)
    # d/df [ (sqrt(f/kf) - omega)/tau ] = 1 / (2 tau sqrt(kf f))
    want = 1.0 / (2.0 * r.tau_spool * np.sqrt(r.kf * veh.hover_thrust))
    assert B[12, 0] == pytest.approx(want, rel=1e-4)


def test_slider_command_drives_the_servo_at_its_stiffness(veh):
    _, B = linearize(veh)
    assert B[15, 1] == pytest.approx(veh.slider.omega_n ** 2, rel=1e-6)
    assert B[16, 2] == pytest.approx(veh.slider.omega_n ** 2, rel=1e-6)


def test_slider_inputs_cannot_touch_yaw(veh):
    """Structural: no slider position produces a moment about the thrust axis."""
    _, B = linearize(veh)
    assert abs(B[11, 1]) < 1e-9
    assert abs(B[11, 2]) < 1e-9


def test_describe_modes_renders(veh):
    A, _ = linearize(veh)
    text = describe_modes(modes(A))
    assert "eigenvalue" in text and "rigid-body integrators" in text
