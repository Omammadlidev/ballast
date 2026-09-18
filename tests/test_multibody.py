"""The multibody model is the part that is easiest to get subtly wrong, so it is
checked against independent constructions rather than against itself."""
import numpy as np
import pytest

from ballast import Vehicle, math3d as m3
from ballast.multibody import angular_acceleration, inertia_terms


@pytest.fixture
def veh():
    return Vehicle()


def slider_at(veh, x, y):
    return veh.slider.position(x, y)


def test_inertia_matches_explicit_two_body_sum(veh):
    """J_G from the reduced-mass identity must equal the brute-force sum."""
    rng = np.random.default_rng(1)
    mb, ms = veh.shell_mass, veh.slider.mass
    for _ in range(200):
        s = slider_at(veh, rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06))
        c = veh.com_body(s)
        rb, rs = -c, s - c
        expected = (veh.shell_inertia
                    + mb * ((rb @ rb) * np.eye(3) - np.outer(rb, rb))
                    + ms * ((rs @ rs) * np.eye(3) - np.outer(rs, rs)))
        J, _, _ = inertia_terms(veh, s, np.zeros(3))
        assert np.allclose(J, expected, atol=1e-15)


def test_inertia_is_symmetric_positive_definite(veh):
    s = slider_at(veh, 0.06, -0.06)
    J, _, _ = inertia_terms(veh, s, np.zeros(3))
    assert np.allclose(J, J.T)
    assert np.all(np.linalg.eigvalsh(J) > 0)


def test_inertia_rate_matches_finite_difference(veh):
    s = slider_at(veh, 0.02, -0.01)
    s_dot = np.array([0.3, -0.2, 0.0])
    h = 1e-7
    Jp, _, _ = inertia_terms(veh, s + h * s_dot, s_dot)
    Jm, _, _ = inertia_terms(veh, s - h * s_dot, s_dot)
    _, J_dot, _ = inertia_terms(veh, s, s_dot)
    assert np.allclose((Jp - Jm) / (2 * h), J_dot, atol=1e-9)


def test_internal_momentum_vanishes_for_radial_motion(veh):
    """Sliding straight through the body origin stores no angular momentum."""
    s = np.array([0.03, 0.0, 0.0])
    _, _, h = inertia_terms(veh, s, np.array([0.5, 0.0, 0.0]))
    assert np.allclose(h, 0.0)


def test_torque_free_angular_momentum_is_conserved(veh):
    """With no applied torque and a frozen slider, world H_G must be constant."""
    s = slider_at(veh, 0.03, 0.0)
    zero = np.zeros(3)

    def f(y):
        w, q = y[0:3], y[3:7]
        r = angular_acceleration(veh, w, s, zero, zero, zero)
        return np.concatenate([r.omega_dot, m3.quat_derivative(q, w)])

    dt = 2e-4
    y = np.concatenate([[0.7, -0.4, 1.1], [1.0, 0.0, 0.0, 0.0]])
    H0 = None
    for _ in range(int(2.0 / dt)):
        k1 = f(y); k2 = f(y + dt / 2 * k1); k3 = f(y + dt / 2 * k2); k4 = f(y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        y[3:7] = m3.quat_normalize(y[3:7])
        r = angular_acceleration(veh, y[0:3], s, zero, zero, zero)
        H = m3.quat_to_rot(y[3:7]) @ r.angular_momentum
        if H0 is None:
            H0 = H.copy()
    assert np.linalg.norm(H - H0) / np.linalg.norm(H0) < 1e-10


def test_torque_free_kinetic_energy_is_conserved(veh):
    """A frozen slider does no work, so rotational kinetic energy is constant."""
    s = slider_at(veh, -0.04, 0.02)
    zero = np.zeros(3)
    w = np.array([1.2, 0.5, -0.8])
    J, _, _ = inertia_terms(veh, s, zero)
    E0 = 0.5 * w @ J @ w
    dt = 2e-4
    for _ in range(int(2.0 / dt)):
        k1 = angular_acceleration(veh, w, s, zero, zero, zero).omega_dot
        k2 = angular_acceleration(veh, w + dt / 2 * k1, s, zero, zero, zero).omega_dot
        k3 = angular_acceleration(veh, w + dt / 2 * k2, s, zero, zero, zero).omega_dot
        k4 = angular_acceleration(veh, w + dt * k3, s, zero, zero, zero).omega_dot
        w = w + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    assert abs(0.5 * w @ J @ w - E0) / E0 < 1e-10


def test_reduces_to_rigid_body_when_slider_is_centred(veh):
    """Slider on the body z-axis with zero rate == plain Euler equations."""
    s = np.array([0.0, 0.0, veh.slider.plane_z])
    zero = np.zeros(3)
    w = np.array([0.3, -0.5, 0.9])
    tau = np.array([0.01, -0.02, 0.005])
    r = angular_acceleration(veh, w, s, zero, zero, tau)
    J, _, _ = inertia_terms(veh, s, zero)
    expected = np.linalg.solve(J, tau - np.cross(w, J @ w))
    assert np.allclose(r.omega_dot, expected, atol=1e-14)


def test_slider_acceleration_produces_reaction_torque(veh):
    """Accelerating the slider must kick the airframe back."""
    s = slider_at(veh, 0.04, 0.0)
    zero = np.zeros(3)
    quiet = angular_acceleration(veh, zero, s, zero, zero, zero)
    kicked = angular_acceleration(veh, zero, s, zero, np.array([0.0, 8.0, 0.0]), zero)
    assert not np.allclose(quiet.omega_dot, kicked.omega_dot)
    # s x s_ddot points along -z for s along +x and s_ddot along +y
    assert kicked.internal_torque[2] > 0
