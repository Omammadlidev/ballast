import numpy as np
import pytest

from ballast import Vehicle, math3d as m3
from ballast.control.allocation import achievable_torque, allocate
from ballast.control.attitude import AttitudeController, desired_rotation
from ballast.control.position import PositionController, PositionGains


@pytest.fixture
def veh():
    return Vehicle()


def test_allocation_inverts_the_torque_map_exactly(veh):
    f = veh.hover_thrust
    rng = np.random.default_rng(0)
    for _ in range(200):
        tau = rng.uniform(-0.05, 0.05, 2)
        a = allocate(veh, tau, f)
        if a.saturated:
            continue
        cx, cy = a.com_cmd
        assert np.allclose(np.array([-cy * f, cx * f]), tau, atol=1e-14)


def test_allocation_saturation_preserves_direction(veh):
    f = veh.hover_thrust
    tau = np.array([0.3, -0.4])
    a = allocate(veh, tau, f)
    assert a.saturated
    assert np.allclose(a.torque_cmd / np.linalg.norm(a.torque_cmd),
                       tau / np.linalg.norm(tau), atol=1e-12)
    assert np.linalg.norm(a.torque_cmd) == pytest.approx(achievable_torque(veh, f),
                                                         rel=1e-12)


def test_allocation_stays_inside_the_rails(veh):
    rng = np.random.default_rng(3)
    for _ in range(500):
        a = allocate(veh, rng.uniform(-2, 2, 2), rng.uniform(1.0, 14.0))
        assert np.linalg.norm(a.slider_cmd) <= veh.slider.travel + 1e-12


def test_authority_scales_with_thrust(veh):
    assert achievable_torque(veh, 2 * veh.hover_thrust) == \
        pytest.approx(2 * achievable_torque(veh, veh.hover_thrust))


def test_desired_rotation_delivers_the_requested_thrust_axis():
    rng = np.random.default_rng(5)
    for _ in range(200):
        b3 = rng.standard_normal(3); b3 /= np.linalg.norm(b3)
        R = m3.quat_to_rot(m3.quat_normalize(rng.standard_normal(4)))
        Rd = desired_rotation(b3, R)
        assert np.allclose(Rd[:, 2], b3, atol=1e-10)
        assert np.allclose(Rd @ Rd.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(Rd), 1.0, atol=1e-10)


def test_desired_rotation_handles_the_degenerate_heading():
    R = np.eye(3)
    Rd = desired_rotation(np.array([1.0, 0.0, 0.0]), R)
    assert np.allclose(Rd[:, 2], [1.0, 0.0, 0.0], atol=1e-12)
    assert np.isclose(np.linalg.det(Rd), 1.0, atol=1e-12)


def test_attitude_error_is_zero_when_aligned():
    ctl = AttitudeController()
    tau = ctl(np.eye(3), np.zeros(3), np.array([0.0, 0.0, 1.0]))
    assert np.allclose(tau, 0.0, atol=1e-12)


def test_attitude_controller_pushes_the_right_way():
    """Tilted toward +x, the controller must command a nose-down (-y) torque."""
    ctl = AttitudeController()
    R = m3.rot_y(np.radians(10))                  # body z leans toward +x
    tau = ctl(R, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    assert tau[1] < 0


def test_attitude_controller_never_asks_for_yaw():
    """The returned torque has two components; yaw is structurally absent."""
    ctl = AttitudeController()
    R = m3.rot_z(np.radians(80)) @ m3.rot_x(np.radians(15))
    tau = ctl(R, np.array([0.1, -0.2, 3.0]), np.array([0.0, 0.0, 1.0]))
    assert len(tau) == 2


def test_position_controller_holds_gravity_at_rest():
    ctl = PositionController()
    cmd = ctl(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), 1e-3)
    assert np.allclose(cmd.b3_des, [0.0, 0.0, 1.0], atol=1e-12)
    assert cmd.accel[2] == pytest.approx(9.80665, abs=1e-9)


def test_position_controller_leans_toward_the_target():
    ctl = PositionController()
    cmd = ctl(np.zeros(3), np.zeros(3), np.array([5.0, 0.0, 0.0]),
              np.zeros(3), np.zeros(3), 1e-3)
    assert cmd.b3_des[0] > 0                      # lean toward +x to fly there


def test_tilt_cone_is_enforced():
    ctl = PositionController(PositionGains(tilt_limit_deg=15.0))
    cmd = ctl(np.zeros(3), np.zeros(3), np.array([500.0, 0.0, 0.0]),
              np.zeros(3), np.zeros(3), 1e-3)
    assert cmd.tilt_clipped
    assert np.degrees(np.arccos(cmd.b3_des[2])) == pytest.approx(15.0, abs=1e-9)


def test_integrator_is_bounded():
    ctl = PositionController(PositionGains(integral_limit=np.array([0.5, 0.5, 0.5])))
    for _ in range(20000):
        ctl(np.array([10.0, 0.0, 0.0]), np.zeros(3), np.zeros(3),
            np.zeros(3), np.zeros(3), 1e-3)
    assert np.all(np.abs(ctl.integral) <= 0.5 + 1e-12)
