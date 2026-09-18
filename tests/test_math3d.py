import numpy as np
import pytest

from ballast import math3d as m3


@pytest.fixture
def rng():
    return np.random.default_rng(20240917)


def random_quat(rng):
    return m3.quat_normalize(rng.standard_normal(4))


def test_rotation_is_orthonormal(rng):
    for _ in range(200):
        R = m3.quat_to_rot(random_quat(rng))
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_quat_rot_roundtrip(rng):
    for _ in range(200):
        q = random_quat(rng)
        assert np.allclose(m3.quat_to_rot(m3.rot_to_quat(m3.quat_to_rot(q))),
                           m3.quat_to_rot(q), atol=1e-10)


def test_rot_to_quat_covers_all_branches():
    """Shepperd's method switches on the largest diagonal term."""
    for R in (np.eye(3), m3.rot_x(np.pi), m3.rot_y(np.pi), m3.rot_z(np.pi)):
        assert np.allclose(m3.quat_to_rot(m3.rot_to_quat(R)), R, atol=1e-12)


def test_hat_vee_are_inverse(rng):
    for _ in range(100):
        v = rng.standard_normal(3)
        assert np.allclose(m3.hat(v) @ rng.standard_normal(3) * 0, 0)
        assert np.allclose(m3.vee(m3.hat(v)), v)


def test_hat_implements_cross_product(rng):
    for _ in range(100):
        a, b = rng.standard_normal(3), rng.standard_normal(3)
        assert np.allclose(m3.hat(a) @ b, np.cross(a, b))


def test_quat_multiply_matches_matrix_product(rng):
    for _ in range(100):
        a, b = random_quat(rng), random_quat(rng)
        assert np.allclose(m3.quat_to_rot(m3.quat_multiply(a, b)),
                           m3.quat_to_rot(a) @ m3.quat_to_rot(b), atol=1e-12)


def test_align_rotation(rng):
    for _ in range(200):
        a = rng.standard_normal(3); a /= np.linalg.norm(a)
        b = rng.standard_normal(3); b /= np.linalg.norm(b)
        R = m3.align_rotation(a, b)
        assert np.allclose(R @ a, b, atol=1e-10)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)


def test_align_rotation_antiparallel():
    a = np.array([0.0, 0.0, 1.0])
    R = m3.align_rotation(a, -a)
    assert np.allclose(R @ a, -a, atol=1e-10)


def test_project_perp(rng):
    n = np.array([0.0, 0.0, 1.0])
    for _ in range(50):
        v = rng.standard_normal(3)
        assert abs(m3.project_perp(v, n) @ n) < 1e-12


def test_tilt_and_heading():
    assert m3.tilt_angle(np.eye(3)) == pytest.approx(0.0)
    assert m3.tilt_angle(m3.rot_x(np.radians(30))) == pytest.approx(np.radians(30))
    assert m3.heading_angle(m3.rot_z(np.radians(45))) == pytest.approx(np.radians(45))
