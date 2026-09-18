"""Minimal SO(3) / quaternion utilities.

Quaternions are stored as ``[w, x, y, z]`` and always represent a rotation that
maps **body** coordinates into **world** coordinates::

    v_world = R(q) @ v_body
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "hat", "vee", "quat_normalize", "quat_multiply", "quat_to_rot",
    "rot_to_quat", "quat_from_axis_angle", "quat_derivative",
    "rot_x", "rot_y", "rot_z", "tilt_angle", "heading_angle",
    "align_rotation", "project_perp",
]


def hat(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix with ``hat(a) @ b == cross(a, b)``."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def vee(S: np.ndarray) -> np.ndarray:
    """Inverse of :func:`hat`, taking the skew part of ``S``."""
    return np.array([S[2, 1] - S[1, 2], S[0, 2] - S[2, 0], S[1, 0] - S[0, 1]]) * 0.5


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = q / n
    return -q if q[0] < 0.0 else q


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Shepperd's method: pick the largest denominator for numerical safety."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return quat_normalize(q)


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([[np.cos(0.5 * angle)], np.sin(0.5 * angle) * (axis / n)])


def quat_derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """``qdot`` for a body rate expressed in body coordinates."""
    return 0.5 * quat_multiply(q, np.concatenate([[0.0], omega_body]))


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def tilt_angle(R: np.ndarray) -> float:
    """Angle between the body z-axis and world up, in radians."""
    return float(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))


def heading_angle(R: np.ndarray) -> float:
    """Yaw of the body x-axis projected onto the world horizontal plane."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def align_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest rotation matrix taking unit vector ``a`` onto unit vector ``b``."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1.0 + 1e-9:                       # anti-parallel: any perpendicular axis
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, axis)
        return quat_to_rot(quat_from_axis_angle(axis, np.pi))
    K = hat(v)
    return np.eye(3) + K + K @ K / (1.0 + c)


def project_perp(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Component of ``v`` perpendicular to unit vector ``n``."""
    return v - float(np.dot(v, n)) * n
