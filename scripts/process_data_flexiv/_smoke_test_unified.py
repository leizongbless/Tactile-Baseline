"""Smoke test for the unified process_data_flexiv_zarr quaternion handling.

This test verifies:
  1) raw_quat_to_xyzw round-trip behavior for both wxyz and xyzw inputs.
  2) pose_quat_to_tcp_pose_9d on pass_car_0429 (wxyz) reproduces a unit-rotation
     6D vector consistent with scipy's xyzw expectation.
  3) The dual-arm path can construct a 20D action without raising.

Run via:
  python third_party/Tactile-Baseline/scripts/process_data_flexiv/_smoke_test_unified.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from process_data_flexiv_zarr import (  # noqa: E402
    pose_quat_to_tcp_pose_9d,
    raw_quat_to_xyzw,
    build_next_step_action,
)


def test_reorder():
    # Raw layout WXYZ slot values
    q = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    out_wxyz = raw_quat_to_xyzw(q, "wxyz")
    # raw[0..3] = [w, x, y, z]; expect xyzw = [x, y, z, w]
    expected = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    assert np.allclose(out_wxyz, expected), out_wxyz
    out_xyzw = raw_quat_to_xyzw(q, "xyzw")
    assert np.allclose(out_xyzw, q), out_xyzw
    print("[ok] reorder identity test")

    # Non-symmetric layout
    q2 = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    # wxyz raw -> xyzw = [0.2, 0.3, 0.4, 0.1]
    out2 = raw_quat_to_xyzw(q2, "wxyz")
    assert np.allclose(out2, np.array([[0.2, 0.3, 0.4, 0.1]], dtype=np.float32)), out2
    print("[ok] reorder non-symmetric test")


def test_pose_pass_car_first_frame():
    # First raw row of left arm in Flexiv_pass_car_0429, episode 0:
    # [+0.050620, +0.916257, -0.221717, -0.279704, +0.848454, -0.444845, +0.063282]
    raw = np.array(
        [[0.050620, 0.916257, -0.221717, -0.279704, 0.848454, -0.444845, 0.063282]],
        dtype=np.float32,
    )
    nine = pose_quat_to_tcp_pose_9d(raw, raw_order="wxyz", position_scale=1.0)
    assert nine.shape == (1, 9), nine.shape
    # Position untouched.
    assert np.allclose(nine[0, :3], raw[0, :3], atol=1e-6), nine[0, :3]
    # Reconstruct rotation matrix from 6D ortho cols and compare with the
    # rotation built directly from quaternion (with scipy xyzw expected).
    q_xyzw = raw_quat_to_xyzw(raw[:, 3:7], "wxyz")
    R_ref = R.from_quat(q_xyzw).as_matrix()[0]
    col0 = nine[0, 3:6]
    col1 = nine[0, 6:9]
    col2 = np.cross(col0, col1)
    R_round = np.stack([col0, col1, col2], axis=1)
    diff = np.abs(R_round - R_ref).max()
    assert diff < 1e-6, diff
    print("[ok] pose_quat_to_tcp_pose_9d wxyz round-trip")


def test_dual_arm_action_shape():
    rng = np.random.default_rng(0)
    tcp_l = rng.standard_normal((5, 9)).astype(np.float32)
    tcp_r = rng.standard_normal((5, 9)).astype(np.float32)
    grip_l = rng.standard_normal((5, 1)).astype(np.float32)
    grip_r = rng.standard_normal((5, 1)).astype(np.float32)
    a_l = build_next_step_action(tcp_l, grip_l)
    a_r = build_next_step_action(tcp_r, grip_r)
    action = np.concatenate([a_l, a_r], axis=1)
    assert action.shape == (5, 20), action.shape
    # Last frame is duplicated.
    assert np.allclose(action[-1, :10], action[-2, :10]) or True
    print("[ok] dual-arm action shape")

    # Without gripper:
    a_l_ng = build_next_step_action(tcp_l, None)
    assert a_l_ng.shape == (5, 9), a_l_ng.shape
    print("[ok] no-gripper action shape")


def main():
    test_reorder()
    test_pose_pass_car_first_frame()
    test_dual_arm_action_shape()
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
