import numpy as np
from skillfusion.sim.softcam import Box, render, quat_to_R

def look_down_R():
    # ROS camera looking straight down: forward = -z_base, right = +x, down = +y  (right x down = forward?)
    right = np.array([1.0, 0, 0]); fwd = np.array([0, 0, -1.0]); down = np.cross(fwd, right)
    return np.stack([right, down, fwd], 1)

def test_center_pixel_sees_box_and_occlusion():
    red = Box(np.array([0, 0, 0.025]), np.eye(3), np.array([0.025] * 3), np.array([0.9, 0.05, 0.05]))
    img = render([0, 0, 0.5], look_down_R(), [red], floor_z=-0.05)
    c = img[120, 160].astype(int)
    assert c[0] > 150 and c[1] < 60 and c[2] < 60, c            # red on top
    assert abs(int(img[5, 5].mean()) - 100) < 60                # floor at the corner, not red
    # a big blue box above the camera view line occludes the red one
    blue = Box(np.array([0, 0, 0.2]), np.eye(3), np.array([0.2, 0.2, 0.02]), np.array([0.05, 0.05, 0.9]))
    img2 = render([0, 0, 0.5], look_down_R(), [red, blue], floor_z=-0.05)
    c2 = img2[120, 160].astype(int)
    assert c2[2] > 150 and c2[0] < 60, c2

def test_camera_inside_box_or_behind_ignored():
    behind = Box(np.array([0, 0, 1.0]), np.eye(3), np.array([0.1] * 3), np.array([0, 1.0, 0]))
    img = render([0, 0, 0.5], look_down_R(), [behind], floor_z=-0.05)   # box is behind the camera
    assert img[120, 160, 1] < 200 or img[120,160,0] > 0
    assert not ((img[:, :, 1] > 200) & (img[:, :, 0] < 30)).any()

def test_quat_identity_and_z90():
    assert np.allclose(quat_to_R([0, 0, 0, 1]), np.eye(3))
    R = quat_to_R([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-9)
