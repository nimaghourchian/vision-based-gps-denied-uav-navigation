import glob
import math
import os
import re
import time

import cv2
import numpy as np
import pymap3d as pm

from config import config_loader
from data.shared_data import SharedArrayWithLock


CONFIG = config_loader.load_config()


def load_sorted_jpgs(folder):
    """Return JPEG frame paths sorted by the first integer in each filename."""
    paths = glob.glob(os.path.join(folder, "*.jpg"))
    paths += glob.glob(os.path.join(folder, "*.jpeg"))

    def key_fn(path):
        match = re.search(r"(\d+)", os.path.basename(path))
        return int(match.group(1)) if match else os.path.basename(path)

    return sorted(paths, key=key_fn)


def compute_shift_feature_matching(frame1, frame2):
    """Estimate inter-frame translation and in-plane rotation using SIFT."""
    gray1 = cv2.cvtColor(frame1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    if descriptors1 is None or descriptors2 is None:
        return 0.0, 0.0, 0.0

    matcher = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=50),
    )
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [
        first
        for first, second in matches
        if first.distance < 0.7 * second.distance
    ]

    if len(good_matches) <= 10:
        return 0.0, 0.0, 0.0

    source_points = np.float32(
        [keypoints1[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 2)
    destination_points = np.float32(
        [keypoints2[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 2)

    transform, _ = cv2.estimateAffinePartial2D(source_points, destination_points)
    if transform is None:
        return 0.0, 0.0, 0.0

    a, b = transform[0, 0], transform[0, 1]
    angle_deg = math.degrees(math.atan2(b, a))
    dx = transform[0, 2]
    dy = transform[1, 2]
    return dx, dy, angle_deg


def main(
    gps_name,
    attitude_name,
    vn_name,
    alt_name,
    initial_position_name,
    gps_ned_name,
    vn_ned_name,
    shape,
    dtype,
):
    """Run relative visual odometry over a recorded sequence of nadir frames."""
    shared_attitude = SharedArrayWithLock(
        attitude_name, shape, dtype, create=False
    )
    shared_vn = SharedArrayWithLock(vn_name, (20, 3), dtype, create=False)
    shared_alt = SharedArrayWithLock(alt_name, (20, 2), dtype, create=False)
    shared_initial_position = SharedArrayWithLock(
        initial_position_name, (1, 4), dtype, create=False
    )
    shared_vn_ned = SharedArrayWithLock(vn_ned_name, shape, dtype, create=False)

    frames_dir = CONFIG.get("visual_odometry", {}).get(
        "frames_dir", "data/frames"
    )
    frame_paths = load_sorted_jpgs(frames_dir)
    if len(frame_paths) < 2:
        raise FileNotFoundError(
            f"At least two JPEG frames are required in {frames_dir!r}."
        )

    step = 20
    scale = 0.6
    fov_vertical = 71.5 * scale
    fov_horizontal = 79.5 * scale

    # MAVLink ATTITUDE yaw is in radians; accumulate image-derived rotation in degrees.
    initial_yaw = shared_attitude.read()[-1, 2]
    yaw_deg = math.degrees(float(initial_yaw))
    position_ne = np.zeros(2, dtype=np.float64)

    for idx in range(len(frame_paths) - 1):
        start = time.time()

        frame1 = cv2.imread(frame_paths[idx])
        frame2 = cv2.imread(frame_paths[idx + 1])
        if frame1 is None or frame2 is None:
            continue

        height, width = frame1.shape[:2]
        dx, dy, angle_deg = compute_shift_feature_matching(frame1, frame2)

        altitude = shared_alt.read()[-1, 0]
        if np.isnan(altitude) or altitude <= 0:
            continue

        gsd_vertical = (
            2 * math.tan(math.radians(fov_vertical / 2)) * altitude / height
        )
        gsd_horizontal = (
            2 * math.tan(math.radians(fov_horizontal / 2)) * altitude / width
        )

        dx_m = dx * gsd_horizontal
        dy_m = dy * gsd_vertical
        yaw_deg += angle_deg

        image_delta = np.array([dx_m, dy_m])
        theta = math.radians(yaw_deg - 90.0)
        rotation = np.array(
            [
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)],
            ]
        )
        position_ne += rotation @ image_delta
        timestamp = time.time()

        with shared_vn_ned.lock:
            array = shared_vn_ned.get()
            array[:-1] = array[1:]
            array[-1] = np.array(
                [position_ne[0], position_ne[1], 0.0, timestamp]
            )

        initial_position = shared_initial_position.read()[0]
        if not np.isnan(initial_position[:2]).any():
            lat, lon, _ = pm.ned2geodetic(
                position_ne[0],
                position_ne[1],
                0.0,
                initial_position[0],
                initial_position[1],
                0.0,
            )
            with shared_vn.lock:
                array = shared_vn.get()
                array[:-1] = array[1:]
                array[-1] = np.array([lat, lon, timestamp])

        target_period = step / 30.0
        time.sleep(max(0.0, target_period - (time.time() - start)))
