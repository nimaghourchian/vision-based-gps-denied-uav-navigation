import json
import queue
import socket
from io import BytesIO

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from data.shared_data import SharedArrayWithLock


IMAGE_PORT = 14553
FOV_HORIZONTAL = 79.5
FOV_VERTICAL = 79.5


def read_jpeg_to_rgb(data):
    """Decode one JPEG payload into an RGB NumPy array."""
    try:
        image = imageio.imread(BytesIO(data))
        if image.ndim == 2:
            return np.stack([image] * 3, axis=-1).astype(np.float32)
        if image.shape[2] in (3, 4):
            return image[..., :3].astype(np.float32)
        raise ValueError("Unsupported image format")
    except Exception as exc:
        print(f"Error reading image: {exc}")
        return None


def extract_timestamp(image_data):
    """Return the EXIF DateTime value when present."""
    try:
        image = Image.open(BytesIO(image_data))
        exif_data = image.getexif()
        return exif_data.get(306) if exif_data else None
    except Exception:
        return None


def compute_shift_feature_matching(frame1, frame2):
    """Estimate pixel translation between consecutive frames using SIFT."""
    gray1 = cv2.cvtColor(frame1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    if descriptors1 is None or descriptors2 is None:
        return 0.0, 0.0

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
        return 0.0, 0.0

    source_points = np.float32(
        [keypoints1[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 2)
    destination_points = np.float32(
        [keypoints2[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 2)

    transform, _ = cv2.estimateAffinePartial2D(source_points, destination_points)
    if transform is None:
        return 0.0, 0.0

    return float(transform[0, 2]), float(transform[1, 2])


def _recv_exact(connection, size):
    payload = bytearray()
    while len(payload) < size:
        chunk = connection.recv(size - len(payload))
        if not chunk:
            raise ConnectionError("Image stream closed unexpectedly")
        payload.extend(chunk)
    return bytes(payload)


def image_processing_process(attitude_name, vn_name, alt_name, shape, dtype):
    """Receive a JPEG stream and accumulate a relative north/east trajectory."""
    shared_attitude = SharedArrayWithLock(
        attitude_name, shape, dtype, create=False
    )
    shared_vn = SharedArrayWithLock(vn_name, (20, 3), dtype, create=False)
    shared_alt = SharedArrayWithLock(alt_name, (20, 2), dtype, create=False)

    frame_queue = queue.Queue(maxsize=2)
    timestamp_queue = queue.Queue(maxsize=2)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("", IMAGE_PORT))
    server.listen(1)
    print(f"Listening for image stream on port {IMAGE_PORT}")

    cumulative_north = 0.0
    cumulative_east = 0.0

    while True:
        connection, address = server.accept()
        print(f"Image stream connected from {address}")

        try:
            while True:
                header_length = int.from_bytes(_recv_exact(connection, 4), "big")
                header = json.loads(
                    _recv_exact(connection, header_length).decode("utf-8")
                )

                image_length = int.from_bytes(_recv_exact(connection, 4), "big")
                image_data = _recv_exact(connection, image_length)

                timestamp = extract_timestamp(image_data)
                if timestamp is None:
                    timestamp = header.get("ts_ns", 0) / 1e9

                frame = read_jpeg_to_rgb(image_data)
                if frame is None:
                    continue

                if frame_queue.full():
                    frame_queue.get()
                    timestamp_queue.get()
                frame_queue.put(frame)
                timestamp_queue.put(timestamp)

                if frame_queue.qsize() < 2:
                    continue

                frame1 = frame_queue.queue[0]
                frame2 = frame_queue.queue[1]
                timestamp2 = timestamp_queue.queue[1]

                dx, dy = compute_shift_feature_matching(frame1, frame2)

                # MAVLink ATTITUDE uses radians, so yaw can be used directly.
                yaw = shared_attitude.read()[-1, 2]
                altitude = shared_alt.read()[-1, 0]
                if np.isnan(yaw) or np.isnan(altitude) or altitude <= 0:
                    frame_queue.get()
                    timestamp_queue.get()
                    continue

                height, width = frame2.shape[:2]
                gsd_x = (
                    altitude
                    * 2
                    * np.tan(np.radians(FOV_HORIZONTAL / 2))
                    / width
                )
                gsd_y = (
                    altitude
                    * 2
                    * np.tan(np.radians(FOV_VERTICAL / 2))
                    / height
                )

                delta_front = dy * gsd_y
                delta_right = -dx * gsd_x

                delta_north = (
                    delta_front * np.cos(yaw) - delta_right * np.sin(yaw)
                )
                delta_east = (
                    delta_front * np.sin(yaw) + delta_right * np.cos(yaw)
                )

                cumulative_north += delta_north
                cumulative_east += delta_east

                with shared_vn.lock:
                    array = shared_vn.get()
                    array[:-1] = array[1:]
                    array[-1] = np.array(
                        [cumulative_north, cumulative_east, timestamp2]
                    )

                frame_queue.get()
                timestamp_queue.get()

        except (ConnectionError, OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"Image-stream connection ended: {exc}")
        finally:
            connection.close()
