"""Development experiment combining SIFT relative localization with periodic AVL fixes.

This script is retained to document the asynchronous AVL/RVL correction experiment.
It is separate from the Kalman-filter implementation in ``core/``. Configure the
dataset, FAISS caches, map tiles, LiteSAM config, and weights before running.
"""

import time
import math
import csv
import cv2
import numpy as np
import pymap3d as pm
from pathlib import Path
from collections import deque
from threading import Thread, Lock, Event
from queue import Queue, Empty

from localization.absolute.avl_localizer import AVLConfig, AVLLocalizer


DATASET_DIR = Path("data/avl_rvl_frames")
GT_CSV = Path("data/avl_rvl_frames/image_gps_log.csv")
OUT_CSV = Path("outputs/est_fused.csv")

SIFT_RATE_HZ = 10.0
ALT_M = 410.0
FOV_DEG = (60.0, 60.0)
MAX_CORRECTION_M = 150.0
HIST_SECONDS = 120.0

GT_IMAGE_COL = "filename"
GT_LAT_COL = "latitude_deg"
GT_LON_COL = "longitude_deg"

cfg = AVLConfig(
    parent_cache=Path("faiss_cache/parent"),
    sub_cache=Path("faiss_cache/subtiles"),
    sub_tiles_dir=Path("tiles_out/subtiles"),
    mode="two",
    topk_stage1=5,
    topk_stage2=5,
    target_size=320,
)
localizer = AVLLocalizer(cfg)


def avl_localize_image(image_path: Path):
    result = localizer.localize(str(image_path), verbose=False)
    if result is None:
        return None, None, 0.0
    return (
        float(result["lat"]),
        float(result["lon"]),
        float(result.get("inlier_ratio", 1.0)),
    )


def compute_relative_motion(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        return None

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if len(good) < 10:
        return None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
    matrix, _ = cv2.estimateAffinePartial2D(src, dst)
    if matrix is None:
        return None

    dx = float(matrix[0, 2])
    dy = float(matrix[1, 2])
    yaw_rad = math.atan2(matrix[0, 1], matrix[0, 0])
    return dx, dy, float(math.degrees(yaw_rad))


def geodetic_to_ne(lat, lon, init_lla):
    north, east, _ = pm.geodetic2ned(
        lat, lon, 0.0, init_lla[0], init_lla[1], init_lla[2]
    )
    return float(north), float(east)


def closest_hist_idx_by_frame(hist, frame_idx_target):
    if not hist:
        return None

    best_i, best_dt = None, float("inf")
    for i, item in enumerate(hist):
        delta = abs(item["frame_idx"] - frame_idx_target)
        if delta < best_dt:
            best_i, best_dt = i, delta
    return best_i


def load_ground_truth(gt_csv: Path):
    gt = {}
    with open(gt_csv, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            gt[row[GT_IMAGE_COL]] = (
                float(row[GT_LAT_COL]),
                float(row[GT_LON_COL]),
            )
    return gt


def rmse_meters(est_rows, gt_dict, init_lla):
    errors_squared = []
    for row in est_rows:
        name = row["image_name"]
        if name not in gt_dict:
            continue

        lat_est, lon_est = row["lat_out"], row["lon_out"]
        if lat_est is None or lon_est is None:
            continue

        lat_gt, lon_gt = gt_dict[name]
        n_est, e_est = geodetic_to_ne(lat_est, lon_est, init_lla)
        n_gt, e_gt = geodetic_to_ne(lat_gt, lon_gt, init_lla)
        errors_squared.append((n_est - n_gt) ** 2 + (e_est - e_gt) ** 2)

    if not errors_squared:
        return None
    return math.sqrt(sum(errors_squared) / len(errors_squared))


class LatestFrameRef:
    def __init__(self):
        self.lock = Lock()
        self.latest_idx = -1
        self.latest_path = None

    def set(self, idx, path):
        with self.lock:
            self.latest_idx = idx
            self.latest_path = path

    def get(self):
        with self.lock:
            return self.latest_idx, self.latest_path


def avl_worker_latest(latest_ref: LatestFrameRef, result_q: Queue, stop_evt: Event):
    last_done_idx = -1

    while not stop_evt.is_set():
        idx, path = latest_ref.get()
        if path is None or idx <= last_done_idx:
            time.sleep(0.01)
            continue

        frame_idx = idx
        image_path = Path(path)
        try:
            lat, lon, trust = avl_localize_image(image_path)
            if lat is None or lon is None:
                raise RuntimeError("AVL returned None")
            result_q.put(
                {
                    "frame_idx": frame_idx,
                    "image_path": str(image_path),
                    "lat": lat,
                    "lon": lon,
                    "trust": trust,
                }
            )
        except Exception as exc:
            result_q.put(
                {
                    "frame_idx": frame_idx,
                    "image_path": str(image_path),
                    "lat": None,
                    "lon": None,
                    "trust": 0.0,
                    "err": str(exc),
                }
            )

        last_done_idx = frame_idx


def run_fused():
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = sorted(
        [path for path in DATASET_DIR.iterdir() if path.suffix.lower() in exts]
    )
    if len(files) < 2:
        raise RuntimeError(f"Need at least 2 images in {DATASET_DIR}")

    gt = load_ground_truth(GT_CSV)
    first_name = files[0].name
    if first_name not in gt:
        raise RuntimeError(f"First image {first_name} not found in ground truth CSV.")

    init_lla = (gt[first_name][0], gt[first_name][1], 0.0)
    print("[INIT] init_lla from GT:", init_lla)

    result_q = Queue()
    latest_ref = LatestFrameRef()
    stop_evt = Event()
    Thread(
        target=avl_worker_latest,
        args=(latest_ref, result_q, stop_evt),
        daemon=True,
    ).start()

    north_raw, east_raw = 0.0, 0.0
    yaw_deg = 0.0
    off_n, off_e = 0.0, 0.0
    hist = deque(maxlen=int(HIST_SECONDS * SIFT_RATE_HZ) + 50)
    out_rows = []

    prev = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
    if prev is None:
        raise FileNotFoundError(files[0])

    dt_target = 1.0 / SIFT_RATE_HZ
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output_handle = open(OUT_CSV, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        output_handle,
        fieldnames=[
            "frame_idx",
            "image_name",
            "north_raw_m",
            "east_raw_m",
            "off_n_m",
            "off_e_m",
            "north_out_m",
            "east_out_m",
            "lat_out",
            "lon_out",
            "avl_lat",
            "avl_lon",
            "avl_trust",
        ],
    )
    writer.writeheader()

    try:
        for i in range(1, len(files)):
            t0_loop = time.time()
            image_path = files[i]
            latest_ref.set(i, image_path)
            image_name = image_path.name

            current = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if current is None:
                print("[WARN] could not read:", image_path)
                continue

            result = compute_relative_motion(prev, current)
            prev = current
            if result is None:
                continue

            dx_px, dy_px, dyaw = result
            height, width = current.shape[:2]
            fov_h, fov_w = FOV_DEG

            gsd_h = 2 * math.tan(math.radians(fov_h / 2)) * ALT_M / height
            gsd_w = 2 * math.tan(math.radians(fov_w / 2)) * ALT_M / width
            dx_m = dx_px * gsd_w
            dy_m = dy_px * gsd_h

            yaw_deg += dyaw
            theta = math.radians(yaw_deg - 90.0)
            rotation = np.array(
                [
                    [math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)],
                ]
            )
            d_north, d_east = (rotation @ np.array([dx_m, dy_m])).tolist()

            north_raw += float(d_north)
            east_raw += float(d_east)
            north_out = north_raw + off_n
            east_out = east_raw + off_e

            hist.append(
                {"frame_idx": i, "n_out": north_out, "e_out": east_out}
            )

            avl_lat = avl_lon = None
            avl_trust = 0.0

            while True:
                try:
                    fix = result_q.get_nowait()
                except Empty:
                    break

                if fix.get("lat") is None or fix.get("lon") is None:
                    continue

                avl_lat = fix["lat"]
                avl_lon = fix["lon"]
                avl_trust = float(fix.get("trust", 1.0))
                idx = closest_hist_idx_by_frame(hist, fix["frame_idx"])
                if idx is None:
                    continue

                n_rvl_then = hist[idx]["n_out"]
                e_rvl_then = hist[idx]["e_out"]
                n_avl, e_avl = geodetic_to_ne(avl_lat, avl_lon, init_lla)

                dn = n_avl - n_rvl_then
                de = e_avl - e_rvl_then
                if (dn * dn + de * de) > MAX_CORRECTION_M**2:
                    print(
                        f"[AVL] REJECT correction too large dn={dn:.1f} de={de:.1f}"
                    )
                    continue

                off_n += dn
                off_e += de
                for k in range(idx, len(hist)):
                    hist[k]["n_out"] += dn
                    hist[k]["e_out"] += de

                north_out = north_raw + off_n
                east_out = east_raw + off_e
                print(
                    f"[AVL] APPLY fix @ frame={fix['frame_idx']}: "
                    f"dn={dn:.2f} de={de:.2f} "
                    f"off=({off_n:.2f},{off_e:.2f}) trust={avl_trust:.2f}"
                )

            lat_out, lon_out, _ = pm.ned2geodetic(
                north_out,
                east_out,
                0.0,
                init_lla[0],
                init_lla[1],
                init_lla[2],
            )

            row = {
                "frame_idx": i,
                "image_name": image_name,
                "north_raw_m": north_raw,
                "east_raw_m": east_raw,
                "off_n_m": off_n,
                "off_e_m": off_e,
                "north_out_m": north_out,
                "east_out_m": east_out,
                "lat_out": float(lat_out),
                "lon_out": float(lon_out),
                "avl_lat": avl_lat,
                "avl_lon": avl_lon,
                "avl_trust": avl_trust,
            }
            out_rows.append(
                {
                    "image_name": image_name,
                    "lat_out": float(lat_out),
                    "lon_out": float(lon_out),
                }
            )
            writer.writerow(row)
            output_handle.flush()

            sleep_time = dt_target - (time.time() - t0_loop)
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        stop_evt.set()
        output_handle.close()

    rmse = rmse_meters(out_rows, gt, init_lla)
    if rmse is None:
        print("RMSE could not be computed.")
    else:
        print(f"RMSE (meters): {rmse:.3f}")
    print("Saved:", OUT_CSV)


if __name__ == "__main__":
    run_fused()
