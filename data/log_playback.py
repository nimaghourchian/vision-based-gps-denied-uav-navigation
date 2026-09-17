import time

import numpy as np
import pandas as pd

from data.shared_data import SharedArrayWithLock


def playback_from_excel(
    merged_excel_path,
    gps_name,
    imu_name,
    pressure_name,
    attitude_name,
    shape,
    dtype,
    stop_event,
):
    """Replay a merged ArduPilot log at approximately its recorded timing."""
    print(f"Loading data from {merged_excel_path} ...")
    df = pd.read_excel(merged_excel_path)
    df = df.sort_values(by="TimeUS").reset_index(drop=True)

    shared_gps = SharedArrayWithLock(gps_name, shape, dtype, create=False)
    shared_imu = SharedArrayWithLock(imu_name, shape, dtype, create=False)
    shared_press = SharedArrayWithLock(pressure_name, (20, 2), dtype, create=False)
    shared_attitude = SharedArrayWithLock(attitude_name, shape, dtype, create=False)

    print("Starting playback simulation...")
    start_time = time.time()
    start_log_time = df.iloc[0]["TimeUS"]

    for i, row in df.iterrows():
        if stop_event.is_set():
            break

        msg_type = str(row["Type"]).upper().strip()
        log_time = row["TimeUS"]

        elapsed_log_time = (log_time - start_log_time) / 1e6
        elapsed_real_time = time.time() - start_time
        delay = elapsed_log_time - elapsed_real_time
        if delay > 0:
            time.sleep(delay)

        timestamp = time.time()

        if msg_type == "IMU":
            new_data = np.array(
                [
                    row.get("AccX", 0),
                    row.get("AccY", 0),
                    row.get("AccZ", 0),
                    timestamp,
                ]
            )
            with shared_imu.lock:
                array = shared_imu.get()
                array[:-1] = array[1:]
                array[-1] = new_data

        elif msg_type == "AHR2":
            attitude_data = np.array(
                [
                    row.get("Roll", 0),
                    row.get("Pitch", 0),
                    row.get("Yaw", 0),
                    timestamp,
                ]
            )
            with shared_attitude.lock:
                array = shared_attitude.get()
                array[:-1] = array[1:]
                array[-1] = attitude_data

            gps_data = np.array(
                [
                    row.get("Lat", 0),
                    row.get("Lng", 0),
                    row.get("Alt", 0),
                    timestamp,
                ]
            )
            with shared_gps.lock:
                array = shared_gps.get()
                array[:-1] = array[1:]
                array[-1] = gps_data

        elif msg_type == "BARO":
            new_data = np.array([row.get("Press", 0), timestamp])
            with shared_press.lock:
                array = shared_press.get()
                array[:-1] = array[1:]
                array[-1] = new_data

        elif msg_type == "GPS":
            new_data = np.array(
                [
                    row.get("Lat", 0),
                    row.get("Lng", 0),
                    row.get("Alt", 0),
                    timestamp,
                ]
            )
            with shared_gps.lock:
                array = shared_gps.get()
                array[:-1] = array[1:]
                array[-1] = new_data

        if i % 100 == 0:
            print(f"[{i}/{len(df)}] Sent {msg_type} at {timestamp:.2f}")

    print("Playback finished.")
