import time

import numpy as np
from pymavlink import mavutil

from config import config_loader
from data.shared_data import SharedArrayWithLock


CONFIG = config_loader.load_config()
MAVLINK_CONNECTION = CONFIG.get("mavlink", {}).get(
    "connection", "udp:127.0.0.1:14550"
)


def mavlink_listener(
    gps_name,
    imu_name,
    pressure_name,
    attitude_name,
    shape,
    dtype,
    stop_event,
):
    """Read ArduPilot MAVLink telemetry into named shared-memory buffers."""
    master = mavutil.mavlink_connection(MAVLINK_CONNECTION)
    if not master.wait_heartbeat(timeout=10):
        raise ConnectionError("Failed to connect to ArduPilot")

    print(f"Connected to ArduPilot through {MAVLINK_CONNECTION}")

    shared_gps = SharedArrayWithLock(gps_name, shape, dtype, create=False)
    shared_imu = SharedArrayWithLock(imu_name, shape, dtype, create=False)
    shared_press = SharedArrayWithLock(pressure_name, (20, 2), dtype, create=False)
    shared_attitude = SharedArrayWithLock(attitude_name, shape, dtype, create=False)

    while not stop_event.is_set():
        msg = master.recv_match(blocking=True)
        if msg is None:
            continue

        timestamp = time.time()
        msg_type = msg.get_type()

        if msg_type == "GLOBAL_POSITION_INT":
            new_data = np.array(
                [msg.lat * 1e-7, msg.lon * 1e-7, msg.alt / 1000.0, timestamp]
            )
            with shared_gps.lock:
                array = shared_gps.get()
                array[:-1] = array[1:]
                array[-1] = new_data

        elif msg_type == "RAW_IMU":
            new_data = np.array(
                [
                    msg.xacc * 9.81 / 1000.0,
                    msg.yacc * 9.81 / 1000.0,
                    msg.zacc * 9.81 / 1000.0,
                    msg.time_usec / 1e6,
                ]
            )
            with shared_imu.lock:
                array = shared_imu.get()
                array[:-1] = array[1:]
                array[-1] = new_data

        elif msg_type == "ATTITUDE":
            new_data = np.array(
                [msg.roll, msg.pitch, msg.yaw, msg.time_boot_ms / 1e3]
            )
            with shared_attitude.lock:
                array = shared_attitude.get()
                array[:-1] = array[1:]
                array[-1] = new_data

        elif msg_type == "SCALED_PRESSURE":
            new_data = np.array([msg.press_abs, msg.time_boot_ms / 1e3])
            with shared_press.lock:
                array = shared_press.get()
                array[:-1] = array[1:]
                array[-1] = new_data

        time.sleep(0.01)
