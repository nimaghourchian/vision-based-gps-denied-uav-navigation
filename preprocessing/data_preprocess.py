import time

import numpy as np
import pymap3d as pm

from config import config_loader
from data.shared_data import SharedArrayWithLock
from preprocessing.utils import accel_transform, add_to_queue, pressure_to_altitude


CONFIG = config_loader.load_config()
ENABLE = CONFIG["initial_state"]["enable"]


def initial_setting(
    gps_name,
    imu_name,
    attitude_name,
    pressure_name,
    initial_position_name,
    initial_imu_name,
    initial_attitude_name,
    initial_pressure_name,
    shape,
    dtype,
    enable=ENABLE,
):
    """Capture the first complete sensor set as the local reference state."""
    if not enable:
        return

    shared_gps = SharedArrayWithLock(gps_name, shape, dtype, create=False)
    shared_imu = SharedArrayWithLock(imu_name, shape, dtype, create=False)
    shared_attitude = SharedArrayWithLock(attitude_name, shape, dtype, create=False)
    shared_pressure = SharedArrayWithLock(pressure_name, (20, 2), dtype, create=False)

    shared_initial_position = SharedArrayWithLock(
        initial_position_name, (1, 4), dtype, create=False
    )
    shared_initial_imu = SharedArrayWithLock(
        initial_imu_name, (1, 4), dtype, create=False
    )
    shared_initial_attitude = SharedArrayWithLock(
        initial_attitude_name, (1, 4), dtype, create=False
    )
    shared_initial_pressure = SharedArrayWithLock(
        initial_pressure_name, (1, 2), dtype, create=False
    )

    gps_data = shared_gps.read()[-1]
    imu_data = shared_imu.read()[-1]
    attitude_data = shared_attitude.read()[-1]
    pressure_data = shared_pressure.read()[-1]

    if any(
        np.isnan(sample).any()
        for sample in (gps_data, imu_data, attitude_data, pressure_data)
    ):
        return

    with shared_initial_position.lock:
        shared_initial_position.get()[0] = gps_data
    with shared_initial_imu.lock:
        shared_initial_imu.get()[0] = imu_data
    with shared_initial_attitude.lock:
        shared_initial_attitude.get()[0] = attitude_data
    with shared_initial_pressure.lock:
        shared_initial_pressure.get()[0] = pressure_data


def pressure2alt(
    pressure_name,
    alt_name,
    initial_pressure_name,
    shape,
    dtype,
    enable=True,
):
    """Convert the pressure stream into altitude relative to the initial pressure."""
    if not enable:
        return

    shared_pressure = SharedArrayWithLock(
        pressure_name, (20, 2), dtype, create=False
    )
    shared_alt = SharedArrayWithLock(alt_name, (20, 2), dtype, create=False)
    shared_initial_pressure = SharedArrayWithLock(
        initial_pressure_name, (1, 2), dtype, create=False
    )

    while True:
        current_pressure = shared_pressure.read()[-1]
        initial_pressure = shared_initial_pressure.read()[0]

        if not np.isnan(current_pressure).any() and not np.isnan(initial_pressure).any():
            altitude = pressure_to_altitude(
                current_pressure[0], initial_pressure[0]
            )
            if altitude is not None:
                sample = np.array([altitude, current_pressure[-1]])
                add_to_queue(shared_alt.get(), sample, shared_alt.lock)

        time.sleep(0.3)


def coordinate_transform(
    gps_name,
    imu_name,
    attitude_name,
    gps_ned_name,
    imu_ned_name,
    initial_position_name,
    initial_imu_name,
    initial_attitude_name,
    shape,
    dtype,
    stop_event,
    enable=ENABLE,
):
    """Transform GPS and IMU measurements into the local NED frame."""
    if not enable:
        return

    shared_gps = SharedArrayWithLock(gps_name, shape, dtype, create=False)
    shared_imu = SharedArrayWithLock(imu_name, shape, dtype, create=False)
    shared_attitude = SharedArrayWithLock(attitude_name, shape, dtype, create=False)

    shared_gps_ned = SharedArrayWithLock(gps_ned_name, shape, dtype, create=False)
    shared_imu_ned = SharedArrayWithLock(imu_ned_name, shape, dtype, create=False)

    shared_initial_position = SharedArrayWithLock(
        initial_position_name, (1, 4), dtype, create=False
    )
    shared_initial_imu = SharedArrayWithLock(
        initial_imu_name, (1, 4), dtype, create=False
    )
    shared_initial_attitude = SharedArrayWithLock(
        initial_attitude_name, (1, 4), dtype, create=False
    )

    while not stop_event.is_set():
        current_gps = shared_gps.read()[-1]
        initial_position = shared_initial_position.read()[-1]

        if not np.isnan(current_gps).any() and not np.isnan(initial_position).any():
            north, east, down = pm.geodetic2ned(
                current_gps[0],
                current_gps[1],
                current_gps[2],
                initial_position[0],
                initial_position[1],
                initial_position[2],
            )
            gps_sample = np.array([north, east, down, current_gps[-1]])
            add_to_queue(shared_gps_ned.get(), gps_sample, shared_gps_ned.lock)

        current_imu = shared_imu.read()[-1]
        current_attitude = shared_attitude.read()[-1]
        initial_imu = shared_initial_imu.read()[0]
        initial_attitude = shared_initial_attitude.read()[0]

        if (
            not np.isnan(current_imu).any()
            and not np.isnan(current_attitude).any()
            and not np.isnan(initial_imu).any()
            and not np.isnan(initial_attitude).any()
        ):
            acceleration_body = np.array(
                [[current_imu[0]], [current_imu[1]], [current_imu[2]]]
            )
            roll, pitch, yaw = current_attitude[:3]
            acceleration_ned = accel_transform(
                acceleration_body, roll, pitch, yaw
            ).reshape(3)

            imu_sample = np.array([*acceleration_ned, current_imu[-1]])
            add_to_queue(shared_imu_ned.get(), imu_sample, shared_imu_ned.lock)

        time.sleep(0.01)
