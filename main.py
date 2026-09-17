import multiprocessing
import time
from multiprocessing import Event, Process, shared_memory

import numpy as np

from config import config_loader
from core import kalman_filter
from data import log_playback, mavlink_input
from data.shared_data import SharedArrayWithLock
from outputs import map_publisher
from preprocessing import data_preprocess


CONFIG = config_loader.load_config()
DT = CONFIG["kalman"]["dt"]
X_INIT = np.array(CONFIG["initial_state"]["x0"])
P_INIT = np.array(CONFIG["initial_state"]["P0"])
SHAPE = (20, 4)
DTYPE = np.float64


def cleanup_all():
    """Close and unlink the named shared-memory blocks created by this process."""
    names = [
        "gps_raw",
        "imu_raw",
        "pressure_scaled",
        "attitude",
        "gps_ned",
        "imu_ned",
        "alt_queue",
        "initial_position",
        "initial_imu",
        "initial_attitude",
        "initial_pressure",
        "state_queue",
        "state_queue_lla",
    ]

    for name in names:
        try:
            block = shared_memory.SharedMemory(name=name)
            block.close()
            block.unlink()
        except FileNotFoundError:
            pass


def main():
    """Start the sensor, preprocessing, estimation, and visualization workers."""
    SharedArrayWithLock("gps_raw", SHAPE, DTYPE, create=True)
    SharedArrayWithLock("imu_raw", SHAPE, DTYPE, create=True)
    SharedArrayWithLock("pressure_scaled", (20, 2), DTYPE, create=True)
    SharedArrayWithLock("attitude", SHAPE, DTYPE, create=True)

    SharedArrayWithLock("gps_ned", SHAPE, DTYPE, create=True)
    SharedArrayWithLock("imu_ned", SHAPE, DTYPE, create=True)
    SharedArrayWithLock("alt_queue", (20, 2), DTYPE, create=True)

    SharedArrayWithLock("initial_position", (1, 4), DTYPE, create=True)
    SharedArrayWithLock("initial_imu", (1, 4), DTYPE, create=True)
    SharedArrayWithLock("initial_attitude", (1, 4), DTYPE, create=True)
    SharedArrayWithLock("initial_pressure", (1, 2), DTYPE, create=True)

    SharedArrayWithLock("state_queue", (20, 7), DTYPE, create=True)
    SharedArrayWithLock("state_queue_lla", SHAPE, DTYPE, create=True)

    stop_event = Event()

    sensor_process = Process(
        target=mavlink_input.mavlink_listener,
        args=(
            "gps_raw",
            "imu_raw",
            "pressure_scaled",
            "attitude",
            SHAPE,
            DTYPE,
            stop_event,
        ),
    )

    # For recorded-log replay, replace sensor_process with:
    # Process(
    #     target=log_playback.playback_from_excel,
    #     args=(
    #         "data/merged_sorted.xlsx",
    #         "gps_raw",
    #         "imu_raw",
    #         "pressure_scaled",
    #         "attitude",
    #         SHAPE,
    #         DTYPE,
    #         stop_event,
    #     ),
    # )

    initialization_process = Process(
        target=data_preprocess.initial_setting,
        args=(
            "gps_raw",
            "imu_raw",
            "attitude",
            "pressure_scaled",
            "initial_position",
            "initial_imu",
            "initial_attitude",
            "initial_pressure",
            SHAPE,
            DTYPE,
            True,
        ),
    )

    barometer_process = Process(
        target=data_preprocess.pressure2alt,
        args=(
            "pressure_scaled",
            "alt_queue",
            "initial_pressure",
            SHAPE,
            DTYPE,
        ),
    )

    transform_process = Process(
        target=data_preprocess.coordinate_transform,
        args=(
            "gps_raw",
            "imu_raw",
            "attitude",
            "gps_ned",
            "imu_ned",
            "initial_position",
            "initial_imu",
            "initial_attitude",
            SHAPE,
            DTYPE,
            stop_event,
            True,
        ),
    )

    kalman_process = Process(
        target=kalman_filter.kalman_process,
        args=(
            X_INIT,
            P_INIT,
            "gps_ned",
            "imu_ned",
            "initial_position",
            "alt_queue",
            "state_queue",
            "state_queue_lla",
            SHAPE,
            DTYPE,
            DT,
        ),
    )

    map_process = Process(
        target=map_publisher.publish_state,
        args=("gps_raw", "state_queue_lla", SHAPE, DTYPE, stop_event, True),
    )

    print("[System] Starting processes...")
    sensor_process.start()
    time.sleep(5)
    initialization_process.start()
    time.sleep(0.3)
    barometer_process.start()
    time.sleep(2)
    transform_process.start()
    time.sleep(10)
    kalman_process.start()
    time.sleep(2)
    map_process.start()

    # The two visual-odometry implementations are kept as separate development
    # components under preprocessing/. They can be connected to the shared
    # vision buffers when reproducing those experiments.


if __name__ == "__main__":
    try:
        main()
        input("\n[System] Press ENTER to stop...\n")
    finally:
        print("[System] Stopping processes...")
        for process in multiprocessing.active_children():
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        cleanup_all()
        print("[System] All done. Clean exit.")
