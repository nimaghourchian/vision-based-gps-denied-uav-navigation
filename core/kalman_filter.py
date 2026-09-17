import time

import numpy as np
import pymap3d as pm

from config import config_loader
from core import state_model as sm
from data.shared_data import SharedArrayWithLock


config = config_loader.load_config()

ENABLE = config["initial_state"]["enable"]
dt = config["kalman"]["dt"]
x_init = np.array(config["initial_state"]["x0"])
P_init = np.array(config["initial_state"]["P0"])


class KalmanFilter:
    def __init__(self, x0, P0, dt):
        self.x = x0
        self.P = P0

        self.Q = sm.get_process_noise()
        self.Hg = sm.get_observation_model_gps()
        self.Hv = sm.get_observation_model_vn()
        self.Hb = sm.get_observation_model_baro()

        self.Rg = np.eye(3) * 0.5   # GPS noise covariance
        self.Rv = np.eye(2) * 1.0   # Vision-navigation noise covariance
        self.Rb = np.eye(1) * 0.5   # Barometer noise covariance

        self.F = sm.get_state_model(dt)
        self.B = sm.get_control_matrix(dt)

    def predict(self, u):
        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, h, r):
        innovation = z - h @ self.x
        innovation_covariance = h @ self.P @ h.T + r
        kalman_gain = self.P @ h.T @ np.linalg.inv(innovation_covariance)
        self.x = self.x + kalman_gain @ innovation
        self.P = self.P - kalman_gain @ h @ self.P

    def get_state(self):
        return self.x

    def get_covariance(self):
        return self.P


def kalman_process(
    x_init,
    p_init,
    gps_ned_name,
    imu_ned_name,
    initial_position_name,
    alt_name,
    state_name,
    state_lla_name,
    shape,
    dtype,
    dt,
    enable=ENABLE,
):
    """Run the six-state position/velocity Kalman filter in a worker process."""
    if not enable:
        return

    shared_gps_ned = SharedArrayWithLock(gps_ned_name, shape, dtype, create=False)
    shared_imu_ned = SharedArrayWithLock(imu_ned_name, shape, dtype, create=False)
    shared_alt = SharedArrayWithLock(alt_name, (20, 2), dtype, create=False)
    shared_initial_position = SharedArrayWithLock(
        initial_position_name, (1, 4), dtype, create=False
    )
    shared_state = SharedArrayWithLock(state_name, (20, 7), dtype, create=False)
    shared_state_lla = SharedArrayWithLock(state_lla_name, shape, dtype, create=False)

    kf = KalmanFilter(x_init, p_init, dt)

    while True:
        imu_ned = shared_imu_ned.read()
        gps_ned = shared_gps_ned.read()
        alt_queue = shared_alt.read()

        # The penultimate IMU sample was used during development to reduce
        # timing mismatch with the other queues.
        u = imu_ned[-2, :3].reshape(3, 1)
        z_gps = gps_ned[-1, :3].reshape(3, 1)
        z_baro = -alt_queue[-1, :1]
        timestamp = time.time()

        if not np.isnan(u).any():
            kf.predict(u)

            if (
                not np.isnan(z_gps).any()
                and abs(gps_ned[-1, -1] - timestamp) < 0.2
            ):
                kf.update(z_gps, kf.Hg, kf.Rg)

            # Vision-position updates were integrated during development but
            # are disabled in this checked-in integration snapshot. The
            # horizontal observation model remains available as kf.Hv.

            if (
                not np.isnan(z_baro).any()
                and abs(alt_queue[-1, 1] - timestamp) < 0.2
            ):
                kf.update(z_baro, kf.Hb, kf.Rb)

        state = kf.get_state().reshape(6)

        with shared_state.lock:
            state_with_time = np.array([*state, timestamp])
            array = shared_state.get()
            array[:-1] = array[1:]
            array[-1] = state_with_time

        with shared_state_lla.lock, shared_initial_position.lock:
            initial_position = shared_initial_position.get()[-1]
            if not np.isnan(initial_position[:3]).any():
                state_lla = pm.ned2geodetic(
                    state[0],
                    state[1],
                    state[2],
                    initial_position[0],
                    initial_position[1],
                    initial_position[2],
                )
                state_lla_with_time = np.array([*state_lla, timestamp])
                array = shared_state_lla.get()
                array[:-1] = array[1:]
                array[-1] = state_lla_with_time

        time.sleep(dt)
