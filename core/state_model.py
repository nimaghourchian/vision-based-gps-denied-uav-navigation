import numpy as np


def get_state_model(dt):
    """Constant-velocity state-transition model for [N, E, D, Vn, Ve, Vd]."""
    F = np.eye(6)
    F[0, 3] = F[1, 4] = F[2, 5] = dt
    return F


def get_control_matrix(dt):
    """Map NED acceleration into position and velocity increments."""
    B = np.zeros((6, 3))
    B[0, 0] = B[1, 1] = B[2, 2] = 0.5 * dt**2
    B[3, 0] = B[4, 1] = B[5, 2] = dt
    return B


def get_u(accel=None):
    """Return acceleration as the 3x1 control-input vector."""
    if accel is None:
        return np.zeros((3, 1))
    return np.asarray(accel, dtype=float).reshape(3, 1)


def get_observation_model_gps():
    H = np.zeros((3, 6))
    H[0, 0] = 1
    H[1, 1] = 1
    H[2, 2] = 1
    return H


def get_observation_model_vn():
    H = np.zeros((2, 6))
    H[0, 0] = 1
    H[1, 1] = 1
    return H


def get_observation_model_baro():
    H = np.zeros((1, 6))
    H[0, 2] = 1
    return H


def get_process_noise():
    return np.eye(6)


def get_measurement_noise_gps(pos_std):
    return np.eye(3) * pos_std**2


def get_measurement_noise_vn(vn_std):
    return np.eye(2) * vn_std**2


def get_measurement_noise_baro(baro_std):
    return np.array([[baro_std**2]])


def get_measurement_noise_orientation(angle_std):
    return np.eye(3) * angle_std**2
