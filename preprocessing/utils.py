import numpy as np


def accel_transform(a_b, phi, theta, psi):
    """Transform a body-frame acceleration vector into NED and remove gravity."""
    c_b_n = np.array(
        [
            [
                np.cos(theta) * np.cos(psi),
                np.cos(theta) * np.sin(psi),
                -np.sin(theta),
            ],
            [
                np.sin(phi) * np.sin(theta) * np.cos(psi)
                - np.cos(phi) * np.sin(psi),
                np.sin(phi) * np.sin(theta) * np.sin(psi)
                + np.cos(phi) * np.cos(psi),
                np.sin(phi) * np.cos(theta),
            ],
            [
                np.cos(phi) * np.sin(theta) * np.cos(psi)
                + np.sin(phi) * np.sin(psi),
                np.cos(phi) * np.sin(theta) * np.sin(psi)
                - np.sin(phi) * np.cos(psi),
                np.cos(phi) * np.cos(theta),
            ],
        ]
    )

    a_n = c_b_n.T @ a_b
    gravity_ned = np.array([[0.0], [0.0], [-9.81]])
    return a_n - gravity_ned


def pressure_to_altitude(pressure_hpa, p0=1013.25):
    """Convert pressure to relative barometric altitude using a standard model."""
    if pressure_hpa <= 0:
        return None
    return 44330.0 * (1.0 - (pressure_hpa / p0) ** 0.1903)


def add_to_queue(array, new_data, lock=None):
    """Append one sample to a fixed-size rolling NumPy buffer."""
    if lock is not None:
        with lock:
            array[:-1] = array[1:]
            array[-1] = new_data
    else:
        array[:-1] = array[1:]
        array[-1] = new_data
