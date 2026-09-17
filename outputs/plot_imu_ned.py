import time

import matplotlib.pyplot as plt
import numpy as np

from data.shared_data import SharedArrayWithLock


def plot_imu_ned(
    shm_name="imu_ned",
    shape=(20, 4),
    dtype=np.float64,
    refresh_rate=10,
):
    """Live plot of north/east/down acceleration versus time."""
    shared_imu_ned = SharedArrayWithLock(
        shm_name, shape=shape, dtype=dtype, create=False
    )

    plt.ion()
    figure = plt.figure(figsize=(8, 8))
    axis_n = figure.add_subplot(311)
    axis_e = figure.add_subplot(312, sharex=axis_n)
    axis_d = figure.add_subplot(313, sharex=axis_n)

    axis_n.set_ylabel("North acc [m/s²]")
    axis_e.set_ylabel("East acc [m/s²]")
    axis_d.set_ylabel("Down acc [m/s²]")
    axis_d.set_xlabel("Time [s]")

    for axis in (axis_n, axis_e, axis_d):
        axis.grid(True)

    line_n, = axis_n.plot([], [])
    line_e, = axis_e.plot([], [])
    line_d, = axis_d.plot([], [])

    times, north, east, down = [], [], [], []

    try:
        while True:
            data = shared_imu_ned.read()
            valid_data = data[~np.isnan(data).any(axis=1)]
            if valid_data.shape[0] == 0:
                time.sleep(1 / refresh_rate)
                continue

            acc_n, acc_e, acc_d, timestamp = valid_data[-1]
            times.append(timestamp)
            north.append(acc_n)
            east.append(acc_e)
            down.append(acc_d)

            line_n.set_data(times, north)
            line_e.set_data(times, east)
            line_d.set_data(times, down)

            for axis in (axis_n, axis_e, axis_d):
                axis.relim()
                axis.autoscale_view()

            plt.pause(1 / refresh_rate)

    except KeyboardInterrupt:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    plot_imu_ned()
