import time

import matplotlib.pyplot as plt
import numpy as np

from data.shared_data import SharedArrayWithLock


def plot_gps_ned(
    shm_name="gps_ned",
    shape=(20, 4),
    dtype=np.float64,
    refresh_rate=5,
):
    """Live 2D plot of the accumulated GPS north/east path."""
    shared_gps_ned = SharedArrayWithLock(
        shm_name, shape=shape, dtype=dtype, create=False
    )

    plt.ion()
    _, axis = plt.subplots(figsize=(6, 6))
    axis.set_title("GPS N-E Path")
    axis.set_xlabel("East [m]")
    axis.set_ylabel("North [m]")
    axis.grid(True)

    line_path, = axis.plot([], [], "o-")
    all_north = []
    all_east = []

    try:
        while True:
            data = shared_gps_ned.read()
            valid_data = data[~np.isnan(data).any(axis=1)]
            if valid_data.shape[0] == 0:
                time.sleep(1 / refresh_rate)
                continue

            north, east = valid_data[-1, :2]
            all_north.append(north)
            all_east.append(east)

            line_path.set_xdata(all_east)
            line_path.set_ydata(all_north)
            axis.relim()
            axis.autoscale_view()
            plt.pause(1 / refresh_rate)

    except KeyboardInterrupt:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    plot_gps_ned()
