from flask import Flask, jsonify
import numpy as np

from data.shared_data import SharedArrayWithLock


app = Flask(__name__)

# Legacy development server for the relative-visual-navigation buffer.
shared_vn = SharedArrayWithLock("vn", (20, 3), np.float64, create=False)
shared_gps = SharedArrayWithLock("gps_raw", (20, 4), np.float64, create=False)


@app.get("/data")
def live_data():
    with shared_vn.lock:
        vn_array = shared_vn.get()[-1].copy()
        vn_lat = float(vn_array[0])
        vn_lon = float(vn_array[1])

    with shared_gps.lock:
        gps_array = shared_gps.get()[-1].copy()
        gps_lat = float(gps_array[0])
        gps_lon = float(gps_array[1])

    return jsonify(
        {
            "vn_lla_lat": vn_lat,
            "vn_lla_lon": vn_lon,
            "gps_lat": gps_lat,
            "gps_lon": gps_lon,
        }
    )


@app.get("/")
def index():
    with open("templates/map.html", encoding="utf-8") as file:
        return file.read()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
