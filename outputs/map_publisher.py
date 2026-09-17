import logging

import numpy as np
from flask import Flask, jsonify, render_template

from data.shared_data import SharedArrayWithLock


logging.getLogger("werkzeug").setLevel(logging.ERROR)


def publish_state(gps_name, state_name, shape, dtype, stop_event, live=True):
    """Serve the latest GPS and estimated geographic state through Flask."""
    app = Flask(__name__, template_folder="../templates")

    shared_gps = SharedArrayWithLock(gps_name, shape, dtype, create=False)
    shared_state_lla = SharedArrayWithLock(state_name, shape, dtype, create=False)

    @app.route("/")
    def index():
        return render_template("map.html")

    @app.route("/position")
    def get_position():
        gps_data = shared_gps.read()
        state_data = shared_state_lla.read()

        gps_valid = gps_data[~np.isnan(gps_data[:, 0])]
        state_valid = state_data[~np.isnan(state_data[:, 0])]

        gps_lat = gps_lon = gps_alt = None
        state_lat = state_lon = state_alt = None

        if gps_valid.shape[0] > 0:
            latest = gps_valid[-1]
            gps_lat, gps_lon, gps_alt = map(float, latest[:3])

        if state_valid.shape[0] > 0:
            latest = state_valid[-1]
            state_lat, state_lon, state_alt = map(float, latest[:3])

        return jsonify(
            {
                "gps": {"lat": gps_lat, "lon": gps_lon, "alt": gps_alt},
                "state": {
                    "lat": state_lat,
                    "lon": state_lon,
                    "alt": state_alt,
                },
            }
        )

    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)
