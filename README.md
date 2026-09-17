# Rapid Assessment Emergency Response: Vision-Based Navigation in GPS-Denied Environments for UAVs

Bachelor's final project in Mechanical Engineering. This repository contains a cleaned public snapshot of the code used while developing a vision-based navigation framework for UAV operation when GPS measurements are unavailable or unreliable.

The code included here covers the **relative visual odometry, sensor preprocessing, Kalman-filter state estimation, MAVLink input, shared-memory communication, simulation support, and live visualization** portions of the project.

> **Scope note:** the complete final project also included an absolute satellite-map localization branch based on image retrieval, local feature matching, RANSAC homography, and conversion of the matched map location to latitude/longitude. That branch was not present in the code archive used to build this repository. A technical overview of the complete project is available on my [portfolio](https://nimaghourchian.github.io/projects/gps-denied-navigation/).

## Included components

- **Relative visual odometry:** SIFT feature detection and description, FLANN matching, Lowe's ratio test, partial-affine motion estimation, image-motion-to-ground-motion conversion, and NED trajectory accumulation.
- **State estimation:** a six-state linear Kalman filter for 3D position and velocity, with NED acceleration used as the control input.
- **Sensor preprocessing:** body-frame acceleration transformation to NED, geodetic-to-NED conversion, and barometric-pressure-to-altitude conversion.
- **ArduPilot / MAVLink interface:** ingestion of GPS, raw IMU, attitude, and pressure messages from a simulation telemetry stream.
- **Multiprocessing architecture:** shared-memory buffers used to exchange sensor and state data between acquisition, preprocessing, estimation, and visualization processes.
- **Visualization:** a Flask/Leaflet interface for comparing GPS and estimated position, plus diagnostic NED plotters.
- **Simulation support:** the Cesium scene/camera asset used during development.

## Repository structure

```text
.
├── config/                         # YAML configuration and loader
├── core/
│   ├── kalman_filter.py            # Position/velocity Kalman filter
│   └── state_model.py              # F, B, H, Q, and R model helpers
├── data/
│   ├── mavlink_input.py            # Live ArduPilot telemetry input
│   ├── log_playback.py             # Recorded-log playback
│   └── shared_data.py              # Named shared-memory wrapper
├── preprocessing/
│   ├── data_preprocess.py          # Coordinate and sensor transforms
│   ├── utils.py                    # Transformation utilities
│   ├── visual_odometry_playback.py # VO from recorded image frames
│   └── visual_odometry_stream.py   # VO from a TCP image stream
├── outputs/                        # Map publisher and diagnostic plots
├── templates/                      # Leaflet map template
├── simulation/                     # Cesium simulation scene
├── experiments/                    # Small development/legacy utilities
├── main.py                         # Multiprocess integration entry point
├── requirements.txt
└── .gitignore
```

## Relative visual odometry

Two development paths are retained because they were used at different stages of the project:

- `visual_odometry_playback.py` processes a recorded sequence of downward-looking frames.
- `visual_odometry_stream.py` receives JPEG frames over TCP for online processing.

Both use SIFT features, FLANN matching, Lowe's ratio test, and an estimated partial affine transform to obtain inter-frame image motion. Camera field of view and altitude are then used to approximate ground sampling distance and convert pixel displacement into horizontal motion.

These modules are retained as separate development components; the checked-in `main.py` does **not** automatically launch a visual-odometry worker.

## Kalman-filter model

The state vector is

```text
x = [N, E, D, V_N, V_E, V_D]^T
```

with NED acceleration used as the control input. The repository contains observation models for GPS position, horizontal vision position, and barometric altitude. In the checked-in integration snapshot, GPS and barometric updates are active while the visual-position update path is documented but disabled.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Project settings are stored in `config/config.yaml`. Adjust the MAVLink connection and the recorded-frame directory for your environment before running the corresponding components.

## Running

The integration entry point is:

```bash
python main.py
```

The original development environment used ArduPilot SITL and simulation-specific telemetry/data sources, so reproducing the complete run requires the corresponding inputs.

## Development status and limitations

This was an undergraduate research prototype, not a flight-certified navigation system. One of the main limitations observed during the complete project was **measurement latency**: absolute visual-localization results could arrive after the UAV had already moved, while the estimator treated accepted measurements as observations of the current state. This timing mismatch could introduce inconsistency and contribute to instability when the fused estimate was used in the closed-loop ArduPilot simulation.

The public snapshot was reorganized from the original development directory, generated caches and duplicate files were removed, machine-specific paths were replaced with configuration entries, comments were standardized, and several obvious execution/portability issues were cleaned up. The navigation code remains representative of the undergraduate development work rather than a production software stack.
