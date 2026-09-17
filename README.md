# Rapid Assessment Emergency Response: Vision-Based Navigation in GPS-Denied Environments for UAVs

Bachelor's final project in Mechanical Engineering. This repository contains a cleaned public snapshot of the code used to develop a vision-based navigation framework for UAV operation when GPS measurements are unavailable or unreliable.

The code included here covers the **relative visual odometry, sensor preprocessing, Kalman-filter state estimation, MAVLink input, shared-memory communication, and live visualization** portions of the project.

> **Scope note:** the complete final project also included an absolute satellite-map localization branch based on image retrieval, local feature matching, RANSAC homography, and conversion of the matched map location to latitude/longitude. That branch was not present in the code archive used to build this repository. A technical overview of the complete project is available on my [portfolio](https://nimaghourchian.github.io/projects/gps-denied-navigation/).

## Included components

- **Relative visual odometry:** SIFT feature detection and description, FLANN matching, Lowe's ratio test, affine motion estimation, image-motion-to-ground-motion conversion, and NED accumulation.
- **State estimation:** a six-state linear Kalman filter for 3D position and velocity with NED acceleration as the control input.
- **Sensor preprocessing:** body-frame acceleration transformation to NED and barometric-pressure-to-altitude conversion.
- **ArduPilot / MAVLink interface:** ingestion of GPS, raw IMU, attitude, and pressure messages from a simulation telemetry stream.
- **Multiprocessing architecture:** shared-memory buffers used to exchange sensor and state data between acquisition, preprocessing, estimation, and visualization processes.
- **Visualization:** a small Flask/Leaflet interface and diagnostic plotting utilities.

## Repository structure

```text
.
├── config/             # YAML configuration and loader
├── core/               # Kalman filter and state-space model
├── data/               # MAVLink input, log playback, shared memory
├── preprocessing/      # Sensor transforms and visual odometry
├── outputs/            # Live map publisher and diagnostic plots
├── templates/          # Leaflet map template
├── main.py             # Multiprocess integration entry point
└── requirements.txt
```

## Kalman-filter model

The filter state is

```text
x = [N, E, D, V_N, V_E, V_D]^T
```

with NED acceleration used as the control input. GPS/NED position, vision-derived horizontal position, and barometric-altitude observation models were developed during integration.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Project settings are stored in `config/config.yaml`. Adjust the MAVLink connection and any local data paths for your own environment before running the complete integration pipeline.

## Running

The integration entry point is:

```bash
python main.py
```

The original development environment used ArduPilot SITL and simulation-specific telemetry/data sources, so reproducing the full run requires the corresponding inputs.

## Development status and limitations

This was an undergraduate research prototype, not a flight-certified navigation system. One of the main limitations observed during the complete project was **measurement latency**: visual-localization results could arrive after the UAV had already moved, while the estimator treated accepted measurements as observations of the current state. This timing mismatch could introduce inconsistency and contribute to instability when the fused estimate was used in the closed-loop ArduPilot simulation.

The public code has been cleaned for portability and readability, but the navigation algorithms remain representative of the development work rather than a production software stack.
