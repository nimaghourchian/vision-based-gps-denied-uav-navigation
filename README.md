# Rapid Assessment Emergency Response: Vision-Based Navigation in GPS-Denied Environments for UAVs

Bachelor's final project in Mechanical Engineering. This repository contains a cleaned public snapshot of the code used while developing a vision-based navigation framework for UAV operation when GPS measurements are unavailable or unreliable.

The repository now includes the main code paths used across the project: **absolute satellite-map localization, relative visual odometry, sensor preprocessing, Kalman-filter state estimation, MAVLink input, shared-memory communication, simulation support, and visualization**.

A technical overview of the complete project is also available on my [portfolio](https://nimaghourchian.github.io/projects/gps-denied-navigation/).

## Navigation pipeline

### 1. Absolute visual localization

The absolute-localization branch estimates a globally referenced position from a downward-looking UAV image and a georeferenced satellite basemap.

The implemented pipeline is:

1. **Global image retrieval** using a pretrained ResNet-50 backbone and GeM pooling.
2. **Hierarchical FAISS search** to first retrieve likely parent map tiles and then search only their associated subtiles.
3. **Fine local feature matching** using the LiteSAM/LoFTR implementation used during the project.
4. **RANSAC homography estimation** between the UAV image and the selected satellite tile.
5. **Projection of the UAV-image center** into the georeferenced tile.
6. **CRS conversion to WGS84 latitude/longitude** using Rasterio.

The reusable implementation is in `localization/absolute/avl_localizer.py`. An earlier monolithic development version is retained in `experiments/two_stage_resnet_litesam.py`.

### 2. Relative visual odometry

The relative-localization branch estimates short-term image motion between consecutive downward-looking frames using SIFT features, FLANN matching, Lowe's ratio test, and a partial affine transform. Camera field of view and altitude are then used to approximate ground sampling distance and convert pixel displacement into horizontal motion.

Two development paths are retained:

- `preprocessing/visual_odometry_playback.py` for recorded image sequences.
- `preprocessing/visual_odometry_stream.py` for a TCP image stream.

### 3. State estimation and fusion

The main estimator in `core/` is a six-state linear Kalman filter with state

```text
x = [N, E, D, V_N, V_E, V_D]^T
```

and NED acceleration as the control input. Observation models for GPS/NED position, horizontal vision position, and barometric altitude were developed during integration.

The repository also retains `experiments/avl_rvl_fusion.py`, an asynchronous development experiment in which periodic absolute-localization fixes are associated with the corresponding relative-odometry history and used to correct accumulated RVL drift. This is a separate experiment from the Kalman-filter implementation in `core/`.

## Repository structure

```text
.
├── config/                              # YAML configuration and loader
├── core/
│   ├── kalman_filter.py                 # Position/velocity Kalman filter
│   └── state_model.py                   # F, B, H, Q, and R model helpers
├── data/
│   ├── mavlink_input.py                 # Live ArduPilot telemetry input
│   ├── log_playback.py                  # Recorded-log playback
│   └── shared_data.py                   # Named shared-memory wrapper
├── localization/
│   └── absolute/
│       └── avl_localizer.py             # Hierarchical map retrieval + homography geolocation
├── preprocessing/
│   ├── data_preprocess.py               # Coordinate and sensor transforms
│   ├── utils.py                         # Transformation utilities
│   ├── visual_odometry_playback.py      # VO from recorded image frames
│   └── visual_odometry_stream.py        # VO from a TCP image stream
├── outputs/                             # Map publisher and diagnostic plots
├── templates/                           # Leaflet map template
├── simulation/                          # Cesium simulation scene
├── experiments/
│   ├── avl_rvl_fusion.py                # Periodic AVL correction of RVL drift
│   ├── two_stage_resnet_litesam.py      # Earlier monolithic AVL prototype
│   └── ...
├── tools/
│   ├── plot_fused_path.py               # Folium trajectory plotter
│   └── tcp_image_sender.py              # TCP image-stream sender
├── main.py                              # Multiprocess integration entry point
├── requirements.txt
└── .gitignore
```

## External AVL assets

The absolute-localization code depends on research assets that are intentionally **not redistributed in this repository**:

- the LiteSAM/LoFTR source tree imported as `src`;
- the LiteSAM configuration file;
- the trained model checkpoint;
- FAISS indices and tile-name arrays generated from the satellite basemap;
- the georeferenced satellite map tiles themselves.

Their paths are supplied through `AVLConfig`. Large local assets, model checkpoints, cached indices, and map tiles are excluded through `.gitignore`.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The external LiteSAM/LoFTR code and its own required assets must also be available for the absolute-localization modules to run.

## Configuration

Project settings for the ArduPilot/state-estimation side are stored in `config/config.yaml`.

For absolute localization, configure `AVLConfig` with the parent FAISS cache, subtile FAISS cache, georeferenced subtile directory, LiteSAM configuration, and model checkpoint.

## Running

The original multiprocess state-estimation entry point is:

```bash
python main.py
```

The absolute localizer is designed to be imported and configured separately through `AVLLocalizer`. The development fusion experiment can be run after configuring its data and AVL asset paths:

```bash
python experiments/avl_rvl_fusion.py
```

Reproducing the complete system requires the corresponding ArduPilot SITL telemetry, simulation imagery, georeferenced basemap, cached descriptors, and model assets.

## Development status and limitations

This was an undergraduate research prototype, not a flight-certified navigation system. One of the main limitations observed during the complete project was **measurement latency**: absolute visual-localization results could arrive after the UAV had already moved, while the estimator treated accepted measurements as observations of the current state. This timing mismatch could introduce inconsistency and contribute to instability when the fused estimate was used in the closed-loop ArduPilot simulation.

The public snapshot was reorganized from the original development directories, generated caches were removed, machine-specific paths were replaced with repository-relative placeholders/configuration, and support scripts were separated from the core modules. The navigation code remains representative of the undergraduate development work rather than a production software stack.
