# Autonomous Vehicle Sensor Fusion

**Team Secure Motion** — Politecnico di Milano, May 2026

> **1st Place — EESTEC Engineering Challenge 2026**

A multi-sensor fusion pipeline for tracking an opponent race car relative to an ego vehicle in real-time, developed as part of the **EESTEC Engineering Challenge** at Politecnico di Milano. The team placed **first** in the competition.

## Team

| Name | Role |
|------|------|
| Gagan J H | Team Member |
| Ashkriti Dewan | Team Member |
| Hardik Nayak | Team Member |

---

## Overview

The goal is to fuse LiDAR, radar, and camera data to produce stable, real-time state estimates of an opponent vehicle during a race. The system outputs **range**, **relative heading**, **absolute velocity**, and **uncertainty bounds** for the tracked target.

**Dataset:** 120 seconds of Yas Marina Circuit racing data — 2,388 synchronized frames stored in a ROS2 SQLite3 (`.db3`) bag with CDR-serialized messages, provided by **[PoliMove](https://a2rl.io/team-detail/5/polimove-autonomous-racing-team)**.

**About PoliMove:** PoliMove is the autonomous racing research team of Politecnico di Milano. The team competes in the Indy Autonomous Challenge and the Abu Dhabi Autonomous Racing League (A2RL) at Yas Marina Circuit, and has won the Indy Autonomous Challenge four consecutive times. The dataset used in this project was collected from their autonomous race car at Yas Marina, Abu Dhabi, providing real-world, high-speed racing sensor data.

---

## Sensor Suite

| Sensor | Qty | Role |
|--------|-----|------|
| Camera | 7   | Object classification & visual confirmation |
| LiDAR  | 3   | Spatial geometry & precise range measurement |
| Radar  | 4   | Direct Doppler velocity & robust tracking |

**Active ROS2 Topics:**
- `camera_[fl, fr, cl, cr, rl, rr, r]`
- `lidar_[front, right, left]`
- `radar_[front, back, right, left]`

---

## Software Architecture

```
ROS2 SQLite Database
       |
  ┌────┴────────────────────┐
  ▼                         ▼                   ▼
LiDAR PointCloud2    Radar PointCloud2    Camera JPEGs
       |                    |                   |
 Range Extraction     Radar Decoding      YOLOv8 Detection
       └────────────────────┴──────────────────┘
                            │
                  Extended Kalman Filter
                    [x, y, vx, vy]
                       ▲        │
                  ego.mat        ▼
             Vehicle Dynamics   Dashboard / MP4 Export
```

**Core executables:** [Task2/run_pipeline.py](Task2/run_pipeline.py) · [Task2/ekf_track.py](Task2/ekf_track.py) · [Task2/dashboard.py](Task2/dashboard.py)

---

## Sensor Fusion Mapping

| Sensor Source     | Functional Contribution |
|-------------------|-------------------------|
| Front LiDAR       | Point cluster segmentation → exact target geometry |
| Front Radar       | Doppler radial velocity measurement |
| Camera Array      | YOLOv8 bounding boxes → visual confirmation + position tracking |
| `ego.mat`         | Dynamic vehicle state → EKF kinematic correction |
| Side/Rear Sensors | 360° situational awareness & clutter visualization |

---

## Pipeline

### Phase 1 — Offline Processing
1. Parse raw SQLite bag data
2. Synchronize multi-rate sensor clocks
3. Track spatial LiDAR features (PointCloud2 blob decoding, bounding-box filtering, 20th-percentile forward depth)
4. Log telemetry baselines to CSV
5. Decode radar Doppler returns
6. Compute ego vehicle kinematics (trapezoidal dead reckoning from GPS/IMU)
7. Execute EKF state tracking
8. Export structured results to CSV

### Phase 2 — Live Insights
- Real-time multi-sensor telemetry visualization
- Synchronized 4-panel monitoring dashboard
- MP4 video export support

**Output CSV files:**

| File | Contents |
|------|----------|
| [Task2/all_frames_measurements.csv](Task2/all_frames_measurements.csv) | Raw per-frame sensor measurements |
| [Task2/all_frames_ekf.csv](Task2/all_frames_ekf.csv) | Full EKF state estimates |
| [Task2/all_frames_ekf_submission.csv](Task2/all_frames_ekf_submission.csv) | Submission-format EKF results |

---

## Algorithms

### Baseline: Linear Kalman Filter

Tracks 1D range and range-rate as a sanity baseline.

**State:** `x = [r, ṙ]`

**Kinematics:** `r_{k+1} = r_k + Δt · ṙ_k`

**Prediction:** `x̂⁻ = Ax̂`, `P⁻ = APA^T + Q`

**Correction:** `K = P⁻H^T(HP⁻H^T + R)⁻¹`, `x̂ = x̂⁻ + K(z − Hx̂⁻)`

### Core Engine: Extended Kalman Filter

Full 2D world-frame tracking with asynchronous sensor updates.

**Tracked state:** `x = [x_w, y_w, v_x, v_y]^T`

- **Asynchronous updates:** Separate measurement models `h(x)` for LiDAR range, radar position, and radar radial velocity
- **Ego-pose integration:** Incorporates true vehicle velocity, yaw rate, and absolute heading via trapezoidal dead reckoning
- **Outputs:** 2D world coordinates, absolute target speed, heading vectors, continuous covariance tracking

---

## Tracking Performance

| Metric | Result |
|--------|--------|
| Positional Variance | ~0.3 m (stable) |
| Velocity Variance   | ~0.2 m/s (stable) |
| Target Speed        | True fused velocity |
| Relative Heading    | Active tracking |
| EKF Bounds          | Strictly bounded; minor lateral expansion during high-G turns |

---

## Real-Time Dashboard

A 4-panel synchronized monitoring UI ([Task2/dashboard.py](Task2/dashboard.py)):

- **Vision Pane:** 7-camera mosaic overlaid with active YOLOv8 bounding boxes
- **Spatial Pane:** Bird's-eye-view of all 3 LiDAR point clusters
- **Radar Pane:** Multi-directional live radar scatter return map
- **HUD Overlay:** Range, speed, heading, EKF covariance, and ego vehicle telemetry strips

![Dashboard](demo/dashboard.png)

---

## Key Scripts

| Script | Description |
|--------|-------------|
| [Task2/run_pipeline.py](Task2/run_pipeline.py) | Top-level pipeline runner — ingests bag, runs KF + EKF, exports CSVs |
| [Task2/ekf_track.py](Task2/ekf_track.py) | Extended Kalman Filter implementation |
| [Task2/kalman_track.py](Task2/kalman_track.py) | Linear Kalman Filter baseline |
| [Task2/dashboard.py](Task2/dashboard.py) | Real-time 4-panel visualization dashboard |
| [Task2/radar_decode.py](Task2/radar_decode.py) | Radar PointCloud2 CDR decoding |
| [Task2/ego_velocity.py](Task2/ego_velocity.py) | Ego vehicle kinematics from GPS/IMU |
| [Task2/final_task2.py](Task2/final_task2.py) | Final submission pipeline for Task 2 |
| [animateLidar.py](animateLidar.py) | LiDAR point cloud animation |
| [lidarOnCamera.py](lidarOnCamera.py) | LiDAR point projection onto camera feed |
| [egoViz.py](egoViz.py) | Ego vehicle trajectory and dynamics visualization |
| [radarOccupancyGrid.py](radarOccupancyGrid.py) | Radar occupancy grid mapping |
| [readRos.py](readRos.py) | ROS2 SQLite bag reader and CDR deserializer |
| [task_3_R1.py](task_3_R1.py) | Task 3 implementation |
| [OdometryData.m](OdometryData.m) | MATLAB odometry data processing |
| [Track_Map.m](Track_Map.m) | MATLAB Yas Marina track map generation |

---

## Ego Vehicle Analysis

- **GPS trajectory** covers the full Yas Marina circuit with speed-intensity color coding
- **Heading angle** shows a sharp transition at ~110 s corresponding to the tight hairpin turn
- **IMU acceleration** peaks approach 8 m/s², correlating with wheel speed drops during braking
- **IMU latency** verified stable at sub-14 ms across all 3 IMU units

---

## Edge Cases Resolved

| Issue | Root Cause | Resolution |
|-------|-----------|------------|
| LiDAR alignment | Raw blob header offset error | Fixed offset correction in decoder |
| Radar tracking | Coordinate system axis mismatch | Inverted axes for sign consistency |
| YOLO accuracy | Formula cars absent from training set | Deployed low-floor confidence cascade |
| Visual crashes | Matplotlib colorbar redraws | Initialized static persistent colorbar |

---

## Future Work

**Algorithmic:**
- Multi-LiDAR extrinsic auto-calibration
- Pixel-level LiDAR-camera projection fusion
- Map/track-aware motion priors
- Radar occupancy mapping

**Platform:**
- Fine-tune custom YOLOv8 weights on formula car data
- Real-time C++ inference pipeline rewrite
- RTK-GPS ground-truth verification

---

## Repository Structure

```
.
├── Task2/                  # Task 2: opponent tracking pipeline & results
│   ├── run_pipeline.py
│   ├── ekf_track.py
│   ├── kalman_track.py
│   ├── dashboard.py
│   ├── radar_decode.py
│   ├── ego_velocity.py
│   ├── final_task2.py
│   └── *.csv               # Output telemetry data
├── Papers/                 # Reference papers and sensor datasheets
├── Presentation/           # Final presentation (PDF)
├── demo/                   # Dashboard and visualization screenshots
├── animateLidar.py
├── calibrateLidar.py
├── combineFrontCameras.py
├── egoViz.py
├── lidarOnCamera.py
├── radarOccupancyGrid.py
├── radarOccupancyGridBasic.py
├── readRos.py
├── task_3_R1.py
├── OdometryData.m
└── Track_Map.m
```

---

## Acknowledgements

- **[PoliMove](https://a2rl.io/team-detail/5/polimove-autonomous-racing-team)** — for providing the real-world autonomous racing dataset collected at Yas Marina Circuit, Abu Dhabi. PoliMove is the autonomous racing team of Politecnico di Milano and four-time winner of the Indy Autonomous Challenge.
- **EESTEC LC Milano** — for organizing the Engineering Challenge.
- **Politecnico di Milano** — for hosting the competition.

---

*EESTEC Engineering Challenge 2026 — Politecnico di Milano — 1st Place*
