"""
ekf_track.py
------------
Extended Kalman Filter (EKF) for target vehicle tracking.

Improvements over the linear KF in kalman_track.py:
  1. Ego-velocity-aware: reads ego speed / heading from ego_interp.csv
     to integrate ego pose in world frame at each step.
  2. Fuses three measurement types per frame:
       a. Lidar range          – nonlinear in world-frame Cartesian state → EKF
       b. Radar body position  – linear (rotation of world state)
       c. Radar absolute radial velocity – nonlinear in state → EKF
  3. World-frame Cartesian state [xt, yt, vxt, vyt] with EKF measurement
     Jacobians for the nonlinear observations.

Radar velocity convention (verified from data):
  The front radar sensor on this vehicle reports *absolute* (ego-compensated)
  target radial velocity in m/s, positive = moving away from ego.

Inputs:
    task2_measurements.csv   (lidar distances + radar timestamps per camera frame)
    ego_interp.csv           (ego vx, vy, heading, ax per camera frame)
    ../Data/2025-10-08_09-35_sensors_raw_1.db3

Outputs:
    task2_ekf.csv            per-frame state: range, speed, heading, kf internals
    task2_ekf_submission.csv final single-row answer: position_m, velocity_m_s,
                                                      heading_deg, confidence
"""

import sqlite3
import numpy as np
import pandas as pd

# ==========================
# CONFIG
# ==========================

DB           = "../Data/2025-10-08_09-35_sensors_raw_1.db3"
RADAR_TOPIC  = "/sensor/radar_front/points"
RADAR_OFFSET = 220
RADAR_STEP   = 32

# EKF noise parameters
# Process noise (constant-velocity model; target can accelerate ±Q_v m/s² per step)
Q_POS = 0.05    # position process noise [m²]
Q_VEL = 0.50    # velocity process noise [(m/s)²]

# Measurement noise (standard deviations then squared to get variances)
R_LIDAR      = 0.40 ** 2   # lidar range: ~0.4 m std  (20th-pct method is noisy)
R_RADAR_XY   = 0.20 ** 2   # radar body position: ~0.2 m std
R_RADAR_VEL  = 0.50 ** 2   # radar radial velocity: ~0.5 m/s std

# Initial state covariance
P0_POS = 1.0 ** 2   # ±1 m initial position uncertainty
P0_VEL = 2.0 ** 2   # ±2 m/s initial velocity uncertainty

# ==========================
# RADAR DTYPE
# ==========================

RADAR_DTYPE = np.dtype([
    ("x",          "<f4"),
    ("y",          "<f4"),
    ("z",          "<f4"),
    ("velocity",   "<f4"),
    ("snr",        "<f4"),
    ("rcs",        "<f4"),
    ("confidence", "<f4"),
    ("vint",       "<f4"),
])


# ==========================
# RADAR DECODE HELPER
# ==========================

def decode_radar_target(raw):
    """
    Decode a radar message blob.
    Returns (x_body, y_body, velocity_abs, confidence) for the closest
    high-confidence forward target, or None if no such target exists.

    velocity_abs: absolute radial velocity [m/s], positive = moving away.
    """
    payload = raw[RADAR_OFFSET:]
    usable  = (len(payload) // RADAR_STEP) * RADAR_STEP
    payload = payload[:usable]
    if usable == 0:
        return None

    pts = np.frombuffer(payload, dtype=RADAR_DTYPE)

    x    = pts["x"].astype(float)
    y    = pts["y"].astype(float)
    v    = pts["velocity"].astype(float)
    conf = pts["confidence"].astype(float)

    # Forward corridor: 3–20 m ahead, ±3.5 m lateral, confidence threshold
    mask = (x > 3.0) & (x < 20.0) & (np.abs(y) < 3.5) & (conf > 10.0)
    if not mask.any():
        return None

    x, y, v, conf = x[mask], y[mask], v[mask], conf[mask]

    # Pick the closest target
    r       = np.sqrt(x ** 2 + y ** 2)
    nearest = np.argmin(r)

    return (
        float(x[nearest]),
        float(y[nearest]),
        float(v[nearest]),
        float(conf[nearest]),
    )


# ==========================
# LOAD INPUTS
# ==========================

meas = pd.read_csv("task2_measurements.csv")
ego  = pd.read_csv("ego_interp.csv")

cam_ts     = meas["camera"].values.astype(float)
lidar_dist = meas["distance_m"].values.astype(float)
radar_ts   = meas["radar"].values.astype(int)

vx_ego   = ego["vx"].values.astype(float)       # body-frame longitudinal [m/s]
vy_ego   = ego["vy"].values.astype(float)       # body-frame lateral      [m/s]
psi      = ego["heading"].values.astype(float)  # world-frame heading [rad], unwrapped

N = len(cam_ts)

# ==========================
# QUERY RADAR FROM DB
# ==========================

conn     = sqlite3.connect(DB)
topic_id = conn.execute(
    "SELECT id FROM topics WHERE name=?", (RADAR_TOPIC,)
).fetchone()[0]

radar_cache = {}   # radar_ts → (x_b, y_b, velocity, confidence) or None

for rts in set(radar_ts):
    row = conn.execute(
        "SELECT data FROM messages WHERE topic_id=? AND timestamp=? LIMIT 1",
        (topic_id, int(rts)),
    ).fetchone()
    radar_cache[rts] = decode_radar_target(row[0]) if row else None

conn.close()

# Per-frame radar results
radar_meas = [radar_cache[rts] for rts in radar_ts]

# ==========================
# EGO POSE INTEGRATION
# ==========================
# Integrate ego position in a local world frame (ego starts at origin).
# Rotation convention: body→world uses the heading (yaw) angle:
#   x_world = x_body * cos(psi) - y_body * sin(psi)
#   y_world = x_body * sin(psi) + y_body * cos(psi)

xe = np.zeros(N)   # ego world-x position
ye = np.zeros(N)   # ego world-y position

for i in range(1, N):
    dt = (cam_ts[i] - cam_ts[i - 1]) / 1e9          # seconds
    c, s = np.cos(psi[i - 1]), np.sin(psi[i - 1])
    # Body → world velocity transform
    vxw = c * vx_ego[i - 1] - s * vy_ego[i - 1]
    vyw = s * vx_ego[i - 1] + c * vy_ego[i - 1]
    xe[i] = xe[i - 1] + vxw * dt
    ye[i] = ye[i - 1] + vyw * dt


# ==========================
# EKF INITIALISATION
# ==========================
# State: x = [xt_w, yt_w, vxt_w, vyt_w]  (world-frame Cartesian)

def init_state(r0, psi0, xe0, ye0, vr0):
    """
    Initialise from first lidar range r0 and first radar velocity vr0.
    Assume target is directly ahead (bearing = 0 in body frame).
    """
    c, s = np.cos(psi0), np.sin(psi0)

    # Target world position: ahead along ego heading at range r0
    xt0 = xe0 + r0 * c
    yt0 = ye0 + r0 * s

    # Target world velocity: it travels in the same direction as ego heading
    # with the radar-reported absolute speed.
    vxt0 = vr0 * c
    vyt0 = vr0 * s

    return np.array([xt0, yt0, vxt0, vyt0])


# Choose first valid radar velocity for initialisation
v_init = 14.75   # sensible default (close to ego speed at window start)
for rm in radar_meas:
    if rm is not None:
        v_init = rm[2]   # absolute radial velocity
        break

x_est = init_state(
    r0   = lidar_dist[0],
    psi0 = psi[0],
    xe0  = xe[0],
    ye0  = ye[0],
    vr0  = v_init,
)

P_est = np.diag([P0_POS, P0_POS, P0_VEL, P0_VEL])

# ==========================
# EKF HELPERS
# ==========================

def predict(x, P, dt):
    """
    Constant-velocity prediction step.
    F is linear, but we keep the EKF formulation for consistency.
    """
    F = np.array([
        [1., 0., dt,  0.],
        [0., 1.,  0., dt],
        [0., 0.,  1.,  0.],
        [0., 0.,  0.,  1.],
    ])
    Q = np.diag([Q_POS, Q_POS, Q_VEL, Q_VEL]) * dt

    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


def update_lidar(x, P, z_range, xe_k, ye_k):
    """
    EKF update: lidar range measurement (nonlinear in state).
    h(x) = sqrt((xt - xe)^2 + (yt - ye)^2)
    """
    dx = x[0] - xe_k
    dy = x[1] - ye_k
    r  = max(np.sqrt(dx ** 2 + dy ** 2), 0.1)   # guard div/0

    h = r
    innovation = z_range - h

    # Measurement Jacobian H (1×4)
    H = np.array([[dx / r, dy / r, 0., 0.]])

    S  = H @ P @ H.T + R_LIDAR
    K  = (P @ H.T) / S[0, 0]          # (4,1) Kalman gain
    x  = x + K[:, 0] * innovation
    P  = (np.eye(4) - np.outer(K[:, 0], H)) @ P
    return x, P


def update_radar_pos(x, P, x_b_meas, y_b_meas, psi_k, xe_k, ye_k):
    """
    EKF update: radar body-frame position (x_b, y_b).
    These are linear in the world-frame state after the known rotation psi_k:
        h_xb = cos(psi)*(xt-xe) + sin(psi)*(yt-ye)
        h_yb = -sin(psi)*(xt-xe) + cos(psi)*(yt-ye)
    Linear → standard KF update, no Jacobian needed beyond H.
    """
    c, s = np.cos(psi_k), np.sin(psi_k)

    dx = x[0] - xe_k
    dy = x[1] - ye_k
    h_xb = c * dx + s * dy
    h_yb = -s * dx + c * dy

    # Measurement Jacobian (2×4)
    H = np.array([
        [ c,  s, 0., 0.],
        [-s,  c, 0., 0.],
    ])

    z     = np.array([x_b_meas, y_b_meas])
    innov = z - np.array([h_xb, h_yb])
    R     = np.eye(2) * R_RADAR_XY
    S     = H @ P @ H.T + R
    K     = P @ H.T @ np.linalg.inv(S)

    x = x + K @ innov
    P = (np.eye(4) - K @ H) @ P
    return x, P


def update_radar_vel(x, P, z_vel_abs, xe_k, ye_k):
    """
    EKF update: absolute radial velocity measurement (nonlinear in state).
    h(x) = (vxt*(xt-xe) + vyt*(yt-ye)) / r

    This is the radial component of the target's absolute world-frame velocity,
    consistent with what the radar reports (ego-compensated Doppler).

    Jacobian H (1×4):
        ∂h/∂xt  = (vxt*dy² - vyt*dx*dy) / r³
        ∂h/∂yt  = (vyt*dx² - vxt*dx*dy) / r³
        ∂h/∂vxt = dx / r
        ∂h/∂vyt = dy / r
    """
    dx   = x[0] - xe_k
    dy   = x[1] - ye_k
    r    = max(np.sqrt(dx ** 2 + dy ** 2), 0.1)
    vxt  = x[2]
    vyt  = x[3]

    h         = (vxt * dx + vyt * dy) / r
    innovation = z_vel_abs - h

    H = np.array([[
        (vxt * dy ** 2 - vyt * dx * dy) / r ** 3,   # ∂h/∂xt
        (vyt * dx ** 2 - vxt * dx * dy) / r ** 3,   # ∂h/∂yt
        dx / r,                                       # ∂h/∂vxt
        dy / r,                                       # ∂h/∂vyt
    ]])

    S  = H @ P @ H.T + R_RADAR_VEL
    K  = (P @ H.T) / S[0, 0]
    x  = x + K[:, 0] * innovation
    P  = (np.eye(4) - np.outer(K[:, 0], H)) @ P
    return x, P


# ==========================
# EKF MAIN LOOP
# ==========================

rows = []
confidences = []

for i in range(N):

    # -- Predict (skip on first step; state already initialised) --
    if i > 0:
        dt     = (cam_ts[i] - cam_ts[i - 1]) / 1e9
        x_est, P_est = predict(x_est, P_est, dt)

    # -- Update: lidar range --
    z_lidar = lidar_dist[i]
    if not np.isnan(z_lidar) and z_lidar > 0:
        x_est, P_est = update_lidar(x_est, P_est, z_lidar, xe[i], ye[i])

    # -- Update: radar position + velocity --
    rm = radar_meas[i]
    if rm is not None:
        x_b, y_b, v_abs, conf = rm

        x_est, P_est = update_radar_pos(
            x_est, P_est, x_b, y_b, psi[i], xe[i], ye[i]
        )
        x_est, P_est = update_radar_vel(
            x_est, P_est, v_abs, xe[i], ye[i]
        )
        confidences.append(conf)

    # -- Derived quantities --
    dx  = x_est[0] - xe[i]
    dy  = x_est[1] - ye[i]
    rng = np.sqrt(dx ** 2 + dy ** 2)

    spd = np.sqrt(x_est[2] ** 2 + x_est[3] ** 2)   # absolute target speed

    # Target heading in world frame, then relative to ego heading
    tgt_heading_world = np.degrees(np.arctan2(x_est[3], x_est[2]))
    ego_heading_world = np.degrees(psi[i] % (2 * np.pi))
    heading_rel       = (tgt_heading_world - ego_heading_world + 180) % 360 - 180

    rows.append({
        "camera":           int(cam_ts[i]),
        "ego_x":            round(xe[i],       3),
        "ego_y":            round(ye[i],       3),
        "target_x_world":   round(x_est[0],    3),
        "target_y_world":   round(x_est[1],    3),
        "target_vx_world":  round(x_est[2],    3),
        "target_vy_world":  round(x_est[3],    3),
        "range_m":          round(rng,          3),
        "speed_m_s":        round(spd,          3),
        "heading_rel_deg":  round(heading_rel,  2),
        "pos_std_m":        round(np.sqrt((P_est[0,0]+P_est[1,1])/2), 4),
        "vel_std_m_s":      round(np.sqrt((P_est[2,2]+P_est[3,3])/2), 4),
    })

# ==========================
# SAVE PER-FRAME RESULTS
# ==========================

df = pd.DataFrame(rows)

print()
print("EKF track (per frame):")
print()
print(df[[
    "camera", "range_m", "speed_m_s", "heading_rel_deg",
    "pos_std_m", "vel_std_m_s"
]].to_string(index=False))
print()

df.to_csv("task2_ekf.csv", index=False)
print("saved task2_ekf.csv")
print()

# ==========================
# FINAL SUBMISSION OUTPUT
# ==========================

# Use the last EKF estimate (most refined after all measurements)
final = df.iloc[-1]
mean_conf = float(np.mean(confidences)) if confidences else 39.9

out = pd.DataFrame({
    "position_m":   [round(final.range_m,          2)],
    "velocity_m_s": [round(final.speed_m_s,         2)],
    "heading_deg":  [round(final.heading_rel_deg,    1)],
    "confidence":   [round(mean_conf,                1)],
})

print("FINAL TRACK (EKF + ego-velocity compensation):")
print()
print(out.to_string(index=False))
print()

out.to_csv("task2_ekf_submission.csv", index=False)
print("saved task2_ekf_submission.csv")
print()

# Compare with original linear KF
print("── Comparison ──────────────────────────────────")
print(f"  Linear KF  position : 7.56 m   velocity : 14.75 m/s  heading : 0°")
print(f"  EKF + ego  position : {final.range_m:.2f} m   velocity : {final.speed_m_s:.2f} m/s  heading : {final.heading_rel_deg:.1f}°")
print()
