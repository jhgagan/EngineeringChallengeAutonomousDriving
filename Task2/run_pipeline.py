"""
run_pipeline.py
---------------
Single entry point. Run with:

    python3 run_pipeline.py          (from Task2/ directory)

What it does
------------
1.  Reads ALL camera-front timestamps from the database (2 388 frames / 120 s).
2.  Matches each camera frame to its nearest lidar and radar timestamp.
3.  Extracts lidar range (20th-percentile forward distance) for every frame.
    → cached in  all_frames_measurements.csv  (re-used on subsequent runs).
4.  Runs the LINEAR KF (range-only, constant-velocity model).
5.  Loads ego velocity / heading from ego.mat and interpolates at each frame.
6.  Decodes radar (body-frame position + absolute radial velocity) per frame.
7.  Integrates ego pose in world frame.
8.  Runs the EXTENDED KALMAN FILTER (nonlinear lidar + radar measurements).
9.  Saves all_frames_lkf.csv, all_frames_ekf.csv,
        all_frames_ekf_submission.csv.
10. Generates kf_vs_ekf_comparison_all.png.

Notes
-----
- Lidar blobs are ~830 KB each.  First run reads ~1.9 GB and takes ~90 s on
  WSL/D-drive.  After that, the cached CSV makes steps 3-onwards instant.
- Radar blobs are ~16 KB each; fetching them takes a few seconds.
"""

import os, sys, time
import sqlite3
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

# All paths are relative to the script's own directory so the script
# can be executed from any working directory.
_HERE       = os.path.dirname(os.path.abspath(__file__))
_DATA       = os.path.join(_HERE, "..", "Data")

DB          = os.path.join(_DATA, "2025-10-08_09-35_sensors_raw_1.db3")
MAT         = os.path.join(_DATA, "ego.mat")
CAM_TOPIC   = "/sensor/camera/camera_fr/image/compressed"
LIDAR_TOPIC = "/sensor/lidar_front/points"
RADAR_TOPIC = "/sensor/radar_front/points"

MEAS_CACHE  = os.path.join(_HERE, "all_frames_measurements.csv")   # lidar range cache

# Lidar blob layout (from task2_pipeline.py)
LIDAR_OFFSET = 244
LIDAR_STEP   = 32
LIDAR_DTYPE  = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("skip","V20")])

# Radar blob layout (from radar_decode.py)
RADAR_OFFSET = 220
RADAR_STEP   = 32
RADAR_DTYPE  = np.dtype([
    ("x","<f4"),("y","<f4"),("z","<f4"),
    ("velocity","<f4"),("snr","<f4"),("rcs","<f4"),
    ("confidence","<f4"),("vint","<f4"),
])

# Linear KF noise  (from kalman_track.py)
LKF_Q = np.array([[0.01, 0.0], [0.0, 0.01]])
LKF_R = 0.1

# EKF noise  (from ekf_track.py)
Q_POS, Q_VEL    = 0.05, 0.50
R_LIDAR         = 0.40 ** 2
R_RADAR_XY      = 0.20 ** 2
R_RADAR_VEL     = 0.50 ** 2
P0_POS, P0_VEL  = 1.0 ** 2, 2.0 ** 2

# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def nearest_ts(query, pool):
    """Return pool[nearest_index] for each value in query (both int64 arrays)."""
    idx  = np.searchsorted(pool, query)
    idx  = np.clip(idx, 0, len(pool) - 1)
    left = np.clip(idx - 1, 0, len(pool) - 1)
    use_left = np.abs(pool[left] - query) < np.abs(pool[idx] - query)
    idx[use_left] = left[use_left]
    return pool[idx]


def batch_fetch(conn, topic_id, ts_set):
    """Return {int_ts: blob} for every timestamp in ts_set."""
    result   = {}
    ts_list  = list(ts_set)
    chunk_sz = 500
    for i in range(0, len(ts_list), chunk_sz):
        sub = ts_list[i : i + chunk_sz]
        ph  = ",".join(["?"] * len(sub))
        for ts, data in conn.execute(
            f"SELECT timestamp, data FROM messages"
            f" WHERE topic_id=? AND timestamp IN ({ph})",
            [topic_id] + sub,
        ).fetchall():
            result[int(ts)] = data
    return result


def extract_lidar_range(raw):
    """20th-percentile forward distance from a raw lidar blob (task2_pipeline logic)."""
    payload = raw[LIDAR_OFFSET:]
    usable  = (len(payload) // LIDAR_STEP) * LIDAR_STEP
    if usable == 0:
        return -1.0
    pts  = np.frombuffer(payload[:usable], dtype=LIDAR_DTYPE)
    x, y = pts["x"], pts["y"]
    mask = (x > 0) & (x < 40) & (np.abs(y) < 8)
    d    = np.sqrt(x[mask] ** 2 + y[mask] ** 2)
    d    = d[d > 3]
    if len(d) == 0:
        return -1.0
    return round(float(np.percentile(d, 20)), 2)


def decode_radar_target(raw):
    """(x_b, y_b, velocity_abs, confidence) for closest forward target, or None."""
    payload = raw[RADAR_OFFSET:]
    usable  = (len(payload) // RADAR_STEP) * RADAR_STEP
    if usable == 0:
        return None
    pts  = np.frombuffer(payload[:usable], dtype=RADAR_DTYPE)
    x    = pts["x"].astype(float)
    y    = pts["y"].astype(float)
    v    = pts["velocity"].astype(float)
    conf = pts["confidence"].astype(float)
    mask = (x > 3.0) & (x < 20.0)
    if not mask.any():
        return None
    r       = np.sqrt(x[mask] ** 2 + y[mask] ** 2)
    nearest = np.argmin(r)
    return float(x[mask][nearest]), float(y[mask][nearest]), \
           float(v[mask][nearest]), float(conf[mask][nearest])


# ═══════════════════════════════════════════════════════════════════
# STEP 1-2 : TIMESTAMPS
# ═══════════════════════════════════════════════════════════════════

print("\n━━━  STEP 1 / 2  Timestamps  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
t_start = time.time()

conn = sqlite3.connect(DB)

def topic_id(name):
    return conn.execute("SELECT id FROM topics WHERE name=?", (name,)).fetchone()[0]

cam_id   = topic_id(CAM_TOPIC)
lid_id   = topic_id(LIDAR_TOPIC)
rad_id   = topic_id(RADAR_TOPIC)

def all_ts(tid):
    return np.array(
        [r[0] for r in conn.execute(
            "SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,)
        ).fetchall()],
        dtype=np.int64,
    )

cam_ts    = all_ts(cam_id)
lidar_all = all_ts(lid_id)
radar_all = all_ts(rad_id)

matched_lidar = nearest_ts(cam_ts, lidar_all)
matched_radar = nearest_ts(cam_ts, radar_all)

N = len(cam_ts)
print(f"  camera : {N} frames  ({(cam_ts[-1]-cam_ts[0])/1e9:.1f} s)")
print(f"  lidar  : {len(lidar_all)} frames   radar: {len(radar_all)} frames")
print(f"  unique lidar ts: {len(set(matched_lidar.tolist()))}"
      f"   unique radar ts: {len(set(matched_radar.tolist()))}")

# ═══════════════════════════════════════════════════════════════════
# STEP 3 : LIDAR RANGES  (with CSV cache)
# ═══════════════════════════════════════════════════════════════════

print("\n━━━  STEP 3  Lidar ranges  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

if os.path.exists(MEAS_CACHE):
    print(f"  Cache found: {MEAS_CACHE}  – skipping blob read.")
    _cache       = pd.read_csv(MEAS_CACHE)
    lidar_range  = _cache["distance_m"].values.astype(float)
    print(f"  {(lidar_range > 0).sum()} valid lidar readings loaded from cache.")
else:
    print(f"  No cache – fetching {len(set(matched_lidar.tolist()))} lidar blobs"
          f"  (~1.9 GB, may take ~90 s on WSL/D-drive) …")
    t0 = time.time()
    lidar_blobs = batch_fetch(conn, lid_id, set(matched_lidar.tolist()))
    print(f"  Blobs fetched in {time.time()-t0:.1f} s")

    t1 = time.time()
    lidar_range = np.array([
        extract_lidar_range(lidar_blobs[int(ts)]) if int(ts) in lidar_blobs else -1.0
        for ts in matched_lidar
    ])
    del lidar_blobs
    print(f"  Ranges extracted in {time.time()-t1:.1f} s"
          f"  ({(lidar_range>0).sum()} valid)")

    # save cache
    pd.DataFrame({
        "camera":     cam_ts,
        "lidar_ts":   matched_lidar,
        "radar_ts":   matched_radar,
        "distance_m": lidar_range,
    }).to_csv(MEAS_CACHE, index=False)
    print(f"  Saved cache → {MEAS_CACHE}")

# ═══════════════════════════════════════════════════════════════════
# STEP 4 : LINEAR KF  (kalman_track.py logic)
# ═══════════════════════════════════════════════════════════════════

print("\n━━━  STEP 4  Linear KF  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

first_v = np.where(lidar_range > 0)[0]
x_lkf   = np.array([lidar_range[first_v[0]], 0.0])
P_lkf   = np.eye(2)
lkf_dist = np.full(N, np.nan)
lkf_vel  = np.full(N, np.nan)

for i in range(N):
    if i > 0:
        dt    = (cam_ts[i] - cam_ts[i - 1]) / 1e9
        A     = np.array([[1., dt], [0., 1.]])
        x_lkf = A @ x_lkf
        P_lkf = A @ P_lkf @ A.T + LKF_Q

    z = lidar_range[i]
    if z > 0:
        H     = np.array([[1., 0.]])
        y     = np.array([z]) - (H @ x_lkf)        # shape (1,)
        S     = H @ P_lkf @ H.T + LKF_R             # shape (1,1)
        K     = P_lkf @ H.T @ np.linalg.inv(S)      # shape (2,1)
        x_lkf = x_lkf + (K @ y)                     # shape (2,)
        P_lkf = (np.eye(2) - K @ H) @ P_lkf         # shape (2,2)

    lkf_dist[i] = x_lkf[0]
    lkf_vel[i]  = x_lkf[1]

print(f"  Done.  Final: dist={lkf_dist[-1]:.2f} m  range-rate={lkf_vel[-1]:.3f} m/s")

# ═══════════════════════════════════════════════════════════════════
# STEP 5 : EGO STATE FROM ego.mat  (ego_velocity.py logic)
# ═══════════════════════════════════════════════════════════════════

print("\n━━━  STEP 5  Ego state from ego.mat  ━━━━━━━━━━━━━━━━━━━━━━")

f           = h5py.File(MAT, "r")
est         = f["log/estimation"]
bag_stamp   = est["bag_stamp"][0]
vx_mat      = est["vx"][0]
vy_mat      = est["vy"][0]
heading_mat = est["heading"][0]
ax_mat      = est["ax"][0]
time_offset = float(f["log/time_offset_nsec"][0, 0])
f.close()

abs_ts_mat = time_offset + bag_stamp * 1e9
cam_f      = cam_ts.astype(float)

vx_ego  = np.interp(cam_f, abs_ts_mat, vx_mat)
vy_ego  = np.interp(cam_f, abs_ts_mat, vy_mat)
psi     = np.interp(cam_f, abs_ts_mat, heading_mat)
ax_ego  = np.interp(cam_f, abs_ts_mat, ax_mat)

print(f"  vx: {vx_ego.min():.2f} .. {vx_ego.max():.2f} m/s")

# ═══════════════════════════════════════════════════════════════════
# STEP 6 : RADAR DECODE  (ekf_track.py logic)
# ═══════════════════════════════════════════════════════════════════

print("\n━━━  STEP 6  Radar decode  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

radar_blobs = batch_fetch(conn, rad_id, set(matched_radar.tolist()))
conn.close()   # done with DB
print(f"  {len(radar_blobs)} radar blobs fetched.")

radar_cache = {int(ts): decode_radar_target(blob)
               for ts, blob in radar_blobs.items()}
del radar_blobs

radar_meas  = [radar_cache.get(int(ts)) for ts in matched_radar]
radar_v_raw = np.array([rm[2] if rm else np.nan for rm in radar_meas])
n_radar_det = int(np.sum([rm is not None for rm in radar_meas]))
print(f"  Target detected in {n_radar_det}/{N} frames.")

# ═══════════════════════════════════════════════════════════════════
# STEP 7 : EGO POSE INTEGRATION  (ekf_track.py logic)
# ═══════════════════════════════════════════════════════════════════

xe = np.zeros(N);  ye = np.zeros(N)
for i in range(1, N):
    dt    = (cam_ts[i] - cam_ts[i - 1]) / 1e9
    c, s  = np.cos(psi[i - 1]), np.sin(psi[i - 1])
    xe[i] = xe[i - 1] + (c * vx_ego[i - 1] - s * vy_ego[i - 1]) * dt
    ye[i] = ye[i - 1] + (s * vx_ego[i - 1] + c * vy_ego[i - 1]) * dt

# ═══════════════════════════════════════════════════════════════════
# STEP 8 : EKF  (ekf_track.py logic)
# ═══════════════════════════════════════════════════════════════════

print("\n━━━  STEP 8  EKF  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

def ekf_predict(x, P, dt):
    F = np.array([[1.,0.,dt, 0.],[0.,1., 0.,dt],[0.,0.,1., 0.],[0.,0.,0., 1.]])
    Q = np.diag([Q_POS, Q_POS, Q_VEL, Q_VEL]) * dt
    return F @ x, F @ P @ F.T + Q

def ekf_update_lidar(x, P, z, xe_k, ye_k):
    dx = x[0]-xe_k;  dy = x[1]-ye_k
    r  = max(np.sqrt(dx*dx+dy*dy), 0.1)
    H  = np.array([[dx/r, dy/r, 0., 0.]])
    K  = (P @ H.T) / (H @ P @ H.T + R_LIDAR)[0, 0]
    x  = x + K[:,0]*(z-r)
    P  = (np.eye(4) - np.outer(K[:,0], H)) @ P
    return x, P

def ekf_update_radar_pos(x, P, x_b, y_b, psi_k, xe_k, ye_k):
    c, s  = np.cos(psi_k), np.sin(psi_k)
    dx, dy = x[0]-xe_k, x[1]-ye_k
    H     = np.array([[ c, s,0.,0.],[-s,c,0.,0.]])
    innov = np.array([x_b,y_b]) - np.array([c*dx+s*dy, -s*dx+c*dy])
    S     = H @ P @ H.T + np.eye(2)*R_RADAR_XY
    K     = P @ H.T @ np.linalg.inv(S)
    x     = x + K @ innov
    P     = (np.eye(4) - K @ H) @ P
    return x, P

def ekf_update_radar_vel(x, P, z_v, xe_k, ye_k):
    dx, dy = x[0]-xe_k, x[1]-ye_k
    r      = max(np.sqrt(dx*dx+dy*dy), 0.1)
    vxt, vyt = x[2], x[3]
    h  = (vxt*dx + vyt*dy) / r
    H  = np.array([[(vxt*dy**2-vyt*dx*dy)/r**3,
                    (vyt*dx**2-vxt*dx*dy)/r**3,
                    dx/r, dy/r]])
    K  = (P @ H.T) / (H @ P @ H.T + R_RADAR_VEL)[0, 0]
    x  = x + K[:,0]*(z_v - h)
    P  = (np.eye(4) - np.outer(K[:,0], H)) @ P
    return x, P

# initialise from first lidar + first radar
v_init = next((rm[2] for rm in radar_meas if rm is not None), 14.0)
r0     = lidar_range[first_v[0]] if lidar_range[first_v[0]] > 0 else 6.0
c0, s0 = np.cos(psi[0]), np.sin(psi[0])
x_ekf  = np.array([r0*c0, r0*s0, v_init*c0, v_init*s0])
P_ekf  = np.diag([P0_POS, P0_POS, P0_VEL, P0_VEL])

ekf_range   = np.full(N, np.nan)
ekf_speed   = np.full(N, np.nan)
ekf_heading = np.full(N, np.nan)
ekf_pos_std = np.full(N, np.nan)
ekf_vel_std = np.full(N, np.nan)
ekf_confs   = []

for i in range(N):
    if i > 0:
        dt = (cam_ts[i] - cam_ts[i - 1]) / 1e9
        x_ekf, P_ekf = ekf_predict(x_ekf, P_ekf, dt)

    z = lidar_range[i]
    if z > 0:
        x_ekf, P_ekf = ekf_update_lidar(x_ekf, P_ekf, z, xe[i], ye[i])

    rm = radar_meas[i]
    if rm is not None:
        xb, yb, v_abs, conf = rm
        x_ekf, P_ekf = ekf_update_radar_pos(x_ekf, P_ekf, xb, yb, psi[i], xe[i], ye[i])
        x_ekf, P_ekf = ekf_update_radar_vel(x_ekf, P_ekf, v_abs, xe[i], ye[i])
        ekf_confs.append(conf)

    dx, dy = x_ekf[0]-xe[i], x_ekf[1]-ye[i]
    rng    = np.sqrt(dx*dx + dy*dy)
    spd    = np.sqrt(x_ekf[2]**2 + x_ekf[3]**2)
    t_hdg  = np.degrees(np.arctan2(x_ekf[3], x_ekf[2]))
    e_hdg  = np.degrees(psi[i] % (2*np.pi))
    hdg    = (t_hdg - e_hdg + 180) % 360 - 180

    ekf_range[i]   = round(rng, 3)
    ekf_speed[i]   = round(spd, 3)
    ekf_heading[i] = round(hdg, 2)
    ekf_pos_std[i] = round(np.sqrt((P_ekf[0,0]+P_ekf[1,1])/2), 4)
    ekf_vel_std[i] = round(np.sqrt((P_ekf[2,2]+P_ekf[3,3])/2), 4)

print(f"  Done.  Final: range={ekf_range[-1]:.2f} m  speed={ekf_speed[-1]:.2f} m/s"
      f"  heading={ekf_heading[-1]:.1f}°")

# ═══════════════════════════════════════════════════════════════════
# STEP 9 : SAVE CSVs
# ═══════════════════════════════════════════════════════════════════

print("\n━━━  STEP 9  Saving CSVs  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

time_s = (cam_f - cam_f[0]) / 1e9

pd.DataFrame({
    "camera":       cam_ts,
    "kf_distance":  np.round(lkf_dist, 4),
    "kf_velocity":  np.round(lkf_vel,  4),
    "ego_vx":       np.round(vx_ego, 4),
}).to_csv(os.path.join(_HERE, "all_frames_lkf.csv"), index=False)

pd.DataFrame({
    "camera":          cam_ts,
    "range_m":         ekf_range,
    "speed_m_s":       ekf_speed,
    "heading_rel_deg": ekf_heading,
    "pos_std_m":       ekf_pos_std,
    "vel_std_m_s":     ekf_vel_std,
    "ego_vx":          np.round(vx_ego, 4),
}).to_csv(os.path.join(_HERE, "all_frames_ekf.csv"), index=False)

mean_conf = float(np.nanmean(ekf_confs)) if ekf_confs else 39.9
pd.DataFrame({
    "position_m":   [round(float(ekf_range[-1]),   2)],
    "velocity_m_s": [round(float(ekf_speed[-1]),   2)],
    "heading_deg":  [round(float(ekf_heading[-1]), 1)],
    "confidence":   [round(mean_conf,              1)],
}).to_csv(os.path.join(_HERE, "all_frames_ekf_submission.csv"), index=False)

print("  all_frames_lkf.csv")
print("  all_frames_ekf.csv")
print("  all_frames_ekf_submission.csv")

# ═══════════════════════════════════════════════════════════════════
# STEP 10 : PLOT
# ═══════════════════════════════════════════════════════════════════

print("\n━━━  STEP 10  Plot  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "lines.linewidth": 1.6,
})

AX_COL   = "#1a1d27"
GRID_COL = "#2a2d3a"
TEXT_COL = "#e0e0e0"
C_RAW    = "#6c7a8f"
C_KF     = "#4ea3e0"
C_EKF    = "#f97316"
C_EGO    = "#a0e88a"
C_RADAR  = "#c084fc"

fig = plt.figure(figsize=(14, 16))
fig.patch.set_facecolor("#0f1117")
gs  = gridspec.GridSpec(4, 2, figure=fig,
                        hspace=0.50, wspace=0.35,
                        left=0.08, right=0.97, top=0.93, bottom=0.06)

def style(ax, title):
    ax.set_facecolor(AX_COL)
    ax.tick_params(colors=TEXT_COL)
    for sp in ax.spines.values(): sp.set_color(GRID_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TEXT_COL)
    ax.grid(color=GRID_COL, linewidth=0.5, linestyle="--", alpha=0.7)
    ax.set_title(title, pad=6, fontweight="bold")
    ax.set_xlabel("Time [s]")

# ── ① Range ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
style(ax1, "① Range to Target  (all frames, 120 s)")

raw_valid = lidar_range.copy(); raw_valid[lidar_range <= 0] = np.nan
ax1.plot(time_s, raw_valid,   color=C_RAW, lw=0.6, alpha=0.4, label="Raw lidar (20th pct)")
ax1.plot(time_s, lkf_dist,    color=C_KF,  lw=1.4, label="Linear KF")
ax1.plot(time_s, ekf_range,   color=C_EKF, lw=1.6, label="EKF (lidar + radar pos)")
ax1.fill_between(time_s,
                 ekf_range - ekf_pos_std,
                 ekf_range + ekf_pos_std,
                 color=C_EKF, alpha=0.18, label="EKF ±1σ")
ax1.set_ylabel("Distance [m]")
ax1.legend(loc="upper right", facecolor=AX_COL, edgecolor=GRID_COL, labelcolor=TEXT_COL)

# ── ② Velocity ──────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, :])
style(ax2, "② Target Absolute Velocity")

# Linear KF: range-rate + ego vx ≈ absolute
kf_abs = lkf_vel + vx_ego
ax2.plot(time_s, vx_ego,        color=C_EGO,   lw=1.2, ls=":", label="Ego vx [m/s]")
ax2.plot(time_s, radar_v_raw,   color=C_RADAR, lw=0.5, alpha=0.5, label="Radar abs vel (raw)")
ax2.plot(time_s, kf_abs,        color=C_KF,    lw=1.4, ls="-.", label="Linear KF range-rate + ego")
ax2.plot(time_s, ekf_speed,     color=C_EKF,   lw=1.6, label="EKF absolute speed")
ax2.fill_between(time_s,
                 ekf_speed - ekf_vel_std,
                 ekf_speed + ekf_vel_std,
                 color=C_EKF, alpha=0.18, label="EKF ±1σ")
ax2.set_ylabel("Speed [m/s]")
ax2.set_ylim(-2, None)
ax2.legend(loc="upper right", facecolor=AX_COL, edgecolor=GRID_COL, labelcolor=TEXT_COL)

# ── ③ EKF uncertainty ───────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
style(ax3, "③ EKF Uncertainty Convergence")
ax3.plot(time_s, ekf_pos_std, color=C_EKF,   label="Position σ [m]")
ax3b = ax3.twinx()
ax3b.plot(time_s, ekf_vel_std, color=C_RADAR, ls="--", label="Velocity σ [m/s]")
ax3b.tick_params(colors=TEXT_COL)
for sp in ax3b.spines.values(): sp.set_color(GRID_COL)
ax3b.set_ylabel("Velocity σ [m/s]", color=TEXT_COL)
ax3b.yaxis.label.set_color(TEXT_COL)
h1, l1 = ax3.get_legend_handles_labels()
h2, l2 = ax3b.get_legend_handles_labels()
ax3.legend(h1+h2, l1+l2, facecolor=AX_COL, edgecolor=GRID_COL, labelcolor=TEXT_COL)
ax3.set_ylabel("Position σ [m]")
ax3.set_xlabel("Time [s]")

# ── ④ Heading ───────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
style(ax4, "④ Target Heading Relative to Ego")
ax4.axhline(0, color=C_KF, lw=1.0, ls=":", label="Linear KF (assumed 0°)")
ax4.plot(time_s, ekf_heading, color=C_EKF, lw=1.4, label="EKF heading_rel [°]")
ax4.fill_between(time_s, ekf_heading, 0, color=C_EKF, alpha=0.12)
ax4.set_ylabel("Heading [°]")
ax4.set_xlabel("Time [s]")
ax4.legend(facecolor=AX_COL, edgecolor=GRID_COL, labelcolor=TEXT_COL)

# ── ⑤ Final bar chart ───────────────────────────────────────────────
ax5 = fig.add_subplot(gs[3, :])
ax5.set_facecolor(AX_COL)
ax5.tick_params(colors=TEXT_COL)
for sp in ax5.spines.values(): sp.set_color(GRID_COL)
ax5.grid(color=GRID_COL, lw=0.5, ls="--", alpha=0.7, axis="y")
ax5.set_title("⑤ Final Estimate Comparison (last frame)", pad=6,
              fontweight="bold", color=TEXT_COL)

metrics   = ["Position [m]", "Velocity [m/s]", "Heading [°]"]
kf_final  = [round(float(lkf_dist[-1]),2), round(float(kf_abs[-1]),2), 0.0]
ekf_final = [round(float(ekf_range[-1]),2), round(float(ekf_speed[-1]),2),
             round(float(ekf_heading[-1]),1)]

x_b, w = np.arange(3), 0.32
bk = ax5.bar(x_b-w/2, kf_final,  w, color=C_KF,  label="Linear KF")
be = ax5.bar(x_b+w/2, ekf_final, w, color=C_EKF, label="EKF + ego velocity")
for bars in [bk, be]:
    for bar in bars:
        h = bar.get_height()
        ax5.text(bar.get_x()+bar.get_width()/2, h+0.15, f"{h:.2f}",
                 ha="center", va="bottom", color=TEXT_COL, fontsize=9, fontweight="bold")
ax5.set_xticks(x_b); ax5.set_xticklabels(metrics, color=TEXT_COL)
ax5.set_ylabel("Value", color=TEXT_COL); ax5.yaxis.label.set_color(TEXT_COL)
ax5.legend(facecolor=AX_COL, edgecolor=GRID_COL, labelcolor=TEXT_COL)

# title + note
fig.suptitle(
    f"Linear KF  vs  EKF  —  {N} frames  ({time_s[-1]:.0f} s)",
    color=TEXT_COL, fontsize=13, fontweight="bold", y=0.965,
)
fig.text(
    0.5, 0.01,
    "Linear KF: lidar range only · no ego dynamics · no heading estimate\n"
    "EKF + ego: lidar range + radar (x,y) + radar absolute velocity · "
    "ego pose integrated · nonlinear measurement Jacobians",
    ha="center", va="bottom", color="#9ca3af", fontsize=8.5, style="italic",
    bbox=dict(boxstyle="round,pad=0.4", fc="#1a1d27", ec=GRID_COL, alpha=0.85),
)

out_png = os.path.join(_HERE, "kf_vs_ekf_comparison_all.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved → {out_png}")

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'━'*60}")
print(f"  Total runtime : {time.time()-t_start:.0f} s")
print(f"  Frames processed: {N}")
print()
print(f"  Linear KF  — position: {lkf_dist[-1]:.2f} m  "
      f"velocity: {kf_abs[-1]:.2f} m/s  heading: 0.0°")
print(f"  EKF + ego  — position: {ekf_range[-1]:.2f} m  "
      f"velocity: {ekf_speed[-1]:.2f} m/s  heading: {ekf_heading[-1]:.1f}°")
print(f"{'━'*60}\n")
