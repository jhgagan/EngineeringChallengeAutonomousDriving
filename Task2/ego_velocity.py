"""
ego_velocity.py
---------------
Extract ego-vehicle motion state from ego.mat (HDF5/MATLAB v7.3)
and interpolate at each camera frame timestamp used in the pipeline.

Inputs:
    ../Data/ego.mat          (ego motion log)
    task2_measurements.csv  (camera timestamps from task2_pipeline.py)

Output:
    ego_interp.csv           columns: camera, vx, vy, heading, ax, yaw_rate
                             vx / vy  : longitudinal / lateral velocity in body frame [m/s]
                             heading  : absolute yaw angle (unwrapped) in radians
                             ax       : longitudinal acceleration [m/s²]
                             yaw_rate : yaw angular rate [rad/s]
"""

import h5py
import numpy as np
import pandas as pd

# ==========================
# CONFIG
# ==========================

MAT = "../Data/ego.mat"

# ==========================
# LOAD EGO.MAT
# ==========================

f = h5py.File(MAT, "r")
est = f["log/estimation"]

bag_stamp   = est["bag_stamp"][0]      # seconds, relative to recording start
vx          = est["vx"][0]             # longitudinal velocity, body frame [m/s]
vy          = est["vy"][0]             # lateral velocity,      body frame [m/s]
heading     = est["heading"][0]        # world-frame yaw angle, unwrapped [rad]
ax          = est["ax"][0]             # longitudinal acceleration [m/s²]
yaw_rate    = est["yaw_rate"][0]       # yaw rate [rad/s]

# time_offset_nsec converts bag_stamp (sec) → absolute ROS nanoseconds
time_offset = float(f["log/time_offset_nsec"][0, 0])

f.close()

# Absolute timestamps in nanoseconds, matching the db3 bag_stamp convention
abs_ts_ns = time_offset + bag_stamp * 1e9

# ==========================
# LOAD CAMERA TIMESTAMPS
# ==========================

meas   = pd.read_csv("task2_measurements.csv")
cam_ts = meas["camera"].values   # nanoseconds

# ==========================
# INTERPOLATE EGO STATE
# ==========================

# np.interp requires xs to be monotonically increasing;
# abs_ts_ns is derived from bag_stamp which is sorted.

vx_i       = np.interp(cam_ts, abs_ts_ns, vx)
vy_i       = np.interp(cam_ts, abs_ts_ns, vy)
heading_i  = np.interp(cam_ts, abs_ts_ns, heading)
ax_i       = np.interp(cam_ts, abs_ts_ns, ax)
yaw_rate_i = np.interp(cam_ts, abs_ts_ns, yaw_rate)

# ==========================
# BUILD AND SAVE
# ==========================

df = pd.DataFrame({
    "camera":   cam_ts,
    "vx":       np.round(vx_i,       4),
    "vy":       np.round(vy_i,       4),
    "heading":  np.round(heading_i,  6),
    "ax":       np.round(ax_i,       4),
    "yaw_rate": np.round(yaw_rate_i, 6),
})

print()
print("Ego state interpolated at camera timestamps:")
print()
print(df.to_string(index=False))
print()

df.to_csv("ego_interp.csv", index=False)

print("saved ego_interp.csv")
print()
print(f"  vx range : {vx_i.min():.3f} .. {vx_i.max():.3f} m/s")
print(f"  vy range : {vy_i.min():.3f} .. {vy_i.max():.3f} m/s")
print(f"  ax range : {ax_i.min():.3f} .. {ax_i.max():.3f} m/s²")
print(f"  yaw_rate : {yaw_rate_i.min():.4f} .. {yaw_rate_i.max():.4f} rad/s")
