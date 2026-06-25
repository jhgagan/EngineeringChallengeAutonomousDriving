"""
compare_kf_ekf.py
-----------------
Side-by-side comparison of the linear KF (kalman_track.py) and the
Extended Kalman Filter (ekf_track.py) over the 20-frame tracking window.

Four panels:
  1. Range to target   – raw lidar | Linear KF | EKF
  2. Target velocity   – raw radar per-frame | EKF absolute speed |
                         Linear KF + ego (approximate absolute) | ego speed
  3. EKF 1-σ uncertainty bands (position std, velocity std)
  4. Bar chart: final answer comparison between both methods

Inputs:
    task2_kf.csv        (linear KF output from kalman_track.py)
    task2_ekf.csv       (EKF output from ekf_track.py)
    ego_interp.csv      (ego velocity from ego_velocity.py)
    task2_measurements.csv   (raw lidar distances + radar timestamps)
    ../Data/2025-10-08_09-35_sensors_raw_1.db3  (for raw per-frame radar velocity)
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ==========================
# LOAD CSVs
# ==========================

kf   = pd.read_csv("task2_kf.csv")
ekf  = pd.read_csv("task2_ekf.csv")
ego  = pd.read_csv("ego_interp.csv")
meas = pd.read_csv("task2_measurements.csv")

# Time axis in seconds (relative to first frame)
t0 = float(meas["camera"].iloc[0])
time_s = (meas["camera"].values.astype(float) - t0) / 1e9

# ==========================
# RAW PER-FRAME RADAR VELOCITY
# (re-decode closest forward target for each radar timestamp)
# ==========================

DB          = "../Data/2025-10-08_09-35_sensors_raw_1.db3"
RADAR_DTYPE = np.dtype([
    ("x",          "<f4"), ("y",          "<f4"), ("z",   "<f4"),
    ("velocity",   "<f4"), ("snr",        "<f4"), ("rcs", "<f4"),
    ("confidence", "<f4"), ("vint",       "<f4"),
])

radar_ts_col = meas["radar"].values   # int64 (pandas reads as int64 from CSV)

conn     = sqlite3.connect(DB)
topic_id = conn.execute(
    "SELECT id FROM topics WHERE name='/sensor/radar_front/points'"
).fetchone()[0]

cache = {}
for rts in set(radar_ts_col):
    row = conn.execute(
        "SELECT data FROM messages WHERE topic_id=? AND timestamp=? LIMIT 1",
        (topic_id, int(rts)),
    ).fetchone()
    if not row:
        cache[rts] = (np.nan, np.nan)
        continue
    raw     = row[0]
    payload = raw[220:]
    payload = payload[:(len(payload) // 32) * 32]
    if not len(payload):
        cache[rts] = (np.nan, np.nan)
        continue
    pts  = np.frombuffer(payload, dtype=RADAR_DTYPE)
    x    = pts["x"].astype(float)
    y    = pts["y"].astype(float)
    v    = pts["velocity"].astype(float)
    mask = (x > 3.0) & (x < 20.0)   # minimal filter; pick the closest return
    if not mask.any():
        cache[rts] = (np.nan, np.nan)
        continue
    r       = np.sqrt(x[mask] ** 2 + y[mask] ** 2)
    nearest = np.argmin(r)
    cache[rts] = (float(v[mask][nearest]), float(r[nearest]))

conn.close()

radar_v_raw   = np.array([cache[rts][0] for rts in radar_ts_col])
radar_r_raw   = np.array([cache[rts][1] for rts in radar_ts_col])

# ==========================
# DERIVED QUANTITIES
# ==========================

# Linear KF: kf_velocity is the *range rate* (relative, m/s).
# Approximate absolute target velocity = kf_velocity + ego vx
kf_abs_vel = kf["kf_velocity"].values + ego["vx"].values

# EKF absolute speed (already world-frame magnitude)
ekf_speed = ekf["speed_m_s"].values

# Ego speed (for reference)
ego_speed = ego["vx"].values

# ==========================
# FIGURE LAYOUT
# ==========================

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
})

fig = plt.figure(figsize=(13, 14))
fig.patch.set_facecolor("#0f1117")

gs = gridspec.GridSpec(
    4, 2,
    figure=fig,
    hspace=0.48,
    wspace=0.35,
    left=0.08, right=0.97,
    top=0.93,  bottom=0.06,
)

AX_COLOR  = "#1a1d27"
GRID_COL  = "#2a2d3a"
TEXT_COL  = "#e0e0e0"

C_RAW  = "#6c7a8f"
C_KF   = "#4ea3e0"
C_EKF  = "#f97316"
C_EGO  = "#a0e88a"
C_RADAR= "#c084fc"

def style_ax(ax, title):
    ax.set_facecolor(AX_COLOR)
    ax.tick_params(colors=TEXT_COL)
    ax.spines[:].set_color(GRID_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TEXT_COL)
    ax.grid(color=GRID_COL, linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_title(title, pad=6, fontweight="bold")
    ax.set_xlabel("Time [s]")

# ── Panel 1: Range to target ──────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
style_ax(ax1, "① Range to Target")

ax1.plot(time_s, meas["distance_m"].values,
         "o--", color=C_RAW, alpha=0.6, ms=4, label="Raw lidar (20th pct)")
ax1.plot(time_s, kf["kf_distance"].values,
         color=C_KF, label="Linear KF (range only)")
ax1.plot(time_s, ekf["range_m"].values,
         color=C_EKF, label="EKF (lidar + radar position)")

ax1.fill_between(time_s,
                 ekf["range_m"].values - ekf["pos_std_m"].values,
                 ekf["range_m"].values + ekf["pos_std_m"].values,
                 alpha=0.2, color=C_EKF, label="EKF ±1σ position")

ax1.set_ylabel("Distance  [m]")
ax1.legend(loc="upper right", facecolor=AX_COLOR, edgecolor=GRID_COL,
           labelcolor=TEXT_COL)

# annotate discrepancy
mid = len(time_s) // 2
ax1.annotate(
    f"Δ ≈ {kf['kf_distance'].iloc[mid] - ekf['range_m'].iloc[mid]:.1f} m\n"
    "(linear KF lacks radar)",
    xy=(time_s[mid], ekf["range_m"].iloc[mid]),
    xytext=(time_s[mid] - 0.15, ekf["range_m"].iloc[mid] + 1.4),
    arrowprops=dict(arrowstyle="->", color=TEXT_COL, lw=0.9),
    color=TEXT_COL, fontsize=8.5,
)

# ── Panel 2: Target absolute velocity ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, :])
style_ax(ax2, "② Target Absolute Velocity")

ax2.plot(time_s, ego_speed,
         color=C_EGO, linestyle=":", linewidth=1.4,
         label="Ego speed  vₓ  [m/s]")

ax2.scatter(time_s, radar_v_raw,
            color=C_RADAR, s=28, zorder=5, alpha=0.75,
            label="Raw radar (absolute, per frame)")

ax2.plot(time_s, kf_abs_vel,
         color=C_KF, linestyle="-.",
         label="Linear KF range-rate + ego  ≈  target abs vel")

ax2.plot(time_s, ekf_speed,
         color=C_EKF, label="EKF absolute speed  √(vₓ²+vy²)")

ax2.fill_between(time_s,
                 ekf_speed - ekf["vel_std_m_s"].values,
                 ekf_speed + ekf["vel_std_m_s"].values,
                 alpha=0.18, color=C_EKF, label="EKF ±1σ velocity")

ax2.set_ylabel("Speed  [m/s]")
ax2.legend(loc="upper right", facecolor=AX_COLOR, edgecolor=GRID_COL,
           labelcolor=TEXT_COL)

# ── Panel 3: EKF uncertainty convergence ──────────────────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
style_ax(ax3, "③ EKF Uncertainty Convergence")

ax3.plot(time_s, ekf["pos_std_m"].values,
         color=C_EKF, label="Position σ  [m]")
ax3b = ax3.twinx()
ax3b.plot(time_s, ekf["vel_std_m_s"].values,
          color=C_RADAR, linestyle="--", label="Velocity σ  [m/s]")
ax3b.tick_params(colors=TEXT_COL)
ax3b.spines[:].set_color(GRID_COL)
ax3b.yaxis.label.set_color(TEXT_COL)
ax3b.set_facecolor(AX_COLOR)
ax3b.set_ylabel("Velocity σ  [m/s]", color=TEXT_COL)

lines1, labs1 = ax3.get_legend_handles_labels()
lines2, labs2 = ax3b.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labs1 + labs2,
           facecolor=AX_COLOR, edgecolor=GRID_COL, labelcolor=TEXT_COL)
ax3.set_ylabel("Position σ  [m]")
ax3.set_xlabel("Time [s]")

# ── Panel 4: EKF heading estimate ─────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
style_ax(ax4, "④ Target Heading Relative to Ego")

ax4.axhline(0, color=C_KF, linewidth=1.0, linestyle=":",
            label="Linear KF (assumed 0°)")
ax4.plot(time_s, ekf["heading_rel_deg"].values,
         color=C_EKF, label="EKF heading_rel  [deg]")
ax4.fill_between(time_s, ekf["heading_rel_deg"].values, 0,
                 alpha=0.15, color=C_EKF)
ax4.set_ylabel("Heading  [°]")
ax4.set_xlabel("Time [s]")
ax4.legend(facecolor=AX_COLOR, edgecolor=GRID_COL, labelcolor=TEXT_COL)

# ── Panel 5: Final answer comparison (bar chart) ───────────────────────────────
ax5 = fig.add_subplot(gs[3, :])
ax5.set_facecolor(AX_COLOR)
ax5.tick_params(colors=TEXT_COL)
ax5.spines[:].set_color(GRID_COL)
ax5.set_facecolor(AX_COLOR)

metrics      = ["Position [m]", "Velocity [m/s]", "Heading [°]"]
kf_final     = [
    round(float(kf["kf_distance"].iloc[-1]),  2),
    round(float(kf_abs_vel[-1]),               2),
    0.0,   # linear KF assumed 0°
]
ekf_final    = [
    round(float(ekf["range_m"].iloc[-1]),      2),
    round(float(ekf["speed_m_s"].iloc[-1]),    2),
    round(float(ekf["heading_rel_deg"].iloc[-1]), 1),
]

x_bar  = np.arange(len(metrics))
width  = 0.32

bars_kf  = ax5.bar(x_bar - width / 2, kf_final,  width, color=C_KF,  label="Linear KF")
bars_ekf = ax5.bar(x_bar + width / 2, ekf_final, width, color=C_EKF, label="EKF + ego velocity")

for bars in [bars_kf, bars_ekf]:
    for bar in bars:
        h = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2, h + 0.15,
                 f"{h:.2f}", ha="center", va="bottom",
                 color=TEXT_COL, fontsize=9, fontweight="bold")

ax5.set_xticks(x_bar)
ax5.set_xticklabels(metrics, color=TEXT_COL, fontsize=10)
ax5.set_ylabel("Value", color=TEXT_COL)
ax5.yaxis.label.set_color(TEXT_COL)
ax5.title.set_color(TEXT_COL)
ax5.set_title("⑤ Final Estimate Comparison", pad=6, fontweight="bold")
ax5.legend(facecolor=AX_COLOR, edgecolor=GRID_COL, labelcolor=TEXT_COL)
ax5.grid(color=GRID_COL, linewidth=0.6, linestyle="--", alpha=0.7, axis="y")

# ── Main title ────────────────────────────────────────────────────────────────
fig.suptitle(
    "Linear KF  vs  Extended Kalman Filter  —  Target Vehicle Tracking",
    color=TEXT_COL, fontsize=13, fontweight="bold", y=0.965,
)

# ── Key differences annotation box ───────────────────────────────────────────
note = (
    "Key differences\n"
    "Linear KF  : lidar range only · no ego dynamics · no heading estimate\n"
    "EKF + ego  : lidar range + radar (x,y) + radar absolute velocity · ego pose integrated · nonlinear measurement Jacobians"
)
fig.text(
    0.5, 0.01, note,
    ha="center", va="bottom", color="#9ca3af",
    fontsize=8.5, style="italic",
    bbox=dict(boxstyle="round,pad=0.4", fc="#1a1d27", ec=GRID_COL, alpha=0.85),
)

plt.savefig("kf_vs_ekf_comparison.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("saved kf_vs_ekf_comparison.png")
