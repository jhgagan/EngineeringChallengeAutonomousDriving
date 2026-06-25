"""
Ego Odometry Simulation Dashboard
===================================
Layout:
    ┌─────────────┬─────────────┬─────────────┐
    │  Velocity   │Acceleration │             │
    ├─────────────┼─────────────┤  Trajectory │
    │   Heading   │  Yaw Rate   │             │
    └─────────────┴─────────────┴─────────────┘

Libraries:
    pip install mat73 h5py numpy matplotlib

Usage:
    python ego_dashboard.py
    python ego_dashboard.py --file Data/ego.mat --speed 1 --window 20 --fps 30
"""

import argparse
import time
import numpy as np
import mat73
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

matplotlib.use("TkAgg")   # change to "Qt5Agg" if TkAgg is unavailable

# ── colour palette ─────────────────────────────────────────────────────────────
BG        = "#0d1117"
BG_AX     = "#0f1621"
GRID_COL  = "#1e2a38"
C_VX      = "#33ff66"
C_VY      = "#ff6633"
C_SPD     = "#00d4ff"
C_AX      = "#00d4ff"
C_AY      = "#ff3355"
C_ACC     = "#ff6b21"
C_HDG     = "#ffd700"
C_YAW     = "#3dff14"
C_TRAJ    = "#00d4ff"
C_GHOST   = "#1e2d3d"
C_DOT     = "#00d4ff"
C_TITLE   = "#00d4ff"
C_TICK    = "#6a8aaa"
C_LABEL   = "#4a6a8a"


# ── helpers ────────────────────────────────────────────────────────────────────
def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(BG_AX)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.tick_params(colors=C_TICK, labelsize=8)
    ax.xaxis.label.set_color(C_LABEL)
    ax.yaxis.label.set_color(C_LABEL)
    ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.9)
    ax.set_title(title, fontsize=10, fontfamily="monospace", color="#b0ccee", pad=6)
    ax.set_xlabel(xlabel, fontsize=8, fontfamily="monospace")
    ax.set_ylabel(ylabel, fontsize=8, fontfamily="monospace")


def auto_ylim(ax, *bufs, margin=0.12):
    """Auto-scale y axis from one or more deque buffers."""
    all_vals = [v for buf in bufs for v in buf]
    if len(all_vals) < 2:
        return
    lo, hi = min(all_vals), max(all_vals)
    pad = (hi - lo) * margin + 0.05
    ax.set_ylim(lo - pad, hi + pad)


def make_line(ax, color, lw=1.2, zorder=2):
    ln, = ax.plot([], [], color=color, linewidth=lw, zorder=zorder,
                  solid_capstyle="round")
    return ln


# ── argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Ego Odometry Dashboard")
    p.add_argument("--file",   default="Data/ego.mat",
                   help="Path to ego.mat  (default: Data/ego.mat)")
    p.add_argument("--speed",  type=float, default=1.0,
                   help="Playback speed multiplier — 1 = real time (default: 1)")
    p.add_argument("--window", type=float, default=20.0,
                   help="Seconds of history in time-series plots (default: 20)")
    p.add_argument("--fps",    type=int,   default=30,
                   help="Target render frame rate (default: 30)")
    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── 1. load data ──────────────────────────────────────────────────────────
    print(f"Loading {args.file} ...")
    data = mat73.loadmat(args.file)
    est  = data["log"]["estimation"]

    t        = np.array(est["stamp__tot"])
    t0       = t - t[0]
    vx       = np.array(est["vx"])
    vy       = np.array(est["vy"])
    ax_      = np.array(est["ax"])
    ay_      = np.array(est["ay"])
    heading  = np.array(est["ekf_heading"])
    yaw_rate = np.array(est["yaw_rate"])
    px       = np.array(est["gps_pos_report__x"])[:, 0]
    py       = np.array(est["gps_pos_report__y"])[:, 0]

    N  = len(t)
    dt = float(np.mean(np.diff(t)))

    print(f"Duration:    {t0[-1]:.1f} s")
    print(f"Sample rate: {1/dt:.0f} Hz")
    print(f"Playback:    {args.speed}x  |  render @ {args.fps} fps")

    # ── 2. precompute all derived signals on CPU (vectorised, done in <5 ms) ──
    print("Precomputing signals ...")
    spd_all = np.sqrt(vx**2  + vy**2)
    acc_all = np.sqrt(ax_**2 + ay_**2)
    hdg_all = np.degrees(heading)
    yr_all  = np.degrees(yaw_rate)
    print("Done.\n")

    RENDER_EVERY = max(1, round((1.0 / dt) / args.fps))   # e.g. 188/30 ≈ 6
    WINDOW       = args.window
    SPEED        = args.speed
    MAX_BUF      = int(WINDOW / dt) + 20

    # ── 3. figure & layout ────────────────────────────────────────────────────
    #
    #   GridSpec 2 rows × 3 cols
    #   col 0-1 → 2×2 time-series grid
    #   col 2   → full-height trajectory (rowspan 0:2)
    #
    fig = plt.figure(figsize=(16, 8), facecolor=BG)
    fig.canvas.manager.set_window_title("Ego Odometry Dashboard")

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.42, wspace=0.30,
        left=0.06, right=0.97, top=0.91, bottom=0.08
    )

    sub_vel  = fig.add_subplot(gs[0, 0])   # top-left
    sub_acc  = fig.add_subplot(gs[0, 1])   # top-middle
    sub_hdg  = fig.add_subplot(gs[1, 0])   # bottom-left
    sub_yaw  = fig.add_subplot(gs[1, 1])   # bottom-middle
    sub_traj = fig.add_subplot(gs[:, 2])   # full right column

    style_ax(sub_vel,  "Velocity",      "Time [s]", "[m/s]")
    style_ax(sub_acc,  "Acceleration",  "Time [s]", "[m/s²]")
    style_ax(sub_hdg,  "Heading",       "Time [s]", "[deg]")
    style_ax(sub_yaw,  "Yaw Rate",      "Time [s]", "[deg/s]")
    style_ax(sub_traj, "GPS Trajectory","X [m]",    "Y [m]")

    # ghost full path on trajectory panel
    sub_traj.plot(px, py, color=C_GHOST, linewidth=0.8, zorder=1)
    sub_traj.set_aspect("equal", adjustable="datalim")

    # ── 4. animated line objects ──────────────────────────────────────────────
    ln_vx   = make_line(sub_vel,  C_VX,  1.2)
    ln_vy   = make_line(sub_vel,  C_VY,  1.2)
    ln_spd  = make_line(sub_vel,  C_SPD, 1.8)

    ln_ax_  = make_line(sub_acc,  C_AX,  1.2)
    ln_ay_  = make_line(sub_acc,  C_AY,  1.2)
    ln_acc  = make_line(sub_acc,  C_ACC, 1.8)

    ln_hdg  = make_line(sub_hdg,  C_HDG, 1.5)
    ln_yaw  = make_line(sub_yaw,  C_YAW, 1.5)

    ln_traj = make_line(sub_traj, C_TRAJ, 1.8, zorder=3)

    # current-position dot + heading arrow on trajectory
    dot, = sub_traj.plot([], [], "o",
                         markersize=9,
                         markerfacecolor=C_DOT,
                         markeredgecolor="white",
                         linewidth=1.2, zorder=5)

    arrow_hdg = sub_traj.annotate(
        "", xy=(px[0], py[0]),
        xytext=(px[0], py[0]),
        arrowprops=dict(arrowstyle="-|>", color="#ff6b35",
                        lw=1.8, mutation_scale=12),
        zorder=6
    )

    # legends
    sub_vel.legend(["Vx", "Vy", "|V|"],
                   facecolor="none", edgecolor="none",
                   labelcolor="white", fontsize=8, loc="upper left")
    sub_acc.legend(["Ax", "Ay", "|A|"],
                   facecolor="none", edgecolor="none",
                   labelcolor="white", fontsize=8, loc="upper left")

    title_obj = fig.suptitle(
        "EGO ODOMETRY — LIVE REPLAY",
        color=C_TITLE, fontsize=13,
        fontfamily="monospace", fontweight="bold"
    )

    plt.ion()
    plt.show(block=False)
    fig.canvas.draw()

    # ── 5. rolling buffers ────────────────────────────────────────────────────
    buf_t   = deque(maxlen=MAX_BUF)
    buf_vx  = deque(maxlen=MAX_BUF)
    buf_vy  = deque(maxlen=MAX_BUF)
    buf_spd = deque(maxlen=MAX_BUF)
    buf_ax  = deque(maxlen=MAX_BUF)
    buf_ay  = deque(maxlen=MAX_BUF)
    buf_acc = deque(maxlen=MAX_BUF)
    buf_hdg = deque(maxlen=MAX_BUF)
    buf_yaw = deque(maxlen=MAX_BUF)
    traj_x  = deque(maxlen=10000)
    traj_y  = deque(maxlen=10000)

    # ── 6. main playback loop ─────────────────────────────────────────────────
    wall_start = time.perf_counter()

    for i in range(N):

        if not plt.fignum_exists(fig.number):
            break

        ti  = t0[i]
        spd = spd_all[i]
        acc = acc_all[i]
        hdg = hdg_all[i]
        yr  = yr_all[i]

        # append every sample to buffers (O(1), very cheap)
        buf_t.append(ti);    buf_vx.append(vx[i]);   buf_vy.append(vy[i])
        buf_spd.append(spd)
        buf_ax.append(ax_[i]); buf_ay.append(ay_[i]); buf_acc.append(acc)
        buf_hdg.append(hdg);   buf_yaw.append(yr)
        traj_x.append(px[i]);  traj_y.append(py[i])

        # ── render every Nth sample only ──────────────────────────────────────
        if i % RENDER_EVERY == 0:

            tarr = np.asarray(buf_t)

            # update time-series lines
            ln_vx.set_data(tarr,  np.asarray(buf_vx))
            ln_vy.set_data(tarr,  np.asarray(buf_vy))
            ln_spd.set_data(tarr, np.asarray(buf_spd))
            ln_ax_.set_data(tarr, np.asarray(buf_ax))
            ln_ay_.set_data(tarr, np.asarray(buf_ay))
            ln_acc.set_data(tarr, np.asarray(buf_acc))
            ln_hdg.set_data(tarr, np.asarray(buf_hdg))
            ln_yaw.set_data(tarr, np.asarray(buf_yaw))

            # update trajectory
            tx = np.asarray(traj_x)
            ty = np.asarray(traj_y)
            ln_traj.set_data(tx, ty)
            dot.set_data([px[i]], [py[i]])

            # heading arrow (30 m long in data units)
            ARROW_LEN = 30
            hdg_rad = float(heading[i])
            tip_x = px[i] + ARROW_LEN * np.cos(hdg_rad)
            tip_y = py[i] + ARROW_LEN * np.sin(hdg_rad)
            arrow_hdg.xy     = (tip_x, tip_y)
            arrow_hdg.xytext = (px[i], py[i])

            # sliding x-window on time-series plots
            x_lo = max(0.0, ti - WINDOW)
            x_hi = max(WINDOW, ti + 0.5)
            for sub in (sub_vel, sub_acc, sub_hdg, sub_yaw):
                sub.set_xlim(x_lo, x_hi)

            # auto y-limits
            auto_ylim(sub_vel, buf_vx, buf_vy, buf_spd)
            auto_ylim(sub_acc, buf_ax, buf_ay, buf_acc)
            auto_ylim(sub_hdg, buf_hdg)
            auto_ylim(sub_yaw, buf_yaw)

            # title
            title_obj.set_text(
                f"EGO ODOMETRY  —  T = {ti:.2f} s  |  "
                f"{spd*3.6:.1f} km/h  |  hdg = {hdg:.1f}°"
            )

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            # ── real-time wall-clock sync ──────────────────────────────────
            target_wall = wall_start + ti / SPEED
            remaining   = target_wall - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)

    print("Playback complete.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()