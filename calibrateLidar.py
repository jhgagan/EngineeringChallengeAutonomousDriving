"""
Interactive LiDAR-on-camera calibration tool.
Adjust sliders until the point cloud aligns with the scene, then close
the window — final values are printed to the terminal for use in lidarOnCamera.py.

NOTE: The gap/notch in the point cloud is a physical blind spot of the LiDAR
sensor and will always appear regardless of calibration.

Sliders:
  FX  — focal length (lower = wider horizontal spread, higher = narrower)
  CX  — horizontal pixel offset; shifts ALL points left/right equally (no depth warp)
  T1  — vertical 3D offset in camera frame (down+ / up-)
  T2  — depth offset along optical axis (fwd+ / back-)
  CY  — vertical pixel offset; shifts ALL points up/down equally
  K1  — arc correction: positive flattens an upward arch, negative flattens a downward arch
"""

import sqlite3
import struct
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image

DB           = "Data/2025-10-08_09-35_sensors_raw_0.db3"
POINT_STEP   = 32
LIDAR_OFFSET = 272
IMG_W, IMG_H = 1928, 500

# LiDAR frame: x=height, y=lateral(right+), z=forward
# Camera frame: x=right,  y=down,           z=forward
R = np.array([[ 0,  1,  0],
              [-1,  0,  0],
              [ 0,  0,  1]], dtype=np.float64)

LIDAR_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("pad", "V4"),
    ("timestamp", "<i4"), ("intensity", "<f4"),
    ("scan_idx", "<u2"), ("scan_id", "<u2"),
    ("ring", "<u2"), ("channel", "u1"), ("pad2", "V1"),
])

# ── Starting values ───────────────────────────────────────────────────────────
INIT = dict(
    fx  = 400.0,
    cx  = IMG_W / 2.0,   # pure horizontal pixel offset (replaces 3D t0)
    t1  =  0.5,
    t2  =  0.1,
    cy  = IMG_H / 2.0 + 30,
    k1  =  0.0,          # arc curvature correction (pixels at image edge)
)

# ── Load first frame ──────────────────────────────────────────────────────────
print("Loading first frame...")
conn = sqlite3.connect(DB)
lidar_id = conn.execute(
    "SELECT id FROM topics WHERE name='/sensor/lidar_front/points'"
).fetchone()[0]
cam_id = conn.execute(
    "SELECT id FROM topics WHERE name='/sensor/camera/camera_fr/image/compressed'"
).fetchone()[0]
CALIB_FRAME = 50   # change this to calibrate on a different frame

lidar_raw = conn.execute(
    "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1 OFFSET ?",
    (lidar_id, CALIB_FRAME)
).fetchone()[0]
cam_raw = conn.execute(
    "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1 OFFSET ?",
    (cam_id, CALIB_FRAME)
).fetchone()[0]
conn.close()


def decode_image(raw):
    off = 4 + 8
    slen = struct.unpack_from("<I", raw, off)[0]; off += 4
    off += slen; off = (off + 3) & ~3
    slen = struct.unpack_from("<I", raw, off)[0]; off += 4
    off += slen; off = (off + 3) & ~3
    dlen = struct.unpack_from("<I", raw, off)[0]; off += 4
    return np.array(Image.open(io.BytesIO(raw[off:off + dlen])))


def parse_lidar(raw):
    payload = raw[LIDAR_OFFSET:]
    usable  = (len(payload) // POINT_STEP) * POINT_STEP
    pts     = np.frombuffer(payload[:usable], dtype=LIDAR_DTYPE)
    valid   = (np.isfinite(pts["x"]) & np.isfinite(pts["y"]) &
               np.isfinite(pts["z"]) & (pts["z"] > 0))
    pts     = pts[valid]
    return (pts["x"].astype(np.float64),
            pts["y"].astype(np.float64),
            pts["z"].astype(np.float64))


cam_img = decode_image(cam_raw)
lx, ly, lz = parse_lidar(lidar_raw)
pts_lidar   = np.vstack([lx, ly, lz])

# ── Static car body region detected from frames 10, 90, 200 ──────────────────
# The car body occupies y > 430, centred around x = 1240.
# This horizontal line is drawn on the image as the alignment target.
CAR_BODY_TOP_Y = 430   # pixel row where the car body starts

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure("LiDAR calibration  (full image)", figsize=(16, 9))
fig.patch.set_facecolor("#1a1a1a")

# Sliders — top portion (6 sliders with 0.06 spacing)
sc_col = "#2a2a2a"
sliders = {}
layout = [
    ("fx", [0.12, 0.91, 0.78, 0.04],  600,   2400, "FX  lower=wider spread  higher=narrower"),
    ("cx", [0.12, 0.85, 0.78, 0.04],    0,  IMG_W, "CX  right(+) / left(-)  uniform pixel shift"),
    ("t1", [0.12, 0.79, 0.78, 0.04],   -3,      3, "T1  down(+)  / up(-)  3D vertical offset"),
    ("t2", [0.12, 0.73, 0.78, 0.04],   -1,      1, "T2  fwd(+)   / back(-)"),
    ("cy", [0.12, 0.67, 0.78, 0.04],   50,    450, "CY  higher=overlay moves down  uniform pixel shift"),
    ("k1", [0.12, 0.61, 0.78, 0.04], -200,    200, "K1  flatten upward arch(+) / flatten downward arch(-)"),
]
for key, rect, vmin, vmax, label in layout:
    ax_s = fig.add_axes(rect, facecolor=sc_col)
    s    = Slider(ax_s, label, vmin, vmax, valinit=INIT[key], color="#4a90d9")
    s.label.set_color("white")
    s.valtext.set_color("white")
    sliders[key] = s

# Info text
info_ax = fig.add_axes([0.03, 0.55, 0.94, 0.05])
info_ax.axis("off")
info_text = info_ax.text(0.5, 0.5, "", transform=info_ax.transAxes,
                         ha="center", va="center", color="lime",
                         fontsize=9, fontfamily="monospace")

# Full image panel
ax_img = fig.add_axes([0.03, 0.02, 0.88, 0.52])
ax_img.axis("off")
img_disp = ax_img.imshow(cam_img)
ax_img.set_xlim(0, IMG_W)
ax_img.set_ylim(IMG_H, 0)
ax_img.autoscale(False)

# Yellow dashed line marks the top of the detected static car body
ax_img.axhline(CAR_BODY_TOP_Y, color="yellow", linewidth=1,
               linestyle="--", alpha=0.7, label="car body top")

sc = ax_img.scatter([], [], s=5, c=[], cmap="turbo",
                    vmin=0, vmax=80, linewidths=0, alpha=0.9)
cbar = plt.colorbar(sc, ax=ax_img, label="depth (m)",
                    fraction=0.015, pad=0.01)
cbar.ax.yaxis.label.set_color("white")
cbar.ax.tick_params(colors="white")


def project(fx, cx, t1, t2, cy, k1):
    T = np.array([0.0, t1, t2])
    K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float64)
    pc  = R @ pts_lidar + T[:, None]
    fwd = pc[2] > 1.0
    pc  = pc[:, fwd]
    uv  = K @ pc
    u   = uv[0] / uv[2]
    v   = uv[1] / uv[2]
    # Arc correction: uniform quadratic offset applied in pixel space.
    # k1 > 0 pulls edge points upward, flattening an upward arch.
    # k1 < 0 pulls edge points downward, flattening a downward arch.
    v_norm = (u - cx) / (IMG_W / 2)
    v = v - k1 * v_norm ** 2
    d   = pc[2]
    ok  = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H) & (d < 150)
    return u[ok], v[ok], d[ok]


def update(_=None):
    fx = sliders["fx"].val
    cx = sliders["cx"].val
    t1 = sliders["t1"].val
    t2 = sliders["t2"].val
    cy = sliders["cy"].val
    k1 = sliders["k1"].val
    u, v, d = project(fx, cx, t1, t2, cy, k1)
    if len(u):
        sc.set_offsets(np.column_stack([u, v]))
        sc.set_array(d)
    else:
        sc.set_offsets(np.empty((0, 2)))
    info_text.set_text(
        f"FX={fx:.0f}   CX={cx:.0f}   CY={cy:.0f}   "
        f"T1={t1:.2f}   T2={t2:.2f}   K1={k1:.1f}   "
        f"points: {len(u)}"
    )
    fig.canvas.draw_idle()


for s in sliders.values():
    s.on_changed(update)

update()


def on_close(event):
    fx = sliders["fx"].val
    cx = sliders["cx"].val
    t1 = sliders["t1"].val
    t2 = sliders["t2"].val
    cy = sliders["cy"].val
    k1 = sliders["k1"].val
    print("\n── Calibration values — paste into lidarOnCamera.py ──────────")
    print(f"FX = FY = {fx:.1f}")
    print(f"CX = {cx:.1f}")
    print(f"CY = {cy:.1f}")
    print(f"T  = np.array([0.000, {t1:.3f}, {t2:.3f}])")
    print(f"K1 = {k1:.3f}   # arc curvature correction")
    print("──────────────────────────────────────────────────────────────")


fig.canvas.mpl_connect("close_event", on_close)
print("Window open — adjust the six sliders to align the overlay.")
print("Close the window when done; calibration values will print here.\n")
plt.show()
