import sqlite3
import struct
import io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio_ffmpeg
from PIL import Image

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

DB           = "Data/2025-10-08_09-35_sensors_raw_0.db3"
POINT_STEP   = 32
LIDAR_OFFSET = 272    # confirmed correct CDR data-start offset
N_FRAMES     = 4000

# ── Camera intrinsics (1928×500 px, calibrated via calibrateLidar.py) ────────
IMG_W, IMG_H = 1928, 500
FX = 1500
FY = 1500
CX = 925
CY = 325
K1 = -25.000   # arc curvature correction — flattens projected scan-ring arch

K = np.array([[FX,  0, CX],
              [ 0, FY, CY],
              [ 0,  0,  1]], dtype=np.float64)

# ── LiDAR → camera_fr extrinsic ──────────────────────────────────────────────
# LiDAR frame: x = height (up+), y = lateral (right+), z = forward
# Camera frame: x = right,        y = down,             z = forward
R = np.array([[ 0,  1,  0],
              [-1,  0,  0],
              [ 0,  0,  1]], dtype=np.float64)

# Camera offset from LiDAR in camera frame [cam_x=right, cam_y=down, cam_z=fwd]
T = np.array([0.000, 1.837, 0.103])
# ─────────────────────────────────────────────────────────────────────────────

LIDAR_DTYPE = np.dtype([
    ("x",         "<f4"),
    ("y",         "<f4"),
    ("z",         "<f4"),
    ("pad",       "V4"),
    ("timestamp", "<i4"),
    ("intensity", "<f4"),
    ("scan_idx",  "<u2"),
    ("scan_id",   "<u2"),
    ("ring",      "<u2"),
    ("channel",   "u1"),
    ("pad2",      "V1"),
])

print("Connecting to database...")
conn = sqlite3.connect(DB)

lidar_id = conn.execute(
    "SELECT id FROM topics WHERE name=?",
    ("/sensor/lidar_front/points",)
).fetchone()[0]

cam_id = conn.execute(
    "SELECT id FROM topics WHERE name=?",
    ("/sensor/camera/camera_fr/image/compressed",)
).fetchone()[0]

print("Fetching data...")
lidar_rows = conn.execute(
    "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT ?",
    (lidar_id, N_FRAMES)
).fetchall()

cam_rows = conn.execute(
    "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
    (cam_id,)
).fetchall()
conn.close()

cam_ts = np.array([r[0] for r in cam_rows], dtype=np.int64)
print(f"Loaded {len(lidar_rows)} LiDAR frames, {len(cam_rows)} camera frames")


def parse_lidar(raw):
    payload  = raw[LIDAR_OFFSET:]
    usable   = (len(payload) // POINT_STEP) * POINT_STEP
    pts      = np.frombuffer(payload[:usable], dtype=LIDAR_DTYPE)
    valid    = (np.isfinite(pts["x"]) & np.isfinite(pts["y"]) &
                np.isfinite(pts["z"]) & (pts["z"] > 0))
    pts      = pts[valid]
    return (pts["x"].astype(np.float64),
            pts["y"].astype(np.float64),
            pts["z"].astype(np.float64),
            pts["intensity"].astype(np.float64))


def decode_image(raw):
    off  = 4 + 8
    slen = struct.unpack_from("<I", raw, off)[0]; off += 4
    off += slen; off = (off + 3) & ~3
    slen = struct.unpack_from("<I", raw, off)[0]; off += 4
    off += slen; off = (off + 3) & ~3
    dlen = struct.unpack_from("<I", raw, off)[0]; off += 4
    return np.array(Image.open(io.BytesIO(raw[off:off + dlen])))


def project(x, y, z):
    pts_cam  = R @ np.vstack([x, y, z]) + T[:, None]
    in_front = pts_cam[2] > 1.0
    pts_cam  = pts_cam[:, in_front]

    uv = K @ pts_cam
    u  = uv[0] / uv[2]
    v  = uv[1] / uv[2]
    d  = pts_cam[2]

    v_norm = (u - CX) / (IMG_W / 2)
    v = v - K1 * v_norm ** 2

    inside = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H) & (d < 150)
    return u[inside], v[inside], d[inside]


print("Setting up figure...")
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis("off")
fig.tight_layout(pad=0)
fig.patch.set_facecolor("black")

img_display = ax.imshow(np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8))
scatter     = ax.scatter([], [], s=4, c=[], cmap="turbo",
                         vmin=0, vmax=80, linewidths=0, alpha=0.9)
plt.colorbar(scatter, ax=ax, label="depth (m)", fraction=0.015, pad=0.01)
title = ax.set_title("", color="white", fontsize=10)


def update(i):
    if i % 10 == 0 or i == len(lidar_rows) - 1:
        print(f"  Frame {i + 1}/{len(lidar_rows)}")

    lidar_ts_ns, lidar_data = lidar_rows[i]
    idx     = int(np.argmin(np.abs(cam_ts - lidar_ts_ns)))
    cam_img = decode_image(cam_rows[idx][1])

    x, y, z, _ = parse_lidar(lidar_data)
    u, v, d     = project(x, y, z)

    img_display.set_data(cam_img)
    if len(u):
        scatter.set_offsets(np.column_stack([u, v]))
        scatter.set_array(d)
    else:
        scatter.set_offsets(np.empty((0, 2)))
    title.set_text(f"LiDAR → camera_fr   frame {i + 1}/{len(lidar_rows)}")
    return img_display, scatter, title


ani = animation.FuncAnimation(
    fig, update, frames=len(lidar_rows), interval=100, blit=True
)

output = "lidar_on_camera.mp4"
print(f"Saving {output} (this may take a while)...")
ani.save(output, writer="ffmpeg", fps=10, dpi=150)
print(f"Saved {output}")

print("Opening interactive window...")
plt.show()
print("Done.")
