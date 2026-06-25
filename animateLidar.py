import os
import sqlite3
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

DB         = "Data/2025-10-08_09-35_sensors_raw_0.db3"
CACHE      = DB + ".lidar_cache_xyz.npz"   # 3-col [x,y,z]; delete old 2-col cache if present
POINT_STEP = 32
OFFSET     = 272

# Output canvas (wide aspect ratio suits a forward-facing perspective view)
W, H = 1280, 720

# Pinhole intrinsics for the virtual perspective camera
FX, FY = 600.0, 600.0
CX, CY = W / 2.0, H / 2.0

# LiDAR → camera frame:  x_lidar=up, y_lidar=right, z_lidar=fwd
#                        →  x_cam=right, y_cam=down,  z_cam=fwd
R = np.array([[ 0,  1,  0],
              [-1,  0,  0],
              [ 0,  0,  1]], dtype=np.float64)

TURBO = (plt.cm.turbo(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)  # (256,3) RGB LUT


def parse_frame(raw):
    # LiDAR frame: x=height(up+), y=lateral(right+), z=forward
    payload = raw[OFFSET:]
    usable  = (len(payload) // POINT_STEP) * POINT_STEP
    dtype   = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("skip", "V20")])
    pts     = np.frombuffer(payload[:usable], dtype=dtype)
    x = pts["x"].astype(np.float32)
    y = pts["y"].astype(np.float32)
    z = pts["z"].astype(np.float32)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > 1.0)
    return np.column_stack([x[mask], y[mask], z[mask]])   # (N, 3)


def load_frames():
    """Return (points, offsets) from cache if available, otherwise parse DB and cache."""
    if os.path.exists(CACHE):
        print(f"Loading cache from {CACHE}...")
        data = np.load(CACHE)
        print(f"Loaded {len(data['offsets']) - 1} frames from cache")
        return data["points"], data["offsets"]

    print("Connecting to database...")
    conn = sqlite3.connect(DB)
    topic_id = conn.execute(
        "SELECT id FROM topics WHERE name=?",
        ("/sensor/lidar_right/points",)
    ).fetchone()[0]

    print("Fetching LiDAR messages...")
    rows = conn.execute(
        "SELECT data FROM messages WHERE topic_id=? ORDER BY rowid LIMIT 2400",
        (topic_id,)
    ).fetchall()
    conn.close()
    print(f"Parsing {len(rows)} frames...")

    all_pts = []
    offsets = [0]
    for i, (raw,) in enumerate(rows):
        if i % 200 == 0:
            print(f"  Parsing frame {i + 1} / {len(rows)}")
        pts = parse_frame(raw)
        all_pts.append(pts)
        offsets.append(offsets[-1] + len(pts))

    points  = np.concatenate(all_pts, axis=0).astype(np.float32)
    offsets = np.array(offsets, dtype=np.int32)

    print(f"Saving cache to {CACHE}...")
    np.savez_compressed(CACHE, points=points, offsets=offsets)
    return points, offsets


def project(pts):
    """Project (N,3) LiDAR [x=up,y=right,z=fwd] points to pixel coords.
    Returns (u, v, depth) arrays for points inside the frame."""
    cam = R @ pts.T                          # (3, N) in camera frame
    in_front = cam[2] > 1.0
    cam = cam[:, in_front]
    depth = cam[2]

    u = FX * cam[0] / depth + CX
    v = FY * cam[1] / depth + CY

    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depth < 100.0)
    return u[inside], v[inside], depth[inside]


def render_frame(pts, label):
    """Render one perspective frame as a (H, W, 3) BGR numpy array."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    if len(pts) == 0:
        return frame

    u, v, depth = project(pts)
    if len(u) == 0:
        return frame

    # Map depth → turbo colour (0–80 m range)
    lut_idx = np.clip((depth / 80.0 * 255).astype(np.int32), 0, 255)
    colors  = TURBO[lut_idx]              # (N, 3) RGB
    colors_bgr = colors[:, ::-1]          # flip to BGR for OpenCV

    # Draw far-to-near so near points appear on top
    order = np.argsort(-depth)
    ui = u[order].astype(np.int32)
    vi = v[order].astype(np.int32)
    col = colors_bgr[order]

    # 3×3 square splat (vectorised, no Python loop)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            yy = np.clip(vi + dy, 0, H - 1)
            xx = np.clip(ui + dx, 0, W - 1)
            frame[yy, xx] = col

    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    return frame


points, offsets = load_frames()
n_frames = len(offsets) - 1

# --- Write MP4 with OpenCV ---
output = "lidar_animation.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output, fourcc, 10, (W, H))

print(f"Rendering and encoding {n_frames} frames with OpenCV...")
for i in range(n_frames):
    if i % 100 == 0 or i == n_frames - 1:
        print(f"  Frame {i + 1} / {n_frames}")
    lo, hi = offsets[i], offsets[i + 1]
    frame  = render_frame(points[lo:hi], f"LiDAR right — frame {i + 1} / {n_frames}")
    writer.write(frame)

writer.release()
print(f"Saved {output}")

# --- Interactive matplotlib preview (renders exact same frames via imshow) ---
print("Opening interactive window...")
fig, ax = plt.subplots(figsize=(12, 7))
ax.axis("off")
fig.patch.set_facecolor("black")
fig.tight_layout(pad=0)

blank   = np.zeros((H, W, 3), dtype=np.uint8)
display = ax.imshow(blank)
title   = ax.set_title("", color="white", fontsize=10)


def update(i):
    lo, hi = offsets[i], offsets[i + 1]
    frame  = render_frame(points[lo:hi], f"LiDAR right — frame {i + 1} / {n_frames}")
    display.set_data(frame[:, :, ::-1])   # BGR → RGB for matplotlib
    title.set_text(f"LiDAR right — frame {i + 1} / {n_frames}")
    return display, title


ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=100, blit=True)
plt.show()
print("Done.")
