"""
radarOccupancyGridBasic.py
==========================
Simple bird's-eye-view occupancy map from all 4 radar sensors.

Steps:
  1. Read radar points from the SQLite bag file
  2. Filter out weak / noisy returns
  3. Convert each sensor's points into a common vehicle frame
  4. Paint occupied cells on a 2D grid
  5. Save the image
"""

import sqlite3
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Settings ──────────────────────────────────────────────────────────────────
DB       = "Data/2025-10-08_09-35_sensors_raw_0.db3"
N_FRAMES = 200          # how many radar frames to fuse (200 ≈ 12 seconds)
RES      = 0.5          # grid cell size in metres
HALF     = 80.0         # grid extends ±80 m around the vehicle

# Quality filters
MIN_SNR  = 13.5         # minimum signal-to-noise ratio (dB)
MIN_CONF = 0.25         # minimum confidence (0–1)
MAX_R    = 150.0        # ignore returns beyond this range (m)

# ── Sensor transforms: rotate each sensor frame → vehicle frame ───────────────
# Vehicle frame:  x = forward,  y = left
# Sensor frame:   y = range (always ≥ 0),  x = lateral
#
# For each radar the rotation matrix R and mounting offset T are:
#   vehicle_xyz = R @ sensor_xyz + T

SENSORS = {
    "/sensor/radar_front/points": {
        "R": np.array([[ 0, 1, 0],   # sensor-y (range)    → vehicle-x (forward)
                       [-1, 0, 0],   # sensor-x (lateral)  → vehicle-y (left, flipped)
                       [ 0, 0, 1]]),
        "T": np.array([3.5, 0.0, 0.5]),   # front bumper position (m)
    },
    "/sensor/radar_back/points": {
        "R": np.array([[ 0,-1, 0],   # sensor-y → backward  → vehicle -x
                       [ 1, 0, 0],   # sensor-x → vehicle +y
                       [ 0, 0, 1]]),
        "T": np.array([-1.5, 0.0, 0.5]),  # rear bumper
    },
    "/sensor/radar_left/points": {
        "R": np.array([[1, 0, 0],    # sensor-x → vehicle +x
                       [0, 1, 0],    # sensor-y (range) → vehicle +y (left)
                       [0, 0, 1]]),
        "T": np.array([0.0, 1.0, 0.5]),   # left side
    },
    "/sensor/radar_right/points": {
        "R": np.array([[-1, 0, 0],   # sensor-x → vehicle -x
                       [ 0,-1, 0],   # sensor-y (range) → vehicle -y (right)
                       [ 0, 0, 1]]),
        "T": np.array([0.0, -1.0, 0.5]),  # right side
    },
}

# ── Step 1: read raw bytes from the bag ───────────────────────────────────────
def read_frames(db, topic, n):
    """Return the first n raw CDR messages for a topic."""
    conn = sqlite3.connect(db)
    tid  = conn.execute("SELECT id FROM topics WHERE name=?", (topic,)).fetchone()
    if tid is None:
        conn.close()
        return []
    rows = conn.execute(
        "SELECT data FROM messages WHERE topic_id=? ORDER BY rowid LIMIT ?",
        (tid[0], n)
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


# ── Step 2: parse one CDR PointCloud2 message ─────────────────────────────────
RADAR_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("velocity", "<f4"), ("snr", "<f4"), ("rcs", "<f4"),
    ("confidence", "<f4"), ("velocity_interval", "<f4"),
])  # 8 × 4 = 32 bytes per point

def parse(raw):
    """Decode CDR bytes → structured numpy array of radar points."""
    # Walk past the CDR/ROS2 header to find the data blob.
    # The header layout is fixed for these radar messages (248 bytes).
    off = 4                                           # CDR encapsulation
    off += 8                                          # stamp (sec + nanosec)
    slen = struct.unpack_from("<I", raw, off)[0]; off += 4
    off += slen                                       # frame_id string
    off = (off + 3) & ~3                              # align to 4
    _h  = struct.unpack_from("<I", raw, off)[0]; off += 4   # height
    n   = struct.unpack_from("<I", raw, off)[0]; off += 4   # width = num points
    nf  = struct.unpack_from("<I", raw, off)[0]; off += 4   # num fields
    for _ in range(nf):                               # skip field descriptors
        fl = struct.unpack_from("<I", raw, off)[0]; off += 4
        off += fl; off = (off + 3) & ~3
        off += 5;  off = (off + 3) & ~3; off += 4
    off += 1; off = (off + 3) & ~3                   # is_bigendian + align
    ps  = struct.unpack_from("<I", raw, off)[0]; off += 4   # point_step
    off += 4                                          # row_step
    dl  = struct.unpack_from("<I", raw, off)[0]; off += 4   # data length
    pts = np.frombuffer(raw[off: off + n * ps], dtype=RADAR_DTYPE)
    return pts


# ── Step 3: filter noisy / invalid returns ────────────────────────────────────
def filter_pts(pts):
    r    = np.sqrt(pts["x"]**2 + pts["y"]**2 + pts["z"]**2)
    keep = (r > 0.5) & (r < MAX_R) & (pts["snr"] >= MIN_SNR) & (pts["confidence"] >= MIN_CONF)
    return pts[keep]


# ── Main ──────────────────────────────────────────────────────────────────────
# Grid: rows = forward/backward axis, cols = left/right axis
n_cells = int(2 * HALF / RES)               # 320 × 320
grid    = np.zeros((n_cells, n_cells), dtype=np.int32)   # 0 = empty, >0 = hit count

def world_to_cell(x, y):
    """Convert vehicle-frame (x, y) in metres → (row, col) grid indices."""
    row = np.round((HALF - x) / RES).astype(int)   # x forward → row from top
    col = np.round((HALF - y) / RES).astype(int)   # y left    → col from left
    return row, col

print("Processing radar frames...")
for topic, cfg in SENSORS.items():
    R, T = cfg["R"], cfg["T"]
    frames = read_frames(DB, topic, N_FRAMES)
    print(f"  {topic.split('/')[-2]:12s}  {len(frames)} frames")

    for raw in frames:
        pts   = parse(raw)
        pts   = filter_pts(pts)
        if len(pts) == 0:
            continue

        # Rotate + translate into vehicle frame (only need x and y for BEV)
        s     = np.column_stack([pts["x"], pts["y"], pts["z"]])  # (N,3) sensor frame
        v     = (R @ s.T).T + T                                  # (N,3) vehicle frame
        vx, vy = v[:, 0], v[:, 1]

        # Paint hits on the grid
        row, col = world_to_cell(vx, vy)
        valid = (row >= 0) & (row < n_cells) & (col >= 0) & (col < n_cells)
        np.add.at(grid, (row[valid], col[valid]), 1)

print(f"\nGrid: {n_cells}×{n_cells} cells, {(grid > 0).sum()} occupied cells")

# ── Visualise ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 9), facecolor="#111111")
ax.set_facecolor("#111111")

# Colour map: black = empty, bright yellow = many hits
display = np.log1p(grid.astype(float))          # log scale so single hits still show
ax.imshow(
    display,
    cmap="hot",
    origin="upper",
    extent=[HALF, -HALF, -HALF, HALF],          # [left, right, bottom, top]
    interpolation="nearest",
    vmin=0,
)

# Vehicle outline
car_w, car_l = 2.0, 4.5
rect = mpatches.FancyBboxPatch(
    (-car_w / 2, -car_l / 2), car_w, car_l,
    boxstyle="round,pad=0.1",
    linewidth=2, edgecolor="cyan", facecolor="none",
)
ax.add_patch(rect)
ax.plot(0, 0, "c+", markersize=12, markeredgewidth=2)

ax.set_xlabel("y — lateral (m)  [left +]", color="white")
ax.set_ylabel("x — forward (m)", color="white")
ax.set_title(f"Radar Occupancy Map  —  {N_FRAMES} frames  ({N_FRAMES/16.7:.0f}s)",
             color="white")
ax.tick_params(colors="white")
ax.grid(color="#333333", linewidth=0.5, linestyle="--")
ax.set_xticks(np.arange(-HALF, HALF + 1, 20))
ax.set_yticks(np.arange(-HALF, HALF + 1, 20))

plt.tight_layout()
plt.savefig("radar_og_basic.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved  radar_og_basic.png")
plt.show()
