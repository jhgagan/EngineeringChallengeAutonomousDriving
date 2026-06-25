"""
radarOccupancyGrid.py
=====================
Generates a bird's-eye-view (BEV) occupancy grid from the four automotive
radar sensors (front, back, left, right) stored in the ROS 2 SQLite bag.

Algorithm
---------
  • Log-odds Bayesian occupancy grid  (Thrun et al., "Probabilistic Robotics")
  • Free-space ray casting (Bresenham) from sensor origin → each detection
  • Radar-specific rotation/translation → common vehicle frame
      Vehicle frame convention: x = forward, y = left, z = up
  • Returns are filtered by SNR and confidence before grid updates

Sensor-frame convention (common to all four radars)
-----------------------------------------------------
  y = boresight / range direction   (always ≥ 0)
  x = lateral (perpendicular to boresight)
  z = elevation

Sensor transforms into vehicle frame
--------------------------------------
  radar_front  – faces +x_vehicle  →  v = R_front  · s  + t_front
  radar_back   – faces -x_vehicle  →  v = R_back   · s  + t_back
  radar_left   – faces +y_vehicle  →  v = R_left   · s  + t_left
  radar_right  – faces -y_vehicle  →  v = R_right  · s  + t_right

Outputs
-------
  radar_occupancy_grid.png        – static BEV snapshot (first N frames fused)
  radar_og_single_frame.png       – single-frame occupancy grid
  radar_occupancy_animation.mp4   – animated grid over time (optional)
"""

import sqlite3
import struct
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches

# ── Database ──────────────────────────────────────────────────────────────────
# Both files together cover the full ~240 s recording (each ~120 s at 16.7 Hz)
DB_PATHS = [
    "Data/2025-10-08_09-35_sensors_raw_0.db3",
    "Data/2025-10-08_09-35_sensors_raw_1.db3",
]
DB_PATH = DB_PATHS[0]   # kept for backward compatibility

# ── Occupancy grid parameters ─────────────────────────────────────────────────
RESOLUTION   = 0.5        # metres per cell
GRID_X_MIN   = -80.0      # vehicle-frame forward extent (metres)
GRID_X_MAX   =  80.0
GRID_Y_MIN   = -80.0      # vehicle-frame lateral extent (metres)
GRID_Y_MAX   =  80.0

# Log-odds update values
L_OCC   =  np.log(0.75 / 0.25)   # ≈  1.099  per occupied hit
L_FREE  =  np.log(0.25 / 0.75)   # ≈ -1.099  per free-ray cell
L_MIN   = -5.0                    # clamp to avoid over-confidence
L_MAX   =  5.0

# Sensor quality thresholds
MIN_SNR        = 13.5      # dB  — discard very weak returns
MIN_CONFIDENCE = 0.25      # 0–1 — discard low-confidence detections
MAX_RANGE      = 150.0     # metres — clip unrealistically far detections

# ── Sensor mounting: rotation matrices + translations in vehicle frame ────────
# Vehicle frame: x=forward, y=left, z=up
#
# Rotation convention:  v_vehicle = R · s_sensor  (+ translation)
# where  s_sensor = (x_sensor, y_sensor, z_sensor)^T
#
# For each sensor:  sensor y → boresight direction in vehicle frame
#                   sensor x → lateral (direction depends on sensor orientation)
#                   sensor z → up (shared convention)

# radar_front  boresight = +x_vehicle
#   s_y → v_x,  s_x → -v_y,  s_z → v_z
R_FRONT = np.array([[ 0,  1,  0],   # v_x = s_y
                    [-1,  0,  0],   # v_y = -s_x
                    [ 0,  0,  1]], dtype=np.float64)
T_FRONT = np.array([3.5, 0.0, 0.5])   # bumper offset (m)

# radar_back  boresight = -x_vehicle
#   s_y → -v_x,  s_x → +v_y,  s_z → v_z
R_BACK  = np.array([[ 0, -1,  0],   # v_x = -s_y
                    [ 1,  0,  0],   # v_y = s_x
                    [ 0,  0,  1]], dtype=np.float64)
T_BACK  = np.array([-1.5, 0.0, 0.5])  # rear bumper offset (m)

# radar_left  boresight = +y_vehicle
#   s_y → +v_y,  s_x → +v_x,  s_z → v_z
R_LEFT  = np.array([[ 1,  0,  0],   # v_x = s_x
                    [ 0,  1,  0],   # v_y = s_y
                    [ 0,  0,  1]], dtype=np.float64)
T_LEFT  = np.array([0.0, 1.0, 0.5])   # left side offset (m)

# radar_right  boresight = -y_vehicle
#   s_y → -v_y,  s_x → -v_x,  s_z → v_z
R_RIGHT = np.array([[-1,  0,  0],   # v_x = -s_x
                    [ 0, -1,  0],   # v_y = -s_y
                    [ 0,  0,  1]], dtype=np.float64)
T_RIGHT = np.array([0.0, -1.0, 0.5])  # right side offset (m)

SENSOR_CONFIG = {
    "/sensor/radar_front/points": (R_FRONT, T_FRONT),
    "/sensor/radar_back/points":  (R_BACK,  T_BACK),
    "/sensor/radar_left/points":  (R_LEFT,  T_LEFT),
    "/sensor/radar_right/points": (R_RIGHT, T_RIGHT),
}

# ── CDR PointCloud2 parser ────────────────────────────────────────────────────
RADAR_DTYPE = np.dtype([
    ("x",                 "<f4"),
    ("y",                 "<f4"),
    ("z",                 "<f4"),
    ("velocity",          "<f4"),
    ("snr",               "<f4"),
    ("rcs",               "<f4"),
    ("confidence",        "<f4"),
    ("velocity_interval", "<f4"),
])   # 8 fields × 4 bytes = 32 bytes/point


def _decode_cdr_string(raw: bytes, offset: int):
    """Read a CDR string (uint32 length + chars incl. null) and return
       (string, new_offset).  Advances offset past the null terminator."""
    length = struct.unpack_from("<I", raw, offset)[0]
    offset += 4
    s = raw[offset: offset + length - 1].decode("utf-8", errors="replace")
    offset += length
    return s, offset


def _align(offset: int, alignment: int) -> int:
    """Round up offset to the next multiple of alignment."""
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)


def parse_radar_frame(raw: bytes):
    """
    Decode a single ROS 2 CDR-encoded sensor_msgs/PointCloud2 radar message.

    Returns
    -------
    pts : np.ndarray of RADAR_DTYPE   (N,)   — raw sensor-frame points
    width : int                               — number of declared points
    """
    offset = 4   # skip 4-byte CDR encapsulation header

    # std_msgs/Header: stamp (sec + nanosec) + frame_id
    offset += 4 + 4    # stamp.sec + stamp.nanosec
    _, offset = _decode_cdr_string(raw, offset)   # frame_id
    offset = _align(offset, 4)

    # PointCloud2 fields: height, width
    height = struct.unpack_from("<I", raw, offset)[0]; offset += 4
    width  = struct.unpack_from("<I", raw, offset)[0]; offset += 4

    # PointField array: skip each field's name/offset/datatype/count
    num_fields = struct.unpack_from("<I", raw, offset)[0]; offset += 4
    for _ in range(num_fields):
        field_name_len = struct.unpack_from("<I", raw, offset)[0]; offset += 4
        offset += field_name_len           # name + null
        offset = _align(offset, 4)
        offset += 4 + 1                    # offset(uint32) + datatype(uint8)
        offset = _align(offset, 4)
        offset += 4                        # count(uint32)

    offset += 1                            # is_bigendian
    offset = _align(offset, 4)

    point_step = struct.unpack_from("<I", raw, offset)[0]; offset += 4
    _row_step   = struct.unpack_from("<I", raw, offset)[0]; offset += 4
    data_len    = struct.unpack_from("<I", raw, offset)[0]; offset += 4

    payload = raw[offset: offset + data_len]
    n_pts   = min(width, len(payload) // point_step)
    pts     = np.frombuffer(payload[: n_pts * point_step], dtype=RADAR_DTYPE)
    return pts, n_pts


def filter_returns(pts):
    """Remove invalid / zero / far / low-quality returns."""
    r = np.sqrt(pts["x"] ** 2 + pts["y"] ** 2 + pts["z"] ** 2)
    mask = (
        (r > 0.5)
        & (r < MAX_RANGE)
        & (pts["snr"]        >= MIN_SNR)
        & (pts["confidence"] >= MIN_CONFIDENCE)
        & np.isfinite(pts["x"])
        & np.isfinite(pts["y"])
        & np.isfinite(pts["z"])
    )
    return pts[mask]


def sensor_to_vehicle(pts, R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Transform sensor-frame points to vehicle frame.

    Parameters
    ----------
    pts : filtered RADAR_DTYPE array   (N,)
    R   : 3×3 rotation matrix
    T   : (3,) translation in vehicle frame

    Returns
    -------
    xyz_v : (N, 3)  vehicle-frame XYZ
    """
    s = np.column_stack([pts["x"], pts["y"], pts["z"]])   # (N, 3)
    v = (R @ s.T).T + T                                   # (N, 3)
    return v


# ── Grid helpers ──────────────────────────────────────────────────────────────
def make_empty_grid():
    """Create a zero-initialised log-odds grid."""
    n_rows = int((GRID_X_MAX - GRID_X_MIN) / RESOLUTION)
    n_cols = int((GRID_Y_MAX - GRID_Y_MIN) / RESOLUTION)
    grid   = np.zeros((n_rows, n_cols), dtype=np.float32)
    # Grid corner (row=0, col=0) corresponds to vehicle (GRID_X_MAX, GRID_Y_MAX)
    origin = (GRID_X_MAX, GRID_Y_MAX)
    return grid, origin


def update_grid(grid, origin, sensor_origin_v: np.ndarray, xyz_v: np.ndarray,
                confidence: np.ndarray):
    """
    Update the log-odds grid with one sensor's detections.

    Uses a vectorised DDA ray-trace for free-space cells and batched
    np.add.at for both free and occupied updates — much faster than
    a pure-Python Bresenham loop.

    Parameters
    ----------
    grid            : (H, W) float32 log-odds array  (modified in place)
    origin          : (ox, oy)  vehicle coords of grid cell (0,0) corner
    sensor_origin_v : (3,)  sensor position in vehicle frame
    xyz_v           : (N,3) vehicle-frame detections
    confidence      : (N,)  per-point confidence [0,1]
    """
    n_rows, n_cols = grid.shape
    N = len(xyz_v)
    if N == 0:
        return

    # Sensor position in grid cells
    sx, sy = float(sensor_origin_v[0]), float(sensor_origin_v[1])
    sr = int((origin[0] - sx) / RESOLUTION)
    sc = int((origin[1] - sy) / RESOLUTION)

    # Target cells for all detections (clamp to grid bounds)
    pr = np.round((origin[0] - xyz_v[:, 0]) / RESOLUTION).astype(int)
    pc = np.round((origin[1] - xyz_v[:, 1]) / RESOLUTION).astype(int)

    # ── Free-space ray cast (vectorised DDA) ──────────────────────────────
    dr = pr - sr        # (N,)
    dc = pc - sc        # (N,)
    n_steps = np.maximum(np.abs(dr), np.abs(dc))   # (N,) — max of |Δrow|,|Δcol|

    all_free_r: list = []
    all_free_c: list = []

    for i in range(N):
        ns = int(n_steps[i])
        if ns < 1:
            continue
        # Parameterise t ∈ [0, 1) so we exclude the endpoint (the occupied cell)
        t = np.arange(ns, dtype=np.float32) / ns
        rr = (sr + t * dr[i]).astype(np.int32)
        cc = (sc + t * dc[i]).astype(np.int32)
        valid = (rr >= 0) & (rr < n_rows) & (cc >= 0) & (cc < n_cols)
        all_free_r.append(rr[valid])
        all_free_c.append(cc[valid])

    if all_free_r:
        fr_all = np.concatenate(all_free_r)
        fc_all = np.concatenate(all_free_c)
        np.add.at(grid, (fr_all, fc_all), L_FREE)

    # ── Occupied update at detection points (vectorised) ─────────────────
    valid_occ = (pr >= 0) & (pr < n_rows) & (pc >= 0) & (pc < n_cols)
    if valid_occ.any():
        conf_weights = (0.5 + 0.5 * confidence[valid_occ]).astype(np.float32)
        np.add.at(grid, (pr[valid_occ], pc[valid_occ]), L_OCC * conf_weights)

    # Clamp once after all updates
    np.clip(grid, L_MIN, L_MAX, out=grid)


def log_odds_to_prob(grid: np.ndarray) -> np.ndarray:
    """Convert log-odds grid → probability [0, 1]."""
    return 1.0 - 1.0 / (1.0 + np.exp(grid))


def grid_to_image(grid: np.ndarray) -> np.ndarray:
    """
    Map log-odds grid → RGB image for display.
      • unknown  (log-odds ≈ 0)   → mid-grey  (128, 128, 128)
      • free     (log-odds < 0)   → white     (255, 255, 255)
      • occupied (log-odds > 0)   → dark/red  (  0,   0,   0) → (200, 0, 0)
    """
    prob  = log_odds_to_prob(grid)
    img   = np.zeros((*grid.shape, 3), dtype=np.uint8)

    # Unknown cells (near prior of 0.5)
    unk_mask  = np.abs(grid) < 0.2
    # Free cells
    free_mask = (grid < -0.2)
    # Occupied cells
    occ_mask  = (grid >  0.2)

    # Unknown → grey
    img[unk_mask]  = [160, 160, 160]

    # Free → light (white to grey gradient)
    free_intensity = np.clip(1.0 - prob[free_mask], 0.5, 1.0)
    img[free_mask, 0] = (free_intensity * 255).astype(np.uint8)
    img[free_mask, 1] = (free_intensity * 255).astype(np.uint8)
    img[free_mask, 2] = (free_intensity * 255).astype(np.uint8)

    # Occupied → dark red  (brighter red = higher confidence)
    occ_prob = prob[occ_mask]
    img[occ_mask, 0] = (occ_prob * 200).astype(np.uint8)
    img[occ_mask, 1] = 0
    img[occ_mask, 2] = 0

    return img


# ── Database I/O ─────────────────────────────────────────────────────────────
def load_all_radar_frames(db_paths):
    """
    Load all radar messages from one or more bag files, concatenated in
    chronological order.

    Parameters
    ----------
    db_paths : str | list[str]
        Path(s) to SQLite .db3 bag files.  Pass a list to span multiple
        files (e.g. both _0.db3 and _1.db3 for the full ~240 s recording).

    Returns
    -------
    all_data : dict  { topic_name → [(timestamp_ns, raw_bytes), ...] }
               Messages are sorted by timestamp across all files.
    """
    if isinstance(db_paths, str):
        db_paths = [db_paths]

    # Accumulate (timestamp, data) per topic across all files
    all_data = {t: [] for t in SENSOR_CONFIG}

    for db_path in db_paths:
        conn = sqlite3.connect(db_path)

        # Map topic name → numeric id for this file
        sensor_rows = {}
        for topic_name in SENSOR_CONFIG:
            row = conn.execute(
                "SELECT id FROM topics WHERE name=?", (topic_name,)
            ).fetchone()
            if row:
                sensor_rows[topic_name] = row[0]
            else:
                print(f"  [WARN] {db_path}: topic not found: {topic_name}")

        # Append messages from this file
        for topic_name, tid in sensor_rows.items():
            rows = conn.execute(
                "SELECT timestamp, data FROM messages "
                "WHERE topic_id=? ORDER BY timestamp",
                (tid,)
            ).fetchall()
            all_data[topic_name].extend(rows)

        conn.close()

    # Sort each sensor's list by timestamp (important when merging two files)
    for topic_name in all_data:
        all_data[topic_name].sort(key=lambda r: r[0])

    # Report
    for topic_name, rows in all_data.items():
        if not rows:
            continue
        dur_s = (rows[-1][0] - rows[0][0]) / 1e9
        print(f"  {topic_name.split('/')[-2]:12s}  "
              f"frames={len(rows):5d}  duration={dur_s:6.1f}s")

    return all_data


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    print("=" * 60)
    print("Radar Occupancy Grid Generator")
    print("=" * 60)

    # ── 1. Load radar data ────────────────────────────────────────────────────
    print("\nLoading radar data from database...")
    # Pass DB_PATHS (both files) to cover the full ~240 s recording.
    # Change to DB_PATH (single file) if you only want the first 120 s.
    all_data = load_all_radar_frames(DB_PATHS)

    # Compute the minimum number of frames across all sensors
    n_frames_per_sensor = {k: len(v) for k, v in all_data.items()}
    print(f"\n  Frames per sensor: {n_frames_per_sensor}")
    n_frames = min(n_frames_per_sensor.values())
    print(f"  Total available: {n_frames} frames  "
          f"≈ {n_frames / 16.7:.0f} seconds at 16.7 Hz")

    # ── 2. Build cumulative occupancy grid ───────────────────────────────────
    # N_CUMULATIVE = n_frames  →  uses every frame from both files (~240 s)
    # N_CUMULATIVE = 2000      →  first file only (~120 s)
    # N_CUMULATIVE = 200       →  first ~12 s (original default)
    N_CUMULATIVE = n_frames    # ← full 240 s; reduce this number to go faster
    print(f"\nBuilding cumulative occupancy grid ({N_CUMULATIVE} frames "
          f"≈ {N_CUMULATIVE / 16.7:.0f}s)...")
    cum_grid, origin = make_empty_grid()
    for fi in range(N_CUMULATIVE):
        if fi % 50 == 0:
            print(f"  Frame {fi}/{N_CUMULATIVE}")
        for topic_name, (R, T) in SENSOR_CONFIG.items():
            if topic_name not in all_data:
                continue
            raw = all_data[topic_name][fi][1]
            pts, _ = parse_radar_frame(raw)
            pts     = filter_returns(pts)
            if len(pts) == 0:
                continue
            xyz_v  = sensor_to_vehicle(pts, R, T)
            update_grid(cum_grid, origin, T, xyz_v, pts["confidence"])

    # ── 3. Save static cumulative grid ───────────────────────────────────────
    print("\nSaving cumulative occupancy grid...")
    _save_grid_figure(
        cum_grid, origin,
        title=f"Radar Occupancy Grid — {N_CUMULATIVE} frames fused",
        filename="radar_occupancy_grid.png",
    )

    # ── 4. Build single-frame grid ────────────────────────────────────────────
    print("\nBuilding single-frame occupancy grid (frame 100)...")
    sf_grid, origin = make_empty_grid()
    SINGLE_FRAME_IDX = min(100, n_frames - 1)
    for topic_name, (R, T) in SENSOR_CONFIG.items():
        if topic_name not in all_data:
            continue
        raw = all_data[topic_name][SINGLE_FRAME_IDX][1]
        pts, _ = parse_radar_frame(raw)
        pts    = filter_returns(pts)
        if len(pts) == 0:
            continue
        xyz_v = sensor_to_vehicle(pts, R, T)
        update_grid(sf_grid, origin, T, xyz_v, pts["confidence"])

    _save_grid_figure(
        sf_grid, origin,
        title=f"Radar Occupancy Grid — single frame (idx {SINGLE_FRAME_IDX})",
        filename="radar_og_single_frame.png",
    )

    # ── 5. Animated version ───────────────────────────────────────────────────
    print("\nBuilding animated occupancy grid (first 400 frames)...")
    _build_animation(all_data, n_frames, origin)

    print("\nDone.  Outputs:")
    print("  radar_occupancy_grid.png")
    print("  radar_og_single_frame.png")
    print("  radar_occupancy_animation.mp4")


def _save_grid_figure(grid, origin, title, filename):
    """Render and save an occupancy grid figure."""
    n_rows, n_cols = grid.shape

    # Axis extents (vehicle frame)
    x_max = GRID_X_MAX
    x_min = GRID_X_MIN
    y_max = GRID_Y_MAX
    y_min = GRID_Y_MIN

    img = grid_to_image(grid)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#111111")
    ax.set_facecolor("#111111")

    # imshow: rows from top → x decreasing downward; cols left → y decreasing rightward
    ax.imshow(
        img,
        extent=[y_max, y_min, x_min, x_max],   # [left, right, bottom, top]
        origin="upper",
        interpolation="nearest",
    )

    # Vehicle outline (rectangle)
    car_w, car_l = 2.0, 4.5
    rect = mpatches.FancyBboxPatch(
        (-car_w / 2, -car_l / 2), car_w, car_l,
        boxstyle="round,pad=0.1",
        linewidth=2, edgecolor="cyan", facecolor="none",
    )
    ax.add_patch(rect)
    ax.plot(0, 0, "c+", markersize=10, markeredgewidth=2)

    # Sensor positions
    sensor_positions = {
        "front": (T_FRONT[0],  T_FRONT[1]),
        "back":  (T_BACK[0],   T_BACK[1]),
        "left":  (T_LEFT[0],   T_LEFT[1]),
        "right": (T_RIGHT[0],  T_RIGHT[1]),
    }
    for name, (sx, sy) in sensor_positions.items():
        ax.plot(sy, sx, "y^", markersize=8, markeredgewidth=1.5,
                markeredgecolor="black")
        ax.text(sy + 1, sx, name, color="yellow", fontsize=7)

    # Grid lines
    ax.set_xticks(np.arange(y_min, y_max + 1, 20), minor=False)
    ax.set_yticks(np.arange(x_min, x_max + 1, 20), minor=False)
    ax.grid(color="#444444", linewidth=0.4, linestyle="--")

    ax.set_xlabel("y — lateral (m)  [left +]", color="white", fontsize=11)
    ax.set_ylabel("x — forward (m)", color="white", fontsize=11)
    ax.set_title(title, color="white", fontsize=12, pad=10)
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")

    # Legend
    legend_patches = [
        mpatches.Patch(color=(160/255, 160/255, 160/255), label="Unknown"),
        mpatches.Patch(color="white",                       label="Free"),
        mpatches.Patch(color=(200/255, 0, 0),               label="Occupied"),
        mpatches.Patch(color="cyan",                         label="Ego vehicle"),
        mpatches.Patch(color="yellow",                       label="Radar sensor"),
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              facecolor="#222222", labelcolor="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {filename}")


def _build_animation(all_data, n_frames, origin):
    """Build and save an animated occupancy grid MP4."""
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    # N_ANIM = n_frames    →  animate full ~240 s (both files)
    # N_ANIM = 2000        →  first file only (~120 s)
    # N_ANIM = 400         →  original ~24 s default
    N_ANIM   = n_frames    # ← full 240 s; STRIDE controls render speed
    STRIDE   = 8            # render 1 in every 8 frames → 500 rendered frames for 4000 total
    anim_frames = list(range(0, N_ANIM, STRIDE))

    print(f"  Rendering {len(anim_frames)} animation frames (stride={STRIDE})...")

    # Running log-odds grid for animation
    run_grid, _ = make_empty_grid()

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#111111")
    ax.set_facecolor("#111111")

    dummy_img = grid_to_image(run_grid)
    im = ax.imshow(
        dummy_img,
        extent=[GRID_Y_MAX, GRID_Y_MIN, GRID_X_MIN, GRID_X_MAX],
        origin="upper",
        interpolation="nearest",
        animated=True,
    )

    # Vehicle / sensor decorations (static)
    car_w, car_l = 2.0, 4.5
    rect = mpatches.FancyBboxPatch(
        (-car_w / 2, -car_l / 2), car_w, car_l,
        boxstyle="round,pad=0.1",
        linewidth=2, edgecolor="cyan", facecolor="none",
    )
    ax.add_patch(rect)
    ax.plot(0, 0, "c+", markersize=10, markeredgewidth=2)
    ax.set_xlabel("y — lateral (m)  [left +]", color="white")
    ax.set_ylabel("x — forward (m)", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#444444", linewidth=0.4, linestyle="--")
    ax.set_xticks(np.arange(GRID_Y_MIN, GRID_Y_MAX + 1, 20))
    ax.set_yticks(np.arange(GRID_X_MIN, GRID_X_MAX + 1, 20))
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")

    title_obj = ax.set_title("", color="white", fontsize=10)

    # Track processed frame indices to avoid re-processing
    last_fi = [-1]

    def anim_update(anim_fi):
        fi = anim_frames[anim_fi]
        # Accumulate all frames up to fi
        for step in range(last_fi[0] + 1, fi + 1):
            for topic_name, (R, T) in SENSOR_CONFIG.items():
                if topic_name not in all_data:
                    continue
                raw = all_data[topic_name][step][1]
                pts, _ = parse_radar_frame(raw)
                pts    = filter_returns(pts)
                if len(pts) == 0:
                    continue
                xyz_v = sensor_to_vehicle(pts, R, T)
                update_grid(run_grid, origin, T, xyz_v, pts["confidence"])
        last_fi[0] = fi

        img = grid_to_image(run_grid)
        im.set_data(img)
        title_obj.set_text(
            f"Radar Occupancy Grid — frame {fi + 1} / {N_ANIM}  "
            f"(res={RESOLUTION}m  grid={GRID_X_MAX-GRID_X_MIN:.0f}×"
            f"{GRID_Y_MAX-GRID_Y_MIN:.0f}m)"
        )
        if anim_fi % 20 == 0:
            print(f"    animation frame {anim_fi + 1}/{len(anim_frames)}")
        return im, title_obj

    ani = animation.FuncAnimation(
        fig, anim_update,
        frames=len(anim_frames),
        interval=150,
        blit=True,
        repeat=False,
    )

    output = "radar_occupancy_animation.mp4"
    ani.save(output, writer="ffmpeg", fps=10, dpi=120)
    print(f"  Saved {output}")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
