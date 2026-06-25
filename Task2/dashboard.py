"""
dashboard.py  —  3-sensor fusion dashboard  (live animation)
=============================================================
Uses ALL 7 cameras, ALL 3 LiDARs, ALL 4 radars.

Layout
------
  ┌─── Camera mosaic (FL FR │ CL CR RL RR R) ───┬─ Radar 4× ─┬─ HUD ─┐
  └─────── LiDAR 3-sensor BEV ─────────────────┘             │        │

Usage
-----
    python3 Task2/dashboard.py              # animate all frames (live window)
    python3 Task2/dashboard.py --fps 3      # playback speed (default 5)
    python3 Task2/dashboard.py --start 268  # start frame
    python3 Task2/dashboard.py --save 268   # save single frame → dashboard.png
    python3 Task2/dashboard.py --video      # render all frames → dashboard.mp4
    python3 Task2/dashboard.py --video dashboard_yolo.mp4 --video-fps 15 --yolo
    python3 Task2/dashboard.py --yolo       # enable YOLOv8 detection (slow)
"""

import os, sys, argparse, sqlite3, warnings
import h5py, numpy as np, pandas as pd
import matplotlib

# Set non-interactive backend BEFORE importing pyplot when rendering to file
if "--video" in sys.argv or "--save" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import matplotlib.animation as animation_mod

warnings.filterwarnings("ignore")

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False

try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    FFMPEG_OK = True
except Exception:
    FFMPEG_OK = False

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "Data")
DB    = os.path.join(_DATA, "2025-10-08_09-35_sensors_raw_1.db3")
MAT   = os.path.join(_DATA, "ego.mat")

# ── sensor topics — 7 cameras · 3 LiDARs · 4 radars ─────────────────────────
# Cameras: [FL, FR, CL, CR, RL, RR, R]  — FR is the primary (in CSV timestamps)
CAM_TOPICS = [
    "/sensor/camera/camera_fl/image/compressed",   # 0  Front-Left   (top row)
    "/sensor/camera/camera_fr/image/compressed",   # 1  Front-Right  (top row, primary)
    "/sensor/camera/camera_cl/image/compressed",   # 2  Center-Left  (strip)
    "/sensor/camera/camera_cr/image/compressed",   # 3  Center-Right (strip)
    "/sensor/camera/camera_rl/image/compressed",   # 4  Rear-Left    (strip)
    "/sensor/camera/camera_rr/image/compressed",   # 5  Rear-Right   (strip)
    "/sensor/camera/camera_r/image/compressed",    # 6  Rear         (strip)
]
CAM_LABELS = ["FL", "FR", "CL", "CR", "RL", "RR", "R"]
CAM_PRIMARY = 1   # index of camera whose timestamp is stored in the CSV

LIDAR_TOPICS = [
    "/sensor/lidar_front/points",   # primary
    "/sensor/lidar_right/points",
    "/sensor/lidar_left/points",
]

RADAR_TOPICS = [
    "/sensor/radar_front/points",   # 0  primary  (opponent detection)
    "/sensor/radar_back/points",    # 1
    "/sensor/radar_right/points",   # 2
    "/sensor/radar_left/points",    # 3
]

# ── data formats ──────────────────────────────────────────────────────────────
# LiDAR: x=height(up+)  y=lateral(right+)  z=forward   [from animateLidar.py]
LIDAR_DTYPE  = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("skip","V20")])
LIDAR_OFFSET = 272

RADAR_DTYPE = np.dtype([
    ("x","<f4"),("y","<f4"),("z","<f4"),("velocity","<f4"),
    ("snr","<f4"),("rcs","<f4"),("confidence","<f4"),("vint","<f4"),
])

# ── args ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--fps",   type=float, default=5.0,
                help="Playback speed (fps, default 5)")
ap.add_argument("--start", type=int,   default=0,
                help="Start frame index")
ap.add_argument("--save",  type=int,   default=None,
                help="Save single frame to dashboard.png")
ap.add_argument("--yolo",      action="store_true",
                help="Enable YOLOv8 detection")
ap.add_argument("--video",     type=str, nargs="?", const="dashboard.mp4",
                metavar="OUT.mp4",
                help="Render all frames to MP4 (default: dashboard.mp4)")
ap.add_argument("--video-fps", type=int, default=10,
                help="Video frame rate (default 10)")
args = ap.parse_args()

VIDEO    = args.video is not None
ANIMATE  = args.save is None and not VIDEO
USE_YOLO = args.yolo and CV2_OK and YOLO_OK

# ── check prerequisites ───────────────────────────────────────────────────────
cache_f = os.path.join(_HERE, "all_frames_measurements.csv")
ekf_f   = os.path.join(_HERE, "all_frames_ekf.csv")
if not os.path.exists(cache_f):
    sys.exit("  Run run_pipeline.py first (builds all_frames_measurements.csv).")

# ── load cached data ──────────────────────────────────────────────────────────
meas   = pd.read_csv(cache_f)
ekf_df = pd.read_csv(ekf_f) if os.path.exists(ekf_f) else None

N_FRAMES = len(meas)
t0_ns    = float(meas["camera"].iloc[0])

# ── ego state from ego.mat ────────────────────────────────────────────────────
_f          = h5py.File(MAT, "r")
_est        = _f["log/estimation"]
abs_ts_mat  = float(_f["log/time_offset_nsec"][0, 0]) + _est["bag_stamp"][0] * 1e9
vx_ego_all  = _est["vx"][0]
vy_ego_all  = _est["vy"][0]
ax_ego_all  = _est["ax"][0]
yr_all      = _est["yaw_rate"][0]
_f.close()

# ── open DB (kept open across all frames) ─────────────────────────────────────
_conn = sqlite3.connect(DB)

def _tid(name):
    r = _conn.execute("SELECT id FROM topics WHERE name=?", (name,)).fetchone()
    return r[0] if r else None

tid_cams   = [_tid(t) for t in CAM_TOPICS]
tid_lidars = [_tid(t) for t in LIDAR_TOPICS]
tid_radars = [_tid(t) for t in RADAR_TOPICS]

def get_blob(topic_id, ts):
    """Exact-timestamp lookup."""
    if topic_id is None: return None
    r = _conn.execute(
        "SELECT data FROM messages WHERE topic_id=? AND timestamp=? LIMIT 1",
        (topic_id, int(ts))
    ).fetchone()
    return r[0] if r else None

_WINDOW_NS = 200_000_000   # ±200 ms search window (uses timestamp_idx)

def get_nearest_blob(topic_id, ts):
    """Nearest-timestamp lookup (used for non-primary sensors).

    Uses a ±200 ms window so the existing timestamp_idx is effective
    (full-scan ORDER BY ABS without a window takes ~15 s per query).
    Falls back to a full scan if no row is found in the window.
    """
    if topic_id is None: return None
    ts = int(ts)
    r = _conn.execute(
        "SELECT data FROM messages "
        "WHERE timestamp BETWEEN ? AND ? AND topic_id=? "
        "ORDER BY ABS(timestamp - ?) LIMIT 1",
        (ts - _WINDOW_NS, ts + _WINDOW_NS, topic_id, ts)
    ).fetchone()
    if r: return r[0]
    # fallback (rare — topic not synchronized or large gap)
    r = _conn.execute(
        "SELECT data FROM messages WHERE topic_id=? "
        "ORDER BY ABS(timestamp - ?) LIMIT 1",
        (topic_id, ts)
    ).fetchone()
    return r[0] if r else None

# ── YOLO model (loaded once if requested) ─────────────────────────────────────
_yolo_model = None
if USE_YOLO:
    print("  Loading YOLOv8n …")
    _yolo_model = YOLO("yolov8n.pt")

# ── colours / style ───────────────────────────────────────────────────────────
BG, AX  = "#0a0c14", "#13161f"
GRID    = "#22263a"
TXT     = "#e8eaf0"
C_OPP   = "#f97316"
C_EGO   = "#22c55e"
C_LID   = "#38bdf8"
C_RAD   = "#c084fc"
C_BBOX  = "#fbbf24"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

# ── figure (created once, updated each frame) ─────────────────────────────────
_cbar = [None]   # persistent reference so we can .remove() before recreating

fig = plt.figure(figsize=(22, 12), facecolor=BG)
gs  = gridspec.GridSpec(
    2, 3,
    figure=fig,
    width_ratios=[2.8, 1.6, 1.4],
    height_ratios=[1.6, 1.0],
    hspace=0.38, wspace=0.28,
    left=0.04, right=0.98, top=0.91, bottom=0.05,
)

ax_cam = fig.add_subplot(gs[0, 0])
ax_lid = fig.add_subplot(gs[1, 0])
ax_rad = fig.add_subplot(gs[:, 1])
ax_hud = fig.add_subplot(gs[:, 2])

# ═══════════════════════════════════════════════════════════════════════════════
# DECODE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _decode_cam(blob):
    """Compressed JPEG blob → RGB ndarray, or None."""
    if not blob or not CV2_OK: return None
    j = blob.find(b'\xff\xd8\xff')
    if j < 0: return None
    arr = np.frombuffer(blob[j:], dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None: return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def _decode_lidar(blob):
    """LiDAR blob → dict(lat, fwd, ht) in vehicle frame, or None."""
    if not blob: return None
    payload = blob[LIDAR_OFFSET:]
    usable  = (len(payload) // 32) * 32
    if not usable: return None
    pts  = np.frombuffer(payload[:usable], dtype=LIDAR_DTYPE)
    lh   = pts["x"].astype(float)   # height  (up+)
    llat = pts["y"].astype(float)   # lateral (right+)
    lfwd = pts["z"].astype(float)   # forward
    scene = (lfwd > 0) & (lfwd < 80) & (np.abs(llat) < 50) & (lh > -5) & (lh < 5)
    if not scene.any(): return None
    return dict(lat=llat[scene], fwd=lfwd[scene], ht=lh[scene])

def _decode_radar(blob):
    """Radar blob → dict(x, y, v, conf), or None."""
    if not blob: return None
    payload = blob[220:]
    usable  = (len(payload) // 32) * 32
    if not usable: return None
    pts = np.frombuffer(payload[:usable], dtype=RADAR_DTYPE)
    return dict(
        x=pts["x"].astype(float), y=pts["y"].astype(float),
        v=pts["velocity"].astype(float), conf=pts["confidence"].astype(float),
    )

# ═══════════════════════════════════════════════════════════════════════════════
# CAMERA MOSAIC BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _build_mosaic(imgs, labels):
    """
    Build a composite image from 7 camera frames.

    Layout:
      ┌────────────────────────────────────────────┐
      │      FL (large)    │      FR (large)       │  ← top row
      ├────────────────────────────────────────────┤
      │  CL  │  CR  │  RL  │  RR  │      R        │  ← strip row
      └────────────────────────────────────────────┘
    """
    W_OUT  = 1928   # composite pixel width (matches a single camera)
    H_TOP  = 230    # height of top-row cameras (FL, FR)
    H_BOT  = 90     # height of strip cameras
    GAP    = 3      # gap between top and strip rows
    BANNER = 18     # label banner height

    def _thumb(img, w, h, label):
        """Resize img to (w, h) and stamp a label banner."""
        if img is None:
            t = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(t, "N/A", (w//2 - 18, h//2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
        else:
            t = cv2.resize(img, (w, h))
        # dark banner + white label text
        cv2.rectangle(t, (0, 0), (w, BANNER), (15, 18, 30), -1)
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
        cv2.putText(t, label, ((w - tw)//2, BANNER - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 210, 255), 1)
        return t

    # ── top row: FL (0) and FR (1) ────────────────────────────────────────
    w_top = W_OUT // 2 - GAP // 2
    top_tiles = [
        _thumb(imgs[0] if len(imgs) > 0 else None, w_top, H_TOP, labels[0]),
        _thumb(imgs[1] if len(imgs) > 1 else None, W_OUT - w_top - GAP, H_TOP, labels[1]),
    ]
    row0 = np.concatenate(
        [top_tiles[0], np.zeros((H_TOP, GAP, 3), np.uint8), top_tiles[1]], axis=1
    )

    # ── strip row: CL(2) CR(3) RL(4) RR(5) R(6) ─────────────────────────
    n_strip = 5
    strip_imgs = [imgs[i] if i < len(imgs) else None for i in range(2, 7)]
    strip_lbls = labels[2:7]
    w_each = W_OUT // n_strip
    tiles  = []
    for idx in range(n_strip):
        w_t = W_OUT - (n_strip - 1) * w_each if idx == n_strip - 1 else w_each
        tiles.append(_thumb(strip_imgs[idx], w_t - (GAP if idx < n_strip - 1 else 0),
                            H_BOT, strip_lbls[idx] if idx < len(strip_lbls) else "?"))
        if idx < n_strip - 1:
            tiles.append(np.zeros((H_BOT, GAP, 3), np.uint8))
    row1 = np.concatenate(tiles, axis=1)
    # trim/pad to exactly W_OUT
    if row1.shape[1] > W_OUT:
        row1 = row1[:, :W_OUT]
    elif row1.shape[1] < W_OUT:
        row1 = np.concatenate(
            [row1, np.zeros((H_BOT, W_OUT - row1.shape[1], 3), np.uint8)], axis=1
        )

    gap_px = np.zeros((GAP, W_OUT, 3), np.uint8)
    return np.concatenate([row0, gap_px, row1], axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# PER-FRAME RENDER
# ═══════════════════════════════════════════════════════════════════════════════

def render(fi):
    """Decode all sensors and redraw every panel for frame index fi."""
    ax_cam.cla(); ax_lid.cla(); ax_rad.cla(); ax_hud.cla()

    cam_ts   = int(meas["camera"].iloc[fi])
    lidar_ts = int(meas["lidar_ts"].iloc[fi])
    radar_ts = int(meas["radar_ts"].iloc[fi])
    t_sec    = (cam_ts - t0_ns) / 1e9

    # ── ego state (interpolated from MAT) ─────────────────────────────────
    q       = float(cam_ts)
    vx_ego  = float(np.interp(q, abs_ts_mat, vx_ego_all))
    vy_ego  = float(np.interp(q, abs_ts_mat, vy_ego_all))
    ax_ego  = float(np.interp(q, abs_ts_mat, ax_ego_all))
    yr_ego  = float(np.interp(q, abs_ts_mat, yr_all))
    ego_spd = float(np.hypot(vx_ego, vy_ego))
    ego_kph = ego_spd * 3.6

    # ── decode all 4 radars ───────────────────────────────────────────────
    rad_dec = [_decode_radar(get_nearest_blob(tid, radar_ts)) for tid in tid_radars]

    # merge for display
    rad_all = dict(x=np.empty(0), y=np.empty(0), v=np.empty(0), conf=np.empty(0))
    for pts in rad_dec:
        if pts:
            for k in rad_all:
                rad_all[k] = np.concatenate([rad_all[k], pts[k]])
    has_radar = rad_all["x"].size > 0

    # opponent target from FRONT radar only (x: 3–20 m ahead of ego)
    radar_tgt = None
    front_pts = rad_dec[0]   # radar_front
    if front_pts:
        rx, ry = front_pts["x"], front_pts["y"]
        rv, rc = front_pts["v"], front_pts["conf"]
        fwd = (rx > 3.0) & (rx < 20.0)
        if fwd.any():
            ni = np.argmin(np.hypot(rx[fwd], ry[fwd]))
            radar_tgt = (float(rx[fwd][ni]), float(ry[fwd][ni]),
                         float(rv[fwd][ni]), float(rc[fwd][ni]))

    # ── decode each LiDAR separately (no blind merge — different extrinsics) ──
    # NOTE: right/left LiDARs are shown in their own sensor frame because no
    #       extrinsic calibration matrices are available for them.
    #       Opponent-cluster detection uses FRONT LiDAR only (known geometry).
    lidar_dec = [_decode_lidar(get_nearest_blob(tid, lidar_ts)) for tid in tid_lidars]
    has_lidar = any(d is not None for d in lidar_dec)

    # LiDAR opponent cluster — front LiDAR only (index 0)
    lidar_bbox    = None
    lidar_cluster = None
    front_lid = lidar_dec[0]
    if front_lid and radar_tgt:
        z_s =  radar_tgt[0]          # radar x → lidar z (forward)
        y_s = -radar_tgt[1]          # radar y (left+) → lidar y (right+)
        ll, lf, lh = front_lid["lat"], front_lid["fwd"], front_lid["ht"]
        cl  = ((np.abs(lf - z_s) < 3.0) &
               (np.abs(ll - y_s) < 3.0) &
               (lh > -1.5) & (lh < 2.5))
        if cl.sum() >= 5:
            lidar_cluster = dict(lat=ll[cl], fwd=lf[cl])
            lidar_bbox    = (float(ll[cl].min()), float(ll[cl].max()),
                             float(lf[cl].min()), float(lf[cl].max()))

    # ── decode all 7 cameras ──────────────────────────────────────────────
    cam_imgs = []
    for i, tid in enumerate(tid_cams):
        blob = (get_blob(tid, cam_ts)
                if i == CAM_PRIMARY
                else get_nearest_blob(tid, cam_ts))
        cam_imgs.append(_decode_cam(blob))

    # YOLO detection on every camera image that has data
    if USE_YOLO and _yolo_model:
        for i, img in enumerate(cam_imgs):
            if img is None:
                continue
            draw = img.copy()
            dets = []
            for cls_f, conf_f in [([2,3,5,7], 0.10), ([2,3,5,7], 0.02), (None, 0.01)]:
                kw = dict(imgsz=1920, verbose=False, conf=conf_f)
                if cls_f is not None:
                    kw["classes"] = cls_f
                res  = _yolo_model(img, **kw)[0]
                seen = set()
                for box in res.boxes:
                    x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
                    key = (x1//10, y1//10, x2//10, y2//10)
                    if key in seen: continue
                    seen.add(key)
                    cf   = float(box.conf[0])
                    name = _yolo_model.names[int(box.cls[0])]
                    dets.append((x1,y1,x2,y2,cf,name))
                if dets: break
            for x1,y1,x2,y2,cf,name in dets:
                cv2.rectangle(draw, (x1,y1), (x2,y2), (0,230,80), 3)
                lbl = f"{name} {cf:.2f}"
                (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(draw, (x1, y1-th-6), (x1+tw+4, y1), (0,180,60), -1)
                cv2.putText(draw, lbl, (x1+2, y1-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10,10,10), 2)
            cam_imgs[i] = draw

    # build 7-camera mosaic
    mosaic = _build_mosaic(cam_imgs, CAM_LABELS) if CV2_OK else None

    # ── EKF / stats ───────────────────────────────────────────────────────
    opp_rng  = float(np.hypot(radar_tgt[0], radar_tgt[1])) if radar_tgt else float("nan")
    opp_spd  = abs(float(radar_tgt[2])) if radar_tgt else float("nan")
    opp_conf = float(radar_tgt[3]) if radar_tgt else float("nan")
    if ekf_df is not None:
        opp_ekf_rng = float(ekf_df["range_m"].iloc[fi])
        opp_ekf_spd = float(ekf_df["speed_m_s"].iloc[fi])
        opp_heading = float(ekf_df["heading_rel_deg"].iloc[fi])
    else:
        opp_ekf_rng = opp_rng
        opp_ekf_spd = opp_spd
        opp_heading = 0.0

    # ══════════════════════════════════════════════════════════════════════
    # PANEL A — Camera array mosaic
    # ══════════════════════════════════════════════════════════════════════
    ax_cam.set_facecolor(AX)
    ax_cam.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax_cam.spines.values():
        sp.set_color(C_OPP); sp.set_linewidth(2)
    cam_title = ("CAMERA ARRAY  ·  7 sensors  ·  YOLOv8"
                 if USE_YOLO else "CAMERA ARRAY  ·  7 sensors")
    ax_cam.set_title(cam_title, color=TXT, fontsize=10, fontweight="bold", pad=5)

    if mosaic is not None:
        ax_cam.imshow(mosaic, aspect="auto")
    else:
        ax_cam.text(0.5, 0.5, "No camera data (cv2 not available)",
                    ha="center", va="center", color=TXT,
                    transform=ax_cam.transAxes, fontsize=12)

    ax_cam.text(0.01, 0.03, "📷  7 Cameras",
                transform=ax_cam.transAxes, color=C_EGO,
                fontsize=8, va="bottom",
                bbox=dict(fc=AX, ec=GRID, alpha=0.8, pad=3))

    # ══════════════════════════════════════════════════════════════════════
    # PANEL B — LiDAR BEV (3-sensor merged)
    # ══════════════════════════════════════════════════════════════════════
    ax_lid.set_facecolor(AX)
    ax_lid.tick_params(colors=TXT, labelsize=8)
    for sp in ax_lid.spines.values(): sp.set_color(GRID)
    ax_lid.set_title(
        "LiDAR  ·  3 sensors  ·  BEV  [m]  "
        "(R/L in sensor frame — no extrinsic cal)",
        color=TXT, fontsize=9, fontweight="bold", pad=5)
    ax_lid.grid(color=GRID, lw=0.5, ls="--", alpha=0.6)

    # Per-sensor style: front → height-coloured; right/left → solid colour
    LID_STYLES = [
        dict(cmap="cool",   s=0.4, alpha=0.55, vmin=-1.5, vmax=2.5, label="Front LiDAR"),
        dict(color="#f97316", s=0.3, alpha=0.35, label="Right LiDAR (sensor frame)"),
        dict(color="#22c55e", s=0.3, alpha=0.35, label="Left LiDAR  (sensor frame)"),
    ]
    if has_lidar:
        for i, (pts, sty) in enumerate(zip(lidar_dec, LID_STYLES)):
            if pts is None: continue
            if i == 0:      # front: colour by height
                hc = np.clip(pts["ht"], sty["vmin"], sty["vmax"])
                ax_lid.scatter(pts["lat"], pts["fwd"],
                               c=hc, cmap=sty["cmap"],
                               s=sty["s"], alpha=sty["alpha"],
                               vmin=sty["vmin"], vmax=sty["vmax"],
                               rasterized=True, label=sty["label"])
            else:           # side: solid colour, raw sensor frame
                ax_lid.scatter(pts["lat"], pts["fwd"],
                               c=sty["color"], s=sty["s"], alpha=sty["alpha"],
                               rasterized=True, label=sty["label"])

        # Opponent cluster (front LiDAR only)
        if lidar_cluster:
            ax_lid.scatter(lidar_cluster["lat"], lidar_cluster["fwd"],
                           c=C_OPP, s=4, alpha=0.9, zorder=4, label="Opponent")
        if lidar_bbox:
            lat_mn, lat_mx, fwd_mn, fwd_mx = lidar_bbox
            ax_lid.add_patch(Rectangle(
                (lat_mn, fwd_mn), lat_mx - lat_mn, fwd_mx - fwd_mn,
                lw=2, edgecolor=C_BBOX, facecolor=C_BBOX, alpha=0.12, zorder=5))
            ax_lid.plot([lat_mn, lat_mx, lat_mx, lat_mn, lat_mn],
                        [fwd_mn, fwd_mn, fwd_mx, fwd_mx, fwd_mn],
                        color=C_BBOX, lw=2, zorder=6)
            ax_lid.text((lat_mn + lat_mx) / 2, fwd_mx + 0.4,
                        f"W={lat_mx-lat_mn:.1f}  D={fwd_mx-fwd_mn:.1f} m",
                        color=C_BBOX, fontsize=7, ha="center", zorder=7,
                        bbox=dict(fc=AX, ec=C_BBOX, alpha=0.8, pad=2))

    ax_lid.plot(0, 0, marker="^", ms=12, color=C_EGO, zorder=9, label="Ego")
    ax_lid.annotate("", xy=(0, 4), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=C_EGO, lw=1.8))
    ax_lid.set_xlabel("Lateral  [m]  ← left  |  right →", color=TXT, fontsize=8)
    ax_lid.set_ylabel("Forward  [m]", color=TXT, fontsize=8)
    ax_lid.set_xlim(-50, 50); ax_lid.set_ylim(0, 50)
    ax_lid.invert_xaxis()
    ax_lid.legend(loc="upper right",
                  facecolor=AX, edgecolor=GRID, labelcolor=TXT, fontsize=6)
    ax_lid.text(0.01, 0.02, "🔵  3× LiDAR",
                transform=ax_lid.transAxes, color=C_LID,
                fontsize=8, va="bottom",
                bbox=dict(fc=AX, ec=GRID, alpha=0.8, pad=3))

    # ══════════════════════════════════════════════════════════════════════
    # PANEL C — Radar scatter (4-sensor merged)
    # ══════════════════════════════════════════════════════════════════════
    ax_rad.set_facecolor(AX)
    ax_rad.tick_params(colors=TXT, labelsize=8)
    for sp in ax_rad.spines.values(): sp.set_color(GRID)
    ax_rad.set_title("Radar  ·  4-sensor  [m]",
                     color=TXT, fontsize=10, fontweight="bold", pad=5)
    ax_rad.grid(color=GRID, lw=0.5, ls="--", alpha=0.6)

    if has_radar:
        spd_all = np.abs(rad_all["v"])
        sc = ax_rad.scatter(rad_all["y"], rad_all["x"],
                            c=spd_all, cmap="plasma",
                            vmin=0, vmax=30, s=18, alpha=0.75, zorder=3)
        # Create colorbar once; update in-place every subsequent frame.
        # (cbar.remove() fails on multi-row GridSpec axes — no subplotspec to restore.)
        if _cbar[0] is None:
            _cbar[0] = plt.colorbar(sc, ax=ax_rad, pad=0.02, fraction=0.05)
            _cbar[0].set_label("Speed [m/s]", color=TXT, fontsize=7)
            _cbar[0].ax.yaxis.set_tick_params(color=TXT, labelcolor=TXT)
            _cbar[0].outline.set_edgecolor(GRID)
        else:
            _cbar[0].update_normal(sc)

        if radar_tgt:
            tx, ty = radar_tgt[0], radar_tgt[1]
            near = (np.abs(rad_all["x"] - tx) < 1.5) & (np.abs(rad_all["y"] - ty) < 1.5)
            if near.any():
                bx, by = rad_all["x"][near], rad_all["y"][near]
                xmn, xmx = bx.min() - 0.3, bx.max() + 0.3
                ymn, ymx = by.min() - 0.3, by.max() + 0.3
                ax_rad.add_patch(Rectangle(
                    (ymn, xmn), ymx - ymn, xmx - xmn,
                    lw=2, edgecolor=C_BBOX, facecolor=C_BBOX, alpha=0.12, zorder=5))
                ax_rad.plot([ymn, ymx, ymx, ymn, ymn],
                            [xmn, xmn, xmx, xmx, xmn],
                            color=C_BBOX, lw=2, zorder=6)
                ax_rad.text((ymn + ymx) / 2, xmx + 0.2,
                            f"v={abs(radar_tgt[2]):.1f} m/s",
                            color=C_BBOX, fontsize=8, ha="center",
                            fontweight="bold", zorder=7)

    ax_rad.plot(0, 0, marker="^", ms=10, color=C_EGO, zorder=9)
    ax_rad.set_xlabel("Lateral  y  [m]", color=TXT, fontsize=8)
    ax_rad.set_ylabel("Forward  x  [m]", color=TXT, fontsize=8)
    # wider y-range to show side-radar returns
    ax_rad.set_xlim(-15, 15); ax_rad.set_ylim(-5, 25)
    ax_rad.invert_xaxis()
    ax_rad.text(0.01, 0.02, "🟣  4× Radar",
                transform=ax_rad.transAxes, color=C_RAD,
                fontsize=8, va="bottom",
                bbox=dict(fc=AX, ec=GRID, alpha=0.8, pad=3))

    # ══════════════════════════════════════════════════════════════════════
    # PANEL D — Stats HUD
    # ══════════════════════════════════════════════════════════════════════
    ax_hud.set_facecolor(AX)
    ax_hud.set_xlim(0, 1); ax_hud.set_ylim(-0.15, 1)
    ax_hud.axis("off")
    ax_hud.set_title("SENSOR FUSION  ·  STATS",
                     color=TXT, fontsize=10, fontweight="bold", pad=5)

    def hline(y, color=GRID):
        ax_hud.axhline(y, color=color, lw=0.7, xmin=0.02, xmax=0.98)

    def lbl(y, key, val, unit="", color=TXT, ksize=9, vsize=13):
        ax_hud.text(0.06, y+0.018, key, color="#9ca3af",
                    fontsize=ksize, va="bottom", fontweight="bold")
        ax_hud.text(0.94, y, val, color=color,
                    fontsize=vsize, va="bottom", ha="right", fontweight="bold")
        if unit:
            ax_hud.text(0.96, y, unit, color="#9ca3af", fontsize=8, va="bottom")

    ax_hud.text(0.5, 0.97, "🏎  OPPONENT CAR", color=C_OPP,
                fontsize=12, fontweight="bold", ha="center", va="top")
    hline(0.93, C_OPP)
    lbl(0.85, "RANGE (radar)",
        f"{opp_rng:.2f}" if not np.isnan(opp_rng) else "n/a", "m", C_OPP)
    lbl(0.76, "RANGE (EKF fused)",
        f"{opp_ekf_rng:.2f}" if not np.isnan(opp_ekf_rng) else "n/a", "m", C_OPP)
    lbl(0.67, "SPEED  (radar)",
        f"{opp_spd:.1f}" if not np.isnan(opp_spd) else "n/a", "m/s", C_OPP)
    lbl(0.59, "SPEED  (EKF fused)",
        f"{opp_ekf_spd:.1f}" if not np.isnan(opp_ekf_spd) else "n/a", "m/s", C_OPP)
    lbl(0.51, "SPEED  (km/h)",
        f"{opp_ekf_spd*3.6:.0f}" if not np.isnan(opp_ekf_spd) else "n/a", "km/h", C_OPP)
    lbl(0.43, "HEADING  (EKF)", f"{opp_heading:+.1f}°", "", C_OPP)
    lbl(0.35, "RADAR CONF",
        f"{opp_conf:.0f}" if not np.isnan(opp_conf) else "n/a", "", C_OPP)
    if lidar_bbox:
        lat_mn, lat_mx, fwd_mn, fwd_mx = lidar_bbox
        lbl(0.27, "LIDAR W×D",
            f"{lat_mx-lat_mn:.1f}×{fwd_mx-fwd_mn:.1f}", "m", C_OPP, ksize=8, vsize=10)
    hline(0.23)

    ax_hud.text(0.5, 0.21, "🚗  EGO VEHICLE", color=C_EGO,
                fontsize=12, fontweight="bold", ha="center", va="top")
    hline(0.175, C_EGO)
    lbl(0.11, "SPEED", f"{ego_spd:.1f}", "m/s", C_EGO)
    lbl(0.03, "SPEED", f"{ego_kph:.0f}", "km/h", C_EGO)
    ax_hud.text(0.06, -0.05,
                f"Accel  {ax_ego:+.2f} m/s²   Yaw  {yr_ego:+.3f} rad/s",
                color="#9ca3af", fontsize=8, va="top")

    # bottom legend
    fig.legend(
        handles=[
            mpatches.Patch(color=C_LID,  label="LiDAR (×3)"),
            mpatches.Patch(color=C_RAD,  label="Radar (×4)"),
            mpatches.Patch(color=C_EGO,  label="Camera (×7)"),
            mpatches.Patch(color=C_BBOX, label="Fused box"),
        ],
        loc="lower center", ncol=4,
        facecolor=AX, edgecolor=GRID, labelcolor=TXT,
        fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle(
        f"3-Sensor Fusion  ·  Frame {fi}/{N_FRAMES-1}"
        f"  ·  t = {t_sec:.1f} s"
        f"  ·  Yas Marina Circuit, Abu Dhabi",
        color=TXT, fontsize=13, fontweight="bold", y=0.975,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

if args.save is not None:
    # ── single-frame PNG ──────────────────────────────────────────────────────
    fi  = max(0, min(args.save, N_FRAMES - 1))
    out = os.path.join(_HERE, "dashboard.png")
    print(f"\n  Rendering frame {fi} …")
    render(fi)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    _conn.close()
    print(f"  Saved → {out}")

elif VIDEO:
    # ── MP4 video — stream frames directly to ffmpeg ──────────────────────────
    if not FFMPEG_OK:
        sys.exit("  ffmpeg not found. Install imageio-ffmpeg:  pip install imageio-ffmpeg")

    from matplotlib.animation import FFMpegWriter

    out = (args.video if os.path.isabs(args.video)
           else os.path.join(_HERE, args.video))
    fps = args.video_fps

    writer = FFMpegWriter(
        fps=fps, bitrate=-1,
        extra_args=["-vcodec", "libx264", "-crf", "20", "-pix_fmt", "yuv420p"],
    )

    print(f"\n  Encoding {N_FRAMES} frames → {out}")
    print(f"  fps={fps}  ·  duration ≈ {N_FRAMES/fps:.0f} s  ·  dpi=120")
    print(f"  Ctrl-C saves a partial (but playable) MP4 up to the last encoded frame\n")

    _frames_done = [0]
    try:
        with writer.saving(fig, out, dpi=120):
            for fi in range(N_FRAMES):
                render(fi)
                writer.grab_frame()
                _frames_done[0] = fi + 1
                # progress bar
                pct  = _frames_done[0] / N_FRAMES
                done = int(40 * pct)
                bar  = "█" * done + "░" * (40 - done)
                print(f"\r  [{bar}] {_frames_done[0]:>4}/{N_FRAMES}  {pct*100:5.1f}%",
                      end="", flush=True)
    except KeyboardInterrupt:
        # context manager's __exit__ already called writer.finish() → ffmpeg finalised
        print(f"\n\n  ⚠  Interrupted at frame {_frames_done[0]}/{N_FRAMES}")
        print(f"  Partial video saved → {out}")
        _conn.close()
        sys.exit(0)

    _conn.close()
    print(f"\n\n  ✓  Saved → {out}")

else:
    # ── live animation window ─────────────────────────────────────────────────
    interval_ms   = max(50, int(1000 / args.fps))
    start_fi      = max(0, min(args.start, N_FRAMES - 1))
    frame_counter = [start_fi]

    def anim_update(_):
        fi = frame_counter[0]
        render(fi)
        frame_counter[0] = (fi + 1) % N_FRAMES
        fig.canvas.draw_idle()

    ani = animation_mod.FuncAnimation(
        fig, anim_update,
        interval=interval_ms,
        cache_frame_data=False,
    )
    plt.show()
    _conn.close()
