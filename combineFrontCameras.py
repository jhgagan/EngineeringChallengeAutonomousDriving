"""
Merge camera_fl (front-left) and camera_fr (front-right) into one wide image.

Horizontal alignment: sliding-window Normalized Cross-Correlation (NCC).
A fixed 400-pixel window anchored at FR col 0 is compared against all
horizontal positions in FL using Pearson r.  The best-matching column gives
the horizontal offset dx, which aligns the dominant far-field scene content
and avoids biasing toward a single near object (parallax-safe).

Vertical alignment: centroid of the dark static stop (track barrier) visible
at the bottom of both images.

The overlap zone uses a hard cut at the column with the lowest mean absolute
pixel difference — no blending, no ghosting.

Usage:
    python combineFrontCameras.py [frame_index]

Default frame_index = 0.
"""

import sqlite3
import struct
import io
import sys
import numpy as np
from PIL import Image

DB    = "Data/2025-10-08_09-35_sensors_raw_1.db3"
FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# ── Config ───────────────────────────────────────────────────────────────────
_NCC_ROW_RANGE = (80, 380)
_NCC_WIN_W     = 400

_FL_STOP_ROWS = (420, 500)
_FL_STOP_COLS = (1300, 1450)
_FR_STOP_ROWS = (420, 500)
_FR_STOP_COLS = (590,  750)
_STOP_THRESH  = 110


def decode_image(raw: bytes) -> np.ndarray:
    off  = 4 + 8
    slen = struct.unpack_from("<I", raw, off)[0]; off += 4
    off += slen; off = (off + 3) & ~3
    slen = struct.unpack_from("<I", raw, off)[0]; off += 4
    off += slen; off = (off + 3) & ~3
    dlen = struct.unpack_from("<I", raw, off)[0]; off += 4
    return np.array(Image.open(io.BytesIO(raw[off:off + dlen])).convert("RGB"))


def _ncc(a, b):
    a_c = a - a.mean();  b_c = b - b.mean()
    d   = np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())
    return float((a_c * b_c).sum() / d) if d > 1e-12 else 0.0


def _find_dx_ncc(fl_arr, fr_arr):
    r0, r1 = _NCC_ROW_RANGE
    win_w  = _NCC_WIN_W
    fl_g   = fl_arr[r0:r1].mean(axis=2).astype(np.float64)
    fr_g   = fr_arr[r0:r1].mean(axis=2).astype(np.float64)
    W      = fl_g.shape[1]

    fr_win   = fr_g[:, :win_w].ravel()
    best_ncc = -np.inf
    best_dx  = 0
    for fl_col in range(0, W - win_w + 1, 2):
        r = _ncc(fl_g[:, fl_col:fl_col + win_w].ravel(), fr_win)
        if r > best_ncc:
            best_ncc = r
            best_dx  = fl_col
    return best_dx, best_ncc


def _stop_centroid_y(arr, row_range, col_range):
    r0, r1 = row_range;  c0, c1 = col_range
    region = arr[r0:r1, c0:c1].mean(axis=2)
    rows, _ = np.where(region < _STOP_THRESH)
    return float(rows.mean() + r0) if len(rows) else float((r0 + r1) / 2)


# ── Load frames ───────────────────────────────────────────────────────────────
conn   = sqlite3.connect(DB)
fl_id  = conn.execute("SELECT id FROM topics WHERE name=?",
                      ("/sensor/camera/camera_fl/image/compressed",)).fetchone()[0]
fr_id  = conn.execute("SELECT id FROM topics WHERE name=?",
                      ("/sensor/camera/camera_fr/image/compressed",)).fetchone()[0]
fl_raw = conn.execute(
    "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1 OFFSET ?",
    (fl_id, FRAME)).fetchone()[0]
fr_raw = conn.execute(
    "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1 OFFSET ?",
    (fr_id, FRAME)).fetchone()[0]
conn.close()

fl = decode_image(fl_raw)
fr = decode_image(fr_raw)

# ── Compute alignment ─────────────────────────────────────────────────────────
dx, ncc_r = _find_dx_ncc(fl, fr)
fl_cy = _stop_centroid_y(fl, _FL_STOP_ROWS, _FL_STOP_COLS)
fr_cy = _stop_centroid_y(fr, _FR_STOP_ROWS, _FR_STOP_COLS)
dy    = int(round(fl_cy - fr_cy))

print(f"NCC dx={dx}  (r={ncc_r:.4f})")
print(f"Stop dy={dy}  (FL row={fl_cy:.1f}  FR row={fr_cy:.1f})")

H, W    = fl.shape[:2]
fl_oy   = max(0, -dy)
fr_oy   = max(0,  dy)
crop_r0 = max(fl_oy, fr_oy)
crop_r1 = min(fl_oy + H, fr_oy + H)
mh      = crop_r1 - crop_r0
mw      = dx + W
fl_r0   = crop_r0 - fl_oy
fr_r0   = crop_r0 - fr_oy

# ── Find seam (minimum per-column pixel difference in overlap) ────────────────
overlap_w    = W - dx
fl_ov        = fl[fl_r0:fl_r0 + mh, dx:W].astype(np.float32)
fr_ov        = fr[fr_r0:fr_r0 + mh, :overlap_w].astype(np.float32)
col_diffs    = np.abs(fl_ov - fr_ov).mean(axis=(0, 2))
seam_in_ovlp = int(col_diffs.argmin())
seam_col     = dx + seam_in_ovlp

print(f"Seam at canvas col {seam_col}  (overlap diff {col_diffs[seam_in_ovlp]:.1f})")

# ── Hard-cut merge ────────────────────────────────────────────────────────────
merged = np.empty((mh, mw, 3), dtype=np.uint8)
merged[:, :seam_col] = fl[fl_r0:fl_r0 + mh, :seam_col]
merged[:, seam_col:] = fr[fr_r0:fr_r0 + mh,
                           seam_in_ovlp:seam_in_ovlp + (mw - seam_col)]

# ── Save ──────────────────────────────────────────────────────────────────────
output = "front_combined.jpg"
Image.fromarray(merged).save(output, quality=95)
print(f"Saved {output}  ({mw}x{mh} px)  [frame {FRAME}]")
