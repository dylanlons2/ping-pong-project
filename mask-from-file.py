import sys
import os
import subprocess
import tempfile
import json
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from datetime import datetime


def is_hdr(path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=color_transfer",
         "-of", "json", path],
        capture_output=True, text=True
    )
    info = json.loads(result.stdout)
    trc = info.get("streams", [{}])[0].get("color_transfer", "")
    return trc in ("arib-std-b67", "smpte2084")


def convert_hdr_to_sdr(path):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    print(f"HDR detected ({path}), converting to SDR...")
    subprocess.run(
        ["ffmpeg", "-y", "-i", path,
         "-vf", "format=gbrpf32le,tonemap=reinhard:peak=1.0:desat=0,format=yuv420p",
         "-c:v", "libx264", "-crf", "18", "-an", tmp.name],
        capture_output=True, check=True
    )
    print("SDR conversion done.")
    return tmp.name


source = sys.argv[1] if len(sys.argv) > 1 else None
if source is None:
    print("Usage: python3 mask-from-file.py <video>")
    exit(1)

tmp_path = None
video_path = source
# For now, disable conversion
# if isinstance(source, str) and os.path.isfile(source) and is_hdr(source):
#     tmp_path = convert_hdr_to_sdr(source)
#     video_path = tmp_path

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Cannot open video file: {video_path}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

os.makedirs("data/output", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"data/output/{timestamp}-out.mp4"
mask_path = f"data/output/{timestamp}-mask.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
mask_out = cv2.VideoWriter(mask_path, fourcc, fps, (width, height), isColor=False)

# lower = np.array([0, 120, 120])
# upper = np.array([255, 255, 255])

# from self-tuning
lower = np.array([0, 144, 158])
upper = np.array([27, 177, 242])

min_area = 5

frame_num = 0
detections = 0
single_hits = 0
x_midpoints = []
y_midpoints = []
timestamps = []
bboxes = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    best_score = 0.0
    valid_count = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        valid_count += 1
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        score = circularity * np.log1p(area)
        if score > best_score:
            best_score = score
            best_contour = c

    if valid_count == 1:
        single_hits += 1

    if best_contour is not None:
        x, y, w, h = cv2.boundingRect(best_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        detections += 1

        x_mid = x + 0.5 * w
        y_mid = y + 0.5 * h
        t = frame_num / fps
        x_midpoints.append(x_mid)
        y_midpoints.append(y_mid)
        timestamps.append(t)
        bboxes.append((frame_num, round(t, 4), x, y, w, h, round(x_mid, 1), round(y_mid, 1)))

        print(f"Bounding Box created at: {x}, {y}, {w}, {h}")

    out.write(frame)
    mask_out.write(mask)
 
    if frame_num % 100 == 0:
        print(f"  Processed {frame_num}/{total_frames} frames...")

print(f"Done. {frame_num} frames processed, {detections} detections.")
print(f"Saved to {output_path}")
print(f"Mask saved to {mask_path}")
print(f"Detections: {detections}")
print(f"Single-hit frames: {single_hits}")

cap.release()
out.release()
mask_out.release()

if tmp_path:
    os.unlink(tmp_path)

csv_path = f"data/output/{timestamp}-tracking.csv"
if bboxes:
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                         "x_midpoint", "y_midpoint", "vx_px_per_s", "vy_px_per_s"])
        for i, row in enumerate(bboxes):
            if i == 0:
                vx = 0.0
                vy = 0.0
            else:
                dt = row[1] - bboxes[i - 1][1]
                if dt > 0:
                    vx = round((row[6] - bboxes[i - 1][6]) / dt, 2)
                    vy = round((row[7] - bboxes[i - 1][7]) / dt, 2)
                else:
                    vx = 0.0
                    vy = 0.0
            writer.writerow(row + (vx, vy))
    print(f"CSV saved to {csv_path}")
else:
    print("No detections — CSV not written.")

if timestamps:
    t = np.array(timestamps)
    x = np.array(x_midpoints)
    y = height - np.array(y_midpoints)

    order = 5
    local_max = argrelextrema(x, np.greater, order=order)[0]
    local_min = argrelextrema(x, np.less, order=order)[0]
    extrema = np.sort(np.concatenate([local_max, local_min]))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, x, "b.-", markersize=3, linewidth=0.8, label="X midpoint")
    ax.plot(t, y, "r.-", markersize=3, linewidth=0.8, label="Y midpoint")

    for idx in extrema:
        ax.axvline(t[idx], color="green", alpha=0.4, linewidth=0.8, linestyle="--")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (px)")
    ax.set_title("Ball position over time (vertical lines = X direction changes)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    print(f"Found {len(extrema)} direction changes in X.")
    plt.tight_layout()
    plt.show()

    vx = np.zeros_like(t)
    vy = np.zeros_like(t)
    dt = np.diff(t)
    mask_nonzero = dt > 0
    vx[1:] = np.where(mask_nonzero, np.diff(x) / dt, 0.0)
    vy[1:] = np.where(mask_nonzero, np.diff(y) / dt, 0.0)

    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(t, vx, "b.-", markersize=2, linewidth=0.8)
    ax1.set_ylabel("Vx (px/s)")
    ax1.set_title("Velocity over time")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, vy, "r.-", markersize=2, linewidth=0.8)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Vy (px/s)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
else:
    print("No detections to plot.")
