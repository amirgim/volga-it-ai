"""Broke pipeline: Haar cascade → Tesseract."""
import csv
import os
import time
from typing import Dict, List

import cv2
import numpy as np

from .detect import detect
from .recognize import read_plate

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _plate_type(crop_bgr: np.ndarray) -> str:
    """Guess type from colour + aspect ratio. No ML."""
    h, w = crop_bgr.shape[:2]
    aspect = w / max(h, 1)

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    yellow = ((hue > 15) & (hue < 40) & (sat > 80) & (val > 100))
    if yellow.mean() > 0.4:
        return "type1b"

    if 1.2 <= aspect <= 2.2:
        return "type1a"
    if 3.5 <= aspect <= 6.0:
        return "type1"
    return "other"


def process_image(img_bgr: np.ndarray) -> List[Dict]:
    out: List[Dict] = []
    for (x, y, w, h) in detect(img_bgr):
        if w < 60 or h < 15:
            continue
        crop = img_bgr[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        text = read_plate(crop)
        if not text:
            continue
        ptype = _plate_type(crop)
        if ptype == "other":
            continue
        out.append({
            "plate_num": text,
            "plate_type": ptype,
            "confidence": 0.5,
        })
    return out


def run_dir(input_dir: str, output_csv: str) -> None:
    files = sorted(
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in EXTS
    )
    rows: List[tuple] = []
    t0 = time.time()
    for f in files:
        img = cv2.imread(os.path.join(input_dir, f))
        if img is None:
            continue
        for r in process_image(img):
            rows.append((f, r["plate_num"], r["plate_type"], r["confidence"]))
    elapsed = time.time() - t0
    with open(output_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["image", "plate_num", "plate_type", "confidence"])
        w.writerows(rows)
    n = max(1, len(files))
    print(f"images={len(files)} rows={len(rows)} "
          f"avg_ms={elapsed / n * 1000:.1f}")
