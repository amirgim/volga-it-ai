"""Extract plate crops from synthetic images using quad annotations.

Filename: <PLATE>_<type>_<n>.png. type1a crops are already "unfolded"
(two lines glued into one) exactly like at inference time.
"""
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.unfold import unfold_1a

META = Path("dataset/meta.csv")
CROPS_DIR = Path("dataset/crops")
SEED = 42
SCALES = [0.92, 1.0, 1.12, 1.25]  # jitter mimics YOLO bbox looseness


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def jitter_quad(rect, scale):
    cx, cy = rect[:, 0].mean(), rect[:, 1].mean()
    return (rect - [cx, cy]) * scale + [cx, cy]


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    if not META.exists():
        raise SystemExit(f"missing {META}")

    count = 0
    with open(META, "r", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            plate = row.get("plate_num", "")
            ptype = row.get("plate_type", "")
            if (row.get("is_synthetic") != "1" or ptype not in ("type1", "type1a", "type1b")
                    or len(plate) not in (8, 9) or "#" in plate):
                continue
            img_path = Path("dataset") / row["image"]
            quad_str = row.get("quad", "")
            if not img_path.exists() or not quad_str:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            rect = order_points(np.array([float(x) for x in quad_str.split(",")]).reshape(4, 2))

            for scale in SCALES:
                r = jitter_quad(rect, scale)
                mw = max(int(np.linalg.norm(r[2] - r[3])), int(np.linalg.norm(r[1] - r[0])))
                mh = max(int(np.linalg.norm(r[1] - r[2])), int(np.linalg.norm(r[0] - r[3])))
                if mw < 40 or mh < 15:
                    continue
                dst = np.array([[0, 0], [mw - 1, 0], [mw - 1, mh - 1], [0, mh - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(r, dst)
                warped = cv2.warpPerspective(img, M, (mw, mh))
                if warped is None or warped.size == 0:
                    continue
                if warped.mean() < 5 and warped.std() < 5:
                    continue
                if ptype == "type1a":
                    warped = unfold_1a(warped)
                cv2.imwrite(str(CROPS_DIR / f"{plate}_{ptype}_{count:06d}.png"), warped)
                count += 1
    print(f"Done. Total crops: {count} in {CROPS_DIR}")


if __name__ == "__main__":
    main()
