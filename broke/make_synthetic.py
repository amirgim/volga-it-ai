"""Draw synthetic plates with OpenCV only. No ML, no torch, no PIL text.

Reproducible via fixed seed. Writes PNG + YOLO label + meta.csv row.
"""
import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

SEED = 42
LETTERS = "ABEKMHOPCTYX"
REGIONS_2 = [f"{i:02d}" for i in range(1, 100)]
REGIONS_3 = [f"{p}{i:02d}" for p in "127" for i in range(1, 100)]
FONT = cv2.FONT_HERSHEY_SIMPLEX
BLACK = (0, 0, 0)


def plate_string() -> str:
    l1 = random.choice(LETTERS)
    d3 = f"{random.randint(0, 999):03d}"
    l2 = random.choice(LETTERS) + random.choice(LETTERS)
    region = (random.choice(REGIONS_2)
              if random.random() < 0.7 else random.choice(REGIONS_3))
    return f"{l1}{d3}{l2}{region}"


def render(plate: str, ptype: str) -> np.ndarray:
    if ptype == "type1a":
        W, H, bg = 290, 170, (255, 255, 255)
    elif ptype == "type1b":
        W, H, bg = 520, 112, (30, 200, 245)
    else:
        W, H, bg = 520, 112, (255, 255, 255)

    img = np.full((H, W, 3), bg, dtype=np.uint8)

    if ptype == "type1a":
        cv2.putText(img, plate[:4], (15, 60), FONT, 1.5, BLACK, 3, cv2.LINE_AA)
        cv2.putText(img, plate[4:9], (15, 130), FONT, 1.5, BLACK, 3, cv2.LINE_AA)
        cv2.putText(img, "RUS", (200, 150), FONT, 0.7, BLACK, 2, cv2.LINE_AA)
    else:
        region_len = 2 if len(plate) == 8 else 3
        cv2.putText(img, plate[:-region_len], (20, 80),
                    FONT, 2.0, BLACK, 5, cv2.LINE_AA)
        cv2.putText(img, plate[-region_len:], (W - 100, 95),
                    FONT, 1.0, BLACK, 2, cv2.LINE_AA)
    return img


def augment(img: np.ndarray) -> np.ndarray:
    if random.random() < 0.5:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    noise = np.random.randint(-18, 18, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/images/synthetic")
    ap.add_argument("--labels", default="dataset/labels")
    ap.add_argument("--meta", default="dataset/meta.csv")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.labels).mkdir(parents=True, exist_ok=True)
    Path(args.meta).parent.mkdir(parents=True, exist_ok=True)

    ptypes = (["type1"] * 30 + ["type1a"] * 30
              + ["type1b"] * 30 + ["other"] * 10)

    write_header = not Path(args.meta).exists()
    with open(args.meta, "a", encoding="utf-8", newline="") as mf:
        w = csv.writer(mf, delimiter=";")
        if write_header:
            w.writerow(["image", "plate_num", "plate_type", "bbox", "quad",
                        "is_vehicle", "is_synthetic", "source", "license",
                        "conditions"])
        for i in range(args.n):
            ptype = random.choice(ptypes)
            plate = plate_string()
            img = augment(render(plate, ptype))
            name = f"syn_{i:06d}.png"
            cv2.imwrite(str(Path(args.out) / name), img)
            H, W = img.shape[:2]
            cls = {"type1": 0, "type1a": 1, "type1b": 2, "other": 3}[ptype]
            with open(Path(args.labels) / f"syn_{i:06d}.txt", "w") as lf:
                lf.write(f"{cls} 0.5 0.5 0.9 0.9\n")
            plate_num = "" if ptype == "other" else plate
            rel = f"images/synthetic/{name}"
            w.writerow([rel, plate_num, ptype, f"0,0,{W},{H}", "",
                        1, 1, "generator", "own_photo", "day"])
    print(f"generated {args.n} images into {args.out}")


if __name__ == "__main__":
    main()
