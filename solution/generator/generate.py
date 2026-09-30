"""Synthetic plate generator. Reproducible with a fixed seed.

Produces images into dataset/images/synthetic/ and appends YOLO labels
into dataset/labels/. Every generated image is recorded in meta.csv.
"""
import argparse
import csv
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SEED = 42
LETTERS = "ABEKMHOPCTYX"
REGIONS_2 = [f"{i:02d}" for i in range(1, 100)]
REGIONS_3 = [f"{p}{i:02d}" for p in "127" for i in range(1, 100)]

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def _font_path() -> str | None:
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _plate_string() -> str:
    l1 = random.choice(LETTERS)
    d3 = f"{random.randint(0, 999):03d}"
    l2 = random.choice(LETTERS) + random.choice(LETTERS)
    region = random.choice(REGIONS_2) if random.random() < 0.7 else random.choice(REGIONS_3)
    return f"{l1}{d3}{l2}{region}"


def _render(plate: str, ptype: str, font_path: str | None) -> Image.Image:
    if ptype == "type1a":
        size = (290, 170)
        bg, fg = (255, 255, 255), (0, 0, 0)
    elif ptype == "type1b":
        size = (520, 112)
        bg, fg = (245, 200, 30), (0, 0, 0)
    else:
        size = (520, 112)
        bg, fg = (255, 255, 255), (0, 0, 0)

    W, H = size
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)

    try:
        f_big = ImageFont.truetype(font_path, int(H * 0.55)) if font_path else ImageFont.load_default()
        f_small = ImageFont.truetype(font_path, int(H * 0.28)) if font_path else ImageFont.load_default()
    except Exception:
        f_big = ImageFont.load_default()
        f_small = ImageFont.load_default()

    if ptype == "type1a":
        d.text((W * 0.06, H * 0.05), plate[:4], fill=fg, font=f_big)
        d.text((W * 0.06, H * 0.55), plate[4:9], fill=fg, font=f_big)
        d.text((W * 0.62, H * 0.72), "RUS", fill=fg, font=f_small)
    else:
        d.text((W * 0.05, H * 0.15), plate[:-2], fill=fg, font=f_big)
        d.text((W * 0.82, H * 0.68), plate[-2:], fill=fg, font=f_small)
    return img


def _find_coeffs(pa, pb):
    m = []
    for p1, p2 in zip(pa, pb):
        m.append([p2[0], p2[1], 1, 0, 0, 0, -p1[0] * p2[0], -p1[0] * p2[1]])
        m.append([0, 0, 0, p2[0], p2[1], 1, -p1[1] * p2[0], -p1[1] * p2[1]])
    A = np.array(m, dtype=np.float64)
    B = np.array([c for p in pa for c in p], dtype=np.float64)
    return np.linalg.solve(A, B).tolist()


def _augment(img: Image.Image) -> Image.Image:
    w, h = img.size

    def jit(scale=0.05):
        return random.uniform(-scale, scale) * w, random.uniform(-scale, scale) * h

    src = [(0 + jit()[0], 0 + jit()[1]),
           (w + jit()[0], 0 + jit()[1]),
           (w + jit()[0], h + jit()[1]),
           (0 + jit()[0], h + jit()[1])]
    coeffs = _find_coeffs(src, [(0, 0), (w, 0), (w, h), (0, h)])
    img = img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0, 1.2)))

    arr = np.array(img).astype(np.int16)
    arr += np.random.randint(-18, 18, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


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
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.labels, exist_ok=True)

    font_path = _font_path()
    ptypes = ["type1"] * 30 + ["type1a"] * 30 + ["type1b"] * 30 + ["other"] * 10

    write_header = not os.path.exists(args.meta)
    with open(args.meta, "a", encoding="utf-8", newline="") as mf:
        w = csv.writer(mf, delimiter=";")
        if write_header:
            w.writerow(["image", "plate_num", "plate_type", "bbox", "quad",
                        "is_vehicle", "is_synthetic", "source", "license", "conditions"])
        for i in range(args.n):
            ptype = random.choice(ptypes)
            plate = _plate_string()
            img = _augment(_render(plate, ptype, font_path))
            name = f"syn_{i:06d}.png"
            img.save(os.path.join(args.out, name))
            W, H = img.size
            # YOLO label: full-frame box
            if i%500 == 0:
                print(f"{i}/{args.n}")
            with open(os.path.join(args.labels, f"syn_{i:06d}.txt"), "w") as lf:
                cls = {"type1": 0, "type1a": 1, "type1b": 2, "other": 3}[ptype]
                lf.write(f"{cls} 0.5 0.5 0.9 0.9\n")
            if ptype == "other":
                plate_num = ""
            else:
                plate_num = plate
            rel = f"images/synthetic/{name}"
            w.writerow([rel, plate_num, ptype, f"0,0,{W},{H}", "",
                        1, 1, "generator", "own", "synthetic"])
    print(f"generated {args.n} images into {args.out}")


if __name__ == "__main__":
    main()
