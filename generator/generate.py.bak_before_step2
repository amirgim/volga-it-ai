"""Synthetic plate generator with background insertion.

Pipeline per image:
  1. pick a real-photo background
  2. Haar-cascade search for existing plates in it
  3. if a plate exists -> paste synthetic plate exactly over it,
     blur any other detected plates so YOLO never sees unlabeled ones
  4. else -> paste into lower third at realistic size
  5. warp + lighting augment, write image + YOLO label + meta.csv
"""
import argparse
import csv
import math
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SEED = 42
LETTERS = "ABEKMHOPCTYX"
REGIONS_2 = [f"{i:02d}" for i in range(1, 100)]
REGIONS_3 = [f"{p}{i:02d}" for p in "127" for i in range(1, 100)]

FONT_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "GOST-R-50577-93.ttf"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "RoadNumbers2.0.ttf"),
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

_CASCADE = None
_PLATE_CACHE = {}


def _font_path():
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _get_cascade():
    global _CASCADE
    if _CASCADE is None:
        path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
        _CASCADE = cv2.CascadeClassifier(path)
        if _CASCADE.empty():
            raise RuntimeError(f"cannot load cascade: {path}")
    return _CASCADE


def _find_plates(bg_path):
    """Detect plates in a background image, cached by path. Returns [(x,y,w,h), ...]."""
    if bg_path in _PLATE_CACHE:
        return _PLATE_CACHE[bg_path]
    img = cv2.imread(bg_path)
    if img is None:
        _PLATE_CACHE[bg_path] = []
        return []
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    boxes = _get_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 15),
    )
    out = []
    for (x, y, w, h) in boxes:
        ar = w / max(h, 1)
        if 2.0 <= ar <= 7.0 and y > H * 0.1:
            out.append((x, y, w, h))
    out.sort(key=lambda b: b[2] * b[3], reverse=True)
    _PLATE_CACHE[bg_path] = out
    return out


def _plate_string():
    l1 = random.choice(LETTERS)
    d3 = f"{random.randint(0, 999):03d}"
    l2 = random.choice(LETTERS) + random.choice(LETTERS)
    region = random.choice(REGIONS_2) if random.random() < 0.7 else random.choice(REGIONS_3)
    return f"{l1}{d3}{l2}{region}"


def _draw_flag(draw, x, y, size):
    draw.rectangle([x, y, x + size, y + size // 3], fill=(255, 255, 255), outline=(0, 0, 0))
    draw.rectangle([x, y + size // 3, x + size, y + 2 * size // 3], fill=(0, 57, 166), outline=(0, 0, 0))
    draw.rectangle([x, y + 2 * size // 3, x + size, y + size], fill=(213, 43, 30), outline=(0, 0, 0))


def _fit_font(draw, text, font_path, max_w, max_h):
    if not font_path:
        return ImageFont.load_default()
    for size in range(int(max_h), 9, -2):
        try:
            font = ImageFont.truetype(font_path, size)
        except Exception:
            return ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        if (bbox[2] - bbox[0]) <= max_w and (bbox[3] - bbox[1]) <= max_h:
            return font
    try:
        return ImageFont.truetype(font_path, 10)
    except Exception:
        return ImageFont.load_default()


def _render_plate(plate, ptype, font_path):
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
    d.rectangle([0, 0, W - 1, H - 1], outline=(100, 100, 100), width=3)
    region_len = 2 if len(plate) == 8 else 3

    if ptype == "type1a":
        top = plate[:4]
        bottom = plate[4:6] + plate[6:]
        f_top = _fit_font(d, top, font_path, W * 0.85, H * 0.38)
        f_bot = _fit_font(d, bottom, font_path, W * 0.85, H * 0.38)
        d.text((W * 0.05, H * 0.04), top, fill=fg, font=f_top)
        d.text((W * 0.05, H * 0.54), bottom, fill=fg, font=f_bot)
        f_small = _fit_font(d, "RUS", font_path, W * 0.20, H * 0.10)
        d.text((W * 0.60, H * 0.76), "RUS", fill=fg, font=f_small)
        _draw_flag(d, W * 0.60, H * 0.88, int(H * 0.08))
    else:
        body = plate[:-region_len]
        region = plate[-region_len:]
        f_body = _fit_font(d, body, font_path, W * 0.72, H * 0.55)
        f_region = _fit_font(d, region, font_path, W * 0.22, H * 0.30)
        d.text((W * 0.04, H * 0.10), body, fill=fg, font=f_body)
        d.text((W * 0.76, H * 0.40), region, fill=fg, font=f_region)
        f_small = _fit_font(d, "RUS", font_path, W * 0.15, H * 0.12)
        d.text((W * 0.04, H * 0.72), "RUS", fill=fg, font=f_small)
        _draw_flag(d, W * 0.04, H * 0.86, int(H * 0.10))
    return img


def _render_other(font_path):
    W, H = 520, 112
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    if random.random() < 0.5:
        strip_w = int(W * 0.07)
        d.rectangle([0, 0, strip_w, H], fill=(0, 51, 153))
        try:
            f_country = ImageFont.truetype(font_path, int(H * 0.20)) if font_path else ImageFont.load_default()
            f_text = ImageFont.truetype(font_path, int(H * 0.55)) if font_path else ImageFont.load_default()
        except Exception:
            f_country = ImageFont.load_default()
            f_text = ImageFont.load_default()
        country = random.choice(["D", "F", "GB", "PL", "LT", "LV", "EST"])
        d.text((strip_w * 0.1, H * 0.10), country, fill=(255, 255, 255), font=f_country)
        latin = "ABCDEFGHJKLMNPRSTUVWXYZ"
        text = f"{random.choice(latin)}{random.choice(latin)} {random.randint(1000, 9999)}"
        d.text((strip_w + W * 0.05, H * 0.10), text, fill=(0, 0, 0), font=f_text)
    else:
        d.rectangle([2, 2, W - 3, H - 3], outline=(0, 0, 0), width=2)
    return img


def _quad_over_box(x, y, w, h, plate_aspect):
    """Fit plate inside (x,y,w,h) keeping aspect, center. Tiny perspective."""
    box_aspect = w / max(h, 1)
    if box_aspect > plate_aspect:
        new_h = h
        new_w = int(h * plate_aspect)
    else:
        new_w = w
        new_h = int(w / plate_aspect)
    cx = x + w / 2
    cy = y + h / 2
    hw, hh = new_w / 2, new_h / 2
    jitter = 0.03
    corners = [
        (-hw * (1 + random.uniform(-jitter, jitter)), -hh * (1 + random.uniform(-jitter, jitter))),
        ( hw * (1 + random.uniform(-jitter, jitter)), -hh * (1 + random.uniform(-jitter, jitter))),
        ( hw * (1 + random.uniform(-jitter, jitter)),  hh * (1 + random.uniform(-jitter, jitter))),
        (-hw * (1 + random.uniform(-jitter, jitter)),  hh * (1 + random.uniform(-jitter, jitter))),
    ]
    return [(cx + dx, cy + dy) for (dx, dy) in corners]


def _random_quad_small(W, H, plate_w, plate_h):
    """Plate in lower third, 6-15% of frame width. For backgrounds with no plate."""
    target_w = random.uniform(0.06, 0.15) * W
    scale = target_w / plate_w
    target_h = plate_h * scale

    margin_x = target_w * 0.6 + 5
    margin_y = target_h * 0.6 + 5
    if margin_x * 2 > W or margin_y * 2 > H:
        return None

    cx = random.uniform(margin_x, W - margin_x)
    cy = random.uniform(max(margin_y, H * 0.60), H - margin_y)

    angle = random.uniform(-3, 3)
    cos_a = math.cos(math.radians(angle))
    sin_a = math.sin(math.radians(angle))

    hw, hh = target_w / 2, target_h / 2
    jitter = 0.03
    corners = [
        (-hw * (1 + random.uniform(-jitter, jitter)), -hh * (1 + random.uniform(-jitter, jitter))),
        ( hw * (1 + random.uniform(-jitter, jitter)), -hh * (1 + random.uniform(-jitter, jitter))),
        ( hw * (1 + random.uniform(-jitter, jitter)),  hh * (1 + random.uniform(-jitter, jitter))),
        (-hw * (1 + random.uniform(-jitter, jitter)),  hh * (1 + random.uniform(-jitter, jitter))),
    ]
    quad = []
    for (dx, dy) in corners:
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        x = max(0.0, min(W - 1.0, cx + rx))
        y = max(0.0, min(H - 1.0, cy + ry))
        quad.append((x, y))
    return quad


def _warp_plate(plate_pil, bg_size, quad):
    W, H = bg_size
    plate_np = np.array(plate_pil)
    plate_bgr = cv2.cvtColor(plate_np, cv2.COLOR_RGB2BGR)
    ph, pw = plate_bgr.shape[:2]
    src = np.float32([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]])
    dst = np.float32(quad)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(plate_bgr, M, (W, H), flags=cv2.INTER_LINEAR)
    mask = np.full((ph, pw), 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(mask, M, (W, H), flags=cv2.INTER_LINEAR)
    return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), warped_mask


def _blur_regions(bg_pil, boxes):
    """Blur rectangular regions (existing plates) so YOLO never sees unlabeled ones."""
    if not boxes:
        return bg_pil
    arr = np.array(bg_pil)
    for (x, y, w, h) in boxes:
        pad = 3
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(arr.shape[1], x + w + pad)
        y2 = min(arr.shape[0], y + h + pad)
        if x2 <= x1 or y2 <= y1:
            continue
        region = arr[y1:y2, x1:x2]
        k = max(7, (min(w, h) // 2) * 2 + 1)
        arr[y1:y2, x1:x2] = cv2.GaussianBlur(region, (k, k), 0)
    return Image.fromarray(arr)


def _composite(bg_pil, warped_np, mask_np):
    bg = np.array(bg_pil).astype(np.float32)
    alpha = (mask_np.astype(np.float32) / 255.0)[..., None]
    out = bg * (1 - alpha) + warped_np.astype(np.float32) * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _augment_scene(img, conditions):
    arr = np.array(img).astype(np.float32)
    if "night" in conditions:
        arr = arr * 0.35
        arr[..., 2] *= 1.15
    if "rain" in conditions:
        for _ in range(random.randint(20, 80)):
            x = random.randint(0, arr.shape[1] - 1)
            y = random.randint(0, max(1, arr.shape[0] - 50))
            length = min(random.randint(20, 50), arr.shape[0] - y)
            arr[y:y + length, x] = np.clip(arr[y:y + length, x] * 0.6 + 100, 0, 255)
    if "glare" in conditions:
        arr = np.clip(arr * 1.3, 0, 255)
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    if "dirt" in conditions:
        for _ in range(random.randint(5, 20)):
            x = random.randint(0, arr.shape[1] - 1)
            y = random.randint(0, arr.shape[0] - 1)
            r = random.randint(3, 15)
            cv2.circle(arr, (x, y), r, (random.randint(30, 80),) * 3, -1)

    out = Image.fromarray(arr)
    if "motion_blur" in conditions:
        out = out.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 2.0)))
    if random.random() < 0.5:
        a = np.array(out).astype(np.int16)
        a += np.random.randint(-18, 18, a.shape)
        out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return out


def _bbox_from_quad(quad):
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return x1, y1, x2 - x1, y2 - y1


def _collect_backgrounds(real_dir, extra_dir):
    bgs = []
    for d in [real_dir, extra_dir]:
        if not d or not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                bgs.append(os.path.join(d, f))
    return bgs


def _load_background(bg_path):
    try:
        bg = Image.open(bg_path).convert("RGB")
    except Exception:
        return None
    if bg.size[0] < 400 or bg.size[1] < 200:
        return None
    max_dim = 1280
    if max(bg.size) > max_dim:
        scale = max_dim / max(bg.size)
        bg = bg.resize(
            (int(bg.size[0] * scale), int(bg.size[1] * scale)),
            Image.LANCZOS,
        )
    return bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/images/synthetic")
    ap.add_argument("--labels", default="dataset/labels")
    ap.add_argument("--meta", default="dataset/meta.csv")
    ap.add_argument("--backgrounds", default="dataset/images/real")
    ap.add_argument("--extra-bg", default="generator/backgrounds")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.labels, exist_ok=True)
    os.makedirs(os.path.dirname(args.meta) or ".", exist_ok=True)

    font_path = _font_path()
    if not font_path:
        print("WARNING: no font found, text will be tiny.")

    bgs = _collect_backgrounds(args.backgrounds, args.extra_bg)
    if not bgs:
        raise SystemExit(f"no backgrounds in {args.backgrounds}")

    # Pre-detect plates in all backgrounds once, cached.
    print("scanning backgrounds for existing plates (Haar)...")
    for i, p in enumerate(bgs):
        _find_plates(p)
        if (i + 1) % 100 == 0:
            print(f"  scanned {i + 1}/{len(bgs)}")

    ptypes = ["type1"] * 30 + ["type1a"] * 30 + ["type1b"] * 30 + ["other"] * 10
    CONDITIONS_POOL = ["day", "day", "day", "night", "angle", "dirt", "glare", "motion_blur"]

    write_header = not os.path.exists(args.meta) or os.path.getsize(args.meta) == 0
    mf = open(args.meta, "a", encoding="utf-8", newline="")
    w = csv.writer(mf, delimiter=";")
    if write_header:
        w.writerow(["image", "plate_num", "plate_type", "bbox", "quad",
                    "is_vehicle", "is_synthetic", "source", "license", "conditions"])

    written = 0
    attempts = 0
    max_attempts = args.n * 3
    paste_over = 0
    paste_free = 0

    while written < args.n and attempts < max_attempts:
        attempts += 1
        ptype = random.choice(ptypes)
        if ptype == "other":
            plate_img = _render_other(font_path)
            plate = ""
        else:
            plate = _plate_string()
            plate_img = _render_plate(plate, ptype, font_path)

        bg_path = random.choice(bgs)
        bg = _load_background(bg_path)
        if bg is None:
            continue

        W, H = bg.size
        existing = _find_plates(bg_path)

        if existing:
            bx, by, bw, bh = existing[0]
            plate_aspect = plate_img.size[0] / plate_img.size[1]
            quad = _quad_over_box(bx, by, bw, bh, plate_aspect)
            paste_over += 1
        else:
            # Если номер не найден, а картинка — крупный план, пропускаем фон.
            if W < 800 or H < 500:
                continue
            quad = _random_quad_small(W, H, plate_img.size[0], plate_img.size[1])
            if quad is None:
                continue
            paste_free += 1

        warped, mask = _warp_plate(plate_img, (W, H), quad)
        scene = _composite(bg, warped, mask)

        conditions = [random.choice(CONDITIONS_POOL)]
        if random.random() < 0.3:
            conditions.append(random.choice(["rain", "snow", "dirt", "glare"]))

        scene = _augment_scene(scene, conditions)

        name = f"syn_{written:06d}.png"
        scene.save(os.path.join(args.out, name))

        bbox = _bbox_from_quad(quad)
        cx = (bbox[0] + bbox[2] / 2) / W
        cy = (bbox[1] + bbox[3] / 2) / H
        nw = bbox[2] / W
        nh = bbox[3] / H
        cls = {"type1": 0, "type1a": 1, "type1b": 2, "other": 3}[ptype]
        with open(os.path.join(args.labels, f"syn_{written:06d}.txt"), "w") as lf:
            lf.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        bbox_str = f"{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}"
        quad_str = ",".join(f"{p[0]:.1f},{p[1]:.1f}" for p in quad)
        rel = f"images/synthetic/{name}"
        w.writerow([rel, plate, ptype, bbox_str, quad_str,
                    1, 1, "generator", "own_photo", ",".join(conditions)])

        written += 1
        if written % 500 == 0:
            print(f"  {written}/{args.n}  (over={paste_over} free={paste_free})")

    mf.close()
    print(f"generated {written} images into {args.out}")
    print(f"  pasted over existing plate: {paste_over}")
    print(f"  pasted into empty scene:   {paste_free}")


if __name__ == "__main__":
    main()
