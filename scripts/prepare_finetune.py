"""Prepare combined dataset for YOLO finetuning using Haar pseudo-labels."""
import cv2
import numpy as np
from pathlib import Path
import os
import sys

# Добавляем корень проекта в пути, чтобы импортировать broke
sys.path.append(str(Path(__file__).resolve().parent.parent))
from broke.detect import detect

DATASET = Path("dataset")
IMG_SYNTH = DATASET / "images" / "synthetic"
LBL_SYNTH = DATASET / "labels" / "synthetic"
IMG_REAL = DATASET / "images" / "real"
LBL_REAL = DATASET / "labels" / "real"

IMG_TRAIN = DATASET / "images" / "train"
LBL_TRAIN = DATASET / "labels" / "train"

def main():
    IMG_TRAIN.mkdir(parents=True, exist_ok=True)
    LBL_TRAIN.mkdir(parents=True, exist_ok=True)
    LBL_REAL.mkdir(parents=True, exist_ok=True)

    print("1. Pseudo-labeling real images with Haar cascade...")
    real_imgs = sorted(IMG_REAL.glob("*.jpg")) + sorted(IMG_REAL.glob("*.png"))
    labeled_real = 0
    
    for img_path in real_imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = detect(img)
        
        lines = []
        for (x, y, bw, bh) in boxes:
            # Haar находит только передние белые номера -> это всегда type1 (класс 0)
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            
        if lines:
            label_path = LBL_REAL / f"{img_path.stem}.txt"
            label_path.write_text("".join(lines))
            labeled_real += 1
            
    print(f"   Found plates in {labeled_real} / {len(real_imgs)} real images.")

    print("2. Linking synthetic images into train folder...")
    for img_path in IMG_SYNTH.glob("*.png"):
        lbl_path = LBL_SYNTH / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        dst_img = IMG_TRAIN / img_path.name
        dst_lbl = LBL_TRAIN / lbl_path.name
        if not dst_img.exists():
            os.symlink(img_path.resolve(), dst_img)
        if not dst_lbl.exists():
            os.symlink(lbl_path.resolve(), dst_lbl)

    print("3. Linking pseudo-labeled real images...")
    for lbl_path in LBL_REAL.glob("*.txt"):
        img_stem = lbl_path.stem
        img_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            p = IMG_REAL / f"{img_stem}{ext}"
            if p.exists():
                img_path = p
                break
        if not img_path:
            continue
            
        dst_img = IMG_TRAIN / img_path.name
        dst_lbl = LBL_TRAIN / lbl_path.name
        if not dst_img.exists():
            os.symlink(img_path.resolve(), dst_img)
        if not dst_lbl.exists():
            os.symlink(lbl_path.resolve(), dst_lbl)

    print("4. Writing data_finetune.yaml...")
    yaml_path = DATASET / "data_finetune.yaml"
    yaml_path.write_text(
        f"path: {DATASET.resolve()}\n"
        "train: images/train\n"
        "val: images/train\n"
        "names:\n"
        "  0: type1\n"
        "  1: type1a\n"
        "  2: type1b\n"
        "  3: other\n"
    )
    print("Done.")

if __name__ == "__main__":
    main()
