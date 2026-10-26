"""Shared OCR input helpers. A two-line type1a plate is cut in half and the
halves are glued side by side, so one single-line CRNN reads all plate types."""
import cv2
import numpy as np

OCR_W, OCR_H = 192, 32


def unfold_1a(crop: np.ndarray) -> np.ndarray:
    h = crop.shape[0]
    mid = max(1, h // 2)
    parts = []
    for p in (crop[:mid], crop[mid:]):
        ph, pw = p.shape[:2]
        if ph == 0 or pw == 0:
            continue
        nw = max(8, int(round(pw * OCR_H / ph)))
        parts.append(cv2.resize(p, (nw, OCR_H), interpolation=cv2.INTER_LINEAR))
    return np.hstack(parts) if len(parts) == 2 else cv2.resize(crop, (OCR_W, OCR_H))
