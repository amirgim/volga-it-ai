import csv
import os
import time
from typing import Dict, List

import cv2
import numpy as np
import torch

from .detector import Detector
from .ocr import CRNN, greedy_decode
from .postprocess import mask_low_conf, normalize, is_valid

MEAN, STD = 0.5, 0.5
OCR_SIZE = (128, 32)  # (W, H)
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _crop(img: np.ndarray, bbox, pad: float = 0.04) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - pad * bw))
    y1 = max(0, int(y1 - pad * bh))
    x2 = min(w, int(x2 + pad * bw))
    y2 = min(h, int(y2 + pad * bh))
    return img[y1:y2, x1:x2]


def _preprocess(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, OCR_SIZE, interpolation=cv2.INTER_LINEAR)
    x = gray.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x[None, None, ...]


class Pipeline:
    def __init__(self, det_weights: str, ocr_weights: str,
                 device: str = "cuda", use_onnx: bool = True):
        self.device = device
        self.detector = Detector(det_weights, device=device)

        self.ort = None
        self.ocr = None
        if use_onnx and ocr_weights.endswith(".onnx"):
            import onnxruntime as ort
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if device == "cuda" else ["CPUExecutionProvider"])
            self.ort = ort.InferenceSession(ocr_weights, providers=providers)
            self.ort_input = self.ort.get_inputs()[0].name
        else:
            self.ocr = CRNN()
            self.ocr.load_state_dict(torch.load(ocr_weights, map_location=device))
            self.ocr.to(device).eval()

    def _ocr(self, x: np.ndarray) -> np.ndarray:
        if self.ort is not None:
            out = self.ort.run(None, {self.ort_input: x})[0]
            return out[0] if out.ndim == 3 else out
        with torch.no_grad():
            t = torch.from_numpy(x).to(self.device)
            return self.ocr(t)[0].cpu().numpy()

    def process(self, img: np.ndarray) -> List[Dict]:
        out: List[Dict] = []
        for d in self.detector(img):
            if d["type"] == "other":
                continue
            crop = _crop(img, d["bbox"])
            if crop.size == 0:
                continue
            x = _preprocess(crop)
            logits = self._ocr(x)
            text, confs = greedy_decode(logits)
            text = normalize(text)
            if not text:
                continue
            plate = mask_low_conf(text, confs, 0.5)
            if all(c == "#" for c in plate):
                continue
            ocr_conf = float(np.mean(confs)) if confs else 0.0
            out.append({
                "plate_num": plate,
                "plate_type": d["type"],
                "confidence": round(float(d["conf"]) * ocr_conf, 4),
            })
        return out


def run_dir(input_dir: str, output_csv: str, det_weights: str,
            ocr_weights: str, device: str = "cuda", use_onnx: bool = True) -> None:
    p = Pipeline(det_weights, ocr_weights, device, use_onnx)
    files = sorted(f for f in os.listdir(input_dir)
                   if os.path.splitext(f)[1].lower() in EXTS)
    rows: List[tuple] = []
    t0 = time.time()
    for f in files:
        img = cv2.imread(os.path.join(input_dir, f))
        if img is None:
            continue
        for r in p.process(img):
            rows.append((f, r["plate_num"], r["plate_type"], r["confidence"]))
    elapsed = time.time() - t0
    with open(output_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["image", "plate_num", "plate_type", "confidence"])
        w.writerows(rows)
    n = max(1, len(files))
    print(f"images={len(files)} rows={len(rows)} avg_ms={elapsed / n * 1000:.1f}")
