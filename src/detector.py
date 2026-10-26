"""YOLOv8 detector wrapper. Haar fallback when weights are missing.

Classes in the YOLO model are expected to be in order:
  0 type1, 1 type1a, 2 type1b, 3 other
"""
from pathlib import Path

import numpy as np

from broke.detect import detect as haar_detect
from broke.pipeline import _plate_type as haar_type


class Detector:
    def __init__(self, weights: str | None, device: str = "cuda",
                 conf: float = 0.5, iou: float = 0.45):
        self.conf = conf
        self.iou = iou
        self.device = device
        self.model = None
        self.names: dict[int, str] = {}
        if weights and Path(weights).exists():
            from ultralytics import YOLO
            self.model = YOLO(weights)
            self.model.to(device)
            self.names = dict(self.model.names)
        else:
            print(f"[detector] weights not found at {weights!r}, using Haar fallback")

    def __call__(self, img: np.ndarray) -> list[dict]:
        if self.model is None:
            return self._haar(img)
        return self._yolo(img)

    def _yolo(self, img: np.ndarray) -> list[dict]:
        res = self.model.predict(img, conf=self.conf, iou=self.iou,
                                 device=self.device, verbose=False)[0]
        out = []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            out.append({
                "bbox": (x1, y1, x2, y2),
                "type": self.names.get(cls, "other"),
                "conf": float(box.conf[0]),
            })
        return out

    def _haar(self, img: np.ndarray) -> list[dict]:
        out = []
        for x, y, w, h in haar_detect(img):
            crop = img[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            out.append({
                "bbox": (x, y, x + w, y + h),
                "type": haar_type(crop),
                "conf": 0.5,
            })
        return out
