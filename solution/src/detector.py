from typing import List, Dict

import numpy as np


TYPE_NAMES = ["type1", "type1a", "type1b", "other"]


class Detector:
    """YOLOv8 wrapper. Tries ONNX first, falls back to Ultralytics."""

    def __init__(self, weights: str, device: str = "cuda"):
        self.weights = weights
        self.device = device
        self._model = None
        self._ort = None
        if weights.endswith(".onnx"):
            import onnxruntime as ort
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if device == "cuda" else ["CPUExecutionProvider"])
            self._ort = ort.InferenceSession(weights, providers=providers)
        else:
            from ultralytics import YOLO
            self._model = YOLO(weights)

    def __call__(self, img: np.ndarray) -> List[Dict]:
        if self._model is not None:
            res = self._model.predict(img, conf=0.25, iou=0.5, verbose=False)[0]
            out = []
            for b in res.boxes:
                out.append({
                    "bbox": b.xyxy[0].cpu().numpy().tolist(),
                    "conf": float(b.conf[0]),
                    "type": TYPE_NAMES[int(b.cls[0])] if int(b.cls[0]) < len(TYPE_NAMES) else "other",
                })
            return out
        # Fallback ONNX path is unused when Ultralytics weights are given.
        return []
