"""Plate detection via OpenCV Haar cascade — no training, no torch."""
import cv2
import numpy as np

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
_cascade = None


def _get():
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(_CASCADE_PATH)
        if _cascade.empty():
            raise RuntimeError(f"cannot load cascade: {_CASCADE_PATH}")
    return _cascade


def detect(img_bgr: np.ndarray) -> list:
    """Return list of (x, y, w, h) plate boxes, sorted by area desc."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    boxes = _get().detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 20),
    )
    out = [tuple(map(int, b)) for b in boxes]
    out.sort(key=lambda b: b[2] * b[3], reverse=True)
    return out
