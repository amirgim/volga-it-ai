"""Character recognition via Tesseract with a plate-specific whitelist."""
import re

import cv2
import numpy as np
import pytesseract

WHITELIST = "0123456789ABEKMHOPCTYX"
TESS_CONFIG = f"--oem 1 --psm 7 -c tessedit_char_whitelist={WHITELIST}"

_CYR_TO_LAT = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "У": "Y", "Х": "X",
})


def _prep(plate_bgr: np.ndarray) -> np.ndarray:
    h, w = plate_bgr.shape[:2]
    target_w = 400
    scale = target_w / max(w, 1)
    plate_bgr = cv2.resize(
        plate_bgr,
        (target_w, max(32, int(h * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def read_plate(plate_bgr: np.ndarray) -> str:
    """Return uppercased Latin string with non-plate chars stripped."""
    img = _prep(plate_bgr)
    raw = pytesseract.image_to_string(img, config=TESS_CONFIG)
    text = raw.upper().translate(_CYR_TO_LAT)
    return re.sub(r"[^0-9ABEKMHOPCTYX]", "", text)
