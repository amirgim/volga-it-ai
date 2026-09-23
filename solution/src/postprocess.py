import re

# Allowed glyphs in RU plates (Latin only, upper case)
LETTERS = set("ABEKMHOPCTYX")
DIGITS = set("0123456789")

_PLATE_RE = re.compile(r"^([ABEKMHOPCTYX])(\d{3})([ABEKMHOPCTYX]{2})(\d{2,3})$")

# Confusion pairs between Cyrillic and Latin glyphs with same shape
CYR_TO_LAT = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "У": "Y", "Х": "X",
})


def normalize(s: str) -> str:
    return s.upper().translate(CYR_TO_LAT).replace(" ", "").strip()


def is_valid(plate: str) -> bool:
    """True if plate matches [L][DDD][LL][DD or DDD with leading 1|2|7]."""
    m = _PLATE_RE.match(plate)
    if not m:
        return False
    region = m.group(4)
    if len(region) == 3 and region[0] not in "127":
        return False
    return True


def mask_low_conf(text: str, confs, threshold: float = 0.5) -> str:
    """Replace characters below threshold with '#'."""
    if len(text) != len(confs):
        return text
    return "".join(ch if c >= threshold else "#" for ch, c in zip(text, confs))
