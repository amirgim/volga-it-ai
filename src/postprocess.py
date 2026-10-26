"""Plate post-processing: Cyrillic→Latin, alphabet filter, confidence mask,
format check (GOST R 50577-2018 type 1 / 1A / 1B)."""
ALPHABET = "0123456789ABEKMHOPCTYX"
LETTERS = set("ABEKMHOPCTYX")
DIGITS = set("0123456789")
REGION_PREFIX_3 = set("127")

_CYR_TO_LAT = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "У": "Y", "Х": "X",
})


def normalize(text: str) -> str:
    """Upper-case, translate Cyrillic, drop anything outside the plate alphabet."""
    text = text.upper().translate(_CYR_TO_LAT)
    return "".join(c for c in text if c in ALPHABET)


def mask_low_conf(text: str, confs: list[float], thresh: float = 0.5) -> str:
    """Replace chars whose per-char confidence is below `thresh` with '#'."""
    out = []
    for i, c in enumerate(text):
        if i < len(confs) and confs[i] < thresh:
            out.append("#")
        else:
            out.append(c)
    return "".join(out)


def is_valid(plate: str) -> bool:
    """Check 8- or 9-char GOST plate format. '#' positions fail validation."""
    if len(plate) not in (8, 9):
        return False
    if plate[0] not in LETTERS or plate[4] not in LETTERS or plate[5] not in LETTERS:
        return False
    for i in (1, 2, 3):
        if plate[i] not in DIGITS:
            return False
    region = plate[6:]
    if not all(c in DIGITS for c in region):
        return False
    if len(region) == 3 and region[0] not in REGION_PREFIX_3:
        return False
    return True


# ---- Step 4 additions: positional repair + strict postprocess ----

_LETTER_POS = (0, 4, 5)
_LETTER_SUBS = {"0": "O", "8": "B", "4": "A", "1": "T", "7": "T", "3": "E"}
_DIGIT_SUBS = {"O": "0", "B": "8", "T": "7", "A": "4", "P": "9",
               "C": "0", "X": "7", "H": "4", "M": "4", "E": "8",
               "Y": "4", "K": "4"}


def fix_by_position(plate: str) -> str:
    """Repair chars per GOST position rules. Unrecognized chars become '#'."""
    out = []
    for i, c in enumerate(plate):
        if c == "#":
            out.append(c)
            continue
        if i in _LETTER_POS:
            if c in LETTERS:
                out.append(c)
            elif c in _LETTER_SUBS:
                out.append(_LETTER_SUBS[c])
            else:
                out.append("#")
        else:
            if c in DIGITS:
                out.append(c)
            elif c in _DIGIT_SUBS:
                out.append(_DIGIT_SUBS[c])
            else:
                out.append("#")
    return "".join(out)


def postprocess(raw: str, confs=None, min_conf: float = 0.45) -> "str | None":
    """normalize -> fix_by_position -> is_valid.
    Returns valid plate, or None if it should be dropped (instead of emitting garbage)."""
    if confs is not None and confs:
        avg = sum(confs) / len(confs)
        if avg < min_conf:
            return None
    text = normalize(raw)
    if not text:
        return None
    text = fix_by_position(text)
    if "#" in text:
        return None
    if not is_valid(text):
        return None
    return text
