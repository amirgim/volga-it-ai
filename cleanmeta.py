
"""Drop all synthetic rows (is_synthetic=1) from dataset/meta.csv.

Safety: refuses to write if the result would be header-only but the
input had data rows. That prevents silent data loss from a corrupted
or already-cleared meta.csv.
"""
import csv
import shutil
import sys
from pathlib import Path

META = Path("dataset/meta.csv")
BAK = Path("dataset/meta.csv.bak")


def main() -> None:
    if not META.exists():
        sys.exit(f"not found: {META}")

    with META.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh, delimiter=";"))

    if not rows:
        sys.exit("meta.csv is empty")

    header = rows[0]
    if "is_synthetic" not in header:
        sys.exit(f"no 'is_synthetic' column: {header}")
    idx_synth = header.index("is_synthetic")

    data_rows = [r for r in rows[1:] if r and any(c.strip() for c in r)]
    if not data_rows:
        sys.exit("meta.csv has no data rows (header only). Restore from .bak or git first.")

    kept = [header]
    dropped = 0
    for row in data_rows:
        if len(row) > idx_synth and row[idx_synth].strip() == "1":
            dropped += 1
            continue
        kept.append(row)

    real_count = len(kept) - 1
    if real_count == 0 and len(data_rows) > 0:
        sys.exit(
            f"REFUSING TO WRITE: all {len(data_rows)} rows would be dropped, "
            f"no real rows survive. meta.csv left untouched."
        )

    if not BAK.exists():
        shutil.copy2(META, BAK)
        print(f"backup -> {BAK}")

    with META.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerows(kept)

    print(f"kept {real_count} real rows, dropped {dropped} synthetic")


if __name__ == "__main__":
    main()