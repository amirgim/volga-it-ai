"""Compare result.csv with the organizers' reference CSV (debug set)."""
import argparse
import csv
from collections import defaultdict


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=";"))


def same(pred, ref):
    return len(pred) == len(ref) and all(r == "#" or p == r for p, r in zip(pred, ref))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="result.csv")
    ap.add_argument("--ref", required=True)
    a = ap.parse_args()
    pred, ref = defaultdict(list), read(a.ref)
    for r in read(a.pred):
        pred[r["image"]].append(r)
    targets = [r for r in ref if r.get("plate_type") != "other"]
    hit = type_ok = 0
    for r in targets:
        for p in pred.get(r["image"], []):
            if same(p["plate_num"], r["plate_num"]):
                hit += 1
                type_ok += p["plate_type"] == r["plate_type"]
                break
    n = max(1, len(targets))
    total_pred = sum(len(v) for v in pred.values())
    print(f"reference plates: {len(targets)}  predicted rows: {total_pred}")
    print(f"exact plate match: {hit}/{len(targets)} = {hit / n:.2%}")
    print(f"  of them correct type: {type_ok}/{max(1, hit)}")


if __name__ == "__main__":
    main()
