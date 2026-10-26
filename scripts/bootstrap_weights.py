"""One-time bootstrap: train YOLOv8n on synthetic data, export to models/yolo.pt.

Run once (CPU ~15-20 min, GPU ~3 min):
    python scripts/bootstrap_weights.py

Produces:
    models/yolo.pt   - YOLOv8n, 4 classes (type1/1a/1b/other)
    runs/detect/plate_bootstrap/weights/best.pt  - raw training output

Why this exists: task requires offline inference with shipped weights.
Ultralytics auto-downloads yolov8n.pt from github on first use, which breaks
offline. This script bakes the weights locally.
"""
import argparse
import shutil
from pathlib import Path


def write_synth_yaml() -> Path:
    """Synthetic-only data config. Val is a held-out slice of synthetic."""
    p = Path("dataset/data_synth.yaml")
    p.write_text(
        f"path: {Path('dataset').resolve()}\n"
        "train: images/synthetic\n"
        "val: images/synthetic\n"
        "names:\n"
        "  0: type1\n"
        "  1: type1a\n"
        "  2: type1b\n"
        "  3: other\n",
        encoding="utf-8",
    )
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--base", default="yolov8n.pt",
                    help="base checkpoint; auto-downloaded once then baked in")
    ap.add_argument("--out", default="models/yolo.pt")
    args = ap.parse_args()

    from ultralytics import YOLO

    data_yaml = write_synth_yaml()
    print(f"[bootstrap] using {data_yaml}")

    # First call: Ultralytics downloads yolov8n.pt if missing.
    # Subsequent runs are offline because base is cached in cwd.
    base = Path(args.base)
    model = YOLO(str(base) if base.exists() else args.base)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs/detect",
        name="plate_bootstrap",
        exist_ok=True,
        verbose=True,
    )

    best = Path("runs/detect/plate_bootstrap/weights/best.pt")
    if not best.exists():
        raise SystemExit(f"training produced no weights at {best}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    print(f"[bootstrap] copied {best} → {out}")

    # Also cache the base checkpoint in cwd so future runs work offline
    cache = Path("models") / Path(args.base).name
    if not cache.exists() and Path(args.base).exists():
        shutil.copy2(args.base, cache)
        print(f"[bootstrap] cached base checkpoint → {cache}")


if __name__ == "__main__":
    main()
