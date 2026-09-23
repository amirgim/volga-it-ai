"""Convert meta.csv to YOLO labels and fine-tune YOLOv8. Run from solution/ root."""
import argparse
import csv
import os

from ultralytics import YOLO

TYPE_TO_CLS = {"type1": 0, "type1a": 1, "type1b": 2, "other": 3}


def write_yolo_labels(meta_csv: str, root: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cache: dict = {}
    with open(meta_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            if row["plate_type"] not in TYPE_TO_CLS:
                continue
            img_rel = row["image"]
            img_path = os.path.join(root, img_rel)
            import cv2
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            x, y, bw, bh = map(float, row["bbox"].split(","))
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            cache.setdefault(img_rel, []).append(
                f"{TYPE_TO_CLS[row['plate_type']]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
            )
    for img_rel, lines in cache.items():
        name = os.path.splitext(os.path.basename(img_rel))[0] + ".txt"
        with open(os.path.join(out_dir, name), "w") as fh:
            fh.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="dataset/meta.csv")
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--labels", default="dataset/labels")
    ap.add_argument("--base", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="models/detector.pt")
    args = ap.parse_args()

    write_yolo_labels(args.meta, args.root, args.labels)
    model = YOLO(args.base)
    model.train(
        data="dataset/data.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=16,
        project="runs",
        name="plate_det",
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.export(format="onnx", imgsz=args.imgsz, half=True)


if __name__ == "__main__":
    main()
