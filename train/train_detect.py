"""Train YOLOv8 detector on dataset/data.yaml (4 classes: type1/1a/1b/other)."""
import argparse

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--base", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="plate")
    args = ap.parse_args()

    model = YOLO(args.base)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, project=args.project, name=args.name)
    print(f"done: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
