"""Main pipeline entry point. Falls back to Haar+Tesseract if weights missing."""
import argparse

from src.pipeline import run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Plate recognition inference")
    ap.add_argument("--input", required=True, help="directory with .jpg/.png images")
    ap.add_argument("--output", default="result.csv")
    ap.add_argument("--det", default="models/yolo.pt")
    ap.add_argument("--ocr", default="models/crnn.onnx")
    ap.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    ap.add_argument("--no-onnx", action="store_true",
                    help="use models/crnn.pt (torch) instead of .onnx")
    args = ap.parse_args()
    run_dir(args.input, args.output, args.det, args.ocr,
            args.device, use_onnx=not args.no_onnx)


if __name__ == "__main__":
    main()
