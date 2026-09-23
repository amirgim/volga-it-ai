import argparse
from src.pipeline import run_dir


def main():
    ap = argparse.ArgumentParser(description="Offline license plate recognizer")
    ap.add_argument("--input", required=True, help="directory with .jpg/.png images")
    ap.add_argument("--output", default="result.csv")
    ap.add_argument("--det", default="models/detector.onnx")
    ap.add_argument("--ocr", default="models/ocr.onnx")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--onnx", action="store_true", default=True)
    args = ap.parse_args()
    run_dir(args.input, args.output, args.det, args.ocr, args.device, args.onnx)


if __name__ == "__main__":
    main()
