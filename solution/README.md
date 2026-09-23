# Non-standard RU plate recognition

Offline pipeline: YOLOv8 detector (type1 / type1a / type1b / other) + CRNN OCR.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python run.py --input /path/to/images --output result.csv
```

Optional flags: `--det`, `--ocr`, `--device`, `--onnx`.

Model weights live in `models/` (`detector.onnx`, `ocr.onnx`).
The pipeline never calls the network.

## Output format

CSV with `;`, UTF-8, header:

```
image;plate_num;plate_type;confidence
```

Unrecognized characters are `#`. Images with no plate produce no rows.

## Verify postprocess logic

```bash
python -m src.selfcheck
```

## Train

```bash
python train/train_detector.py
python train/train_ocr.py
```

## Generate synthetic data

```bash
python generator/generate.py --n 5000 --seed 42
```

## Speed target

Average <=100 ms per image on Intel i5-7600 + GTX 1050 Ti 4 GB.
Use ONNX + FP16 for the reference configuration.
