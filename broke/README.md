# broke/

Fallback pipeline for machines that cannot run PyTorch.

**Detection:** OpenCV Haar cascade `haarcascade_russian_plate_number.xml`
(ships with OpenCV, ~900 KB).

**Recognition:** Tesseract OCR with character whitelist
`0123456789ABEKMHOPCTYX`, page-segmentation mode 7 (single line).

**Type classification:** heuristic on crop colour (yellow → type1b) +
aspect ratio (square → type1a, wide → type1).

## What it does NOT do

- No training. Haar cascade and Tesseract are pre-trained.
- No GPU. No CUDA. No torch.
- No neural network of any kind.

## Install (native, no Docker)

Requires the **Tesseract binary**, not just the Python wrapper.

- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `apt install tesseract-ocr`
- macOS: `brew install tesseract`

Then:

```bash
pip install -r broke/requirements.txt
```

## Run

Generate synthetic data:

```bash
python -m broke.make_synthetic --n 5000 --seed 42
```

Inference on a folder:

```bash
python -m broke.run --input test_images --output result.csv
```

## Docker

```bash
docker compose -f docker-compose.broke.yml build
docker compose -f docker-compose.broke.yml run --rm generate
docker compose -f docker-compose.broke.yml run --rm infer
```

## Accuracy

Expected exact-match on real photos: **20–40%**.
The neural stack (YOLO + CRNN) reaches **70–85%**.

Use this only if PyTorch cannot be installed on the target machine.

## Known limitations

- Haar cascade detects only white rectangular plates — misses type1a and
  type1b partially.
- Tesseract mixes up `B`/`8`, `O`/`0`, `C`/`G` — the whitelist blocks
  Cyrillic but not Latin/Cyrillic shape ambiguity.
- No confidence score from Tesseract — outputs `0.5` as a placeholder.
- Two-line (type1a) plates are not split — Tesseract reads them as one line.
