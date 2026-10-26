
## Честные замечания (актуально на момент сдачи)

- OCR: CRNN (CTC) обучен на синтетических кропах (`scripts/extract_crops.py` → `train/train_ocr.py`).
  Двухстрочный type1a «разворачивается» в одну строку (`src/unfold.py`) и читается той же моделью.
- Реальные данные: 410 фото Wikimedia Commons (лицензии в `meta.csv`). Квоты ТЗ по реальным
  type1a (150) и type1b (300) НЕ выполнены — таких фото в свободных источниках мало.
- Разметка `dataset/labels/real` получена АВТОМАТИЧЕСКИ (Haar-каскад OpenCV), не вручную, класс везде type1.
  Ручной разметкой она не является. Реальные `plate_num`, `bbox`, `quad` в `meta.csv` не размечены.
- Запуск: `python -m src.run --input <каталог> --output result.csv --det models/yolo.pt --ocr models/crnn.onnx --device cpu`
  (`--no-onnx` — использовать `models/crnn.pt`).
