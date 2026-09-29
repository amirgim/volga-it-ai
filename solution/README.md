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



text
# Распознавание нестандартных ГРЗ РФ

Офлайн-пайплайн: детектор YOLOv8 (type1 / type1a / type1b / other) + OCR CRNN.

## Установка

```bash
pip install -r requirements.txt
```

## Запуск
```bash
python run.py --input /path/to/images --output result.csv
```
Опциональные флаги: --det, --ocr, --device, --onnx.

Веса моделей лежат в models/ (detector.onnx, ocr.onnx).
Пайплайн никогда не обращается к сети.

## Формат вывода
```bash
CSV с разделителем ;, кодировка UTF-8, заголовок:
```
text
image;plate_num;plate_type;confidence
Нераспознанные символы — #. Изображения без знаков не дают строк.

## Проверка логики постобработки
```bash
python -m src.selfcheck
```
## Обучение
```bash
python train/train_detector.py
python train/train_ocr.py
```
## Генерация синтетических данных
```bash
python generator/generate.py --n 5000 --seed 42
```
## Целевая скорость
Среднее время — не более 100 мс на изображение на Intel i5-7600 + GTX 1050 Ti 4 ГБ.
Для референсной конфигурации использовать ONNX + FP16.

