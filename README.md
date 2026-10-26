# Распознавание нестандартных ГРЗ

Пайплайн детекции и распознавания номерных знаков типов **1**, **1A** (квадратные)
и **1B** (жёлтые) по ГОСТ Р 50577-2018, с корректным отсеиванием `other`.

Полуфинал дисциплины «Искусственный интеллект и анализ данных», куратор АИС Город.

## Структура

```
.
├── src/                 # основной пайплайн (YOLO + CRNN, ONNX)
│   ├── detector.py      # YOLOv8 + fallback на Haar
│   ├── ocr.py           # CRNN (CTC)
│   ├── postprocess.py   # нормализация, маска по confidence, валидация
│   ├── pipeline.py      # сборка + fallback на Tesseract
│   └── run.py           # CLI: --input DIR --output CSV
├── broke/               # CPU-only fallback: Haar + Tesseract
├── train/
│   ├── train_detect.py  # обучение YOLOv8
│   └── train_ocr.py     # обучение CRNN (stub, см. README)
├── generator/
│   └── generate.py      # синтетика (5000 картинок, seed=42)
├── scripts/
│   ├── fetch_open_plates.py  # Wikimedia Commons (только CC)
│   └── bootstrap_weights.py  # обучение YOLO → models/yolo.pt
├── dataset/
│   ├── images/real/     # 400 фотографий с Commons
│   ├── images/synthetic/# 5000 синтетических
│   ├── labels/          # YOLO .txt
│   ├── meta.csv         # разметка (один ряд — один знак)
│   └── data.yaml        # конфиг YOLO
├── models/              # веса (.pt / .onnx) — кладутся сюда
├── test_images/         # для быстрой проверки
├── Dockerfile.broke     # образ: OpenCV + Tesseract, без torch
└── docker-compose.broke.yml
```

## Запуск

### Быстрый старт (Docker, CPU, без обучения)

Ничего не надо устанавливать — образ самодостаточный (OpenCV + Tesseract).

Инференс на реальных фотографиях, скачанных с Commons:

```bash
docker compose -f docker-compose.broke.yml build
docker compose -f docker-compose.broke.yml run --rm infer
```

Результат — ```result.csv``` в корне. Ожидаемое качество exact-match: **20–40%**
(Tesseract не читает двухстрочные `1A`; см. «Ограничения»).

На своей папке:

```bash
docker compose -f docker-compose.broke.yml run --rm infer \
  python -m broke.run --input test_images --output result.csv
```

### Основной пайплайн (YOLO + CRNN)

Требует PyTorch и веса в ```models/```. Если весов нет — автоматически откатится
на `broke` (Haar + Tesseract), предупредив в stderr.

```bash
pip install -r requirements.txt
python -m src.run --input dataset/images/real --output result.csv \
  --det models/yolo.pt --ocr models/crnn.onnx --device cuda
```

Флаг ```--no-onnx``` — использовать ```models/crnn.pt``` вместо ONNX.

### Только синтетика (проверить генератор)

```bash
docker compose -f docker-compose.broke.yml run --rm generate
```

или нативно:

```bash
pip install -r generator/requirements.txt
python generator/generate.py --n 5000 --seed 42
```

## Формат результата

CSV, разделитель ```;```, UTF-8, заголовок обязателен:

```
image;plate_num;plate_type;confidence
img_0042.jpg;A123BC716;type1a;0.93
img_0043.jpg;K555OP25;type1b;0.88
```

- ```image``` — имя файла, без пути.
- ```plate_num``` — номер формата `A123BC456`, латиница, верхний регистр, `#` для нечитаемых.
- ```plate_type``` — ```type1``` | ```type1a``` | ```type1b``` | ```other```.
- ```confidence``` — [0.0, 1.0], точка как разделитель.

Если на изображении несколько знаков — несколько строк. Если ноль — строк нет.

## Датасет

### Реальные данные

@dataset/images/real/``` — 400 фото с Wikimedia Commons.
Источник, лицензия и авто-извлечённый `plate_num` — в `dataset/meta.csv`.

**Что не доделано:**

- @bbox``` и @quad``` **пустые** — детектор на этих данных не обучится, пока
  не разметить вручную.
- @plate_type``` по умолчанию `type1` — нужно проверить глазами.
- Квоты ТЗ (`type1a` ≥ 150 / 50 уник., `type1b` ≥ 300 / 100 уник.) **не
  выполнены** — на Commons таких фото десятки, не сотни.

Что делать: доразметить 150 `type1a` + 300 `type1b` (своя съёмка на парковке,
свободные фотобанки) и записать в `meta.csv` реальные `bbox`/`quad`.

### Синтетика

```generator/generate.py``` — 5000 изображений, `seed=42`, воспроизводимо.

**Ограничение:** во всех YOLO-разметках bbox = `0.5 0.5 0.9 0.9`
(знак на весь кадр). Детектор на этом научится только «есть знак по центру».
Для нормального обучения нужно вставление знака в фон + реальные боксы.

### Валидатор организаторов

Перед сдачей:

```bash
python validate_dataset.py dataset/
```

Отчёт приложить к решению.

## Train

### Детектор

```bash
python -m train.train_detect --epochs 50 --imgsz 640
# → runs/detect/plate/weights/best.pt
cp runs/detect/plate/weights/best.pt models/yolo.pt
```

Требует @dataset/data.yaml``` с реальной разметкой (bbox для ```train: images/real```).
С текущей разметкой (пустые bbox) обучение упадёт — это ожидаемо.

### OCR

@train/train_ocr.py``` пока **заглушка**. План:

1. Прогнать детектор по `dataset/images/*`, получить bbox.
2. Вырезать кропы, ресайз в 128×32, сохранить как `dataset/crops/<plate_num>.png`.
3. `python -m train.train_ocr --crops dataset/crops --out models/crnn.onnx`.

Кроп-экстрактор не реализован. До этого OCR работает только через fallback
(Tesseract), что даёт ~20–40% exact match.

## Веса

| Компонент | Файл | Откуда взять |
|---|---|---|
| YOLOv8 детектор | `models/yolo.pt` (~6 МБ) | `python scripts/bootstrap_weights.py` |
| CRNN OCR | `models/crnn.onnx` | **не реализовано**, используется Tesseract |
| Haar cascade | — | уже внутри `opencv-python`, файл-константа |
| Tesseract LSTM | — | уже внутри `tesseract-ocr`, системный пакет |

### Как получить веса для сдачи

```bash
# один раз, нужен интернет только на первую загрузку yolov8n.pt
pip install -r requirements.txt
python scripts/bootstrap_weights.py --epochs 20
```

После этого `models/yolo.pt` **кладётся в архив с решением** (в `.gitignore`
он исключён, но в финальный zip — обязателен). Дальше инференс работает
полностью офлайн:

```bash
python -m src.run --input /path/to/hidden_set --output result.csv \
  --det models/yolo.pt --ocr models/crnn.onnx --device cpu
```

Если `models/yolo.pt` отсутствует — `src/detector.py` автоматически откатывается
на Haar cascade и печатает предупреждение в stderr. То есть **решение не падает
даже без весов**, просто качество падает до уровня `broke/`.

### Почему CRNN не натренирован

Для CRNN нужно:
1. Вырезать кропы номеров из `dataset/images/*` по bbox.
2. Ресайз в 128×32, сохранить как `dataset/crops/<plate_num>.png`.
3. Прогнать CTC-loss 30 эпох.

Шаг 1–2 не реализован (см. `train/train_ocr.py`). Пока OCR = Tesseract.
Это даёт ~20–40% exact match против целевых 70–85%, но **работает из коробки**.

## Ограничения

- **type1A (двухстрочные)** основным CRNN **не читаются** — модель обучена на
  однострочных кропах. Нужен либо отдельный OCR для двух строк, либо
  двухголовый выход. В fallback (Tesseract `--psm 7`) тоже не читаются.
- **Синтетика однотипная** — фиксированный шрифт, bbox на весь кадр.
- **Bbox в `meta.csv` пустой** для реальных фото — детектор нельзя обучить,
  пока не доразмечено.
- **Нет CRNN-весов** — OCR через Tesseract fallback.
- **Скорость** не замерена на референсе (i5-7600 + GTX 1050 Ti). На broke
  ~20–40 мс/кадр в Docker (без GPU), на YOLO+CRNN нужно мерить после обучения.

## Воспроизводимость

- Синтетика: `seed=42` зафиксирован в `generator/generate.py` (по умолчанию).
- Commons: `scripts/fetch_open_plates.py` использует `User-Agent`, `maxlag=5`,
  retry-after — не банимся.
- Все веса — в `models/` рядом с решением, офлайн.
- YOLO веса воспроизводятся командой `python scripts/bootstrap_weights.py`.

## Команды одной строкой (что нужно комиссии)

```bash
docker compose -f docker-compose.broke.yml build
docker compose -f docker-compose.broke.yml run --rm infer \
  python -m broke.run --input /path/to/hidden_set --output result.csv
```

Это работает **без** обучения, **без** GPU, **без** интернета — только
OpenCV и Tesseract внутри образа.

## Честные замечания (актуально на момент сдачи)

- OCR: CRNN (CTC) обучен на синтетических кропах (`scripts/extract_crops.py` → `train/train_ocr.py`).
  Двухстрочный type1a «разворачивается» в одну строку (`src/unfold.py`) и читается той же моделью.
- Реальные данные: 410 фото Wikimedia Commons (лицензии в `meta.csv`). Квоты ТЗ по реальным
  type1a (150) и type1b (300) НЕ выполнены — таких фото в свободных источниках мало.
- Разметка `dataset/labels/real` получена АВТОМАТИЧЕСКИ (Haar-каскад OpenCV), не вручную, класс везде type1.
  Ручной разметкой она не является. Реальные `plate_num`, `bbox`, `quad` в `meta.csv` не размечены.
- Запуск: `python -m src.run --input <каталог> --output result.csv --det models/yolo.pt --ocr models/crnn.onnx --device cpu`
  (`--no-onnx` — использовать `models/crnn.pt`).

## Честные замечания (актуально на момент сдачи)

- OCR: CRNN (CTC) обучен на синтетических кропах (`scripts/extract_crops.py` → `train/train_ocr.py`).
  Двухстрочный type1a «разворачивается» в одну строку (`src/unfold.py`) и читается той же моделью.
- Реальные данные: 410 фото Wikimedia Commons (лицензии в `meta.csv`). Квоты ТЗ по реальным
  type1a (150) и type1b (300) НЕ выполнены — таких фото в свободных источниках мало.
- Разметка `dataset/labels/real` получена АВТОМАТИЧЕСКИ (Haar-каскад OpenCV), не вручную, класс везде type1.
  Ручной разметкой она не является. Реальные `plate_num`, `bbox`, `quad` в `meta.csv` не размечены.
- Запуск: `python -m src.run --input <каталог> --output result.csv --det models/yolo.pt --ocr models/crnn.onnx --device cpu`
  (`--no-onnx` — использовать `models/crnn.pt`).
