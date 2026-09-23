# Dataset

Structure:

```
dataset/
  images/real/        # participant-collected real photos
  images/synthetic/   # generator output
  labels/             # one .txt per image (YOLO format)
  meta.csv            # one row per plate
  generator/          # generator script + requirements
  data.yaml           # YOLO data config
  LICENSE
```

`meta.csv` columns: `image;plate_num;plate_type;bbox;quad;is_vehicle;is_synthetic;source;license;conditions`.

Recommended minimums:

- type1a: >=150 images, >=50 unique plates
- type1b: >=300 images, >=100 unique plates
- other:  >=50 negative examples
- synthetic: >=5000 images

Do not include images from the organizers' debug set.

Run generator:

```bash
python generator/generate.py --n 5000
```

Then run the organizers' `validate_dataset.py` and attach the report.
