# Synthetic plate generator

Deterministic: fixed seed (default 42). Regenerates the same dataset.

```bash
python generator/generate.py --n 5000 --seed 42
```

Outputs:

- `dataset/images/synthetic/syn_XXXXXX.png`
- `dataset/labels/syn_XXXXXX.txt` (YOLO: `cls cx cy w h`)
- appends rows to `dataset/meta.csv`

Type distribution is weighted: 30% type1, 30% type1a, 30% type1b, 10% other.
Augmentations: random perspective, Gaussian blur, Gaussian pixel noise.
