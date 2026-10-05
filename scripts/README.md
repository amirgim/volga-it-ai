# scripts/

## fetch_open_plates.py

Fetches CC-licensed plate photos from Wikimedia Commons.

```bash
python scripts/fetch_open_plates.py --limit 200
```

Dry-run first (writes nothing):

```bash
python scripts/fetch_open_plates.py --limit 200 --dry-run
```

### What it fills

| Field | Source |
|-------|--------|
| `image` | `images/real/commons_<sha1>.<ext>` |
| `plate_num` | extracted from filename/description via regex; empty if not found |
| `plate_type` | `type1` placeholder — **verify manually** |
| `bbox`, `quad` | empty — **fill manually** |
| `is_vehicle` | `1` |
| `is_synthetic` | `0` |
| `source` | Commons page URL |
| `license` | e.g. `CC BY-SA 4.0` |
| `conditions` | empty — **fill manually** |

### What it does NOT fill

- **`bbox` / `quad`** — would require running a detector. Auto-bbox from a
  downloaded model = fake annotation. Left empty on purpose.
- **`plate_num`** when the filename has no plate pattern — same reason.
- **`plate_type`** — needs a visual check. Script guesses `type1` as a
  starting value.
- **`conditions`** — needs human eyes.

### How plate extraction works

Filename and description are scanned for the regex
`[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}` after Cyrillic→Latin
translation. Example filenames that will match:

```
Moscow_A123BC777.jpg            → A123BC777
Toyota_with_plate_K555OP25.png  → K555OP25
Грузовик_Е777КХ99.jpg           → E777KX99
```

If nothing matches, `plate_num` stays empty and you fill it in.

### Expected yield

Wikimedia has **10–200 usable Russian-plate photos** in total. Not
thousands. Budget 1–2 runs of `--limit 200` and then move to your own
photos or another public dataset.

### Related

- `dataset/images/real/` — where images land
- `dataset/meta.csv` — where rows are appended
- `docker-compose.broke.yml` — CPU-only environment that runs this script
  without PyTorch
