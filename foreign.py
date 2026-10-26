import csv, re, shutil, urllib.parse
from pathlib import Path
META = Path("dataset/meta.csv")
KW = re.compile(r"german|deutsch|polish|poland|ukrain|belarus|kazakh|latvia|lithuan|estonia|finland|bulgar|czech|"
                r"slovak|serbia|georgia|armenia|azerbaijan|china|japan|usa|france|italy|rome|spain|british|dutch|"
                r"nether|swed|norw|austria|swiss|switzerland|turk|israel|korea|vietnam|hungary|moto|trailer|"
                r"diplomatic|police|military|transit|eu_license|waymo|mitsubishi", re.I)
EXTRA = Path("foreign.txt")
extra = {l.strip() for l in EXTRA.read_text().split() if l.strip()} if EXTRA.exists() else set()
if not Path("dataset/meta.csv.bak_foreign").exists():
    shutil.copy2(META, "dataset/meta.csv.bak_foreign")
rows = list(csv.reader(open(META, encoding="utf-8-sig", newline=""), delimiter=";"))
h = rows[0]; ix = {k: i for i, k in enumerate(h)}
n = 0
for r in rows[1:]:
    if r[ix["is_synthetic"]] != "0":
        continue
    name = urllib.parse.unquote(r[ix["source"]].split("/wiki/")[-1])
    stem = Path(r[ix["image"]]).stem
    if KW.search(name) or stem in extra:
        r[ix["plate_type"]] = "other"; r[ix["plate_num"]] = ""
        lab = Path("dataset/labels/real") / f"{stem}.txt"
        if lab.exists(): lab.unlink()
        n += 1
csv.writer(open(META, "w", encoding="utf-8-sig", newline=""), delimiter=";").writerows(rows)
print("помечено как other:", n)
