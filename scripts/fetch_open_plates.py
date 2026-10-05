"""Fetch CC-licensed plate photos from Wikimedia Commons.

Polite client: retries on 429 with exponential backoff, reads Retry-After,
uses maxlag=5, sends a proper User-Agent with contact info.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# Wikimedia UA policy: name/version + contact. Anonymous clients with
# useless UAs get the harshest rate limits.
USER_AGENT = (
    "PlateDatasetBot/0.3 "
    "(https://example.org/plate-dataset; contact: plate-dataset@example.org)"
)

META_HEADER = [
    "image", "plate_num", "plate_type", "bbox", "quad",
    "is_vehicle", "is_synthetic", "source", "license", "conditions",
]

ALLOWED_LICENSES = re.compile(
    r"(CC0|Public domain|CC BY(-SA)?(\s|-)?[0-9]|attribution)",
    re.IGNORECASE,
)

MIME_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

DEFAULT_CATEGORIES = [
    "License plates of Russia",
    "License plates of the Soviet Union",
    "Vehicle registration plates of Russia",
    "Russian license plates",
    "Taxis of Russia",
    "Police vehicles of Russia",
]

DEFAULT_QUERIES = [
    "Russian license plate",
    "Russian registration plate",
    "Автомобильный номер",
    "Регистрационный знак",
]

PLATE_RE = re.compile(r"[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}")

CYR_TO_LAT = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "У": "Y", "Х": "X",
})


def api_get(params: dict, use_post: bool = False, retries: int = 5,
            base_wait: float = 2.0) -> dict:
    """Wikimedia API call with retry on 429. Returns parsed JSON."""
    params = dict(params)
    params["format"] = "json"
    params["maxlag"] = "5"     # polite: fail fast if replicas lag
    data = urlencode(params).encode("utf-8")

    last_exc = None
    for attempt in range(retries):
        try:
            if use_post:
                req = Request(
                    COMMONS_API,
                    data=data,
                    headers={
                        "User-Agent": USER_AGENT,
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Accept": "application/json",
                    },
                )
            else:
                req = Request(
                    COMMONS_API + "?" + data.decode("utf-8"),
                    headers={
                        "User-Agent": USER_AGENT,
                        "Accept": "application/json",
                    },
                )
            with urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))

        except HTTPError as exc:
            if exc.code not in (429, 503):
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            if retry_after and retry_after.isdigit():
                wait = int(retry_after)
            else:
                wait = base_wait * (2 ** attempt)
            wait = min(wait, 90)
            print(f"  HTTP {exc.code}, sleeping {wait}s "
                  f"(attempt {attempt + 1}/{retries})", file=sys.stderr)
            time.sleep(wait)
            last_exc = exc
            continue

        except (URLError, TimeoutError) as exc:
            wait = base_wait * (2 ** attempt)
            print(f"  network error, sleeping {wait}s "
                  f"(attempt {attempt + 1}/{retries})", file=sys.stderr)
            time.sleep(wait)
            last_exc = exc
            continue

    raise last_exc if last_exc else RuntimeError("api_get failed")


def category_files(category: str, limit: int, api_sleep: float) -> list:
    titles = []
    cont = None
    while len(titles) < limit:
        p = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmtype": "file",
            "cmlimit": min(500, limit - len(titles)),
        }
        if cont:
            p["cmcontinue"] = cont
        try:
            data = api_get(p, use_post=True)
        except Exception as exc:
            print(f"  category '{category}' failed: {exc}", file=sys.stderr)
            break
        members = data.get("query", {}).get("categorymembers", [])
        if not members:
            break
        titles.extend(m["title"] for m in members)
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
        time.sleep(api_sleep)
    return titles[:limit]


def search_files(query: str, limit: int) -> list:
    try:
        data = api_get({
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srnamespace": "6",
            "srlimit": min(500, limit),
        }, use_post=True)
    except Exception as exc:
        print(f"  search '{query}' failed: {exc}", file=sys.stderr)
        return []
    return [hit["title"] for hit in data.get("query", {}).get("search", [])]


def fetch_imageinfo(titles: list) -> dict:
    data = api_get({
        "action": "query",
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "titles": "|".join(titles),
    }, use_post=True)
    return data.get("query", {}).get("pages", {})


def short_license(extmeta: dict) -> str:
    return extmeta.get("LicenseShortName", {}).get("value", "").strip()


def is_allowed(lic: str) -> bool:
    return bool(lic) and bool(ALLOWED_LICENSES.search(lic))


def extract_plate(text: str) -> str:
    norm = text.upper().translate(CYR_TO_LAT)
    matches = PLATE_RE.findall(norm)
    if not matches:
        return ""
    return max(matches, key=len)


def hashed_name(url: str, ext: str) -> str:
    return "commons_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12] + ext


def download(url: str, dest: Path) -> bool:
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=120) as resp:
            dest.write_bytes(resp.read())
        return True
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"    download failed: {exc}", file=sys.stderr)
        return False


def load_existing(meta_path: Path) -> set:
    if not meta_path.exists():
        return set()
    seen = set()
    with meta_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            src = row.get("source", "").strip()
            if src:
                seen.add(src)
    return seen


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    ap.add_argument("--limit", type=int, default=200,
                    help="max files per source")
    ap.add_argument("--target", default="dataset/images/real")
    ap.add_argument("--meta", default="dataset/meta.csv")
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="pause between downloads")
    ap.add_argument("--sleep-api", type=float, default=1.5,
                    help="pause between API requests")
    ap.add_argument("--batch", type=int, default=10,
                    help="titles per imageinfo request")
    ap.add_argument("--min-width", type=int, default=600)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)
    meta = Path(args.meta)
    meta.parent.mkdir(parents=True, exist_ok=True)

    seen = load_existing(meta)
    new_rows = []
    processed = set()

    def process_titles(label: str, titles: list) -> None:
        nonlocal new_rows
        for i in range(0, len(titles), args.batch):
            time.sleep(args.sleep_api)
            chunk = titles[i:i + args.batch]
            try:
                pages = fetch_imageinfo(chunk)
            except Exception as exc:
                print(f"  imageinfo failed for batch {i}-{i+len(chunk)}: {exc}",
                      file=sys.stderr)
                continue

            for page in pages.values():
                info = (page.get("imageinfo") or [{}])[0]
                extmeta = info.get("extmetadata", {})
                lic = short_license(extmeta)
                if not is_allowed(lic):
                    continue

                url = info.get("url")
                mime = info.get("mime", "")
                width = info.get("width", 0)
                if not url or mime not in MIME_EXT or url in processed:
                    continue
                if width and width < args.min_width:
                    continue

                page_title = page.get("title", "").replace(" ", "_")
                source_url = ("https://commons.wikimedia.org/wiki/"
                              + quote(page_title))
                if source_url in seen:
                    continue

                processed.add(url)
                name = hashed_name(url, MIME_EXT[mime])
                dest = target / name

                description = extmeta.get("ImageDescription", {}).get("value", "")
                plate = extract_plate(page_title) or extract_plate(description)
                tag = f"plate={plate}" if plate else "plate=?"
                print(f"  + {name}  [{lic}]  {tag}  via {label}")

                if args.dry_run:
                    continue
                if dest.exists():
                    continue
                if not download(url, dest):
                    continue

                new_rows.append([
                    f"images/real/{name}",
                    plate,
                    "type1",
                    "",
                    "",
                    1,
                    0,
                    source_url,
                    lic,
                    "",
                ])
                time.sleep(args.sleep)

    for cat in args.categories:
        print(f"[category] {cat}")
        titles = category_files(cat, args.limit, args.sleep_api)
        print(f"  {len(titles)} files")
        if titles:
            process_titles(f"cat:{cat}", titles)
        time.sleep(args.sleep_api)

    for q in args.queries:
        print(f"[search] {q}")
        titles = search_files(q, args.limit)
        print(f"  {len(titles)} files")
        if titles:
            process_titles(f"q:{q}", titles)
        time.sleep(args.sleep_api)

    if args.dry_run:
        print(f"\n[dry-run] would add {len(new_rows)} rows")
        return

    if not new_rows:
        print("\nnothing new to add")
        return

    write_header = (not meta.exists()) or meta.stat().st_size == 0
    with meta.open("a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        if write_header:
            w.writerow(META_HEADER)
        w.writerows(new_rows)

    n_auto = sum(1 for r in new_rows if r[1])
    print(f"\nadded {len(new_rows)} rows ({n_auto} with auto plate_num)")
    print(f"file: {meta}")
    print()
    print("MANUAL STEPS for each new row:")
    print("  1. verify plate_type (script guesses 'type1')")
    print("  2. fill bbox and quad")
    print("  3. verify plate_num (script fills from filename when possible)")
    print("  4. fill conditions (day/night/angle/...)")


if __name__ == "__main__":
    main()
