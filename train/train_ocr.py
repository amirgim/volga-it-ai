"""Train CRNN (CTC) on plate crops. Validation = exact-match on held-out PLATES."""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.ocr import BLANK, CHAR2IDX, CRNN, IDX2CHAR, NUM_CLASSES
from src.unfold import OCR_H, OCR_W


def plate_of(path: Path) -> str:
    s = path.stem.split("_")[0].upper()
    return "".join(c for c in s if c in CHAR2IDX)


def split_files(crops_dir, val_frac=0.1, seed=42):
    by = {}
    for f in sorted(Path(crops_dir).glob("*.png")):
        pl = plate_of(f)
        if len(pl) in (8, 9):
            by.setdefault(pl, []).append(f)
    plates = sorted(by)
    random.Random(seed).shuffle(plates)
    nv = max(1, int(len(plates) * val_frac))
    val_plates = plates[:nv]
    train = [f for p in plates[nv:] for f in by[p]]
    val = [f for p in val_plates for f in by[p]]
    return train, val


def augment(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    if random.random() < 0.7:  # loose/tight box jitter
        dx, dy = int(w * 0.04), int(h * 0.12)
        img = img[random.randint(0, dy):h - random.randint(0, dy),
                  random.randint(0, dx):w - random.randint(0, dx)]
    img = cv2.resize(img, (OCR_W, OCR_H), interpolation=cv2.INTER_LINEAR)
    if random.random() < 0.5:  # low resolution
        s = random.uniform(0.35, 0.8)
        small = cv2.resize(img, (max(8, int(OCR_W * s)), max(4, int(OCR_H * s))))
        img = cv2.resize(small, (OCR_W, OCR_H), interpolation=cv2.INTER_LINEAR)
    if random.random() < 0.4:  # blur / motion blur
        k = random.choice([3, 5])
        if random.random() < 0.5:
            img = cv2.GaussianBlur(img, (k, k), 0)
        else:
            ker = np.zeros((k, k), np.float32)
            ker[k // 2, :] = 1.0 / k
            img = cv2.filter2D(img, -1, ker)
    f = img.astype(np.float32)
    if random.random() < 0.7:  # contrast/brightness/gamma
        f = f * random.uniform(0.5, 1.3) + random.uniform(-40, 40)
        f = np.clip(f, 0, 255)
        g = random.uniform(0.7, 1.5)
        f = 255.0 * (f / 255.0) ** g
    if random.random() < 0.5:
        f = f + np.random.normal(0, random.uniform(3, 15), f.shape)
    img = np.clip(f, 0, 255).astype(np.uint8)
    if random.random() < 0.4:  # jpeg
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(25, 80)])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return img


class PlateDataset(Dataset):
    def __init__(self, files, train):
        self.files, self.train = files, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        plate = plate_of(path)
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.full((OCR_H, OCR_W), 128, np.uint8)
        if self.train:
            img = augment(img)
        else:
            img = cv2.resize(img, (OCR_W, OCR_H), interpolation=cv2.INTER_LINEAR)
        x = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        target = torch.tensor([CHAR2IDX[c] for c in plate], dtype=torch.long)
        return torch.from_numpy(x).unsqueeze(0), target, plate


def collate(batch):
    imgs, targets, texts = zip(*batch)
    lens = torch.tensor([len(t) for t in targets], dtype=torch.long)
    return torch.stack(imgs, 0), torch.cat(targets), lens, texts


def decode_batch(logits: torch.Tensor):
    idx = logits.argmax(-1).cpu().numpy()
    out = []
    for row in idx:
        s, prev = [], -1
        for i in row:
            if i != prev and i != BLANK:
                s.append(IDX2CHAR[int(i)])
            prev = i
        out.append("".join(s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="dataset/crops")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="models/crnn.onnx")
    args = ap.parse_args()

    if not Path(args.crops).exists():
        raise SystemExit(f"no crops in {args.crops}. Run scripts/extract_crops.py first.")
    Path("models").mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_f, val_f = split_files(args.crops)
    print(f"device={device} train={len(train_f)} val={len(val_f)}")
    train_dl = DataLoader(PlateDataset(train_f, True), batch_size=args.batch, shuffle=True,
                          collate_fn=collate, num_workers=args.workers, drop_last=True)
    val_dl = DataLoader(PlateDataset(val_f, False), batch_size=args.batch, shuffle=False,
                        collate_fn=collate, num_workers=args.workers)

    model = CRNN(num_classes=NUM_CLASSES).to(device)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3, steps_per_epoch=len(train_dl), epochs=args.epochs)

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tl = 0.0
        for imgs, targets, tlens, _ in train_dl:
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)                      # B x T x C
            lp = logits.permute(1, 0, 2).log_softmax(2)
            ilens = torch.full((imgs.size(0),), lp.size(0), dtype=torch.long)
            loss = ctc(lp, targets, ilens, tlens)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tl += loss.item()
        model.eval()
        ok = n = 0
        with torch.no_grad():
            for imgs, _, _, texts in val_dl:
                preds = decode_batch(model(imgs.to(device)))
                ok += sum(p == t for p, t in zip(preds, texts))
                n += len(texts)
        acc = ok / max(1, n)
        print(f"Epoch {ep:02d}/{args.epochs} | loss {tl / len(train_dl):.4f} | val exact-match {acc:.3f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), "models/crnn.pt")
    print(f"best val exact-match: {best:.3f}")

    model.load_state_dict(torch.load("models/crnn.pt", map_location="cpu"))
    model.cpu().eval()
    dummy = torch.randn(1, 1, OCR_H, OCR_W)
    torch.onnx.export(model, dummy, args.out, opset_version=17,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
    print(f"Done. Saved {args.out} and models/crnn.pt")


if __name__ == "__main__":
    main()
