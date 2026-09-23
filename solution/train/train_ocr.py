"""Train CRNN on plate crops. Run from solution/ root."""
import argparse
import csv
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.ocr import ALPHABET, BLANK, CHAR2IDX, CRNN, NUM_CLASSES


class PlateDataset(Dataset):
    def __init__(self, meta_csv: str, root: str, img_size=(128, 32)):
        self.root = root
        self.img_size = img_size
        self.items = []
        with open(meta_csv, encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter=";"):
                if not row["plate_num"] or "#" in row["plate_num"]:
                    continue
                if row["plate_type"] == "other":
                    continue
                self.items.append(row)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r = self.items[i]
        path = os.path.join(self.root, r["image"])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"missing {path}")
        # Full-image crop is a lazy fallback; real training should use bbox/quad from meta.
        img = cv2.resize(img, self.img_size)
        x = img.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        text = r["plate_num"]
        y = torch.tensor([CHAR2IDX[c] for c in text], dtype=torch.long)
        return torch.from_numpy(x[None, ...]), y


def collate(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs)
    target_lengths = torch.tensor([len(y) for y in ys], dtype=torch.long)
    targets = torch.cat(ys)
    return xs, targets, target_lengths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="dataset/meta.csv")
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models/ocr.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = PlateDataset(args.meta, args.root)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=4, collate_fn=collate)
    model = CRNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    for ep in range(args.epochs):
        model.train()
        total = 0.0
        for x, targets, target_lengths in dl:
            x = x.to(device)
            logits = model(x)                     # B x T x C
            log_probs = logits.log_softmax(-1).permute(1, 0, 2)  # T x B x C
            T, B, _ = log_probs.shape
            input_lengths = torch.full((B,), T, dtype=torch.long)
            loss = ctc(log_probs, targets, input_lengths, target_lengths)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {ep+1} loss {total / len(dl):.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
