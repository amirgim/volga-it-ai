from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

ALPHABET = "0123456789ABEKMHOPCTYX"
CHAR2IDX = {c: i for i, c in enumerate(ALPHABET)}
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}
NUM_CLASSES = len(ALPHABET) + 1  # + CTC blank
BLANK = len(ALPHABET)


class CRNN(nn.Module):
    """Small CRNN for single-line plate OCR. Input: B x 1 x 32 x 128."""

    def __init__(self, num_classes: int = NUM_CLASSES, hidden: int = 128):
        super().__init__()

        def block(ci: int, co: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, 1, 1),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )

        self.cnn = nn.Sequential(
            block(1, 32),     # 32 x 16 x 64
            block(32, 64),    # 64 x 8 x 32
            block(64, 128),   # 128 x 4 x 16
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 128 x 2 x 16
        )
        self.rnn = nn.LSTM(128 * 2, hidden, num_layers=2,
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)                       # B x 128 x 2 x 16
        b, c, h, w = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, w, c * h)
        out, _ = self.rnn(f)
        return self.fc(out)                   # B x T x C


def greedy_decode(logits: np.ndarray) -> Tuple[str, List[float]]:
    """logits: T x C -> (text, per-char confidence)."""
    x = torch.from_numpy(logits) if isinstance(logits, np.ndarray) else logits
    probs = x.softmax(-1).cpu().numpy()
    idx = probs.argmax(-1)
    chars: List[str] = []
    confs: List[float] = []
    prev = -1
    for t, i in enumerate(idx.tolist()):
        if i != prev and i != BLANK:
            chars.append(IDX2CHAR.get(i, ""))
            confs.append(float(probs[t, i]))
        prev = i
    return "".join(chars), confs
