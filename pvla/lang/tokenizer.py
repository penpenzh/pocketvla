"""Simple word-level tokenizer (vocabulary source for the HF PocketVLATokenizer)."""
from __future__ import annotations

import numpy as np

COLORS = ("red", "green", "blue")
TEMPLATES = (
    "pick the {c} ball and put it in the basket",
    "put the {c} ball into the bin",
    "grab the {c} ball and place it in the box",
    "take the {c} ball to the basket",
    "fetch the {c} ball and drop it into the bin",
)


def make_instruction(color: str, template_id: int) -> str:
    return TEMPLATES[template_id].format(c=color)


class Tokenizer:
    """Whitespace splitting with a fixed small vocabulary, <pad>=0."""

    def __init__(self):
        words: dict[str, int] = {"<pad>": 0}
        for tid in range(len(TEMPLATES)):
            for c in COLORS:
                for w in make_instruction(c, tid).split():
                    if w not in words:
                        words[w] = len(words)
        words["<unk>"] = len(words)
        self.vocab = words
        self.inv = {v: k for k, v in words.items()}
        self.pad_id = 0
        self.unk_id = words["<unk>"]

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(w, self.unk_id) for w in text.lower().split()]

    def pad(self, ids: list[int], length: int) -> np.ndarray:
        a = np.zeros(length, dtype=np.int64)
        n = min(len(ids), length)
        a[:n] = ids[:n]
        return a

    def encode_padded(self, text: str, length: int) -> np.ndarray:
        return self.pad(self.encode(text), length)

    def decode(self, tokens) -> str:
        toks = [int(t) for t in tokens]
        return " ".join(self.inv.get(t, "?") for t in toks if t != self.pad_id)


TOKENIZER = Tokenizer()
VOCAB_SIZE = len(TOKENIZER)
