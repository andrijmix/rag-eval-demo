"""A dependency-free BM25 retriever used as the baseline for evaluation."""
import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        if not docs:
            raise ValueError("docs must not be empty")
        self.ids = list(docs)
        self.k1, self.b = k1, b
        tokens = {i: tokenize(t) for i, t in docs.items()}
        self.len = {i: len(t) for i, t in tokens.items()}
        self.avgdl = sum(self.len.values()) / len(self.len)
        self.tf = {i: Counter(t) for i, t in tokens.items()}
        df: Counter = Counter()
        for t in tokens.values():
            df.update(set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        q = tokenize(query)
        scored = []
        for i in self.ids:
            score = 0.0
            for w in q:
                f = self.tf[i].get(w, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avgdl)
                    score += self.idf.get(w, 0.0) * f * (self.k1 + 1) / norm
            scored.append((i, score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
