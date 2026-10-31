#!/usr/bin/env python3
"""BM25 (Okapi) over the uniform textual representation. Table 4, row 1.

Self-contained: no rank_bm25, no sklearn. BM25 is twenty lines and the
parameters matter more than the library, so they are visible here --
k1 = 1.2, b = 0.75, the standard Okapi setting, and the IDF variant is
Robertson's with the +0.5 smoothing floored at a small positive value so a
term occurring in more than half the corpus cannot contribute a NEGATIVE
score and push a genuinely matching document down the ranking.

Tokenisation is lowercase alphanumeric runs with a short English stoplist.
It is deliberately plain: a lexical baseline that has been tuned is no longer
a reference point for what the dense retrievers are worth.

Usage
-----
  python bm25_retriever.py --gold $Q --resources $R \\
      --output pools/bm25_top100.json --top-k 100
"""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

from retrieval_io import (load_corpus, query_text, report_texts,
                          write_candidates)

STOP = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "how", "i", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "their", "then", "there", "these", "this", "to", "was",
    "what", "when", "where", "which", "who", "why", "will", "with", "you",
}
TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [t for t in TOKEN.findall(str(text or "").lower())
            if t not in STOP and len(t) > 1]


class BM25:
    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.2,
                 b: float = 0.75, idf_floor: float = 1e-3):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.tf: List[Counter] = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=np.float64)
        self.avgdl = float(self.len.mean()) if self.N else 0.0
        df: Counter = Counter()
        for t in self.tf:
            df.update(t.keys())
        # Robertson IDF. The floor matters: without it a term in >50% of the
        # corpus gets a negative weight, so a document containing it scores
        # LOWER than one that does not -- which on a corpus of same-subject
        # material (where "chapter", "explains", "class" are near-universal)
        # inverts the ranking for exactly the common words a learner query is
        # full of.
        self.idf = {t: max(idf_floor,
                           math.log((self.N - n + 0.5) / (n + 0.5) + 1.0))
                    for t, n in df.items()}
        # Postings, so scoring a query touches only the documents that can
        # score above zero rather than all N.
        self.post: Dict[str, List[int]] = {}
        for i, t in enumerate(self.tf):
            for term in t:
                self.post.setdefault(term, []).append(i)

    def scores(self, query: Sequence[str]) -> np.ndarray:
        s = np.zeros(self.N, dtype=np.float64)
        norm = self.k1 * (1 - self.b + self.b * self.len /
                          max(self.avgdl, 1e-9))
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in self.post[term]:
                f = self.tf[i][term]
                s[i] += idf * f * (self.k1 + 1) / (f + norm[i])
        return s


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--resources", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--k1", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.75)
    ns = ap.parse_args()

    gold, resources = load_corpus(ns.gold, ns.resources)
    texts = report_texts(resources)
    docs = [tokenize(t) for t in texts]
    empty_docs = sum(1 for d in docs if not d)
    if empty_docs:
        print(f"  [WARN] {empty_docs} resources tokenise to nothing and can "
              f"never be retrieved by BM25.")
    print(f"vocabulary: {len({t for d in docs for t in d}):,} terms   "
          f"mean length: {np.mean([len(d) for d in docs]):.1f} tokens")

    bm = BM25(docs, ns.k1, ns.b)
    qtexts = [query_text(it) for it in gold]
    k = min(ns.top_k, len(resources))

    order, scores, n_zero = [], [], 0
    for i, qt in enumerate(qtexts):
        s = bm.scores(tokenize(qt))
        idx = np.argpartition(-s, k - 1)[:k] if k < len(s) else np.arange(len(s))
        # Stable secondary sort by index, so ties (very common in BM25, where
        # many documents score exactly 0) are broken the same way on every run
        # instead of by whatever argpartition happened to do.
        idx = idx[np.lexsort((idx, -s[idx]))]
        if s[idx[0]] <= 0:
            n_zero += 1
        order.append(idx.tolist())
        scores.append(s[idx].tolist())
        if (i + 1) % 50 == 0 or i + 1 == len(qtexts):
            print(f"  [{i + 1}/{len(qtexts)}]")

    if n_zero:
        print(f"  [WARN] {n_zero} queries matched NO term in any document; "
              f"their pool is an\n         arbitrary slice of the corpus and "
              f"every metric on those rows is noise.")

    write_candidates(ns.output, gold, qtexts, resources, order, scores, "bm25")
    print("Okapi BM25, k1=%.2f b=%.2f, over index_text()." % (ns.k1, ns.b))


if __name__ == "__main__":
    main()
