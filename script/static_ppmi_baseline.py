"""
static_ppmi_baseline.py
=======================
Lightweight static distributional baseline for the transformer analysis.

This builds term-context PPMI vectors from the Founders corpus using a shared
context vocabulary. Because both periods use the same context-word dimensions,
cosine comparisons do not require embedding-space alignment.

It is not the paper's main method; it is a reviewer-friendly baseline beside
the corpus-adapted transformer outputs.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_core import (
    CONFIG,
    _cos,
    all_pair_relationships,
    clean_text,
    extract_year,
    period_of,
    relational_comparison,
    second_order_change,
    sentences,
    term_drift,
    within_period_similarity,
)


TOKEN_RE = re.compile(r"[a-z][a-z'-]*")
STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "have", "not",
    "are", "was", "were", "his", "her", "their", "our", "you", "your",
    "but", "all", "any", "can", "had", "has", "will", "would", "shall",
    "may", "one", "who", "which", "upon", "there", "been", "they",
    "them", "than", "then", "such", "into", "more", "when", "what",
    "where", "its", "out", "about", "after", "before", "over", "under",
}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower())
            if len(t) > 2 and t not in STOPWORDS]


def build_context_vocab(corpus_path: str, periods: dict, vocab_size: int) -> list[str]:
    counts = Counter()
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            year = extract_year(rec.get("content", ""))
            if year is None or period_of(year, periods) is None:
                continue
            counts.update(tokenize(clean_text(rec.get("content", ""))))
    return [word for word, _ in counts.most_common(vocab_size)]


def build_cooccurrence(corpus_path: str, periods: dict, terms: list[str],
                       vocab: list[str], window: int) -> tuple[dict, dict]:
    vocab_index = {w: i for i, w in enumerate(vocab)}
    term_patterns = {t: re.compile(rf"\b{re.escape(t)}\b") for t in terms}
    counts = {p: np.zeros((len(terms), len(vocab)), dtype=np.float64)
              for p in periods}
    available = {p: {t: 0 for t in terms} for p in periods}

    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            year = extract_year(rec.get("content", ""))
            if year is None:
                continue
            period = period_of(year, periods)
            if period is None:
                continue
            for sent in sentences(clean_text(rec.get("content", ""))):
                low = sent.lower()
                toks = tokenize(low)
                if not toks:
                    continue
                for term_i, term in enumerate(terms):
                    if not term_patterns[term].search(low):
                        continue
                    available[period][term] += 1
                    positions = [i for i, tok in enumerate(toks) if tok == term]
                    if not positions:
                        positions = list(range(len(toks)))
                    for pos in positions:
                        lo = max(0, pos - window)
                        hi = min(len(toks), pos + window + 1)
                        for ctx in toks[lo:hi]:
                            j = vocab_index.get(ctx)
                            if j is not None and ctx != term:
                                counts[period][term_i, j] += 1
    return counts, available


def ppmi(matrix: np.ndarray) -> np.ndarray:
    total = matrix.sum()
    if total == 0:
        return matrix
    row = matrix.sum(axis=1, keepdims=True)
    col = matrix.sum(axis=0, keepdims=True)
    expected = row @ col / total
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.log2((matrix * total) / expected)
    values[~np.isfinite(values)] = 0.0
    values[values < 0] = 0.0
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=5000)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--outputs", default="analysis_outputs_static_ppmi")
    args = parser.parse_args()

    cfg = CONFIG
    out = Path(args.outputs)
    out.mkdir(exist_ok=True)
    labels = list(cfg["periods"])
    terms = cfg["key_terms"] + [t for t in cfg["landmark_terms"]
                                if t not in cfg["key_terms"]]

    print("Building shared context vocabulary...")
    vocab = build_context_vocab(cfg["corpus_path"], cfg["periods"],
                                args.vocab_size)
    pd.DataFrame({"context_word": vocab}).to_csv(out / "context_vocab.csv",
                                                 index=False)

    print("Counting term-context co-occurrences...")
    counts, available = build_cooccurrence(cfg["corpus_path"], cfg["periods"],
                                           terms, vocab, args.window)

    embeddings = {}
    for label in labels:
        mat = ppmi(counts[label])
        embeddings[label] = {term: mat[i] for i, term in enumerate(terms)}
        pd.DataFrame(mat, index=terms, columns=vocab).to_csv(
            out / f"ppmi_vectors_{label}.csv")

    count_rows = []
    for term in terms:
        row = {"term": term}
        for label in labels:
            row[f"{label}_available_contexts"] = available[label][term]
        count_rows.append(row)
    pd.DataFrame(count_rows).to_csv(out / "context_counts.csv", index=False)

    a, b = labels[0], labels[1]
    term_drift(embeddings[a], embeddings[b], cfg["key_terms"]).to_csv(
        out / "semantic_drift.csv", index=False)
    relational_comparison(embeddings[a], embeddings[b], a, b,
                          cfg["term_pairs"]).to_csv(
        out / "relational_comparison.csv", index=False)
    second_order_change(embeddings[a], embeddings[b], cfg["key_terms"],
                        cfg["landmark_terms"]).to_csv(
        out / "second_order_change.csv", index=False)
    for label in labels:
        within_period_similarity(embeddings[label], cfg["key_terms"]).to_csv(
            out / f"similarity_within_{label}.csv")
    all_pair_relationships(embeddings[a], embeddings[b], a, b, terms).to_csv(
        out / "all_pair_relationships_all_terms.csv", index=False)

    print(f"Done. Tables in {out}/")


if __name__ == "__main__":
    main()
