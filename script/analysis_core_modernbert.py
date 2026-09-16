"""
analysis_core_modernbert.py
===========================
Bidirectional encoder baseline for the Founders semantic-drift analysis.

This script preserves the corpus sampling and reporting logic from
analysis_core.py, but swaps the TensorFlow GPT-2 measuring instrument for a
PyTorch encoder model running on Apple Silicon MPS when available.

Use this as a robustness check, not a replacement for the corpus-trained
GPT-2-architecture analysis:

    python3 analysis_core_modernbert.py 5
    python3 analysis_core_modernbert.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from analysis_core import (
    CONFIG as GPT2_CONFIG,
    _cos,
    _word_spans,
    context_count_table,
    gather_contexts,
    relational_comparison,
    second_order_change,
    term_drift,
    within_period_similarity,
    write_context_samples,
    write_reusable_outputs,
)


CONFIG = {
    **GPT2_CONFIG,
    "model_name": "answerdotai/ModernBERT-base",
    "max_seq_length": 512,
    "outputs_dir": "./analysis_outputs_modernbert",
}


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_encoder(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def contextual_token_vectors(tokenizer, model, device: torch.device, term: str,
                             contexts: list[str], max_len: int) -> np.ndarray:
    """
    Return one contextual token vector per sampled sentence for `term`.
    """
    vectors = []
    for ctx in contexts:
        enc = tokenizer(
            ctx,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        spans = _word_spans(ctx.lower(), term)
        if not spans:
            continue
        tok_idx = [
            i for i, (ts, te) in enumerate(offsets)
            if te > ts and any(ts < ce and te > cs for cs, ce in spans)
        ]
        if not tok_idx:
            continue

        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        last_hidden = out.last_hidden_state[0].detach().cpu().numpy()
        vectors.append(last_hidden[tok_idx].mean(axis=0))

    if not vectors:
        return np.empty((0, 0))
    return np.vstack(vectors)


def contextual_embedding(tokenizer, model, device: torch.device, term: str,
                         contexts: list[str], max_len: int) -> np.ndarray | None:
    """
    Embed `term` in context with a bidirectional encoder. For each sentence,
    locate the target word's sub-tokens via tokenizer offsets, average their
    final-layer states, then average across sampled contexts.
    """
    vectors = contextual_token_vectors(tokenizer, model, device, term,
                                       contexts, max_len)
    if vectors.size == 0:
        return None
    return np.mean(vectors, axis=0)


def build_period_embeddings(tokenizer, model, device: torch.device,
                            buckets: dict, period: str, terms: list[str],
                            max_len: int, min_warn: int,
                            usage_vectors: dict | None = None) -> dict[str, np.ndarray]:
    embeddings = {}
    for term in terms:
        contexts = buckets[period][term]
        if len(contexts) < min_warn:
            print(f"  [warn] '{term}' ({period}): only {len(contexts)} "
                  f"contexts - embedding may be unstable")
        vectors = contextual_token_vectors(tokenizer, model, device, term,
                                           contexts, max_len)
        if vectors.size == 0:
            print(f"  [skip] '{term}' ({period}): not found")
        else:
            embeddings[term] = np.mean(vectors, axis=0)
            if usage_vectors is not None:
                usage_vectors.setdefault(period, {})[term] = vectors
    return embeddings


def main(max_contexts: int | None = None):
    cfg = CONFIG if max_contexts is None else {**CONFIG,
                                               "max_contexts_per_term": max_contexts}
    if max_contexts is not None:
        print(f"*** MODERNBERT SMOKE TEST: max_contexts_per_term = {max_contexts} ***")

    out = Path(cfg["outputs_dir"])
    out.mkdir(exist_ok=True)
    labels = list(cfg["periods"])
    a, b = labels[0], labels[1]
    all_terms = cfg["key_terms"] + [l for l in cfg["landmark_terms"]
                                    if l not in cfg["key_terms"]]

    print("Scanning the full corpus and sampling term contexts...")
    buckets, available_counts, scan_stats = gather_contexts(
        cfg["corpus_path"],
        all_terms,
        cfg["periods"],
        cfg["max_contexts_per_term"],
        cfg["sample_seed"],
    )

    counts = context_count_table(all_terms, labels, buckets, available_counts)
    counts.to_csv(out / "context_counts.csv", index=False)
    print(counts.to_string(index=False))
    pd.DataFrame([scan_stats]).to_csv(out / "scan_stats.csv", index=False)
    write_context_samples(out, buckets, labels, all_terms)

    device = best_device()
    print(f"\nLoading {cfg['model_name']} on {device}...")
    tokenizer, model = load_encoder(cfg["model_name"], device)

    emb = {}
    usage_vectors = {}
    for p in labels:
        print(f"[{p}]")
        emb[p] = build_period_embeddings(tokenizer, model, device, buckets, p,
                                         all_terms, cfg["max_seq_length"],
                                         cfg["min_contexts_warn"],
                                         usage_vectors)

    print("\n== Semantic drift (ModernBERT baseline) ==")
    drift = term_drift(emb[a], emb[b], cfg["key_terms"])
    drift.to_csv(out / "semantic_drift.csv", index=False)
    print(drift.to_string(index=False))

    print("\n== Relational comparison of key pairs ==")
    rel = relational_comparison(emb[a], emb[b], a, b, cfg["term_pairs"])
    rel.to_csv(out / "relational_comparison.csv", index=False)
    print(rel.to_string(index=False))

    print("\n== Second-order change (robustness check) ==")
    so = second_order_change(emb[a], emb[b], cfg["key_terms"],
                             cfg["landmark_terms"])
    so.to_csv(out / "second_order_change.csv", index=False)
    print(so.to_string(index=False))

    for p in labels:
        within_period_similarity(emb[p], cfg["key_terms"]).to_csv(
            out / f"similarity_within_{p}.csv")

    print(f"\nDone. Tables in {out}/")
    write_reusable_outputs(out, emb, labels, cfg["key_terms"], all_terms,
                           usage_vectors)
    print("Reusable cache: embeddings.npz, embeddings_manifest.json, "
          "usage_embeddings.npz, usage_embeddings_manifest.json, "
          "all_pair_relationships_*.csv, relationship_change_matrix_*.csv, "
          "nearest_neighbors_*.csv")


if __name__ == "__main__":
    import sys
    _override = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(_override)
