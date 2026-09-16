"""
analysis_core.py
=================
Standalone analysis engine for diachronic semantic analysis of the Founders
corpus. Separate from the Streamlit UI (app.py) — app.py should import from
this module so the tool and the paper can never diverge.

DESIGN: single model, time-sliced contexts
-------------------------------------------
This version uses ONE GPT-2-architecture model trained from scratch on the
Founders corpus (trained_tf_model_epoch_3) as a fixed measuring instrument.
The diachronic signal comes from the CORPUS, not from separate models:

  * For each term, we collect sentences from 1790s documents and, separately,
    from 1800s documents.
  * We embed the term IN CONTEXT — running each real sentence through the model
    and averaging the hidden states over the term's own tokens.
  * The 1790s embedding and the 1800s embedding therefore come from the SAME
    model, i.e. the same coordinate system. They are directly comparable, with
    NO alignment step required. The only thing that differs is the historical
    context the term sat in.

This avoids the two flaws of the original app: it is genuinely contextual
(not isolated-word lookups), and it has no cross-model alignment problem
(there is only one model).

WHAT YOU MUST CHECK BEFORE RUNNING
----------------------------------
  * CONFIG["model_dir"] can be a Hugging Face model id or a local checkpoint.
    Override it with FOUNDERS_MODEL_DIR if needed.
  * CONFIG["corpus_path"] points to data/data.jsonl by default. Override it
    with FOUNDERS_CORPUS_PATH if your corpus lives elsewhere.
  * extract_year() is heuristic — it parses the year out of the document text
    because data.jsonl has no date field. Spot-check it on a sample (see the
    diagnostics CSV the run writes out).

Runtime note: embedding is one forward pass per context sentence
(~terms x periods x max_contexts passes). On CPU this is minutes, not seconds.

Not executed here (model weights are not in this workspace) — run locally and
sanity-check.
"""

from __future__ import annotations
import csv
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from transformers import AutoTokenizer, TFGPT2LMHeadModel
import tensorflow as tf


# ============================================================
# CONFIG — every interpretive decision in one place
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    # The corpus-trained GPT-2-architecture model. This may be a Hugging Face
    # model id or a local folder with TensorFlow checkpoint files.
    "model_dir": os.environ.get("FOUNDERS_MODEL_DIR", "ladew222/founders-gpt2"),
    # If the model folder lacks tokenizer files, point this at one that has
    # tokenizer.json / vocab.json / merges.txt. Otherwise leave as None.
    "tokenizer_dir": None,

    "corpus_path": os.environ.get(
        "FOUNDERS_CORPUS_PATH",
        str(PROJECT_ROOT / "data" / "data.jsonl"),
    ),

    # Period label -> inclusive (start, end) year range.
    # Two eras with a wide gap: the Founding/Revolutionary decades vs the
    # Early National period. The ~1790s sit BETWEEN the two ranges and are
    # deliberately excluded, so the comparison is across a real ~40-year gap
    # (decade-adjacent comparisons showed no signal above within-period noise).
    "periods": {"founding": (1770, 1789), "early_national": (1800, 1819)},

    # The conceptual vocabulary for the demonstration through-line
    # (liberty / republic -> commerce / property).
    # NB: "republican" and "republic" are the period-native words and are
    # abundant in the corpus; "republicanism" (the -ism abstraction) is rare
    # and somewhat anachronistic for the 1790s-1800s. It is kept here so the
    # run still reports it, but the term_pairs below measure the concept
    # through "republican". Discuss "republicanism" as the historiographical
    # concept (Rodgers); measure it through the word the Founders used.
    "key_terms": [
        "liberty", "freedom", "republic", "republican", "republicanism",
        "democracy", "virtue", "commerce", "property", "capital", "trade",
        "wealth", "rights",
        # Promoted from landmark_terms: the analysis flagged these as genuine
        # changers, so they are objects of study, not stable reference points.
        "government", "state", "land",
        # Added for the slavery / founding debate (Morgan; Holton 2021).
        "slavery", "bondage", "servitude",
    ],
    # Term pairs the paper reports — anchored on the period-native "republican".
    "term_pairs": [
        ("liberty", "commerce"), ("liberty", "property"), ("liberty", "virtue"),
        ("republican", "commerce"), ("republican", "virtue"),
        ("republican", "democracy"),
        # Slavery / founding debate.
        ("liberty", "slavery"), ("property", "slavery"), ("rights", "slavery"),
        # Land-as-property, and the "myth of statelessness" debate.
        ("land", "property"), ("government", "state"),
    ],
    # Landmark words assumed relatively stable; used for the second-order check.
    # government, state, and land were removed: the analysis showed they moved,
    # so they fail the "stable anchor" assumption and are now key terms above.
    # The seven below all had low, non-significant drift in the first run.
    "landmark_terms": [
        "people", "country", "law", "nation", "public", "time", "war",
    ],
    "max_contexts_per_term": 100,  # sentences sampled per term per period
    "min_contexts_warn": 20,       # warn if a term has fewer than this
    "max_seq_length": 256,
    "sample_seed": 1729,           # reproducible full-corpus reservoir sample
    "outputs_dir": "./analysis_outputs",
}


# ============================================================
# STEP 1 — Corpus reading and date extraction
# ============================================================
_YEAR_RE = re.compile(r"\b(1[78]\d\d)\b")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def extract_year(content: str, head_chars: int = 800) -> int | None:
    """
    data.jsonl has no date field — the year is inside the document text
    (Founders Online datelines). Take the first plausible 1740-1830 year,
    preferring the document head where datelines sit. HEURISTIC: spot-check it.
    """
    for window in (content[:head_chars], content):
        for m in _YEAR_RE.finditer(window):
            y = int(m.group(1))
            if 1740 <= y <= 1830:
                return y
    return None


def period_of(year: int, periods: dict) -> str | None:
    for label, (lo, hi) in periods.items():
        if lo <= year <= hi:
            return label
    return None


def clean_text(content: str) -> str:
    """Collapse the heavy whitespace/newlines in the scraped content."""
    return re.sub(r"\s+", " ", content).strip()


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if len(s.split()) >= 4]


# ============================================================
# STEP 2 — Full-corpus context counts + reservoir sample
# ============================================================
def gather_contexts(corpus_path: str, terms: list[str], periods: dict,
                    max_per: int, seed: int = 1729) -> tuple[
                        dict[str, dict[str, list[str]]],
                        dict[str, dict[str, int]],
                        dict[str, int],
                    ]:
    """
    Full pass over data.jsonl. For every document: extract its year, map it to
    a period, count every matching sentence for each term, and keep a
    reproducible reservoir sample of up to `max_per` contexts per term-period.

    Returns:
      buckets[period][term] -> sampled sentences
      available_counts[period][term] -> all matching sentences found
      scan_stats -> document-level scan counts
    """
    rng = random.Random(seed)
    buckets = {p: {t: [] for t in terms} for p in periods}
    available_counts = {p: {t: 0 for t in terms} for p in periods}
    patterns = {t: re.compile(rf"\b{re.escape(t)}\b") for t in terms}

    n_docs = n_used = 0
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_docs += 1
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
            n_used += 1
            for sent in sentences(clean_text(rec["content"])):
                low = sent.lower()
                for t in terms:
                    if patterns[t].search(low):
                        available_counts[period][t] += 1
                        seen = available_counts[period][t]
                        bucket = buckets[period][t]
                        if len(bucket) < max_per:
                            bucket.append(sent)
                        else:
                            j = rng.randrange(seen)
                            if j < max_per:
                                bucket[j] = sent
    print(f"  scanned all {n_docs} documents; "
          f"{n_used} fell inside a target period")
    return buckets, available_counts, {"documents_scanned": n_docs,
                                       "target_period_documents": n_used}


def context_count_table(terms: list[str], labels: list[str],
                        buckets: dict[str, dict[str, list[str]]],
                        available_counts: dict[str, dict[str, int]]) -> pd.DataFrame:
    rows = []
    for term in terms:
        row = {"term": term}
        for label in labels:
            row[f"{label}_available_contexts"] = available_counts[label][term]
            row[f"{label}_sampled_contexts"] = len(buckets[label][term])
        rows.append(row)
    return pd.DataFrame(rows)


def write_context_samples(out: Path, buckets: dict[str, dict[str, list[str]]],
                          labels: list[str], terms: list[str]) -> None:
    with open(out / "context_samples.csv", "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["period", "term", "context_number", "sentence"])
        writer.writeheader()
        for period in labels:
            for term in terms:
                for i, sentence in enumerate(buckets[period][term], start=1):
                    writer.writerow({
                        "period": period,
                        "term": term,
                        "context_number": i,
                        "sentence": sentence,
                    })


# ============================================================
# STEP 3 — Contextual embedding (one shared model)
# ============================================================
def load_model(model_dir: str, tokenizer_dir: str | None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir or model_dir)
    model = TFGPT2LMHeadModel.from_pretrained(model_dir)
    return tokenizer, model


def _word_spans(text_lower: str, term: str) -> list[tuple[int, int]]:
    pat = re.compile(rf"\b{re.escape(term.lower())}\b")
    return [(m.start(), m.end()) for m in pat.finditer(text_lower)]


def contextual_token_vectors(tokenizer, model, term: str, contexts: list[str],
                             max_len: int) -> np.ndarray:
    """
    Return one contextual token vector per sampled sentence for `term`.
    """
    vectors = []
    for ctx in contexts:
        enc = tokenizer(ctx, return_offsets_mapping=True,
                        truncation=True, max_length=max_len)
        offsets = enc["offset_mapping"]
        spans = _word_spans(ctx.lower(), term)
        if not spans:
            continue
        tok_idx = [
            i for i, (ts, te) in enumerate(offsets)
            if te > ts and any(ts < ce and te > cs for cs, ce in spans)
        ]
        if not tok_idx:
            continue
        out = model(tf.constant([enc["input_ids"]]), output_hidden_states=True)
        last_hidden = out.hidden_states[-1][0].numpy()      # (seq_len, hidden)
        vectors.append(last_hidden[tok_idx].mean(axis=0))
    if not vectors:
        return np.empty((0, 0))
    return np.vstack(vectors)


def contextual_embedding(tokenizer, model, term: str, contexts: list[str],
                         max_len: int) -> np.ndarray | None:
    """
    Embed `term` in context. For each sentence: run it through the model, take
    the last hidden layer, average the hidden states over exactly the sub-tokens
    of the target term. Then average across all sentences.
    """
    vectors = contextual_token_vectors(tokenizer, model, term, contexts, max_len)
    if vectors.size == 0:
        return None
    return np.mean(vectors, axis=0)


def build_period_embeddings(tokenizer, model, buckets: dict, period: str,
                            terms: list[str], max_len: int,
                            min_warn: int,
                            usage_vectors: dict | None = None) -> dict[str, np.ndarray]:
    embeddings = {}
    for term in terms:
        contexts = buckets[period][term]
        if len(contexts) < min_warn:
            print(f"  [warn] '{term}' ({period}): only {len(contexts)} "
                  f"contexts — embedding may be unstable")
        vectors = contextual_token_vectors(tokenizer, model, term, contexts, max_len)
        if vectors.size == 0:
            print(f"  [skip] '{term}' ({period}): not found")
        else:
            embeddings[term] = np.mean(vectors, axis=0)
            if usage_vectors is not None:
                usage_vectors.setdefault(period, {})[term] = vectors
    return embeddings


# ============================================================
# STEP 4 — Comparisons (all valid: a single model = one space)
# ============================================================
def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def term_drift(emb_a: dict, emb_b: dict, terms: list[str]) -> pd.DataFrame:
    """
    Quick summary measure: how far each term's own averaged contextual vector
    moved between the two periods. Valid WITHOUT alignment because both vectors
    come from the same model. This is the weak "prototype" measure — APD in
    contextual_change_analysis.py is the measure to report. drift = 1 - cosine.
    """
    rows = [{"term": t, "drift": round(1 - _cos(emb_a[t], emb_b[t]), 4)}
            for t in terms if t in emb_a and t in emb_b]
    return pd.DataFrame(rows).sort_values("drift", ascending=False)


# NOTE: APD, within-period dispersion, normalized APD, JSD sense-clustering,
# and the APD permutation significance test all live in
# contextual_change_analysis.py, which reads the usage_embeddings cache this
# module writes. They are kept there (not here) so there is a single code path
# for the usage-level measures and no risk of two divergent APD numbers.


def within_period_similarity(embeddings: dict, terms: list[str]) -> pd.DataFrame:
    terms = [t for t in terms if t in embeddings]
    mat = np.array([[_cos(embeddings[a], embeddings[b]) for b in terms]
                    for a in terms])
    return pd.DataFrame(mat, index=terms, columns=terms)


def all_pair_relationships(emb_a: dict, emb_b: dict, label_a: str,
                           label_b: str, terms: list[str]) -> pd.DataFrame:
    """All unordered term-pair similarities and period-to-period changes."""
    terms = [t for t in terms if t in emb_a and t in emb_b]
    rows = []
    for i, t1 in enumerate(terms):
        for t2 in terms[i + 1:]:
            s_a = _cos(emb_a[t1], emb_a[t2])
            s_b = _cos(emb_b[t1], emb_b[t2])
            change = s_b - s_a
            rows.append({
                "term_1": t1,
                "term_2": t2,
                f"sim_{label_a}": round(s_a, 4),
                f"sim_{label_b}": round(s_b, 4),
                "change": round(change, 4),
                "abs_change": round(abs(change), 4),
            })
    return pd.DataFrame(rows).sort_values("abs_change", ascending=False)


def relationship_change_matrix(emb_a: dict, emb_b: dict,
                               terms: list[str]) -> pd.DataFrame:
    """Symmetric matrix of pairwise similarity change: period_b - period_a."""
    terms = [t for t in terms if t in emb_a and t in emb_b]
    mat = np.zeros((len(terms), len(terms)))
    for i, t1 in enumerate(terms):
        for j, t2 in enumerate(terms):
            mat[i, j] = _cos(emb_b[t1], emb_b[t2]) - _cos(emb_a[t1], emb_a[t2])
    return pd.DataFrame(mat, index=terms, columns=terms)


def nearest_neighbors_by_period(embeddings_by_period: dict, labels: list[str],
                                terms: list[str], top_n: int = 5) -> pd.DataFrame:
    """Nearest words for each term inside each period's semantic space."""
    rows = []
    for label in labels:
        emb = embeddings_by_period[label]
        usable = [t for t in terms if t in emb]
        for term in usable:
            sims = [
                (other, _cos(emb[term], emb[other]))
                for other in usable
                if other != term
            ]
            sims.sort(key=lambda item: item[1], reverse=True)
            for rank, (neighbor, sim) in enumerate(sims[:top_n], start=1):
                rows.append({
                    "period": label,
                    "term": term,
                    "rank": rank,
                    "neighbor": neighbor,
                    "similarity": round(sim, 4),
                })
    return pd.DataFrame(rows)


def save_embedding_cache(out: Path, embeddings_by_period: dict,
                         labels: list[str], terms: list[str]) -> None:
    """
    Save period-term vectors so later analyses can compare words without
    rerunning the transformer model.
    """
    arrays = {}
    manifest = {"periods": labels, "terms": {}}
    for label in labels:
        period_terms = [t for t in terms if t in embeddings_by_period[label]]
        manifest["terms"][label] = period_terms
        if period_terms:
            arrays[label] = np.vstack([embeddings_by_period[label][t]
                                       for t in period_terms])
        else:
            arrays[label] = np.empty((0, 0))
    np.savez_compressed(out / "embeddings.npz", **arrays)
    with open(out / "embeddings_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def save_usage_embedding_cache(out: Path, usage_vectors_by_period: dict,
                               labels: list[str], terms: list[str]) -> None:
    """
    Save per-context token vectors for APD, dispersion, and sense clustering.
    Keys in the NPZ file are stable manifest ids, not human-facing labels.
    """
    arrays = {}
    manifest = {"periods": labels, "terms": {}, "keys": {}}
    for label in labels:
        manifest["terms"][label] = []
        manifest["keys"][label] = {}
        for term in terms:
            vectors = usage_vectors_by_period.get(label, {}).get(term)
            if vectors is None or vectors.size == 0:
                continue
            key = f"{label}__{term}".replace(" ", "_")
            arrays[key] = vectors
            manifest["terms"][label].append(term)
            manifest["keys"][label][term] = key
    np.savez_compressed(out / "usage_embeddings.npz", **arrays)
    with open(out / "usage_embeddings_manifest.json", "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def load_embedding_cache(out: str | Path) -> tuple[dict[str, dict[str, np.ndarray]],
                                                   list[str], list[str]]:
    """Load vectors saved by save_embedding_cache()."""
    out = Path(out)
    with open(out / "embeddings_manifest.json", encoding="utf-8") as fh:
        manifest = json.load(fh)
    data = np.load(out / "embeddings.npz")
    embeddings = {}
    all_terms = []
    for label in manifest["periods"]:
        period_terms = manifest["terms"][label]
        all_terms.extend(period_terms)
        arr = data[label]
        embeddings[label] = {
            term: arr[i]
            for i, term in enumerate(period_terms)
        }
    return embeddings, manifest["periods"], list(dict.fromkeys(all_terms))


def load_usage_embedding_cache(out: str | Path) -> tuple[
        dict[str, dict[str, np.ndarray]], list[str], list[str]]:
    """Load per-context token vectors saved by save_usage_embedding_cache()."""
    out = Path(out)
    with open(out / "usage_embeddings_manifest.json", encoding="utf-8") as fh:
        manifest = json.load(fh)
    data = np.load(out / "usage_embeddings.npz")
    usage = {}
    all_terms = []
    for label in manifest["periods"]:
        usage[label] = {}
        for term in manifest["terms"][label]:
            all_terms.append(term)
            usage[label][term] = data[manifest["keys"][label][term]]
    return usage, manifest["periods"], list(dict.fromkeys(all_terms))


def write_reusable_outputs(out: Path, embeddings_by_period: dict,
                           labels: list[str], key_terms: list[str],
                           all_terms: list[str],
                           usage_vectors_by_period: dict | None = None) -> None:
    """
    Write cache and comprehensive relationship tables. These are the files to
    use for later exploration without rebuilding embeddings.
    """
    a, b = labels[0], labels[1]
    save_embedding_cache(out, embeddings_by_period, labels, all_terms)
    if usage_vectors_by_period is not None:
        save_usage_embedding_cache(out, usage_vectors_by_period, labels, all_terms)

    drift_all = term_drift(embeddings_by_period[a], embeddings_by_period[b],
                           all_terms)
    drift_all.to_csv(out / "semantic_drift_all_terms.csv", index=False)

    key_pairs = all_pair_relationships(
        embeddings_by_period[a], embeddings_by_period[b], a, b, key_terms)
    key_pairs.to_csv(out / "all_pair_relationships_key_terms.csv", index=False)

    full_pairs = all_pair_relationships(
        embeddings_by_period[a], embeddings_by_period[b], a, b, all_terms)
    full_pairs.to_csv(out / "all_pair_relationships_all_terms.csv", index=False)

    relationship_change_matrix(
        embeddings_by_period[a], embeddings_by_period[b], key_terms
    ).to_csv(out / "relationship_change_matrix_key_terms.csv")

    relationship_change_matrix(
        embeddings_by_period[a], embeddings_by_period[b], all_terms
    ).to_csv(out / "relationship_change_matrix_all_terms.csv")

    nearest_neighbors_by_period(
        embeddings_by_period, labels, key_terms
    ).to_csv(out / "nearest_neighbors_key_terms.csv", index=False)

    nearest_neighbors_by_period(
        embeddings_by_period, labels, all_terms
    ).to_csv(out / "nearest_neighbors_all_terms.csv", index=False)


def relational_comparison(emb_a: dict, emb_b: dict, label_a: str, label_b: str,
                          pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for t1, t2 in pairs:
        if all(t in emb_a and t in emb_b for t in (t1, t2)):
            s_a, s_b = _cos(emb_a[t1], emb_a[t2]), _cos(emb_b[t1], emb_b[t2])
            rows.append({"term_1": t1, "term_2": t2,
                         f"sim_{label_a}": round(s_a, 4),
                         f"sim_{label_b}": round(s_b, 4),
                         "change": round(s_b - s_a, 4)})
        else:
            rows.append({"term_1": t1, "term_2": t2,
                         f"sim_{label_a}": None, f"sim_{label_b}": None,
                         "change": None})
    return pd.DataFrame(rows)


def second_order_change(emb_a: dict, emb_b: dict, target_terms: list[str],
                        landmark_terms: list[str]) -> pd.DataFrame:
    """Robustness check: compare each term's similarity profile to stable
    landmark words across the two periods."""
    landmarks = [l for l in landmark_terms if l in emb_a and l in emb_b]
    if not landmarks:
        return pd.DataFrame(
            [{"term": t, "profile_change": None, "n_landmarks": 0}
             for t in target_terms if t in emb_a and t in emb_b]
        )
    rows = []
    for t in target_terms:
        if t not in emb_a or t not in emb_b:
            continue
        pa = np.array([_cos(emb_a[t], emb_a[l]) for l in landmarks])
        pb = np.array([_cos(emb_b[t], emb_b[l]) for l in landmarks])
        rows.append({"term": t,
                     "profile_change": round(1 - _cos(pa, pb), 4),
                     "n_landmarks": len(landmarks)})
    return pd.DataFrame(rows).sort_values("profile_change", ascending=False)


def cluster_linkage(sim_df: pd.DataFrame):
    """Ward-linkage clustering on a DISTANCE matrix (1 - cosine)."""
    dist = 1.0 - sim_df.values
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    return linkage(squareform(dist, checks=False), method="ward")


# ============================================================
# MAIN
# ============================================================
def main(max_contexts: int | None = None):
    # An optional command-line number overrides max_contexts_per_term, so a
    # quick smoke test ("python analysis_core.py 5") needs no file editing.
    cfg = CONFIG if max_contexts is None else {**CONFIG,
                                               "max_contexts_per_term": max_contexts}
    if max_contexts is not None:
        print(f"*** SMOKE TEST: max_contexts_per_term = {max_contexts} ***")
    out = Path(cfg["outputs_dir"])
    out.mkdir(exist_ok=True)
    labels = list(cfg["periods"])
    a, b = labels[0], labels[1]

    # Embeddings are needed for the key terms AND the landmark terms — the
    # second-order robustness check compares key terms against the landmarks.
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

    # Diagnostic: how many contexts each term got per period (report this in
    # the paper — it tells the reader how much text backs each embedding).
    counts = context_count_table(all_terms, labels, buckets, available_counts)
    counts.to_csv(out / "context_counts.csv", index=False)
    print(counts.to_string(index=False))
    pd.DataFrame([scan_stats]).to_csv(out / "scan_stats.csv", index=False)
    write_context_samples(out, buckets, labels, all_terms)

    print("\nLoading the model (once) and building contextual embeddings...")
    tokenizer, model = load_model(cfg["model_dir"], cfg["tokenizer_dir"])
    emb = {}
    usage_vectors = {}
    for p in labels:
        print(f"[{p}]")
        emb[p] = build_period_embeddings(tokenizer, model, buckets, p,
                                         all_terms, cfg["max_seq_length"],
                                         cfg["min_contexts_warn"],
                                         usage_vectors)

    print("\n== Semantic drift (quick prototype-vector summary) ==")
    print("Rough first look only. The measures to report — APD, normalized")
    print("APD, JSD, and the permutation significance test — are produced by")
    print("contextual_change_analysis.py from the usage_embeddings cache.")
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

    write_reusable_outputs(out, emb, labels, cfg["key_terms"], all_terms,
                           usage_vectors)

    print(f"\nDone. Tables in {out}/")
    print("Reusable cache: embeddings.npz, embeddings_manifest.json, "
          "usage_embeddings.npz, usage_embeddings_manifest.json, "
          "all_pair_relationships_*.csv, relationship_change_matrix_*.csv, "
          "nearest_neighbors_*.csv")
    print("Next: read the actual passages behind the largest drift values "
          "before interpreting — the method points you where to read.")


if __name__ == "__main__":
    import sys
    _override = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(_override)
