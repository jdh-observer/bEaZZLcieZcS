"""
analysis_from_cache.py
======================
Explore saved semantic-drift outputs without rebuilding embeddings.

Run the model scripts once:

    python3 analysis_core.py
    python3 analysis_core_modernbert.py

Then use this script for quick follow-up tables:

    python3 analysis_from_cache.py --outputs analysis_outputs
    python3 analysis_from_cache.py --outputs analysis_outputs --terms liberty freedom virtue commerce property
    python3 analysis_from_cache.py --compare analysis_outputs analysis_outputs_modernbert
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis_core import (
    all_pair_relationships,
    load_embedding_cache,
    nearest_neighbors_by_period,
    relationship_change_matrix,
    term_drift,
)


def summarize_output_dir(outputs: Path, terms: list[str] | None,
                         top_n: int) -> None:
    embeddings, labels, cached_terms = load_embedding_cache(outputs)
    selected_terms = terms or cached_terms
    selected_terms = [t for t in selected_terms if t in cached_terms]
    a, b = labels[0], labels[1]

    drift = term_drift(embeddings[a], embeddings[b], selected_terms)
    pairs = all_pair_relationships(embeddings[a], embeddings[b], a, b,
                                   selected_terms)
    neighbors = nearest_neighbors_by_period(embeddings, labels, selected_terms,
                                            top_n=top_n)
    matrix = relationship_change_matrix(embeddings[a], embeddings[b],
                                        selected_terms)

    suffix = "_custom" if terms else "_cached_terms"
    drift_path = outputs / f"cache_query_drift{suffix}.csv"
    pairs_path = outputs / f"cache_query_pair_changes{suffix}.csv"
    neighbors_path = outputs / f"cache_query_neighbors{suffix}.csv"
    matrix_path = outputs / f"cache_query_change_matrix{suffix}.csv"

    drift.to_csv(drift_path, index=False)
    pairs.to_csv(pairs_path, index=False)
    neighbors.to_csv(neighbors_path, index=False)
    matrix.to_csv(matrix_path)

    print(f"\nOutputs: {outputs}")
    print("\n== Drift ==")
    print(drift.head(top_n).to_string(index=False))
    print("\n== Strongest relationship changes ==")
    print(pairs.head(top_n).to_string(index=False))
    print("\nWrote:")
    print(f"  {drift_path}")
    print(f"  {pairs_path}")
    print(f"  {neighbors_path}")
    print(f"  {matrix_path}")


def compare_output_dirs(outputs_a: Path, outputs_b: Path) -> None:
    emb_a, labels_a, terms_a = load_embedding_cache(outputs_a)
    emb_b, labels_b, terms_b = load_embedding_cache(outputs_b)
    if labels_a != labels_b:
        raise ValueError("Cannot compare caches with different period labels.")

    p0, p1 = labels_a[0], labels_a[1]
    common_terms = [t for t in terms_a if t in terms_b]
    drift_a = term_drift(emb_a[p0], emb_a[p1], common_terms).rename(
        columns={"drift": f"{outputs_a.name}_drift"})
    drift_b = term_drift(emb_b[p0], emb_b[p1], common_terms).rename(
        columns={"drift": f"{outputs_b.name}_drift"})
    comparison = drift_a.merge(drift_b, on="term", how="inner")
    comparison[f"{outputs_a.name}_rank"] = comparison[
        f"{outputs_a.name}_drift"].rank(ascending=False, method="min").astype(int)
    comparison[f"{outputs_b.name}_rank"] = comparison[
        f"{outputs_b.name}_drift"].rank(ascending=False, method="min").astype(int)
    comparison = comparison.sort_values(
        [f"{outputs_a.name}_rank", f"{outputs_b.name}_rank"])

    out = Path(f"cache_model_comparison_{outputs_a.name}_vs_{outputs_b.name}.csv")
    comparison.to_csv(out, index=False)

    print("\n== Model/cache drift comparison ==")
    print(comparison.to_string(index=False))
    print(f"\nWrote: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="analysis_outputs",
                        help="Output directory containing embeddings.npz.")
    parser.add_argument("--terms", nargs="*",
                        help="Optional subset of terms to compare.")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--compare", nargs=2, metavar=("OUTPUTS_A", "OUTPUTS_B"),
                        help="Compare drift rankings across two cached runs.")
    args = parser.parse_args()

    if args.compare:
        compare_output_dirs(Path(args.compare[0]), Path(args.compare[1]))
    else:
        summarize_output_dir(Path(args.outputs), args.terms, args.top_n)


if __name__ == "__main__":
    main()
