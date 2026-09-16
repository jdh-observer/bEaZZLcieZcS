"""
relationship_analysis.py
========================
Relationship-first analysis of cached transformer embeddings.

This is the layer for the article's larger argument: not just whether single
words drift, but how a corpus-trained/fine-tuned transformer lets us inspect
semantic relationships among Founding-era political concepts.

It reads embeddings.npz, so it does not rerun a model.

Examples:

    python3 relationship_analysis.py --outputs analysis_outputs
    python3 relationship_analysis.py --outputs analysis_outputs_modernbert
    python3 relationship_analysis.py --outputs analysis_outputs --top-n 8 --edge-threshold 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_core import _cos, all_pair_relationships, load_embedding_cache


def similarity_table(embeddings: dict[str, np.ndarray], terms: list[str]) -> pd.DataFrame:
    usable = [t for t in terms if t in embeddings]
    rows = []
    for i, t1 in enumerate(usable):
        for t2 in usable[i + 1:]:
            rows.append({
                "term_1": t1,
                "term_2": t2,
                "similarity": round(_cos(embeddings[t1], embeddings[t2]), 4),
            })
    return pd.DataFrame(rows).sort_values("similarity", ascending=False)


def centrality_table(embeddings_by_period: dict, labels: list[str],
                     terms: list[str], edge_threshold: float) -> pd.DataFrame:
    rows = []
    for label in labels:
        emb = embeddings_by_period[label]
        usable = [t for t in terms if t in emb]
        for term in usable:
            sims = [
                _cos(emb[term], emb[other])
                for other in usable
                if other != term
            ]
            above = [s for s in sims if s >= edge_threshold]
            rows.append({
                "period": label,
                "term": term,
                "mean_similarity_to_all_terms": round(float(np.mean(sims)), 4),
                "median_similarity_to_all_terms": round(float(np.median(sims)), 4),
                "max_similarity_to_any_term": round(float(np.max(sims)), 4),
                "weighted_degree_above_threshold": round(float(np.sum(above)), 4),
                "degree_above_threshold": len(above),
            })
    return pd.DataFrame(rows)


def centrality_change(centrality: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    a, b = labels[0], labels[1]
    left = centrality[centrality["period"] == a].drop(columns=["period"])
    right = centrality[centrality["period"] == b].drop(columns=["period"])
    merged = left.merge(right, on="term", suffixes=(f"_{a}", f"_{b}"))
    for col in [
        "mean_similarity_to_all_terms",
        "median_similarity_to_all_terms",
        "max_similarity_to_any_term",
        "weighted_degree_above_threshold",
        "degree_above_threshold",
    ]:
        merged[f"{col}_change"] = (
            merged[f"{col}_{b}"] - merged[f"{col}_{a}"]
        ).round(4)
    return merged.sort_values("mean_similarity_to_all_terms_change",
                              ascending=False)


def neighbor_list(embeddings: dict[str, np.ndarray], terms: list[str],
                  term: str, top_n: int) -> list[str]:
    sims = [
        (other, _cos(embeddings[term], embeddings[other]))
        for other in terms
        if other != term and other in embeddings
    ]
    sims.sort(key=lambda item: item[1], reverse=True)
    return [other for other, _ in sims[:top_n]]


def neighborhood_change(embeddings_by_period: dict, labels: list[str],
                        terms: list[str], top_n: int) -> pd.DataFrame:
    a, b = labels[0], labels[1]
    rows = []
    usable = [t for t in terms if t in embeddings_by_period[a]
              and t in embeddings_by_period[b]]
    for term in usable:
        neighbors_a = neighbor_list(embeddings_by_period[a], usable, term, top_n)
        neighbors_b = neighbor_list(embeddings_by_period[b], usable, term, top_n)
        set_a, set_b = set(neighbors_a), set(neighbors_b)
        rows.append({
            "term": term,
            f"neighbors_{a}": json.dumps(neighbors_a),
            f"neighbors_{b}": json.dumps(neighbors_b),
            "retained_neighbors": json.dumps(sorted(set_a & set_b)),
            "lost_neighbors": json.dumps([n for n in neighbors_a if n not in set_b]),
            "gained_neighbors": json.dumps([n for n in neighbors_b if n not in set_a]),
            "neighbor_retention_rate": round(len(set_a & set_b) / top_n, 4),
        })
    return pd.DataFrame(rows).sort_values("neighbor_retention_rate")


def term_relationship_profiles(embeddings_by_period: dict, labels: list[str],
                               terms: list[str]) -> pd.DataFrame:
    a, b = labels[0], labels[1]
    usable = [t for t in terms if t in embeddings_by_period[a]
              and t in embeddings_by_period[b]]
    rows = []
    for focal in usable:
        for other in usable:
            if focal == other:
                continue
            sim_a = _cos(embeddings_by_period[a][focal],
                         embeddings_by_period[a][other])
            sim_b = _cos(embeddings_by_period[b][focal],
                         embeddings_by_period[b][other])
            rows.append({
                "focal_term": focal,
                "related_term": other,
                f"sim_{a}": round(sim_a, 4),
                f"sim_{b}": round(sim_b, 4),
                "change": round(sim_b - sim_a, 4),
                "abs_change": round(abs(sim_b - sim_a), 4),
            })
    return pd.DataFrame(rows).sort_values(["focal_term", "abs_change"],
                                          ascending=[True, False])


def run(outputs: Path, top_n: int, edge_threshold: float) -> None:
    embeddings_by_period, labels, terms = load_embedding_cache(outputs)
    a, b = labels[0], labels[1]

    for label in labels:
        similarity_table(embeddings_by_period[label], terms).to_csv(
            outputs / f"relationship_edges_{label}.csv", index=False)

    all_pair_relationships(
        embeddings_by_period[a], embeddings_by_period[b], a, b, terms
    ).to_csv(outputs / "relationship_edge_changes.csv", index=False)

    centrality = centrality_table(
        embeddings_by_period, labels, terms, edge_threshold)
    centrality.to_csv(outputs / "relationship_centrality_by_period.csv",
                      index=False)

    centrality_change(centrality, labels).to_csv(
        outputs / "relationship_centrality_change.csv", index=False)

    neighborhood_change(embeddings_by_period, labels, terms, top_n).to_csv(
        outputs / "relationship_neighborhood_change.csv", index=False)

    term_relationship_profiles(embeddings_by_period, labels, terms).to_csv(
        outputs / "relationship_profiles_by_term.csv", index=False)

    changes = pd.read_csv(outputs / "relationship_edge_changes.csv")
    print(f"\nOutputs: {outputs}")
    print("\n== Strongest relationship increases ==")
    print(changes.sort_values("change", ascending=False).head(top_n).to_string(index=False))
    print("\n== Strongest relationship decreases ==")
    print(changes.sort_values("change", ascending=True).head(top_n).to_string(index=False))
    print("\nWrote relationship_* CSVs.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="analysis_outputs")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    args = parser.parse_args()
    run(Path(args.outputs), args.top_n, args.edge_threshold)


if __name__ == "__main__":
    main()
