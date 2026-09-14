"""
make_figures.py
===============
Generates the manuscript figures from the analysis output CSVs.

Run from the repository root or from the script directory. It reads analysis
outputs from the repository root by default and writes PNGs into ./media/.
Set ANALYSIS_OUTPUT_ROOT to point elsewhere if the analysis outputs live
outside the article repository.

    python3 make_figures.py

Every figure is built only from values the analysis actually produced; the
only hard-coded numbers are the per-decade document counts in Figure 1,
which come from the corpus scan reported in Section 3.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AOUT = Path(os.environ.get("ANALYSIS_OUTPUT_ROOT", str(ROOT)))
FIG = ROOT / "media"
FIG.mkdir(exist_ok=True)

LANDMARKS = {"people", "country", "law", "nation", "public", "time", "war"}
NAVY, RUST, GREEN, GREY, RED = "#1f4e79", "#a6611a", "#5aae61", "#bdbdbd", "#d7301f"


# ---- Figure 1: corpus composition by decade --------------------------------
decades = ["1740s", "1750s", "1760s", "1770s", "1780s", "1790s",
           "1800s", "1810s", "1820s", "1830s"]
counts = [115, 1999, 1135, 19900, 27317, 29416, 28809, 14564, 6967, 292]
founding, early = {"1770s", "1780s"}, {"1800s", "1810s"}
bar_colors = [NAVY if d in founding else RUST if d in early else GREY
              for d in decades]

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(decades, counts, color=bar_colors)
ax.set_ylabel("documents")
ax.set_title("Founders corpus: documents per decade (160,280 documents total)")
ax.legend(handles=[mpatches.Patch(color=NAVY, label="Founding era (1770-1789)"),
                   mpatches.Patch(color=RUST, label="Early National (1800-1819)"),
                   mpatches.Patch(color=GREY, label="outside the compared periods")],
          frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig1_corpus.png", dpi=150)
plt.close(fig)


# ---- Figure 2: APD permutation results, GPT-2 ------------------------------
g = pd.read_csv(AOUT / "analysis_outputs/usage_change_apd_dispersion_clusters.csv")
g = g.sort_values("apd_z").reset_index(drop=True)


def category(row):
    if row["term"] in LANDMARKS:
        return "landmark word"
    return "key term, p<=0.05" if row["apd_p_perm"] <= 0.05 else "key term, n.s."


cmap = {"key term, p<=0.05": NAVY, "key term, n.s.": "#9ecae1",
        "landmark word": "#888888"}
fig, ax = plt.subplots(figsize=(8, 7.5))
ax.barh(g["term"], g["apd_z"],
        color=[cmap[category(r)] for _, r in g.iterrows()])
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("APD permutation z-score (GPT-2)")
ax.set_title("Measured change per term against a permutation null")
ax.legend(handles=[mpatches.Patch(color=c, label=l) for l, c in cmap.items()],
          frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "fig2_apd_significance.png", dpi=150)
plt.close(fig)


# ---- Figure 3: do the three instruments agree on what changed? -------------
mb = pd.read_csv(
    AOUT / "analysis_outputs_modernbert/usage_change_apd_dispersion_clusters.csv")
pp = pd.read_csv(AOUT / "analysis_outputs_static_ppmi/semantic_drift.csv")
key_terms = list(pp["term"])

scores = pd.DataFrame({
    "GPT-2": g.set_index("term")["apd_z"].reindex(key_terms),
    "ModernBERT": mb.set_index("term")["apd_z"].reindex(key_terms),
    "PPMI": pp.set_index("term")["drift"].reindex(key_terms),
})
ranks = scores.rank(ascending=False)
cols = ["GPT-2", "ModernBERT", "PPMI"]
highlight = {"republican": RED, "government": NAVY, "state": GREEN}

fig, ax = plt.subplots(figsize=(7, 7.5))
for term in key_terms:
    ys = [ranks.loc[term, c] for c in cols]
    hl = term in highlight
    ax.plot(range(3), ys, marker="o", markersize=6 if hl else 4,
            color=highlight.get(term, "#cfcfcf"),
            lw=2.4 if hl else 0.8, zorder=3 if hl else 1)
    if hl:
        ax.text(2.05, ranks.loc[term, "PPMI"], term, va="center",
                fontsize=9, color=highlight[term])
ax.set_xticks(range(3))
ax.set_xticklabels(cols)
ax.set_ylabel("rank by measured change  (1 = changed most)")
ax.invert_yaxis()
ax.set_title("Cross-instrument agreement on the ranking of change")
fig.tight_layout()
fig.savefig(FIG / "fig3_cross_instrument.png", dpi=150)
plt.close(fig)


# ---- Figure 4: relational change, GPT-2 vs PPMI ----------------------------
gr = pd.read_csv(AOUT / "analysis_outputs/relational_comparison.csv")
pr = pd.read_csv(AOUT / "analysis_outputs_static_ppmi/relational_comparison.csv")
for df in (gr, pr):
    df["pair"] = df["term_1"] + "-" + df["term_2"]
merged = gr[["pair", "change"]].merge(
    pr[["pair", "change"]], on="pair", suffixes=("_gpt2", "_ppmi"))

y = np.arange(len(merged))
h = 0.38
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(y + h / 2, merged["change_gpt2"], h, color=NAVY,
        label="GPT-2 (contextual embeddings)")
ax.barh(y - h / 2, merged["change_ppmi"], h, color=RUST,
        label="PPMI (co-occurrence counts)")
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels(merged["pair"])
ax.set_xlabel("change in similarity, Founding era -> Early National")
ax.set_title("Relational change: where the two instruments disagree")
ax.legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "fig4_relational_change.png", dpi=150)
plt.close(fig)

# ---- Figure 5: shared-space semantic relationship map ----------------------
# One PCA fit across ALL term vectors from BOTH periods (legitimate here: a
# single model means a single coordinate system). Vectors are L2-normalized so
# Euclidean distance in the map approximates cosine distance, and mean-centered
# so the projection is not dominated by the anisotropy axis. Both panels use
# the same projection, so positions are directly comparable.
manifest = json.loads(
    (AOUT / "analysis_outputs/embeddings_manifest.json").read_text())
npz = np.load(AOUT / "analysis_outputs/embeddings.npz")
periods = manifest["periods"]
emb = {p: {t: npz[p][i] for i, t in enumerate(manifest["terms"][p])}
       for p in periods}

groups = {
    "liberty & rights": (["liberty", "freedom", "rights", "virtue"], NAVY),
    "government & republic": (["government", "state", "republic", "republican",
                               "republicanism", "democracy"], GREEN),
    "political economy": (["commerce", "property", "capital", "trade",
                           "wealth", "land"], RUST),
    "slavery": (["slavery", "bondage", "servitude"], RED),
}
term_color, term_order = {}, []
for label, (ts, c) in groups.items():
    for t in ts:
        if t in emb[periods[0]] and t in emb[periods[1]]:
            term_color[t] = c
            term_order.append(t)

X = np.vstack([emb[p][t] for p in periods for t in term_order]).astype(float)
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
Xc = X - X.mean(axis=0)
_, _, Vt = np.linalg.svd(Xc, full_matrices=False)
coords = Xc @ Vt[:2].T
n = len(term_order)
pos = {periods[0]: coords[:n], periods[1]: coords[n:]}

fig, axes = plt.subplots(1, 2, figsize=(13, 6.3), sharex=True, sharey=True)
titles = {"founding": "Founding era (1770-1789)",
          "early_national": "Early National (1800-1819)"}
for ax, p in zip(axes, periods):
    P = pos[p]
    for i, t in enumerate(term_order):
        ax.scatter(P[i, 0], P[i, 1], color=term_color[t], s=42, zorder=3)
        ax.annotate(t, (P[i, 0], P[i, 1]), xytext=(5, 0),
                    textcoords="offset points", fontsize=8, va="center")
    ax.axhline(0, color="#e3e3e3", lw=0.6)
    ax.axvline(0, color="#e3e3e3", lw=0.6)
    ax.set_title(titles.get(p, p))
    ax.set_xlabel("component 1")
axes[0].set_ylabel("component 2")
axes[1].legend(handles=[mpatches.Patch(color=c, label=l)
                        for l, (ts, c) in groups.items()],
               frameon=False, fontsize=8, loc="best")
fig.suptitle("Semantic relationship map: key terms in one shared coordinate "
             "space (GPT-2)")
fig.tight_layout()
fig.savefig(FIG / "fig5_relationship_map.png", dpi=150)
plt.close(fig)

print("Wrote 5 figures to", FIG.resolve())
for f in sorted(FIG.glob("*.png")):
    print(" ", f.name)
