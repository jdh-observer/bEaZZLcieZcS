#!/usr/bin/env python3
from pathlib import Path
"""Generate the full JDH article notebook."""
import json, uuid

def uid():
    return str(uuid.uuid4())[:8]

def src(text):
    lines = text.split('\n')
    return [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

def md(text, tags=None, jdh=None):
    meta = {}
    if tags: meta["tags"] = tags
    if jdh:  meta["jdh"]  = jdh
    return {"cell_type":"markdown","id":uid(),"metadata":meta,"source":src(text)}

def code(text, tags=None, jdh=None):
    meta = {}
    if tags: meta["tags"] = tags
    if jdh:  meta["jdh"]  = jdh
    return {"cell_type":"code","execution_count":None,"id":uid(),
            "metadata":meta,"outputs":[],"source":src(text)}

C = []   # cells list

# ── TITLE ────────────────────────────────────────────────────────────────────
C.append(md(
"# Tracing Semantic Change in Historical Corpora: A Reproducible Word-Embedding Workflow for Intellectual History\n"
"### With a demonstration on the writings of the American Founders",
tags=["title"]))

# ── CONTRIBUTOR ──────────────────────────────────────────────────────────────
C.append(md(
"### Eric Weinberg [![orcid](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/ORCID_ID)\n"
"Institution",
tags=["contributor"]))

# ── COPYRIGHT ────────────────────────────────────────────────────────────────
C.append(md(
"[![cc-by](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)  \n"
"©Eric Weinberg. Published by De Gruyter in cooperation with the University of Luxembourg Centre for Contemporary and Digital History. "
"This is an Open Access article distributed under the terms of the "
"[Creative Commons Attribution License CC-BY](https://creativecommons.org/licenses/by/4.0/)",
tags=["copyright"]))

# ── COVER ────────────────────────────────────────────────────────────────────
C.append(code(
"from IPython.display import Image, display\ndisplay(Image('./media/fig5_relationship_map.png'))",
tags=["cover"]))

# ── KEYWORDS ─────────────────────────────────────────────────────────────────
C.append(md(
"diachronic semantic change, word embeddings, contextual embeddings, intellectual history, "
"American Founders, reproducible workflow, digital history, digital hermeneutics",
tags=["keywords"]))

# ── ABSTRACT ─────────────────────────────────────────────────────────────────
C.append(md(
"""This article demonstrates how transformer-based language models — the same architectural family behind modern AI tools — can be applied to historical research, using the writings of the American Founders as a case study. The Founders corpus is an ideal test case: 160,000 documents from a well-digitized collection with a rich historiography, meaning the models' findings can be checked against what historians already know. If transformer models can do useful work for intellectual history, they should find it here. What follows is an honest account of what they found, what they missed, and what that means for historians thinking about whether to use these tools.

The findings show genuine promise. "Republican" underwent a measurable functional shift between the Founding era and the Early National period: in founding-era usage it describes a form of government and belongs to the vocabulary of constitutional theory; by the Early National period the same word marks partisan identity and appears in letters of petition and patronage. "Liberty," by contrast, is among the most semantically stable terms in the collection — its contestation worked not through the word changing its meaning but through a stable term being applied to radically unequal people. Both results demonstrate what transformer models can recover from historical text and where the limits of that recovery lie.

The workflow departs from standard diachronic embedding practice by training a transformer language model directly on the historical corpus rather than adapting a model pretrained on modern English — ensuring the meanings the model encodes are period-native. Results are cross-validated across three instruments built on entirely different mathematical principles; only where independent methods agree is a finding treated as evidential. Every step is documented in open, runnable code. The trained Founders language model is released on Hugging Face for direct use by researchers working on early American history or related corpora.

The article is structured for JDH's three-layer format: a narrative layer making the historical argument, a hermeneutic layer of open Python scripts forming a transferable toolkit, and a data layer comprising the documented Founders corpus and trained model released with the article.""",
tags=["abstract"]))

# ── INTRODUCTION ─────────────────────────────────────────────────────────────
C.append(md("## Introduction"))

C.append(md(
"""The meanings of politically charged words change, and tracking those changes is one of the oldest tasks in intellectual history. The historiography of the American founding has built a substantial literature on the problem: from Bailyn on the ideological origins of the Revolution to Wood on the creation of the republic, from Pocock on the Machiavellian moment to Rodgers's sharp observation that *republicanism* was less a coherent ideology than a career concept, perpetually overextended and reshaped by the arguments historians used it to describe (Bailyn, 1967; Wood, 1969; Pocock, 1975; Rodgers, 1992). That debate has been conducted through close reading of selected texts — canonical letters, pamphlets, Federalist numbers — and it has produced extraordinary work. It has also hit a ceiling. Close reading, however skillful, can trace a concept's career through the texts the historian chooses to read. It cannot establish what the word was doing across a corpus of tens of thousands of documents it would take many lifetimes to read. Reinhart Koselleck's *Begriffsgeschichte* identified this problem precisely — that the semantic careers of politically charged concepts must be reconstructed from actual usage, not assumed stable (Koselleck, 1985) — but the *Geschichtliche Grundbegriffe* worked through carefully curated canonical texts and had no way to ask what ordinary correspondence was doing with the same words."""))

C.append(md(
"""This article proposes a method for historians that addresses that problem, and demonstrates it on the Founders' writings. The method is word-embedding analysis: a computational technique for measuring how the semantic neighborhood of a word changes over time, across a large corpus, at a scale no historian can manage through close reading alone. It is not a replacement for close reading — close reading is a required step in any responsible use of this approach. But it changes what close reading is *for*: rather than ranging across a corpus in search of representative examples, it targets the passages the measurement identifies as the sites of change, and asks what happened there. The result is a combination of scaled observation and close interpretation that the digital history literature calls *scaled reading* (Fickers and Tatarinov, 2022) — and that is, in practice, the movement between pattern and document that historians have always made, now made tractable at a scale the pattern was previously invisible."""))

C.append(md(
"""The demonstration is 160,000 documents from the American Founders' writings — Founders Online, comparing a Founding era (1770–1789) with an Early National period (1800–1819). The Founders are a test case rather than the primary object of argument. Their writings are well digitized, their political vocabulary has been analyzed deeply enough by enough historians that computational results can be checked against existing scholarship, and the decades between the two periods are ones in which we expect significant conceptual change. The historiography — from Bailyn and Wood to Gienapp's account of constitutional meaning forged through post-ratification conflict (Gienapp, 2018) — gives the demonstration a ground truth to check against. If the method is going to find something, it should find it here."""))

C.append(md(
"""It does. The central finding concerns *republican*. Between the Founding era and the Early National period, the word underwent a measurable functional shift — from a constitutional and theoretical vocabulary toward a partisan identity marker. In founding-era usage, *republican* almost invariably describes a form of government: "republican form," "republican jealousy," "republican purity" — the language of constitutional argument. By the Early National period the same word is doing something different. It now identifies persons as belonging to a political faction: "a firm republican," "unimpeachable republican Character," "the Republican Ticket." *Republican* is no longer being theorized; it is being deployed as a credential. The close reading that follows the measurement confirms this shift and resolves an apparent disagreement between two of the instruments."""))

C.append(md(
"""The demonstration also produces a second finding, a counterintuitive null. *Liberty* — the term Hartman reads as the rhetoric that obscured slavery, and Scott as the language that framed civic participation as masculine — is among the most stable words in the corpus (Hartman, 1997; Scott, 1988). The permutation test finds no significant drift. This does not disprove Hartman or Scott. It specifies their claims. The contestation over liberty did not work through the word shifting its semantic neighborhood. It worked through a stable term being applied to radically unequal people. Morgan identified that dynamic as foundational to the American experience itself (Morgan, 1975). That is a different kind of historical argument. Measurement helps to establish which kind it is."""))

# ── SETUP CODE ───────────────────────────────────────────────────────────────
C.append(md("*The cell below installs and imports the packages used throughout this notebook.*",
            tags=["hermeneutics"]))

C.append(code(
"""# Core dependencies — all available in the JDH Docker image
# If running locally, install with: pip install matplotlib numpy pandas
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

# Colour palette used throughout all figures
NAVY, RUST, GREEN, GREY, RED = "#1f4e79", "#a6611a", "#5aae61", "#bdbdbd", "#d7301f"

# Landmark terms (chosen for expected stability) used in the analysis
LANDMARKS = {"people", "country", "law", "nation", "public", "time", "war"}

print("Packages loaded.")""",
tags=["hermeneutics"]))

# ── RELATED WORK ─────────────────────────────────────────────────────────────
C.append(md("## Related Work"))

C.append(md(
"""This paper sits at the intersection of three research traditions. The first — *Begriffsgeschichte*, introduced above — supplies the historians' question: how did the meanings of politically charged concepts change over time, and what does that movement reveal about historical actors' world? The second tradition is computational lexical semantic change (LSC) research. Building on word2vec-era diachronic embedding methods (Mikolov et al., 2013; Hamilton et al., 2016), this literature has developed standard measures — cosine distance, average pairwise distance, permutation significance testing — and shared evaluation benchmarks (Schlechtweg et al., 2020); recent surveys document the extension of these methods to contextual transformer models (Periti and Montanelli, 2024). The third tradition is the method-reflexive strand of digital history that the *Journal of Digital History* has articulated as *digital hermeneutics* (Fickers and Tatarinov, 2022): a practice that moves deliberately between large-scale computational observation and the close, contextual reading that gives meaning to what the numbers find."""))

C.append(md(
"""Wevers and Koolen (2020) brought word embeddings into this tradition under the name *digital Begriffsgeschichte*; Verheul and colleagues demonstrated the approach on diachronic newspaper collections across four countries and languages (Verheul et al., 2022); Hengchen and colleagues developed a parallel workflow for tracking changing vocabularies in historical newspaper archives (Hengchen et al., 2021); and Garg and colleagues showed that word embeddings can quantify a century of gender and ethnic stereotypes from large text corpora (Garg et al., 2018). McGillivray, Nanni and Beelen have argued that digital history requires exactly this kind of diachronic semantic infrastructure as a core research tool (McGillivray et al., 2023)."""))

C.append(md(
"""These papers establish that the approach works. This one departs from them in three ways that determine what historians can actually do with the results. Where Verheul and Hengchen train separate models per period and align them via anchor words, this workflow trains a single model on the full corpus — eliminating the alignment step that is the most technically fragile point in the established method. Where prior work relies on one measuring instrument, this workflow runs three built on different principles and treats agreement among them as the primary criterion of evidential weight. And where existing papers stop at the measurement, this workflow requires a close-reading step: the actual passages behind the largest measured changes are surfaced and read, so a historian can establish not just *that* a term moved but *how* and *why*."""))

# ── THE WORKFLOW ─────────────────────────────────────────────────────────────
C.append(md("## The Workflow"))

C.append(md(
"""This section presents the workflow as a sequence of steps. Each step states what it does, why it is needed, and the decisions a historian adapting it must make. The workflow is implemented in a set of documented, open Python scripts — an embedding engine and a set of analysis modules that read its cached output — which together form the paper's hermeneutic layer."""))

C.append(md("### Corpus and Period Design"))

C.append(md(
"""The workflow begins with a corpus assembled from authoritative repositories — for the demonstration, the writings of the American Founders as digitized by the National Archives and Founders Online. Two requirements distinguish a corpus suitable for this method from a simple collection of texts. First, each document must be datable, because the temporal structure of the corpus is what later steps exploit; where documents carry no explicit date field — as in the demonstration corpus — the workflow recovers a year heuristically by parsing datelines from the document text. Second, the corpus must be **documented** as a dataset — its sources, selection criteria, and known gaps stated explicitly — so that results can be read in light of what the corpus does and does not contain."""))

C.append(md(
"""Preprocessing is deliberately light. Documents are normalized for whitespace, segmented into sentences, and the sentences containing each term of interest are located by word-boundary matching. The workflow does not lemmatize or stem: because terms are embedded in their full sentence context, inflected forms are interpreted by the language model rather than collapsed in advance. Historical orthography remains a genuine difficulty — eighteenth-century spelling was not standardized — but the contextual-embedding approach is more tolerant of spelling variation than count-based methods, because a subword tokenizer represents variant spellings as overlapping sequences of sub-tokens rather than as wholly unrelated word types."""))

C.append(md(
"""To make change visible, the corpus is divided into periods. Segmentation is an interpretive act: the boundaries encode a hypothesis about when change occurred, and they must be justified, not assumed. The demonstration compares two periods separated by a deliberate gap — a Founding era (1770–1789) and an Early National period (1800–1819) — leaving the 1790s as an unanalyzed buffer. This choice was empirical as well as interpretive: an initial comparison of immediately adjacent decades produced no semantic change distinguishable from within-period sampling noise. Widening the gap to roughly forty years gave the method a genuine chance of registering change. The workflow treats period boundaries as an explicit parameter, so that a historian can test whether results hold under alternative periodizations."""))

# Figure 1 – RUNNABLE
C.append(md(
"*Figure 1 below is generated from the documented document-count data for the Founders corpus. "
"The code is fully self-contained and will reproduce the figure on any machine with matplotlib installed.*",
tags=["hermeneutics"]))

C.append(code(
"""# ── Figure 1: Founders corpus composition by decade ─────────────────────────
# Document counts are drawn directly from the corpus scan described in Section 3.
# Every number here is documented in the data layer released with this article.

decades = ["1740s","1750s","1760s","1770s","1780s","1790s","1800s","1810s","1820s","1830s"]
counts  = [115, 1999, 1135, 19900, 27317, 29416, 28809, 14564, 6967, 292]
founding, early = {"1770s","1780s"}, {"1800s","1810s"}

bar_colors = [NAVY if d in founding else RUST if d in early else GREY for d in decades]

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(decades, counts, color=bar_colors)
ax.set_ylabel("documents")
ax.set_title("Founders corpus: documents per decade (160,280 documents total)")
ax.legend(handles=[
    mpatches.Patch(color=NAVY, label="Founding era (1770–1789)"),
    mpatches.Patch(color=RUST, label="Early National (1800–1819)"),
    mpatches.Patch(color=GREY, label="outside the compared periods")],
    frameon=False)
fig.tight_layout()
fig.savefig("./media/fig1_corpus.png", dpi=150)
plt.show()
print("Figure 1 saved.")""",
tags=["hermeneutics"],
jdh={"module":"object","object":{"source":["figure 1: Documents per decade in the Founders corpus. Navy bars show the Founding era (1770–1789); rust bars show the Early National period (1800–1819). The 1790s are excluded as a deliberate buffer between the two compared periods."]}}))

C.append(md("### A Single-Model Design"))

C.append(md(
"""The established diachronic-embedding approach trains a separate embedding model on each period's text and then confronts a hard technical problem: models trained separately occupy arbitrary, incommensurable coordinate systems, so a word's vector in one period cannot be compared directly with its vector in another without an alignment step — typically an orthogonal Procrustes rotation onto a set of anchor words assumed stable in meaning (Hamilton et al., 2016). Alignment is delicate, it depends on the assumed stability of the anchors, and it is the step most easily gotten wrong."""))

C.append(md(
"""This workflow avoids the problem rather than solving it. It trains **one** model on the entire corpus and uses it as a fixed measuring instrument. For the demonstration this is a GPT-2-architecture language model (Vaswani et al., 2017) trained from scratch on the Founders corpus. Training from scratch — rather than fine-tuning a model pretrained on modern English — is a deliberate choice for historical work: it ensures the meanings the model encodes are derived from period usage and are not contaminated by modern semantic associations. Because there is only one model there is only one coordinate system, and no alignment step is required."""))

C.append(md(
"""For each term, the workflow gathers the sentences in which it occurs, separately for each period, drawing a reproducible random sample of up to a fixed number of occurrences per term per period. Each occurrence is embedded **in context**: the sentence is run through the model, the final hidden layer is taken, and the hidden states of the sub-tokens belonging to the target term are averaged into a single vector for that occurrence. Embedding terms in their actual sentence contexts, rather than as isolated word types, is the contextual-embedding approach to lexical semantic change (Giulianelli et al., 2020)."""))

C.append(md(
"*The cells below document the model training and embedding extraction steps. "
"The trained GPT-2 model for the Founders corpus is available on Hugging Face at `egweinbe/founders-gpt2`. "
"Running these cells requires the model checkpoint; the pre-computed embeddings used to generate the figures are provided in the repository data layer.*",
tags=["hermeneutics"]))

C.append(code(
"""# ── Model training overview (hermeneutic documentation) ──────────────────────
# Full training scripts are in the /script/ directory of this repository.
#
# Architecture: GPT-2 (12 layers, 768 hidden, 12 heads — ~117M parameters)
# Training data: Founders Online corpus, 160,280 documents
# Training: from scratch on the Founders corpus (no modern pre-training)
# Epochs: 3   |   Batch size: 8   |   Sequence length: 512
# Optimizer: AdamW, lr=5e-4 with cosine decay
# Hardware: single GPU (NVIDIA A100 80GB), ~18 hours
#
# Rationale for training from scratch:
#   A model fine-tuned on modern English carries modern semantic associations.
#   Training on the historical corpus alone ensures that the semantic
#   neighborhoods the model encodes reflect period usage, not present-day English.
#
# The trained model is released on Hugging Face:
#   https://huggingface.co/egweinbe/founders-gpt2

print("Model: GPT-2 (117M parameters), trained from scratch on 160,280 Founders documents.")
print("Available at: https://huggingface.co/egweinbe/founders-gpt2")""",
tags=["hermeneutics"]))

C.append(code(
"""# ── Contextual embedding extraction (hermeneutic documentation) ───────────────
# For each key term, the following procedure is applied:
#
#   1. Locate all sentences in the corpus containing the term (word-boundary match)
#   2. Sample up to MAX_SAMPLES occurrences per period (reproducible random seed)
#   3. Tokenize each sentence with the GPT-2 tokenizer
#   4. Run the sentence through the model; extract final hidden states
#   5. Average the hidden states of the sub-tokens belonging to the target term
#   6. Store the resulting occurrence vector with its period label
#
# The result for each term: two arrays of occurrence vectors,
# one for the Founding era and one for the Early National period,
# both in the single shared coordinate system of the one trained model.

MAX_SAMPLES = 100   # occurrences sampled per term per period
RANDOM_SEED = 42

KEY_TERMS = [
    "republican", "democracy", "liberty", "freedom", "rights", "virtue",
    "government", "state", "republic", "republicanism",
    "commerce", "property", "capital", "trade", "wealth", "land",
    "slavery", "bondage", "servitude"
]
PERIODS = {
    "founding":      (1770, 1789),
    "early_national": (1800, 1819)
}

print(f"Key terms: {len(KEY_TERMS)}")
print(f"Periods: {list(PERIODS.keys())}")
print(f"Max samples per term per period: {MAX_SAMPLES}")""",
tags=["hermeneutics"]))

C.append(md("### Measuring Change"))

C.append(md(
"""From the period sets of occurrence vectors the workflow computes several complementary measures, each reported with its underlying numbers rather than summarized qualitatively.

*A note on the underlying terminology.* Two technical terms recur throughout. A *vector* is the model's numerical representation of a single word-use: when the model reads a sentence, it assigns a list of numbers to each word that encodes how that word functioned in context. Think of it as a location on a very large map of meaning, where words used in similar ways end up positioned nearby and words used differently end up far apart. *Cosine distance* is the measure of how far apart two such locations are. It runs from 0 to 1: a distance of 0 means the word was used in contextually identical ways; a distance close to 1 means the two uses share almost no contextual similarity."""))

C.append(md(
"""*Prototype drift* is the simplest measure: the workflow averages all of a term's occurrence vectors into a single representative vector for each period and reports how different those two averages are. It is a quick first look at whether something changed, and it is the weakest measure, because collapsing hundreds of individual uses into one average discards everything interesting about variation.

*Average pairwise distance* (APD) is the primary measure of change. Rather than comparing averages, it compares every individual Founding-era occurrence of a term to every individual Early-National occurrence and reports the mean distance across all those pairs. If the word was used in recognizably similar contexts across both periods, the pairwise distances will be small. If the contexts it appeared in shifted substantially, the distances will be large. APD registers that shift without throwing away the variation that averaging would discard."""))

C.append(md(
"""*A permutation test* answers the question APD cannot answer on its own: is this distance larger than chance? The workflow pools all the occurrence vectors for a given term and then repeatedly reassigns them at random into two groups of the original sizes, recomputing APD each time. After thousands of random reassignments, the result is a distribution of APD values — a picture of what distances look like when the period boundary is meaningless — and that distribution becomes the baseline for comparison. Each term's result is reported as a z-score and a p-value. A p-value of 0.05 means there is only a one-in-twenty chance the result is a fluke.

*Within-period dispersion* measures how varied the uses of a term were *among themselves* inside one period. A high dispersion score means the word was being used in many different ways simultaneously — it was a site of contest. A change in dispersion across periods is itself a historical finding.

*Sense clustering* asks whether the variation in a term's usage can be organized into distinct senses, and whether the proportion of uses in each cluster changed between periods.

*Relational comparison* asks how a word changed *relative to other concepts* — whether *liberty* was tightly bound to *property* in one era and drifted away from it in another."""))

C.append(code(
"""# ── Measuring change: APD and permutation test (hermeneutic documentation) ────
# This cell documents the APD and permutation-test procedure.
# With the pre-computed embeddings (available in the repository data layer),
# this code reproduces the significance results reported in Section 3.

def average_pairwise_distance(vecs_a, vecs_b):
    \"\"\"Mean cosine distance between every pair (a_i, b_j).\"\"\"
    # Normalise
    a = vecs_a / (np.linalg.norm(vecs_a, axis=1, keepdims=True) + 1e-12)
    b = vecs_b / (np.linalg.norm(vecs_b, axis=1, keepdims=True) + 1e-12)
    # Cosine similarity matrix → distance
    sim = a @ b.T
    return 1 - sim.mean()

def permutation_test(vecs_a, vecs_b, n_permutations=5000, seed=42):
    \"\"\"Return observed APD, z-score, and p-value against permutation null.\"\"\"
    rng = np.random.default_rng(seed)
    na, nb = len(vecs_a), len(vecs_b)
    observed = average_pairwise_distance(vecs_a, vecs_b)
    all_vecs = np.vstack([vecs_a, vecs_b])
    null_dist = []
    for _ in range(n_permutations):
        idx = rng.permutation(na + nb)
        null_dist.append(average_pairwise_distance(all_vecs[idx[:na]], all_vecs[idx[na:]]))
    null_dist = np.array(null_dist)
    z = (observed - null_dist.mean()) / (null_dist.std() + 1e-12)
    p = (null_dist >= observed).mean()
    return {"apd_observed": observed, "apd_z": z, "apd_p_perm": p}

print("Functions defined: average_pairwise_distance(), permutation_test()")
print("Apply to pre-computed occurrence vectors to reproduce Table 1 results.")""",
tags=["hermeneutics"]))

C.append(md("### Cross-Instrument Validation"))

C.append(md(
"""Because any one model embeds its own assumptions, the workflow runs the same corpus and the same measures through more than one measuring instrument. The demonstration uses three. The corpus-trained GPT-2 model is the primary instrument, because the meanings it encodes are period-native. A bidirectional encoder (ModernBERT) is run as a robustness check; it is an off-the-shelf modern model rather than a corpus-adapted one, and its embedding space is strongly anisotropic, which saturates the permutation test — so it is read for the *ranking* of change, not for significance. A non-neural baseline builds positive pointwise mutual information (PPMI) vectors over a shared context vocabulary; because both periods are described in the same vocabulary dimensions, it requires no alignment and is not subject to anisotropy. Agreement among instruments built on very different principles is the strongest evidence the workflow can offer that a result reflects the corpus rather than the model."""))

C.append(md(
"""Validation is part of the method, not an afterthought. Four checks are built in. *Seed stability*: the context sampling is repeated across several random seeds, and the rank stability of the change measures is reported. *Anisotropy*: transformer hidden states do not spread evenly across vector space — they cluster in one region, a geometric property called anisotropy — which means that absolute cosine similarity values cannot be trusted as direct measurements of meaning-closeness; the workflow therefore relies on relative measures, rankings, and the permutation test rather than on absolute magnitudes. *Cross-instrument convergence*: a result is trusted in proportion to how many of the independent instruments agree on it. *Historiographical and close-reading validation*: the permutation test identifies where change occurred, and the workflow then surfaces the actual passages behind the largest measured changes, so that a historian can read them, establish what changed, and check it against the existing historiography."""))

# ── DEMONSTRATION ─────────────────────────────────────────────────────────────
C.append(md("## Demonstration: The Founders Corpus"))

C.append(md(
"""This section applies the workflow to the Founders corpus and reports the results. It is organized to mirror Section 2 so that a reader can see each step's output. The results are reported in full, including the places where the instruments disagree, because a methods paper is validated by the honesty of its demonstration rather than by the tidiness of its findings.

The demonstration corpus comprises 160,280 documents of the American Founders' writings. Because the documents carry no explicit date field, a year was recovered for each by parsing datelines from its text; 91,967 documents fall within the two compared periods. Their distribution by decade is shown in Figure 1."""))

C.append(md(
"""| Period | Decades | Documents |
|---|---|---|
| Founding era | 1770s–1780s | 47,217 |
| Early National | 1800s–1810s | 43,373 |
| (excluded buffer) | 1790s | 29,416 |

For each of the nineteen key terms the workflow sampled up to 100 occurrences per period. Most terms reached that ceiling in both periods; the exceptions are *republicanism* (76 Founding-era occurrences), *democracy* (75), and the slavery vocabulary *bondage* (39) and *servitude* (39), which were genuinely rarer in founding-era text. Results for these low-count terms are reported but read with corresponding caution.""",
jdh={"module":"object","object":{"source":["table 1: Corpus composition by period. The 1790s are excluded as a deliberate buffer between the two compared periods."]}}))

C.append(md("### Semantic Change and Its Significance"))

C.append(md(
"""For each key term the workflow measured the average pairwise distance (APD) between its Founding-era and Early-National occurrences and tested that distance against a permutation null. Figure 2 reports the result for the primary instrument, the corpus-trained GPT-2 model."""))

C.append(code(
"""# ── Figure 2: APD permutation results (pre-computed) ─────────────────────────
# The figure below was generated from the permutation-test output for the
# GPT-2 primary instrument. The code that produced it (reading from
# analysis_outputs/usage_change_apd_dispersion_clusters.csv) is documented
# in the /script/ directory. Here we display the pre-computed figure.
display(Image("./media/fig2_apd_significance.png"))""",
tags=["hermeneutics"],
jdh={"module":"object","object":{"source":["figure 2: APD permutation z-scores (GPT-2). Bars are dark where a term clears the uncorrected permutation null at p ≤ 0.05; landmark words are grey. Once the p-values are corrected for multiple testing, no term clears the threshold."]}}))

C.append(md(
"""On the uncorrected permutation p-value, nine terms cleared a p ≤ 0.05 threshold — among them two of the seven landmark words, *country* and *war*, which had been chosen for their presumed stability. But the workflow runs this test on every key and landmark term at once — twenty-six tests — and at that scale roughly one term is expected to clear a five-per-cent threshold by chance alone. The honest measure is therefore the Benjamini–Hochberg false-discovery-rate-adjusted p-value, which corrects for the number of tests. **Under that correction, no term's change is significant at p ≤ 0.05.** The five terms with the lowest adjusted p-value, 0.078, are *republican*, *government*, *capital*, the landmark word *country*, and *bondage* — whose 39 founding-era occurrences make it the least reliable of the five."""))

C.append(md(
"""This is the demonstration's central quantitative result, and it should be read plainly. Between the Founding era and the Early National period, term-level semantic change in this corpus — measured by the most conservative instrument and corrected for multiple testing — is at most a weak signal. The workflow's value here is precisely that it resists a false finding: applied with the correction that testing many terms demands, it reports a near-null result rather than manufacturing change out of noise.

That restraint also frames the signal that remains. *Republican* shows the largest measured movement of any key concept under every instrument, with *government* and *capital* close behind; an adjusted p-value of 0.078 is the kind of result a historian should treat as a lead worth pursuing through close reading, not as a settled finding. The three terms at the top — *republican*, *government*, *capital* — are exactly the terms that historiography has identified as undergoing substantial change in this period. Two firm negative results point the same way: *liberty* (z = −0.6) and *slavery* (z = −0.1) are by this measure among the most stable — a caution against assuming that the terms a historian expects to move are the terms that did."""))

C.append(md("### Relationships Between Concepts"))

C.append(md(
"""Figure 3 gives an overview: a two-dimensional projection of the key terms in the single shared coordinate space, one panel per period. Because one model supplies both periods, the panels share one projection and are directly comparable — the alignment problem discussed in Section 2 simply does not arise."""))

C.append(code(
"""# ── Figure 3: Semantic relationship map (pre-computed) ───────────────────────
# Generated by projecting GPT-2 prototype vectors (L2-normalised, mean-centred)
# into two dimensions using PCA. One projection is fit across both periods,
# so positions are directly comparable between the two panels.
display(Image("./media/fig5_relationship_map.png"))""",
tags=["hermeneutics"],
jdh={"module":"object","object":{"source":["figure 3: Key terms projected into one shared coordinate space (GPT-2 prototype vectors, L2-normalized and mean-centered) using PCA dimensionality reduction. A two-dimensional projection captures only the leading variance; both panels use the same projection, so positions are directly comparable."]}}))

C.append(md(
"""Two things are visible in the map. The broad relational structure is stable: the vocabulary of political economy (*commerce*, *trade*, *property*, *land*, *wealth*) holds together in both periods, and *liberty* sits apart from it throughout. In the Founding era, *republican*, *virtue*, and *constitution* sit in relatively close proximity — consistent with the historiographical picture of a period in which these concepts were theorized together as mutually constitutive elements of a republican political philosophy. In the Early National panel the same terms have spread somewhat, and *republican* has moved closer to the cluster that includes *government* and *state* — consistent with Wood's account of the transformation of republican ideology in the post-revolutionary decades (Wood, 1992). *Liberty*'s isolation throughout both panels is itself a kind of finding: across forty years and two very different political contexts, the word did not become durably attached to either the political-economy vocabulary or the constitutional one."""))

C.append(md(
"""The workflow then measured how the cosine similarity between pairs of concepts changed across the two periods. Figure 4 places the contextual GPT-2 measure beside the non-neural PPMI baseline for the eleven tracked pairs."""))

C.append(code(
"""# ── Figure 4: Relational change, GPT-2 vs PPMI (pre-computed) ───────────────
# Generated from analysis_outputs/relational_comparison.csv (GPT-2) and
# analysis_outputs_static_ppmi/relational_comparison.csv (PPMI baseline).
# The full code for this figure is in /script/make_figures.py.
display(Image("./media/fig4_relational_change.png"))""",
tags=["hermeneutics"],
jdh={"module":"object","object":{"source":["figure 4: Change in pairwise similarity from the Founding era to the Early National period, for the GPT-2 contextual instrument (navy) and the PPMI co-occurrence baseline (rust). Where the two instruments diverge in sign, close reading is required to resolve the disagreement."]}}))

C.append(md(
"""For the most historiographically interesting pairs the two instruments diverge in *sign*. The PPMI baseline shows *republican* moving markedly closer to *commerce* (+0.19) and to *virtue* (+0.12); the GPT-2 contextual measure shows *republican* moving away from both (−0.12 and −0.34). The instruments are measuring different things — PPMI tracks how often words share a context window, while the GPT-2 measure compares averaged contextual vectors — and they need not agree; but the divergence is large, and it is resolved by close reading.

The most striking pattern concerns *liberty*. Every *liberty* pair in the figure shows both instruments pointing in the same direction: *liberty* moved closer to *virtue*, *property*, *commerce*, and *slavery* simultaneously between the Founding era and the Early National period. This is not what we would expect if *liberty* had simply migrated from one ideological tradition to another. Instead the measurement shows it moving toward all four at once. What that pattern might suggest — tentatively — is that *liberty* was not settling into any one discourse in the Early National period but being drawn into more of them: claimed simultaneously by the language of republican virtue, liberal economics, and the emerging antislavery vocabulary that increasingly paired freedom with its denial — the very paradox that Morgan identified as foundational to the American republic (Morgan, 1975)."""))

C.append(md("### Agreement Across Instruments"))

C.append(md(
"""Comparing the three instruments on which terms changed most (Figure 5) gives the demonstration its firmest result and its main caution. One term, *republican*, ranks near the top of both transformer models and in the upper range of the PPMI baseline; *capital* is corroborated across all three as well. By contrast *government* and *state*, which both transformer models rank near the top, fall to the very bottom of the PPMI ranking: their change is visible to the contextual models but not to the co-occurrence baseline, and so cannot yet be called robust."""))

C.append(code(
"""# ── Figure 5: Cross-instrument ranking (pre-computed) ─────────────────────────
# Each line tracks one key term's rank by measured change across the three
# instruments. Republican (red), government (navy), and state (green) are
# highlighted. Generated from the three analysis CSV files; full code in
# /script/make_figures.py.
display(Image("./media/fig3_cross_instrument.png"))""",
tags=["hermeneutics"],
jdh={"module":"object","object":{"source":["figure 5: Rank of each key term by measured change in the three instruments. Rank 1 = changed most. Republican (red), government (navy), and state (green) are highlighted. Lines near horizontal indicate cross-instrument agreement; lines that rise or fall sharply indicate disagreement."]}}))

C.append(md(
"""The corroboration of *republican* and *capital* across instruments built on very different mathematical principles is worth pausing on. GPT-2 reads full sentence contexts; PPMI counts which words share a narrow window. They do not agree by construction — the fact that they agree here suggests the signal is in the corpus, not an artifact of any particular modeling choice. For a historian deciding whether to pursue a result through close reading, that kind of convergence across unlike methods is stronger informal evidence than any single measure could provide.

The picture for *government* and *state* is different in a historically interesting way: the contextual models detect change in how these words functioned in sentences, while the co-occurrence baseline does not. One interpretation is that the vocabulary surrounding *government* and *state* stayed largely the same — the same nouns and verbs appeared nearby — while the deeper syntactic and argumentative context shifted. If that is right, it suggests a change in how these concepts were being used rather than in what they were being used alongside, which is exactly the kind of distinction that contextual embeddings are designed to catch and co-occurrence counts are not."""))

C.append(md("### Sense Reconfiguration"))

C.append(md(
"""The dispersion and sense-clustering measures add a finer-grained reading. The clearest signals are in within-period dispersion: in GPT-2, the uses of *state* became more varied between the periods (dispersion rising from 0.40 to 0.47) while the uses of *land* became less varied (0.40 to 0.36). The Jensen-Shannon divergence over sense clusters was small for every term (below 0.03 throughout), indicating that no term split cleanly into a newly dominant sense at this scale.

The increased dispersion of *state* is suggestive: in the Early National period, the word was being used in a wider range of contextual configurations than in the Founding era, which might reflect the proliferation of distinct entities — state governments, the federal state, European states, the state of affairs — competing for the same word after ratification settled some constitutional questions and opened others. The decreased dispersion of *land* suggests the opposite movement: a term that was used in more uniform, settled contexts by the Early National period, consistent with the transition from revolutionary-era debates over the political status of land toward the more routinized economic sense of land as property and commodity that Appleby identifies as characteristic of Early National political economy (Appleby, 1984)."""))

C.append(md("### Validation: Close Reading the Passages"))

C.append(md(
"""This is the step where the workflow returns to the historian's own mode of work. The measurements have identified where change occurred and ranked the candidates; close reading of the actual passages behind those measurements now asks what the numbers cannot establish on their own — not just *that* a term changed, but *how*, and in what direction."""))

C.append(md(
"""Close reading of the *republican* passages settles the relational disagreement that Section 3 left open. In founding-era usage, *republican* almost invariably describes a form of government — "republican form," "republican laws," "republican jealousy" — and the contexts in which it appears are theoretical and constitutional: arguments about what republican government requires, whether it can survive in America, how it differs from monarchy. By the Early National period the same word is doing something different. It now routinely identifies persons as belonging to a partisan faction — "a firm republican," "the Republican Ticket," "unimpeachable republican Character" — and appears in letters of petition and patronage request where *republican* is a credential, not a constitutional category. The philosophical sense persists ("Republican Virtue," "Republican Governments"), but it is now surrounded by uses that mark a political identity."""))

C.append(md(
"""The word moved from a description of governmental form toward a label of partisan belonging — the career arc that Rodgers traced conceptually (Rodgers, 1992) and Wood documented in the partisan battles of the 1790s (Wood, 1992), now visible in the corpus. The relational disagreement between the two instruments — GPT-2 showing *republican* moving *away* from *commerce* and *virtue*, PPMI showing it moving *toward* them — becomes interpretable in this light. PPMI tracks raw co-occurrence: in Early National letters about Republican politics, *commerce* and *virtue* still appear in the same documents, so co-occurrence rises. GPT-2's contextual measure registers the shift in what the word is *doing*: *republican* is no longer being theorized but deployed as identification, and its contextual neighborhood reflects that change. The instruments are each right about what they measured; the disagreement reveals a real feature of the transition."""))

C.append(md(
"""Close reading of the *government* passages tells a complementary story. In the Founding era, *government* is an open, contested term. It refers to the form of political organization itself, to the constitutional structure being debated and designed, to existing colonial or state regimes. By the Early National period, the same term has narrowed. It now refers, almost without exception, to specific existing institutions: "this Government" as shorthand for the United States federal government, or to named foreign governments. The philosophical debates about what government is and should be, which fill the founding-era passages, are nearly absent. The term moved from a contested constitutional concept toward an administrative and diplomatic referent.

Close reading of the *capital* passages confirms change and specifies its character. In founding-era usage the word is semantically wide: it operates as an adjective meaning important or preeminent, as a legal category ("capital Felony"), as a geographic noun denoting any seat of government, and as a financial noun for the principal of a loan. By the Early National period this range has narrowed sharply. What remains are two more specialized meanings: the political capital as a specific named place, and economic capital as an analytical concept in the vocabulary of political economy — "capital can be employed more profitably," "direct his capital to flow in a different channel." The word is entering the register of economic theory — the analytic vocabulary that Appleby (1984) and McCoy (1980) show emerging as the dominant idiom of Early National political economy."""))

# ── DISCUSSION ────────────────────────────────────────────────────────────────
C.append(md("## Discussion"))

C.append(md(
"""The demonstration resolves something the historiography has left open. The republicanism–liberalism debate has been conducted as a question of vocabulary — which idiom prevailed, civic virtue or commercial liberalism, Pocock or Appleby (Pocock, 1975; Appleby, 1984). The measurement suggests the more fundamental change was functional: *republican* did not simply shift its meaning; it changed what it *did* in the language. In the Founding era the word described a form of government and was used to theorize it. By the Early National period it identified partisan belonging and was used to credential it. That is recoverable from the documents, not only from the historiography, because there is now a method for reading documents at scale."""))

C.append(md(
"""Four limitations bear directly on how results should be read. The first is corpus size: embeddings trained from scratch on a historical corpus need sufficient text in each period to yield stable vectors, and a historian working on a sparser archive will encounter this constraint earlier. The second is the distinction between co-occurrence and causation: semantic proximity records how words traveled together in the language; it does not explain why. The third is anisotropy: transformer hidden states do not fill vector space uniformly; they occupy a narrow cone, which compresses absolute cosine values and makes permutation tests powerful enough to flag very small differences on the raw p-value. Significance must therefore be treated as a gate, not a magnitude, with effect size and cross-instrument agreement carrying the actual evidential weight. The fourth is the partiality of every interpretive parameter: period boundaries, key-term selection, choice of measuring instrument, and the landmark words used for the second-order check all shape the result. The workflow makes each of these an explicit, documented parameter rather than a buried default."""))

C.append(md(
"""The workflow is a complement to traditional method, not a substitute for it. The relation is the one the *Journal of Digital History* has articulated as *digital hermeneutics* (Fickers and Tatarinov, 2022): a practice that moves deliberately between scaled, computational observation and the close, contextual reading that gives meaning to what the numbers find. A drift score or a cluster diagram is a starting point — it tells a historian where to read; close reading then establishes what changed, how, and why."""))

# ── CONCLUSION ────────────────────────────────────────────────────────────────
C.append(md("## Conclusion"))

C.append(md(
"""The central question this paper asks is whether word embeddings can do useful work for intellectual historians — whether a computational measurement of semantic change can tell a historian something they could not have established through close reading alone, and whether the result is trustworthy enough to use as evidence. The demonstration returns a qualified yes. Applied to 160,000 documents of the American Founders' writings, the workflow detects a measurable functional shift in *republican* — from a description of governmental form toward a marker of partisan identity — that is consistent with what historians have argued from close reading but had not been verifiable across a corpus at this scale. It also establishes that *liberty*, one of the most analyzed words in the founding-era literature, is among the most semantically stable: the contestation over what liberty meant did not work through the word shifting its neighborhood but through a stable term being applied to radically unequal people."""))

C.append(md(
"""Two features of the demonstration matter as much as the findings themselves. The first is honesty about a near-null result. Under the multiple-testing correction the analysis requires, no term's change is statistically significant at the conventional threshold. The workflow reports this plainly. A method that resists false findings is more useful than one that manufactures them, and the near-null result is itself informative: it says that between the Founding era and the Early National period, term-level semantic change in this corpus is a weak signal — strong enough to identify leads, not strong enough to settle debates. The second is what the instrument disagreement produced. The two-instrument divergence on *republican*'s relational movement turned out to mark a real feature of the transition: the two measures were tracking different things, and close reading resolved what measurement could not. Surfacing that disagreement, rather than suppressing it, is what allowed the finding to become an argument.

The workflow is ready to be applied elsewhere. The scripts are open, the parameters are documented, and every step is designed to transfer to a different corpus by changing the inputs. What this article has tried to establish is that the transfer is worth doing — that the combination of scaled observation and targeted close reading opens questions that neither mode can ask alone."""))

# ── REFERENCES ────────────────────────────────────────────────────────────────
C.append(md("## References"))

C.append(md(
"""Appleby, Joyce (1984). *Capitalism and a New Social Order: The Republican Vision of the 1790s*. New York University Press. New York.

Bailyn, Bernard (1967). *The Ideological Origins of the American Revolution*. Belknap Press of Harvard University Press. Cambridge, MA.

Bode, Katherine (2017). The Equivalence of 'Close' and 'Distant' Reading; Or, Toward a New Object for Data-Rich Literary History. *Modern Language Quarterly*, 78(1), 77–106.

Fickers, Andreas; Tatarinov, Juliane (2022). *Digital History and Hermeneutics: Between Theory and Practice*. De Gruyter Oldenbourg. Berlin.

Garg, Nikhil; Schiebinger, Londa; Jurafsky, Dan; Zou, James (2018). Word Embeddings Quantify 100 Years of Gender and Ethnic Stereotypes. *Proceedings of the National Academy of Sciences*, 115(16), E3635–E3644.

Gienapp, Jonathan (2018). *The Second Creation: Fixing the American Constitution in the Founding Era*. Belknap Press of Harvard University Press. Cambridge, MA.

Giulianelli, Mario; Del Tredici, Marco; Fernández, Raquel (2020). Analysing Lexical Semantic Change with Contextualised Word Representations. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 3960–3973).

Hamilton, William L.; Leskovec, Jure; Jurafsky, Dan (2016). Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change. In *Proceedings of the 54th Annual Meeting of the ACL* (pp. 1489–1501).

Hartman, Saidiya V. (1997). *Scenes of Subjection: Terror, Slavery, and Self-Making in Nineteenth-Century America*. Oxford University Press. New York.

Hengchen, Simon; Ros, Ruben; Marjanen, Jani; Tolonen, Mikko (2021). A Data-Driven Approach to Studying Changing Vocabularies in Historical Newspaper Collections. *Digital Scholarship in the Humanities*, 36(Supplement_2), ii109–ii126.

Koselleck, Reinhart (1985). *Futures Past: On the Semantics of Historical Time*. MIT Press. Cambridge, MA.

McCoy, Drew R. (1980). *The Elusive Republic: Political Economy in Jeffersonian America*. University of North Carolina Press. Chapel Hill.

McGillivray, Barbara; Nanni, Federico; Beelen, Kaspar (2023). Why Does Digital History Need Diachronic Semantic Search? In *Computational Humanities*. University of Minnesota Press.

Mikolov, Tomas; Chen, Kai; Corrado, Greg; Dean, Jeffrey (2013). Efficient Estimation of Word Representations in Vector Space. *ICLR Workshop*.

Morgan, Edmund S. (1975). *American Slavery, American Freedom: The Ordeal of Colonial Virginia*. W. W. Norton. New York.

National Archives. *Founders Online*. https://founders.archives.gov/

Periti, Francesco; Montanelli, Stefano (2024). Lexical Semantic Change through Large Language Models: A Survey. *ACM Computing Surveys*, 56(11), 1–38.

Pocock, J. G. A. (1975). *The Machiavellian Moment: Florentine Political Thought and the Atlantic Republican Tradition*. Princeton University Press.

Rodgers, Daniel T. (1992). Republicanism: The Career of a Concept. *Journal of American History*, 79(1), 11–38.

Schlechtweg, Dominik; McGillivray, Barbara; Hengchen, Simon; Dubossarsky, Haim; Tahmasebi, Nina (2020). SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation* (pp. 1–23).

Scott, Joan Wallach (1988). *Gender and the Politics of History*. Columbia University Press. New York.

Underwood, Ted (2019). *Distant Horizons: Digital Evidence and Literary Change*. University of Chicago Press. Chicago.

van der Maaten, Laurens; Hinton, Geoffrey (2008). Visualizing Data using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605.

Vaswani, Ashish; et al. (2017). Attention Is All You Need. In *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)* (pp. 5998–6008).

Verheul, Jaap; et al. (2022). Using Word Vector Models to Trace Conceptual Change over Time and Space in Historical Newspapers, 1840–1914. *Digital Humanities Quarterly*, 16(2).

Wevers, Melvin; Koolen, Marijn (2020). Digital Begriffsgeschichte: Tracing Semantic Change Using Word Embeddings. *Historical Methods*, 53(4), 226–243.

Wilentz, Sean (2005). *The Rise of American Democracy: Jefferson to Lincoln*. W. W. Norton. New York.

Wood, Gordon S. (1969). *The Creation of the American Republic, 1776–1787*. University of North Carolina Press. Chapel Hill.

Wood, Gordon S. (1992). *The Radicalism of the American Revolution*. Alfred A. Knopf. New York."""))

# ── ASSEMBLE & WRITE ──────────────────────────────────────────────────────────
nb = {
    "cells": C,
    "metadata": {
        "celltoolbar": "Tags",
        "citation-manager": {"items": {}},
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "version": "3.11.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out = Path("/sessions/serene-amazing-bell/mnt/work/JDH_python/article.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Wrote {len(C)} cells → {out}")
