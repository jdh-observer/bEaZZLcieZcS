# Script Layer

This folder contains the runnable analysis materials referenced in the article.
The scripts implement the full workflow: corpus scanning, date recovery,
context sampling, contextual embedding extraction, drift measurement,
relationship comparison, cross-model baselines, cache-based exploration, and
figure generation.

## Main Pipeline

Run the corpus-trained GPT-2 analysis:

```bash
python3 script/analysis_core.py
```

For a quick smoke test using only five sampled contexts per term:

```bash
python3 script/analysis_core.py 5
```

The script scans the full corpus, samples contexts for the configured terms,
builds contextual embeddings with the corpus-trained model, and writes reusable
outputs to `analysis_outputs/`, including:

- `semantic_drift.csv`
- `relational_comparison.csv`
- `second_order_change.csv`
- `context_counts.csv`
- `context_samples.csv`
- `embeddings.npz`
- `usage_embeddings.npz`
- all-pair relationship tables and change matrices

By default, `analysis_core.py` loads the released model
`ladew222/founders-gpt2` from Hugging Face and expects the corpus at
`data/data.jsonl`. These defaults can be overridden:

```bash
FOUNDERS_MODEL_DIR=/path/to/model \
FOUNDERS_CORPUS_PATH=/path/to/data.jsonl \
python3 script/analysis_core.py
```

## Supporting Instruments

Run the ModernBERT robustness check:

```bash
python3 script/analysis_core_modernbert.py
```

Run the static PPMI co-occurrence baseline:

```bash
python3 script/static_ppmi_baseline.py
```

Explore cached embeddings without rerunning the transformer model:

```bash
python3 script/analysis_from_cache.py --outputs analysis_outputs
python3 script/relationship_analysis.py --outputs analysis_outputs
```

Repeat the sampling procedure across random seeds:

```bash
python3 script/sampling_stability.py --model modernbert
```

Regenerate manuscript figures from the saved analysis outputs:

```bash
python3 script/make_figures.py
```

