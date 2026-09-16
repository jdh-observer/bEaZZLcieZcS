"""
sampling_stability.py
=====================
Repeat the full-corpus reservoir-sampling pipeline across random seeds.

This is designed for robustness reporting: do the top drift terms or strongest
relationship changes persist under different context samples?

Examples:

    python3 sampling_stability.py --model modernbert --seeds 1001 1002 1003
    python3 sampling_stability.py --model gpt2 --seeds 1001 1002 1003

GPT-2 is much slower than ModernBERT on this machine.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import pandas as pd


def run_one(model_name: str, seed: int, max_contexts: int | None) -> Path:
    if model_name == "modernbert":
        module = importlib.import_module("analysis_core_modernbert")
        base_outputs = "analysis_outputs_modernbert"
    elif model_name == "gpt2":
        module = importlib.import_module("analysis_core")
        base_outputs = "analysis_outputs"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    outputs = Path(f"{base_outputs}_seed_{seed}")
    module.CONFIG = {
        **module.CONFIG,
        "sample_seed": seed,
        "outputs_dir": str(outputs),
    }
    module.main(max_contexts)
    return outputs


def summarize_runs(model_name: str, outputs_dirs: list[Path]) -> Path:
    rows = []
    for outputs in outputs_dirs:
        seed = int(outputs.name.rsplit("_", 1)[-1])
        drift = pd.read_csv(outputs / "semantic_drift_all_terms.csv")
        drift["seed"] = seed
        drift["rank"] = drift["drift"].rank(ascending=False,
                                            method="min").astype(int)
        drift["model"] = model_name
        rows.append(drift)

    all_drift = pd.concat(rows, ignore_index=True)
    summary = (
        all_drift.groupby(["model", "term"], as_index=False)
        .agg(
            mean_drift=("drift", "mean"),
            sd_drift=("drift", "std"),
            min_rank=("rank", "min"),
            median_rank=("rank", "median"),
            max_rank=("rank", "max"),
            runs=("seed", "count"),
        )
        .sort_values(["median_rank", "mean_drift"], ascending=[True, False])
    )

    out = Path(f"sampling_stability_{model_name}.csv")
    summary.to_csv(out, index=False)
    all_drift.to_csv(Path(f"sampling_stability_{model_name}_all_runs.csv"),
                     index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["modernbert", "gpt2"],
                        default="modernbert")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[1001, 1002, 1003, 1004, 1005])
    parser.add_argument("--max-contexts", type=int, default=None,
                        help="Optional smoke-test context cap.")
    args = parser.parse_args()

    outputs = []
    for seed in args.seeds:
        print(f"\n=== {args.model} seed {seed} ===")
        outputs.append(run_one(args.model, seed, args.max_contexts))

    summary_path = summarize_runs(args.model, outputs)
    print(f"\nWrote: {summary_path}")
    print(pd.read_csv(summary_path).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
