"""
Phase 5 - Step 5.4: Full comparison experiment.
Compares three methods across all 25 selectivity configs:
  1. Local only (paper baseline)
  2. Small-sample (our contribution)
  3. Oracle (ground truth)
"""

import os
import sys
import json
import glob
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.openai_llm import OpenAILLM
from operators.lang_filter import make_lang_filter
from operators.genre_filter import make_genre_filter
from operators.pipeline import QueryPipeline
from optimizer.local_optimizer import find_best_ordering_local
from optimizer.sample_enhanced_optimizer import (
    find_best_ordering_sampled,
    find_best_ordering_oracle,
)


def run_comparison(config_path: str, sample_size: int = 20,
                   n_rows: int = None) -> dict:
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    df = pd.read_csv(config_path)
    if n_rows:
        df = df.head(n_rows)

    tuples = df.to_dict("records")
    n = len(df)

    print(f"\n--- {config_name} ({n} rows) ---")

    # Ground truth selectivity
    lang_sel = (df["language"].str.lower() == "english").mean()
    genre_sel = (df["genre"].str.lower() == "comedy").mean()
    print(f"  True selectivity: lang={lang_sel:.2f}, genre={genre_sel:.2f}")

    # Method 1: Local only
    llm_local = OpenAILLM()
    lang_op = make_lang_filter(llm_local)
    genre_op = make_genre_filter(llm_local)
    local_result = find_best_ordering_local([lang_op, genre_op], tuples, n)
    local_pick = local_result["best"]["ordering"]

    # Method 2: Small-sample
    llm_sample = OpenAILLM()
    lang_op_s = make_lang_filter(llm_sample)
    genre_op_s = make_genre_filter(llm_sample)
    sample_result = find_best_ordering_sampled(
        [lang_op_s, genre_op_s], df,
        sample_size=sample_size,
        ground_truth_cols={"lang_filter": "language", "genre_filter": "genre"},
    )
    sample_pick = sample_result["best"]["ordering"]
    sample_overhead = sample_result["total_sample_cost"]

    # Method 3: Oracle
    llm_oracle = OpenAILLM()
    lang_op_o = make_lang_filter(llm_oracle)
    genre_op_o = make_genre_filter(llm_oracle)
    oracle_result = find_best_ordering_oracle([lang_op_o, genre_op_o], df)
    oracle_pick = oracle_result["best"]["ordering"]

    print(f"  Local pick:  {local_pick}")
    print(f"  Sample pick: {sample_pick}")
    print(f"  Oracle pick: {oracle_pick}")

    # Run actual execution for both orderings to get real costs
    real_results = {}
    for name, ops in [
        ("lang_first", [make_lang_filter(OpenAILLM()), make_genre_filter(OpenAILLM())]),
        ("genre_first", [make_genre_filter(OpenAILLM()), make_lang_filter(OpenAILLM())]),
    ]:
        pipeline = QueryPipeline(ops)
        _, stats = pipeline.execute(df, show_progress=False)
        real_results[name] = stats
        print(f"  Real {name}: tokens={stats['total_tokens']}, "
              f"latency={stats['total_latency']:.2f}s, cost=${stats['total_cost']:.6f}")

    # Determine actual best
    actual_best = "lang_first" if real_results["lang_first"]["total_tokens"] <= real_results["genre_first"]["total_tokens"] else "genre_first"

    def ordering_to_key(ordering):
        return "lang_first" if "lang_filter" in ordering.split(" -> ")[0] else "genre_first"

    return {
        "config": config_name,
        "lang_selectivity": lang_sel,
        "genre_selectivity": genre_sel,
        "local_pick": ordering_to_key(local_pick),
        "sample_pick": ordering_to_key(sample_pick),
        "oracle_pick": ordering_to_key(oracle_pick),
        "actual_best": actual_best,
        "local_correct": ordering_to_key(local_pick) == actual_best,
        "sample_correct": ordering_to_key(sample_pick) == actual_best,
        "oracle_correct": ordering_to_key(oracle_pick) == actual_best,
        "sample_overhead_cost": sample_overhead,
        "real_results": real_results,
        "sample_stats": sample_result.get("sample_stats", {}),
    }


def run_all_configs(config_dir: str = "data/configs",
                    output_dir: str = "results/comparison",
                    sample_size: int = 20, n_rows: int = None):
    os.makedirs(output_dir, exist_ok=True)

    configs = sorted(glob.glob(os.path.join(config_dir, "lang*_genre*.csv")))
    if not configs:
        print(f"No config files found in {config_dir}")
        return

    all_results = []
    for config_path in configs:
        try:
            result = run_comparison(config_path, sample_size=sample_size,
                                    n_rows=n_rows)
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    summary_df = pd.DataFrame([{
        "config": r["config"],
        "lang_sel": r["lang_selectivity"],
        "genre_sel": r["genre_selectivity"],
        "local_correct": r["local_correct"],
        "sample_correct": r["sample_correct"],
        "oracle_correct": r["oracle_correct"],
        "actual_best": r["actual_best"],
    } for r in all_results])

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Local accuracy:  {summary_df['local_correct'].mean():.1%}")
    print(f"Sample accuracy: {summary_df['sample_correct'].mean():.1%}")
    print(f"Oracle accuracy: {summary_df['oracle_correct'].mean():.1%}")
    print(f"\n{summary_df.to_string(index=False)}")

    summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    with open(os.path.join(output_dir, "full_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    sample_sz = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    run_all_configs(n_rows=n, sample_size=sample_sz)
