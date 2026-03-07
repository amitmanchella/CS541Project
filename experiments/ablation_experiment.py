"""
Phase 6 - Step 6.1: Sample budget ablation.
Varies sample_size in {5, 10, 20} and measures ordering accuracy.
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


def run_ablation(config_dir: str = "data/configs",
                 output_dir: str = "results/ablation",
                 sample_sizes: list = None,
                 n_rows: int = None):
    if sample_sizes is None:
        sample_sizes = [5, 10, 20]

    os.makedirs(output_dir, exist_ok=True)
    configs = sorted(glob.glob(os.path.join(config_dir, "lang*_genre*.csv")))

    all_results = []

    for sample_size in sample_sizes:
        print(f"\n{'='*60}")
        print(f"Sample size: {sample_size}")
        print(f"{'='*60}")

        correct_count = 0
        total_count = 0

        for config_path in configs:
            config_name = os.path.splitext(os.path.basename(config_path))[0]
            df = pd.read_csv(config_path)
            if n_rows:
                df = df.head(n_rows)

            tuples = df.to_dict("records")

            # Get actual best by running both orderings
            real_costs = {}
            for name, ops in [
                ("lang_first", [make_lang_filter(OpenAILLM()), make_genre_filter(OpenAILLM())]),
                ("genre_first", [make_genre_filter(OpenAILLM()), make_lang_filter(OpenAILLM())]),
            ]:
                pipeline = QueryPipeline(ops)
                _, stats = pipeline.execute(df, show_progress=False)
                real_costs[name] = stats["total_tokens"]

            actual_best = "lang_first" if real_costs["lang_first"] <= real_costs["genre_first"] else "genre_first"

            # Sample-enhanced prediction
            llm = OpenAILLM()
            lang_op = make_lang_filter(llm)
            genre_op = make_genre_filter(llm)
            result = find_best_ordering_sampled(
                [lang_op, genre_op], df,
                sample_size=sample_size,
                ground_truth_cols={"lang_filter": "language", "genre_filter": "genre"},
            )
            pick = "lang_first" if "lang_filter" in result["best"]["ordering"].split(" -> ")[0] else "genre_first"

            is_correct = pick == actual_best
            correct_count += int(is_correct)
            total_count += 1

            all_results.append({
                "sample_size": sample_size,
                "config": config_name,
                "predicted": pick,
                "actual": actual_best,
                "correct": is_correct,
                "real_costs": real_costs,
            })

            print(f"  {config_name}: predicted={pick}, actual={actual_best}, correct={is_correct}")

        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"  Accuracy: {accuracy:.1%} ({correct_count}/{total_count})")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, "ablation_results.csv"), index=False)

    # Summary by sample size
    summary = results_df.groupby("sample_size")["correct"].mean().reset_index()
    summary.columns = ["sample_size", "accuracy"]
    summary.to_csv(os.path.join(output_dir, "ablation_summary.csv"), index=False)
    print(f"\nSummary:\n{summary.to_string(index=False)}")

    return all_results


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_ablation(n_rows=n)
