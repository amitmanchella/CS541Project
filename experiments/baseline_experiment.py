"""
Phase 4 - Step 4.3: Baseline experiment.
Runs both orderings (title->plot, plot->title) on a dataset,
records real tokens/latency/cost, and checks if local optimizer picks correctly.
"""

import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.openai_llm import OpenAILLM
from operators.lang_filter import make_lang_filter
from operators.genre_filter import make_genre_filter
from operators.pipeline import QueryPipeline
from optimizer.local_optimizer import find_best_ordering_local


def run_baseline(config_path: str, output_dir: str = "results/baseline",
                 n_rows: int = None):
    os.makedirs(output_dir, exist_ok=True)
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    df = pd.read_csv(config_path)
    if n_rows:
        df = df.head(n_rows)
    print(f"\n{'='*60}")
    print(f"Baseline Experiment: {config_name} ({len(df)} rows)")
    print(f"{'='*60}")

    tuples = df.to_dict("records")

    # Local optimizer prediction
    llm = OpenAILLM()
    lang_op = make_lang_filter(llm)
    genre_op = make_genre_filter(llm)

    local_result = find_best_ordering_local(
        [lang_op, genre_op], tuples, n_total=len(df)
    )
    print(f"\nLocal optimizer predicts: {local_result['best']['ordering']}")
    for plan in local_result["all_plans"]:
        print(f"  {plan['ordering']}: est. cost = {plan['total_cost']:.1f}")

    # Run both orderings with real LLM calls
    results = {}
    for ordering_name, ops in [
        ("title_then_plot", [make_lang_filter(OpenAILLM()), make_genre_filter(OpenAILLM())]),
        ("plot_then_title", [make_genre_filter(OpenAILLM()), make_lang_filter(OpenAILLM())]),
    ]:
        print(f"\nRunning {ordering_name}...")
        pipeline = QueryPipeline(ops)
        result_df, stats = pipeline.execute(df)
        results[ordering_name] = stats

        print(f"  Result: {stats['result_count']} rows")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Total latency: {stats['total_latency']:.2f}s")
        print(f"  Total cost: ${stats['total_cost']:.6f}")
        for op_stat in stats["per_operator"]:
            print(f"    {op_stat['operator']}: {op_stat['tuples_processed']} tuples, "
                  f"{op_stat['total_input_tokens']+op_stat['total_output_tokens']} tokens")

    # Check if local optimizer was correct
    t_then_p_cost = results["title_then_plot"]["total_tokens"]
    p_then_t_cost = results["plot_then_title"]["total_tokens"]
    actual_best = "title_then_plot" if t_then_p_cost <= p_then_t_cost else "plot_then_title"
    predicted_best = "title_then_plot" if "lang_filter" in local_result["best"]["ordering"].split(" -> ")[0] else "plot_then_title"

    correct = actual_best == predicted_best
    print(f"\nActual best: {actual_best} ({min(t_then_p_cost, p_then_t_cost)} tokens)")
    print(f"Local predicted: {predicted_best}")
    print(f"Correct: {correct}")

    # Save results
    output = {
        "config": config_name,
        "n_rows": len(df),
        "local_prediction": local_result,
        "actual_results": results,
        "actual_best": actual_best,
        "predicted_best": predicted_best,
        "correct": correct,
    }

    with open(os.path.join(output_dir, f"{config_name}.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)

    return output


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "data/configs/lang50_genre50.csv"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    run_baseline(config, n_rows=n)
