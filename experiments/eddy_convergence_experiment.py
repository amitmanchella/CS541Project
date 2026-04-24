"""
Eddy convergence experiment.
Measures how quickly the eddy router locks in to the correct ordering.
Tracks per-tuple routing decisions and selectivity estimates over time.
"""

import os
import sys
import json
import glob
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.factory import get_llm
from operators.lang_filter import make_lang_filter
from operators.genre_filter import make_genre_filter
from operators.pipeline import QueryPipeline
from eddy.routing_policies import ThompsonSamplingPolicy
from eddy.router import EddyRouter


def run_convergence(config_path: str, n_rows: int = None) -> dict:
    """Run eddy on a single config and return detailed convergence data."""
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    df = pd.read_csv(config_path)
    if n_rows:
        df = df.head(n_rows)

    n = len(df)
    print(f"\n--- Convergence: {config_name} ({n} rows) ---")

    # Ground truth
    lang_sel = (df["language"].str.lower() == "english").mean()
    genre_sel = (df["genre"].str.lower() == "comedy").mean()

    # Get actual best by running both fixed orderings
    real_costs = {}
    for name, ops in [
        ("lang_first", [make_lang_filter(get_llm()), make_genre_filter(get_llm())]),
        ("genre_first", [make_genre_filter(get_llm()), make_lang_filter(get_llm())]),
    ]:
        pipeline = QueryPipeline(ops)
        _, stats = pipeline.execute(df, show_progress=False)
        real_costs[name] = stats["total_tokens"]

    actual_best = ("lang_first" if real_costs["lang_first"] <= real_costs["genre_first"]
                   else "genre_first")
    actual_best_ordering = ("lang_filter -> genre_filter" if actual_best == "lang_first"
                            else "genre_filter -> lang_filter")

    print(f"  Actual best: {actual_best} (tokens: lang={real_costs['lang_first']}, genre={real_costs['genre_first']})")

    # Run eddy
    llm = get_llm()
    lang_op = make_lang_filter(llm)
    genre_op = make_genre_filter(llm)
    policy = ThompsonSamplingPolicy()
    router = EddyRouter([lang_op, genre_op], policy, df)
    _, eddy_stats = router.execute(df, show_progress=False)

    # Compute per-tuple correctness
    correct_at = []
    cumulative_correct = 0
    for i, entry in enumerate(router.routing_log):
        is_correct = (entry["ordering"] == actual_best_ordering)
        cumulative_correct += int(is_correct)
        correct_at.append({
            "tuple_idx": i,
            "ordering": entry["ordering"],
            "correct": is_correct,
            "cumulative_accuracy": cumulative_correct / (i + 1),
            "tokens": entry["tokens"],
        })

    conv_idx, conv_ordering = router.get_convergence_point(window=10)
    print(f"  Convergence point: tuple {conv_idx}")
    print(f"  Final accuracy: {correct_at[-1]['cumulative_accuracy']:.1%}")

    return {
        "config": config_name,
        "n_rows": n,
        "lang_selectivity": lang_sel,
        "genre_selectivity": genre_sel,
        "actual_best": actual_best,
        "convergence_idx": conv_idx,
        "convergence_ordering": conv_ordering,
        "final_accuracy": correct_at[-1]["cumulative_accuracy"],
        "per_tuple": correct_at,
        "selectivity_history": eddy_stats.get("selectivity_history", {}),
    }


def run_convergence_experiment(config_dir: str = "data/configs",
                               output_dir: str = "results/eddy_convergence",
                               n_rows: int = None):
    """Run convergence analysis on all configs."""
    os.makedirs(output_dir, exist_ok=True)

    configs = sorted(glob.glob(os.path.join(config_dir, "lang*_genre*.csv")))
    if not configs:
        print(f"No config files found in {config_dir}")
        return

    all_results = []
    for config_path in configs:
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        ckpt_path = os.path.join(output_dir, "checkpoints", f"{config_name}.json")
        if os.path.exists(ckpt_path):
            print(f"\n--- {config_name} (loaded from checkpoint) ---")
            with open(ckpt_path) as f:
                all_results.append(json.load(f))
            continue

        for retry in range(5):
            try:
                result = run_convergence(config_path, n_rows=n_rows)
                os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
                with open(ckpt_path, "w") as f:
                    json.dump(result, f, indent=2, default=str)
                all_results.append(result)
                break
            except Exception as e:
                import time as _t
                wait = min(2 ** retry * 30, 300)
                print(f"  ERROR (attempt {retry+1}/5): {e}")
                if retry < 4:
                    print(f"  Retrying in {wait}s...")
                    _t.sleep(wait)
                else:
                    print(f"  GIVING UP on {config_name} after 5 attempts.")

    if not all_results:
        print("No results collected.")
        return all_results

    # Summary
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "config": r["config"],
            "lang_sel": r["lang_selectivity"],
            "genre_sel": r["genre_selectivity"],
            "actual_best": r["actual_best"],
            "convergence_idx": r["convergence_idx"],
            "final_accuracy": r["final_accuracy"],
        })
    summary_df = pd.DataFrame(summary_rows)

    avg_conv = summary_df["convergence_idx"].dropna().mean()
    avg_acc = summary_df["final_accuracy"].mean()
    print(f"\n{'='*60}")
    print(f"CONVERGENCE SUMMARY")
    print(f"{'='*60}")
    print(f"Avg convergence point: {avg_conv:.1f} tuples")
    print(f"Avg final accuracy:    {avg_acc:.1%}")
    print(f"\n{summary_df.to_string(index=False)}")

    summary_df.to_csv(os.path.join(output_dir, "convergence_summary.csv"), index=False)
    with open(os.path.join(output_dir, "convergence_details.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_convergence_experiment(n_rows=n)
