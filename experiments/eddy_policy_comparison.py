"""
Eddy routing policy comparison.
Runs the eddy with each of the 4 routing policies across all configs
and compares accuracy, convergence speed, and total cost.
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
from eddy.routing_policies import (
    ThompsonSamplingPolicy,
    UCBPolicy,
    EpsilonGreedyPolicy,
    LotteryPolicy,
)
from eddy.router import EddyRouter


def ordering_to_key(ordering):
    return "lang_first" if "lang_filter" in ordering.split(" -> ")[0] else "genre_first"


def run_policy_comparison(config_path: str, n_rows: int = None) -> list:
    """Run all 4 policies on a single config, return list of results."""
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    df = pd.read_csv(config_path)
    if n_rows:
        df = df.head(n_rows)

    n = len(df)
    print(f"\n--- Policy comparison: {config_name} ({n} rows) ---")

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

    policies = [
        ThompsonSamplingPolicy(),
        UCBPolicy(c=1.0),
        EpsilonGreedyPolicy(epsilon_0=0.3, decay_rate=50.0),
        LotteryPolicy(base_tickets=100, update_interval=10),
    ]

    results = []
    for policy in policies:
        llm = get_llm()
        lang_op = make_lang_filter(llm)
        genre_op = make_genre_filter(llm)
        router = EddyRouter([lang_op, genre_op], policy, df)
        _, eddy_stats = router.execute(df, show_progress=False)

        # Dominant ordering
        ordering_counts = {}
        for entry in eddy_stats["routing_log"]:
            o = entry["ordering"]
            ordering_counts[o] = ordering_counts.get(o, 0) + 1
        eddy_dominant = max(ordering_counts, key=ordering_counts.get)
        eddy_pick = ordering_to_key(eddy_dominant)

        conv_idx, _ = router.get_convergence_point(window=10)

        results.append({
            "config": config_name,
            "policy": policy.name,
            "eddy_pick": eddy_pick,
            "actual_best": actual_best,
            "correct": eddy_pick == actual_best,
            "convergence_idx": conv_idx,
            "total_tokens": eddy_stats["total_tokens"],
            "total_cost": eddy_stats["total_cost"],
            "total_latency": eddy_stats["total_latency"],
            "ordering_counts": ordering_counts,
        })

        print(f"  {policy.name:20s}: pick={eddy_pick}, correct={eddy_pick == actual_best}, "
              f"conv={conv_idx}, tokens={eddy_stats['total_tokens']}")

    return results


def run_all_policy_comparisons(config_dir: str = "data/configs",
                                output_dir: str = "results/eddy_policies",
                                n_rows: int = None):
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
                all_results.extend(json.load(f))
            continue

        for retry in range(5):
            try:
                results = run_policy_comparison(config_path, n_rows=n_rows)
                os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
                with open(ckpt_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                all_results.extend(results)
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

    results_df = pd.DataFrame(all_results)

    # Summary by policy
    summary = results_df.groupby("policy").agg(
        accuracy=("correct", "mean"),
        avg_convergence=("convergence_idx", lambda x: x.dropna().mean()),
        avg_tokens=("total_tokens", "mean"),
        avg_cost=("total_cost", "mean"),
    ).reset_index()

    print(f"\n{'='*60}")
    print("POLICY COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(summary.to_string(index=False))

    summary.to_csv(os.path.join(output_dir, "policy_summary.csv"), index=False)
    results_df.to_csv(os.path.join(output_dir, "policy_details.csv"), index=False)
    with open(os.path.join(output_dir, "policy_full_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_all_policy_comparisons(n_rows=n)
