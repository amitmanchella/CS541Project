"""
Eddy non-stationary experiment.
Tests adaptive routing on datasets where selectivity shifts partway through.
Demonstrates that the eddy detects and reacts to distribution shifts,
while fixed-ordering methods (local, fixed-sample) cannot adapt.
"""

import os
import sys
import json
import glob
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.factory import get_llm
from operators.lang_filter import make_lang_filter
from operators.genre_filter import make_genre_filter
from operators.pipeline import QueryPipeline
from optimizer.local_optimizer import find_best_ordering_local
from optimizer.sample_enhanced_optimizer import find_best_ordering_sampled
from eddy.routing_policies import ThompsonSamplingPolicy
from eddy.router import EddyRouter


def ordering_to_key(ordering):
    return "lang_first" if "lang_filter" in ordering.split(" -> ")[0] else "genre_first"


def run_nonstationary(config_path: str, n_rows: int = None) -> dict:
    """Run local, fixed-sample, and eddy on a non-stationary config."""
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    df = pd.read_csv(config_path)
    if n_rows:
        df = df.head(n_rows)

    n = len(df)
    print(f"\n--- Non-stationary: {config_name} ({n} rows) ---")

    # Compute per-segment selectivity to show the shift
    segment_size = max(1, n // 10)
    segment_stats = []
    for i in range(0, n, segment_size):
        seg = df.iloc[i:i + segment_size]
        segment_stats.append({
            "start": i,
            "end": min(i + segment_size, n),
            "lang_sel": (seg["language"].str.lower() == "english").mean(),
            "genre_sel": (seg["genre"].str.lower() == "comedy").mean(),
        })

    for seg in segment_stats:
        print(f"  rows {seg['start']:4d}-{seg['end']:4d}: "
              f"lang={seg['lang_sel']:.2f}, genre={seg['genre_sel']:.2f}")

    # Method 1: Local optimizer (commits to one ordering for all rows)
    llm_local = get_llm()
    tuples = df.to_dict("records")
    lang_op = make_lang_filter(llm_local)
    genre_op = make_genre_filter(llm_local)
    local_result = find_best_ordering_local([lang_op, genre_op], tuples, n)
    local_pick = ordering_to_key(local_result["best"]["ordering"])

    # Execute local's chosen ordering
    if local_pick == "lang_first":
        local_ops = [make_lang_filter(get_llm()), make_genre_filter(get_llm())]
    else:
        local_ops = [make_genre_filter(get_llm()), make_lang_filter(get_llm())]
    local_pipeline = QueryPipeline(local_ops)
    _, local_stats = local_pipeline.execute(df, show_progress=False)

    print(f"  Local pick: {local_pick}, tokens={local_stats['total_tokens']}")

    # Method 2: Fixed-sample (samples from first 20 rows, commits)
    llm_sample = get_llm()
    lang_op_s = make_lang_filter(llm_sample)
    genre_op_s = make_genre_filter(llm_sample)
    sample_result = find_best_ordering_sampled(
        [lang_op_s, genre_op_s], df,
        sample_size=20,
        ground_truth_cols={"lang_filter": "language", "genre_filter": "genre"},
    )
    sample_pick = ordering_to_key(sample_result["best"]["ordering"])

    if sample_pick == "lang_first":
        sample_ops = [make_lang_filter(get_llm()), make_genre_filter(get_llm())]
    else:
        sample_ops = [make_genre_filter(get_llm()), make_lang_filter(get_llm())]
    sample_pipeline = QueryPipeline(sample_ops)
    _, sample_stats = sample_pipeline.execute(df, show_progress=False)

    print(f"  Sample pick: {sample_pick}, tokens={sample_stats['total_tokens']}")

    # Method 3: Eddy (adapts per-tuple)
    llm_eddy = get_llm()
    lang_op_e = make_lang_filter(llm_eddy)
    genre_op_e = make_genre_filter(llm_eddy)
    policy = ThompsonSamplingPolicy()
    router = EddyRouter([lang_op_e, genre_op_e], policy, df)
    _, eddy_stats = router.execute(df, show_progress=False)

    print(f"  Eddy tokens: {eddy_stats['total_tokens']}")

    # Analyze eddy routing over time (per-segment breakdown)
    eddy_segment_routing = []
    for seg in segment_stats:
        seg_entries = [e for e in router.routing_log
                       if seg["start"] <= e["tuple_idx"] < seg["end"]]
        if seg_entries:
            lang_first_count = sum(
                1 for e in seg_entries
                if "lang_filter" in e["ordering"].split(" -> ")[0]
            )
            total = len(seg_entries)
            eddy_segment_routing.append({
                "start": seg["start"],
                "end": seg["end"],
                "lang_first_frac": lang_first_count / total,
                "genre_first_frac": 1 - lang_first_count / total,
            })

    # Also get per-tuple routing for detailed plotting
    per_tuple_routing = [
        {
            "tuple_idx": e["tuple_idx"],
            "ordering": e["ordering"],
            "tokens": e["tokens"],
            "is_lang_first": "lang_filter" in e["ordering"].split(" -> ")[0],
        }
        for e in router.routing_log
    ]

    return {
        "config": config_name,
        "n_rows": n,
        "segment_stats": segment_stats,
        "local_pick": local_pick,
        "local_tokens": local_stats["total_tokens"],
        "local_cost": local_stats["total_cost"],
        "sample_pick": sample_pick,
        "sample_tokens": sample_stats["total_tokens"],
        "sample_cost": sample_stats["total_cost"],
        "eddy_tokens": eddy_stats["total_tokens"],
        "eddy_cost": eddy_stats["total_cost"],
        "eddy_segment_routing": eddy_segment_routing,
        "per_tuple_routing": per_tuple_routing,
        "selectivity_history": eddy_stats.get("selectivity_history", {}),
    }


def run_nonstationary_experiment(config_dir: str = "data/configs",
                                  output_dir: str = "results/eddy_nonstationary",
                                  n_rows: int = None):
    os.makedirs(output_dir, exist_ok=True)

    configs = sorted(glob.glob(os.path.join(config_dir, "nonstat_*.csv")))
    if not configs:
        print(f"No non-stationary config files found in {config_dir}")
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
                result = run_nonstationary(config_path, n_rows=n_rows)
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
    print(f"\n{'='*60}")
    print("NON-STATIONARY SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        print(f"\n{r['config']}:")
        print(f"  Local  ({r['local_pick']:12s}): {r['local_tokens']:6d} tokens, "
              f"${r['local_cost']:.6f}")
        print(f"  Sample ({r['sample_pick']:12s}): {r['sample_tokens']:6d} tokens, "
              f"${r['sample_cost']:.6f}")
        print(f"  Eddy   (adaptive       ): {r['eddy_tokens']:6d} tokens, "
              f"${r['eddy_cost']:.6f}")

    with open(os.path.join(output_dir, "nonstationary_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary CSV
    summary_df = pd.DataFrame([{
        "config": r["config"],
        "local_pick": r["local_pick"],
        "local_tokens": r["local_tokens"],
        "sample_pick": r["sample_pick"],
        "sample_tokens": r["sample_tokens"],
        "eddy_tokens": r["eddy_tokens"],
        "eddy_savings_vs_local": 1 - r["eddy_tokens"] / max(r["local_tokens"], 1),
        "eddy_savings_vs_sample": 1 - r["eddy_tokens"] / max(r["sample_tokens"], 1),
    } for r in all_results])
    summary_df.to_csv(os.path.join(output_dir, "nonstationary_summary.csv"),
                       index=False)

    return all_results


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_nonstationary_experiment(n_rows=n)
