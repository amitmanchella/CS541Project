"""
Master script to run the full experiment pipeline.
Usage:
    python3 run_all.py [n_rows] [sample_size]

n_rows: number of rows per config to process (default 50, use 1000 for full)
sample_size: LLM calls per operator for sampling (default 20)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def main():
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    print("=" * 60)
    print("CS541 SemOrder - Full Experiment Pipeline")
    print(f"n_rows={n_rows}, sample_size={sample_size}")
    print("=" * 60)

    # Step 1: Build selectivity configs (no LLM calls needed)
    print("\n[1/5] Building selectivity configs...")
    os.chdir(os.path.join(os.path.dirname(__file__), "data"))
    from data.build_selectivity_configs import build_all_configs
    os.chdir(os.path.dirname(__file__))
    build_all_configs()

    # Step 2: Generate token distribution figure (no LLM calls)
    print("\n[2/5] Generating token distribution figure...")
    from experiments.plot_results import figure1_token_distribution
    figure1_token_distribution()

    # Step 3: Run comparison experiment
    print("\n[3/5] Running comparison experiment across all configs...")
    from experiments.comparison_experiment import run_all_configs
    run_all_configs(n_rows=n_rows, sample_size=sample_size)

    # Step 4: Run ablation
    print("\n[4/5] Running ablation experiment...")
    from experiments.ablation_experiment import run_ablation
    run_ablation(n_rows=n_rows, sample_sizes=[5, 10, 20])

    # Step 5: Generate all figures
    print("\n[5/5] Generating all figures...")
    from experiments.plot_results import generate_all
    generate_all()

    print("\n" + "=" * 60)
    print("DONE! Check results/ directory for outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
