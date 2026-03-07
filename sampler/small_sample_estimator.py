"""
Phase 5 - Step 5.1: Small-sample estimator.
Runs a small number of real LLM calls to estimate selectivity, accuracy, and cost.
"""

import numpy as np
import pandas as pd


class SmallSampleEstimator:
    def __init__(self, sample_size: int = 20):
        self.sample_size = sample_size

    def estimate(self, operator, relation: pd.DataFrame,
                 ground_truth_col: str = None) -> dict:
        """
        Samples `sample_size` tuples, runs them through the LLM,
        and estimates operator statistics.

        Args:
            operator: SemanticOperator instance
            relation: DataFrame with tuples
            ground_truth_col: column name with ground truth labels (for accuracy)

        Returns dict with:
            - estimated_selectivity
            - estimated_accuracy (if ground_truth_col provided)
            - mean_input_tokens
            - mean_output_tokens
            - mean_latency
        """
        n = min(self.sample_size, len(relation))
        sample = relation.sample(n=n, random_state=42)
        tuples = sample.to_dict("records")

        results = []
        input_tokens_list = []
        output_tokens_list = []
        latency_list = []

        for row in tuples:
            res = operator.apply(row)
            results.append(res)
            meta = res.get("_meta", {})
            input_tokens_list.append(meta.get("input_tokens", 0))
            output_tokens_list.append(meta.get("output_tokens", 0))
            latency_list.append(meta.get("latency", 0))

        # Selectivity: fraction that pass the filter
        n_pass = sum(1 for r in results if r["passes_filter"])
        selectivity = n_pass / n if n > 0 else 0.0

        # Accuracy: compare predictions to ground truth
        accuracy = None
        if ground_truth_col and ground_truth_col in sample.columns:
            correct = 0
            for row, res in zip(tuples, results):
                gt = str(row[ground_truth_col]).strip().lower()
                pred = str(res["prediction"]).strip().lower()
                if gt == pred:
                    correct += 1
            accuracy = correct / n if n > 0 else 0.0

        return {
            "operator": operator.name,
            "sample_size": n,
            "estimated_selectivity": selectivity,
            "estimated_accuracy": accuracy,
            "mean_input_tokens": np.mean(input_tokens_list),
            "mean_output_tokens": np.mean(output_tokens_list),
            "mean_latency": np.mean(latency_list),
            "total_sample_tokens": sum(input_tokens_list) + sum(output_tokens_list),
            "total_sample_cost": sum(input_tokens_list) * 0.15 / 1e6 + sum(output_tokens_list) * 0.60 / 1e6,
        }
