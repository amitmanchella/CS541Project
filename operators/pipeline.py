"""Phase 3 - Step 3.4: Query pipeline that executes operators in order."""

import pandas as pd
import time


class QueryPipeline:
    def __init__(self, operators: list):
        self.operators = operators

    def execute(self, relation: pd.DataFrame, batch_size: int = 16,
                show_progress: bool = True) -> tuple:
        """
        Execute operators in sequence. Each operator filters tuples;
        only passing tuples proceed to the next operator.

        Returns: (result_df, execution_stats)
        """
        tuples = relation.to_dict("records")
        all_stats = []
        total_tokens = 0
        total_latency = 0.0
        total_cost = 0.0

        pipeline_start = time.time()

        for op in self.operators:
            if len(tuples) == 0:
                all_stats.append({
                    "operator": op.name,
                    "tuples_processed": 0,
                    "tuples_passed": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_latency": 0,
                    "total_cost": 0,
                })
                continue

            results, stats = op.apply_batch(tuples, batch_size=batch_size,
                                            show_progress=show_progress)
            all_stats.append(stats)
            total_tokens += stats["total_input_tokens"] + stats["total_output_tokens"]
            total_latency += stats["total_latency"]
            total_cost += stats["total_cost"]

            # Filter: keep only tuples that passed
            tuples = [t for t, r in zip(tuples, results) if r["passes_filter"]]

        pipeline_time = time.time() - pipeline_start
        result_df = pd.DataFrame(tuples) if tuples else pd.DataFrame()

        execution_stats = {
            "ordering": " -> ".join(op.name for op in self.operators),
            "total_tokens": total_tokens,
            "total_latency": total_latency,
            "total_cost": total_cost,
            "pipeline_wall_time": pipeline_time,
            "result_count": len(tuples),
            "per_operator": all_stats,
            "tuples_per_stage": [s["tuples_processed"] for s in all_stats],
        }
        return result_df, execution_stats
