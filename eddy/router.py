"""
Core Eddy Router for adaptive semantic operator ordering.

Instead of committing to a fixed operator ordering upfront, the EddyRouter
decides *per-tuple* which ordering to use, based on running statistics
collected during execution.  Every LLM call contributes to both the query
result AND the optimizer's cost/selectivity estimates -- zero wasted calls.

Reference: Avnur & Hellerstein, "Eddies: Continuously Adaptive Query
Processing", SIGMOD 2000.
"""

import time
import pandas as pd
from tqdm import tqdm

from cost_model.local_estimator import estimate_compute_cost
from eddy.routing_policies import RoutingPolicy, UCBPolicy


class EddyRouter:
    """
    Adaptive router that processes tuples through semantic operators,
    choosing the ordering per-tuple via a pluggable routing policy.
    """

    def __init__(self, operators: list, policy: RoutingPolicy,
                 relation: pd.DataFrame):
        """
        Args:
            operators: list of SemanticOperator instances
            policy:    a RoutingPolicy that decides ordering per-tuple
            relation:  the input DataFrame (used for local cost pre-computation)
        """
        self.operators = operators
        self.policy = policy

        # Pre-compute local per-tuple costs (token-based, zero LLM calls)
        tuples = relation.to_dict("records")
        self.local_costs = {}
        for op in operators:
            est = estimate_compute_cost(
                op.prompt_template, op.input_attr, tuples
            )
            self.local_costs[op.name] = {
                "mean_compute_cost": est["mean_compute_cost"],
                "mean_input_tokens": est["mean_input_tokens"],
            }

        # Running statistics per operator (updated during execution)
        self.op_stats = {
            op.name: {
                "passes": 0,
                "fails": 0,
                "total_invocations": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_latency": 0.0,
                "total_cost": 0.0,
                "selectivity_history": [],
            }
            for op in operators
        }

        # Detailed per-tuple log for convergence analysis
        self.routing_log = []

    def execute(self, relation: pd.DataFrame, show_progress: bool = True):
        """
        Process all tuples through the operator pipeline with adaptive routing.

        Returns:
            (result_df, execution_stats)
        """
        tuples = relation.to_dict("records")
        surviving_tuples = []

        total_tokens = 0
        total_latency = 0.0
        total_cost = 0.0

        pipeline_start = time.time()

        iterator = range(len(tuples))
        if show_progress:
            iterator = tqdm(iterator, desc="  eddy", unit="tuple")

        for idx in iterator:
            t = tuples[idx]

            # 1. Ask policy which ordering to use for this tuple
            ordering = self.policy.choose(
                self.operators, self.op_stats, self.local_costs, idx
            )

            # 2. Execute operators in chosen order
            passed_all = True
            tuple_tokens = 0
            tuple_latency = 0.0
            tuple_cost = 0.0

            for op in ordering:
                result = op.apply(t)
                meta = result.get("_meta", {})

                in_tok = meta.get("input_tokens", 0)
                out_tok = meta.get("output_tokens", 0)
                lat = meta.get("latency", 0.0)
                cst = meta.get("cost", 0.0)

                # Update running stats for this operator
                s = self.op_stats[op.name]
                s["total_invocations"] += 1
                s["total_input_tokens"] += in_tok
                s["total_output_tokens"] += out_tok
                s["total_latency"] += lat
                s["total_cost"] += cst

                tuple_tokens += in_tok + out_tok
                tuple_latency += lat
                tuple_cost += cst

                if result["passes_filter"]:
                    s["passes"] += 1
                else:
                    s["fails"] += 1
                    passed_all = False
                    break  # early termination -- skip remaining operators

            # 3. If UCB policy, record observed cost for the chosen ordering
            if isinstance(self.policy, UCBPolicy):
                self.policy.record(ordering, tuple_tokens)

            # 4. Record selectivity snapshot for every operator
            for op in self.operators:
                s = self.op_stats[op.name]
                inv = s["passes"] + s["fails"]
                if inv > 0:
                    s["selectivity_history"].append(s["passes"] / inv)

            # 5. Log this routing decision
            self.routing_log.append({
                "tuple_idx": idx,
                "ordering": " -> ".join(op.name for op in ordering),
                "tokens": tuple_tokens,
                "latency": tuple_latency,
                "cost": tuple_cost,
                "passed_all": passed_all,
            })

            total_tokens += tuple_tokens
            total_latency += tuple_latency
            total_cost += tuple_cost

            if passed_all:
                surviving_tuples.append(t)

        pipeline_time = time.time() - pipeline_start
        result_df = (pd.DataFrame(surviving_tuples)
                     if surviving_tuples else pd.DataFrame())

        # Compute per-operator summary stats
        per_operator = []
        for op in self.operators:
            s = self.op_stats[op.name]
            per_operator.append({
                "operator": op.name,
                "tuples_processed": s["total_invocations"],
                "tuples_passed": s["passes"],
                "total_input_tokens": s["total_input_tokens"],
                "total_output_tokens": s["total_output_tokens"],
                "total_latency": s["total_latency"],
                "total_cost": s["total_cost"],
            })

        execution_stats = {
            "method": "eddy",
            "policy": self.policy.name,
            "total_tokens": total_tokens,
            "total_latency": total_latency,
            "total_cost": total_cost,
            "pipeline_wall_time": pipeline_time,
            "result_count": len(surviving_tuples),
            "per_operator": per_operator,
            "routing_log": self.routing_log,
            "op_stats": {
                name: {k: v for k, v in stats.items()
                       if k != "selectivity_history"}
                for name, stats in self.op_stats.items()
            },
            "selectivity_history": {
                name: stats["selectivity_history"]
                for name, stats in self.op_stats.items()
            },
        }

        return result_df, execution_stats

    def get_convergence_point(self, window: int = 10):
        """
        Find the tuple index where the eddy 'locks in' to a stable ordering.
        Defined as the first point after which the last `window` consecutive
        routing decisions all chose the same ordering.

        Returns:
            (convergence_idx, final_ordering) or (None, None) if never converged
        """
        if len(self.routing_log) < window:
            return None, None

        for i in range(window - 1, len(self.routing_log)):
            recent = self.routing_log[i - window + 1:i + 1]
            orderings = [r["ordering"] for r in recent]
            if len(set(orderings)) == 1:
                return i - window + 1, orderings[0]

        return None, None
