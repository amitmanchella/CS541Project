"""
Phase 4 - Step 4.1: Local token-based cost estimator.
Implements Equations 1-5 from the SemOrder paper.
Makes ZERO LLM calls - only runs the tokenizer locally.
"""

import numpy as np
from utils.tokenizer import count_tokens

# GPT-4o-mini pricing per token
INPUT_PRICE_PER_TOKEN = 0.15 / 1_000_000
OUTPUT_PRICE_PER_TOKEN = 0.60 / 1_000_000

# Default constants
ALPHA = 1.0       # weight for output token scaling
BETA = 0.5        # weight between network and compute latency
NET_LATENCY = 0.1 # baseline network latency (seconds)
C_OUTPUT = 5      # constant output tokens for selection operators (bool/enum)


def estimate_input_tokens(prompt_template: str, input_attr: str,
                          tuples: list[dict]) -> list[int]:
    """Tokenize the embedded prompt for each tuple to get per-tuple input token counts."""
    token_counts = []
    for row in tuples:
        prompt = prompt_template.format(input=row[input_attr])
        token_counts.append(count_tokens(prompt))
    return token_counts


def estimate_compute_cost(prompt_template: str, input_attr: str,
                          tuples: list[dict], c_output: int = C_OUTPUT,
                          alpha: float = ALPHA) -> dict:
    """Equation 4: cost_compute = tokens(P, i) + alpha * C"""
    input_tokens = estimate_input_tokens(prompt_template, input_attr, tuples)
    per_tuple_costs = [t + alpha * c_output for t in input_tokens]
    return {
        "mean_input_tokens": np.mean(input_tokens),
        "std_input_tokens": np.std(input_tokens),
        "mean_compute_cost": np.mean(per_tuple_costs),
        "total_compute_cost": np.sum(per_tuple_costs),
        "per_tuple_input_tokens": input_tokens,
    }


def estimate_latency(prompt_template: str, input_attr: str,
                     tuples: list[dict], beta: float = BETA,
                     net: float = NET_LATENCY, c_output: int = C_OUTPUT) -> dict:
    """Equation 2: cost_latency = (1-beta)*net + beta*cost_compute"""
    compute = estimate_compute_cost(prompt_template, input_attr, tuples, c_output)
    mean_latency = (1 - beta) * net + beta * compute["mean_compute_cost"]
    total_latency = mean_latency * len(tuples)
    return {
        "mean_latency_estimate": mean_latency,
        "total_latency_estimate": total_latency,
        **compute,
    }


def estimate_monetary_cost(prompt_template: str, input_attr: str,
                           tuples: list[dict], c_output: int = C_OUTPUT,
                           price_in: float = INPUT_PRICE_PER_TOKEN,
                           price_out: float = OUTPUT_PRICE_PER_TOKEN) -> dict:
    """Equation 5: cost_monetary = $i * tokens(P,i) + $o * C"""
    compute = estimate_compute_cost(prompt_template, input_attr, tuples, c_output)
    mean_monetary = (compute["mean_input_tokens"] * price_in + c_output * price_out)
    total_monetary = mean_monetary * len(tuples)
    return {
        "mean_monetary_cost": mean_monetary,
        "total_monetary_cost": total_monetary,
        **compute,
    }


def estimate_operator_cost(operator, tuples: list[dict], n_tuples: int = None) -> dict:
    """Full cost estimate for a semantic operator on a relation."""
    if n_tuples is None:
        n_tuples = len(tuples)

    compute = estimate_compute_cost(
        operator.prompt_template, operator.input_attr, tuples
    )
    monetary = estimate_monetary_cost(
        operator.prompt_template, operator.input_attr, tuples
    )

    return {
        "operator": operator.name,
        "n_tuples": n_tuples,
        "mean_input_tokens": compute["mean_input_tokens"],
        "mean_compute_cost": compute["mean_compute_cost"],
        "total_compute_cost": compute["mean_compute_cost"] * n_tuples,
        "mean_monetary_cost": monetary["mean_monetary_cost"],
        "total_monetary_cost": monetary["mean_monetary_cost"] * n_tuples,
    }
