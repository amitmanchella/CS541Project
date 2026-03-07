"""Phase 3 - Step 3.1: Semantic operator base class."""

import time
from tqdm import tqdm


class SemanticOperator:
    def __init__(self, name: str, input_attr: str, prompt_template: str,
                 output_key: str, filter_value: str, llm):
        self.name = name
        self.input_attr = input_attr
        self.prompt_template = prompt_template
        self.output_key = output_key
        self.filter_value = filter_value
        self.llm = llm

    def build_prompt(self, row: dict) -> str:
        return self.prompt_template.format(input=row[self.input_attr])

    def apply(self, row: dict) -> dict:
        prompt = self.build_prompt(row)
        result = self.llm.complete(prompt, {})
        prediction = result.get(self.output_key, "").strip()
        passes = prediction.lower() == self.filter_value.lower()
        return {
            "prediction": prediction,
            "passes_filter": passes,
            "_meta": result.get("_meta", {}),
        }

    def apply_batch(self, tuples: list[dict], batch_size: int = 16,
                    show_progress: bool = True) -> list[dict]:
        results = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_latency = 0.0
        total_cost = 0.0

        iterator = range(0, len(tuples), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"  {self.name}", unit="batch")

        for i in iterator:
            batch = tuples[i:i + batch_size]
            for row in batch:
                res = self.apply(row)
                results.append(res)
                meta = res.get("_meta", {})
                total_input_tokens += meta.get("input_tokens", 0)
                total_output_tokens += meta.get("output_tokens", 0)
                total_latency += meta.get("latency", 0)
                total_cost += meta.get("cost", 0)

        stats = {
            "operator": self.name,
            "tuples_processed": len(tuples),
            "tuples_passed": sum(1 for r in results if r["passes_filter"]),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_latency": total_latency,
            "total_cost": total_cost,
        }
        return results, stats
