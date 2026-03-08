"""Phase 2 - Step 2.2: OpenAI GPT-4o-mini wrapper."""

import json
import time
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from models.base import BaseLLM

load_dotenv()

# GPT-4o-mini pricing (per 1M tokens)
INPUT_PRICE_PER_M = 0.15
OUTPUT_PRICE_PER_M = 0.60


class OpenAILLM(BaseLLM):
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = OpenAI()
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency = 0.0
        self.total_cost = 0.0
        self.call_count = 0

    def count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    def complete(self, prompt: str, output_schema: dict, max_retries: int = 5) -> dict:
        start = time.time()
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait = min(2 ** attempt * 5, 60)
                    print(f"    Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError(f"Failed after {max_retries} retries")
        latency = time.time() - start

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = (input_tokens * INPUT_PRICE_PER_M + output_tokens * OUTPUT_PRICE_PER_M) / 1_000_000

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_latency += latency
        self.total_cost += cost
        self.call_count += 1

        try:
            result = json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, IndexError):
            result = {"error": "parse_failed", "raw": response.choices[0].message.content}

        result["_meta"] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency": latency,
            "cost": cost,
        }
        return result

    def get_stats(self) -> dict:
        return {
            "model": self.model,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_latency": self.total_latency,
            "total_cost": self.total_cost,
            "call_count": self.call_count,
        }

    def reset_stats(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency = 0.0
        self.total_cost = 0.0
        self.call_count = 0
