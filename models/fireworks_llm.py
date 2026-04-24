"""LLM wrapper via Fireworks AI API."""

import json
import os
import time
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from models.base import BaseLLM

load_dotenv()

# Fireworks API configuration
FIREWORKS_API_BASE = "https://api.fireworks.ai/inference/v1"
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
FIREWORKS_MODEL = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/deepseek-v3p2")

# Fireworks pricing (per 1M tokens) - Llama 3.3 70B Instruct
INPUT_PRICE_PER_M = 0.90
OUTPUT_PRICE_PER_M = 0.90


class FireworksLLM(BaseLLM):
    def __init__(self, model: str = None):
        self.model = model or FIREWORKS_MODEL
        self.client = OpenAI(base_url=FIREWORKS_API_BASE, api_key=FIREWORKS_API_KEY)
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency = 0.0
        self.total_cost = 0.0
        self.call_count = 0

    def count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    def complete(self, prompt: str, output_schema: dict, max_retries: int = 10) -> dict:
        RETRYABLE = ["rate_limit", "429", "503", "overloaded", "no healthy upstream",
                      "connection", "reset before headers", "timed out", "timeout",
                      "500", "502", "504"]
        for attempt in range(max_retries):
            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                break
            except Exception as e:
                err_str = str(e).lower()
                if any(keyword in err_str for keyword in RETRYABLE):
                    wait = min(2 ** attempt * 5, 120)
                    print(f"    Retryable error, waiting {wait}s (attempt {attempt+1}/{max_retries}): {str(e)[:80]}")
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
