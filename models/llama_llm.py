"""Phase 2 - Step 2.3: Llama 3.1-8B wrapper (via OpenAI-compatible API)."""

import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from models.base import BaseLLM

load_dotenv()

LLAMA_ENDPOINT = os.getenv("LLAMA_ENDPOINT", "http://localhost:11434/v1")


class LlamaLLM(BaseLLM):
    def __init__(self, model: str = "llama3.1:8b", endpoint: str = None):
        self.model = model
        self.endpoint = endpoint or LLAMA_ENDPOINT
        self.client = OpenAI(base_url=self.endpoint, api_key="not-needed")
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency = 0.0
        self.call_count = 0

        # Try loading HF tokenizer for accurate local counting
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.1-8B", trust_remote_code=True
            )
        except Exception:
            import tiktoken
            self.tokenizer = None
            self._enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(self._enc.encode(text))

    def complete(self, prompt: str, output_schema: dict) -> dict:
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        latency = time.time() - start

        input_tokens = response.usage.prompt_tokens if response.usage else self.count_tokens(prompt)
        output_tokens = response.usage.completion_tokens if response.usage else 0

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_latency += latency
        self.call_count += 1

        try:
            result = json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, IndexError):
            result = {"error": "parse_failed", "raw": response.choices[0].message.content}

        result["_meta"] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency": latency,
            "cost": 0.0,  # local model = free
        }
        return result

    def get_stats(self) -> dict:
        return {
            "model": self.model,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_latency": self.total_latency,
            "total_cost": 0.0,
            "call_count": self.call_count,
        }

    def reset_stats(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency = 0.0
        self.call_count = 0
