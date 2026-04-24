"""LLM provider factory. Returns the configured LLM backend."""

import os
from dotenv import load_dotenv
from models.base import BaseLLM

load_dotenv()


def get_llm(provider: str = None, use_cache: bool = True) -> BaseLLM:
    """
    Return an LLM instance based on the provider string or LLM_PROVIDER env var.
    
    Supported providers:
        - "fireworks" (default): Llama 3.3 70B via Fireworks AI
        - "openai": GPT-4o-mini via OpenAI
        - "llama": Llama 3.1-8B via local Ollama

    Args:
        provider: override LLM_PROVIDER env var
        use_cache: if True (default), wrap with disk-backed cache
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "fireworks").lower().strip()

    if provider == "fireworks":
        from models.fireworks_llm import FireworksLLM
        inner = FireworksLLM()
    elif provider == "openai":
        from models.openai_llm import OpenAILLM
        inner = OpenAILLM()
    elif provider == "llama":
        from models.llama_llm import LlamaLLM
        inner = LlamaLLM()
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'fireworks', 'openai', or 'llama'.")

    if use_cache:
        from models.cache import CachedLLM
        return CachedLLM(inner)
    return inner
