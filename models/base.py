"""Phase 2 - Step 2.1: Base LLM interface."""


class BaseLLM:
    def complete(self, prompt: str, output_schema: dict) -> dict:
        """Returns structured output as dict."""
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError
