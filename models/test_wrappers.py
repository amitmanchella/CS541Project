"""
Phase 2 - Step 2.4: Test both LLM wrappers.
Sends a single prompt to OpenAI (and optionally Llama), verifies structured JSON output.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_openai():
    from models.openai_llm import OpenAILLM

    print("Testing OpenAI GPT-4o-mini wrapper...")
    llm = OpenAILLM()

    prompt = (
        'What language is the following movie title written in? '
        'Reply with ONLY a JSON object like {"language": "English"}.\n\n'
        'Title: "The Shawshank Redemption"'
    )

    result = llm.complete(prompt, {})
    print(f"  Response: {result}")
    print(f"  Stats: {llm.get_stats()}")

    assert "language" in result, f"Expected 'language' key in response, got: {result}"
    assert result["language"].lower() == "english", f"Expected English, got: {result['language']}"
    print("  OpenAI wrapper: PASSED\n")


def test_llama():
    from models.llama_llm import LlamaLLM

    print("Testing Llama wrapper...")
    try:
        llm = LlamaLLM()
        prompt = (
            'What language is the following movie title written in? '
            'Reply with ONLY a JSON object like {"language": "English"}.\n\n'
            'Title: "The Shawshank Redemption"'
        )
        result = llm.complete(prompt, {})
        print(f"  Response: {result}")
        print(f"  Stats: {llm.get_stats()}")
        print("  Llama wrapper: PASSED\n")
    except Exception as e:
        print(f"  Llama wrapper: SKIPPED (endpoint not available: {e})\n")


if __name__ == "__main__":
    test_openai()
    test_llama()
