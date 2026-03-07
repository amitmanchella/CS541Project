import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def tokenize(text: str) -> list[int]:
    return _enc.encode(text)
