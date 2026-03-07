"""Phase 3 - Step 3.3: Language filter operator."""

from operators.semantic_selection import SemanticSelection

LANG_PROMPT = (
    'What language is the following movie title written in? '
    'Reply with ONLY a JSON object like {{"language": "English"}}.\n\n'
    'Title: "{input}"'
)


def make_lang_filter(llm) -> SemanticSelection:
    return SemanticSelection(
        name="lang_filter",
        input_attr="title",
        prompt_template=LANG_PROMPT,
        output_key="language",
        filter_value="English",
        llm=llm,
    )
