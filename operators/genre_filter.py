"""Phase 3 - Step 3.3: Genre filter operator."""

from operators.semantic_selection import SemanticSelection

GENRE_PROMPT = (
    'What is the primary genre of this movie based on its plot? '
    'Reply with ONLY a JSON object like {{"genre": "Comedy"}}.\n\n'
    'Plot: "{input}"'
)


def make_genre_filter(llm) -> SemanticSelection:
    return SemanticSelection(
        name="genre_filter",
        input_attr="plot",
        prompt_template=GENRE_PROMPT,
        output_key="genre",
        filter_value="Comedy",
        llm=llm,
    )
