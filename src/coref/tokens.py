"""Coreference resolution on pre-tokenized text, handling blank/newline/whitespace tokens in order to keep token indices and word positions in sync."""
from .coreference import run


def tokens_to_document(sentences: list[list[str]]) -> tuple[str, list[int]]:
    """Convert sentences of tokens into the input text of `run` (with `is_tokenized=True`).

    Blank tokens are omitted and whitespace within a token is replaced by "_". Empty sentences are skipped.

    Args:
        sentences: Tokens per sentence.

    Returns:
        The text (one sentence per line, words separated by spaces) and `token_indices`, where `token_indices[p]` is
        the index, in the flat sequence of all tokens of all sentences, of the word at position p of the text.
    """
    lines = []
    token_indices = []
    flat_index = 0
    for sentence in sentences:
        words = []
        for token in sentence:
            word = "_".join(token.split())
            if word:
                words.append(word)
                token_indices.append(flat_index)
            flat_index += 1
        if words:
            lines.append(" ".join(words))
    return "\n".join(lines), token_indices


def map_spans_to_tokens(spans: dict, token_indices: list[int]) -> dict:
    """Map mention spans given as word positions in a document text to token indices.

    Args:
        spans: {entity: [(start, end), ...]} with inclusive word positions, as returned by `coreference.run`.
        token_indices: Mapping of word positions to token indices, as returned by `tokens_to_document`.
    """
    return {
        entity: [(token_indices[start], token_indices[end]) for start, end in mention_spans]
        for entity, mention_spans in spans.items()
    }


def resolve_coreference_tokens(sentences: list[list[str]], model, **run_kwargs) -> dict:
    """Run coreference resolution on one tokenized document.

    Args:
        sentences: Tokens per sentence.
        model: Loaded coreference model (see `load_coreference_model`).
        **run_kwargs: Further arguments of `coreference.run`, e.g. `batch_size`.

    Returns:
        {"text": text with mentions in [N ... ] brackets (without blank tokens),
         "spans": {entity: [(start, end), ...]} with inclusive indices into the flat sequence of all input tokens}
    """
    text, token_indices = tokens_to_document(sentences)
    if not text:
        null_result = {
            "text": "",
            "spans": {},
        }
        return null_result
    result = run(
        documents={"document": text},
        model=model,
        is_tokenized=True,
        **run_kwargs
    )["document"]
    return {
        "text": result["text"],
        "spans": map_spans_to_tokens(result["spans"], token_indices),
    }
