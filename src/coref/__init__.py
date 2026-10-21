"""Coreference resolution via a seq2seq transition-based mT5 system (Bohnet et al., 2022)."""
from .coreference import (State, create_document, load_coreference_model,
                          predict_coreferences, render_annotated_text, run)
from .paths import find_model
from .tokens import (map_spans_to_tokens, resolve_coreference_tokens,
                     tokens_to_document)

__all__ = [
    "run",
    "load_coreference_model",
    "find_model",
    "resolve_coreference_tokens",
    "tokens_to_document",
    "map_spans_to_tokens",
    "State",
    "create_document",
    "predict_coreferences",
    "render_annotated_text",
]
