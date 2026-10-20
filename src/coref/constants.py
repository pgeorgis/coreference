"""Constants and defaults for coreference resolution."""

# Hugging Face Hub ID of the mT5 model whose (SentencePiece) tokenizer the coreference model uses
MT5_TOKENIZER = "google/mt5-xxl"

# Environment variable that may point to the directory of a converted coreference model
MODEL_DIR_ENV_VAR = "COREF_MODEL_DIR"
