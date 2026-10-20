"""Locating a converted coreference model."""
import os
from pathlib import Path

from .constants import MODEL_DIR_ENV_VAR


def find_model(model_path: str | os.PathLike = None) -> str:
    """Return the directory of a converted (HuggingFace format) coreference model to load.

    Search order: `model_path` (a local directory or a Hugging Face Hub model ID),
    then the directory in the environment variable COREF_MODEL_DIR.
    """
    if model_path is not None:
        path = str(model_path)
        if Path(path).expanduser().exists():
            return str(Path(path).expanduser())
        if path.startswith(("/", "./", "../", "~")):
            # Looks like a file system path, not a Hub ID: fail clearly instead of letting
            # Hugging Face report a confusing "Repo id must be in the form ..." error
            raise FileNotFoundError(f"Model directory not found: {path}")
        return path  # otherwise assumed to be a Hugging Face Hub model ID
    env_dir = os.environ.get(MODEL_DIR_ENV_VAR)
    if env_dir:
        directory = Path(env_dir).expanduser()
        if not (directory / "config.json").is_file():
            raise FileNotFoundError(
                f"${MODEL_DIR_ENV_VAR} is set to '{env_dir}', but no model (config.json) was found there."
            )
        return str(directory)
    raise FileNotFoundError(
        "No coreference model specified. Pass `model_path` (a local directory of a converted model) "
        f"or set the environment variable {MODEL_DIR_ENV_VAR} to the model directory (see the README "
        "for how to obtain and convert the model checkpoint)."
    )
