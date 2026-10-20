"""Optional Hugging Face access token from a .env file."""
import os

from dotenv import find_dotenv, load_dotenv


def load_hf_token() -> bool:
    """Read HF_ACCESS_TOKEN from the environment or a .env file in the working directory.

    The token is exposed to `huggingface_hub` for this process only, as HF_TOKEN
    (an HF_TOKEN that is already set takes precedence). Nothing is written to disk.
    Returns True if a token is in use.
    """
    load_dotenv(find_dotenv(usecwd=True))  # search from the working directory, not from this package
    token = os.getenv("HF_ACCESS_TOKEN")
    if token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = token
    return bool(os.environ.get("HF_TOKEN"))
