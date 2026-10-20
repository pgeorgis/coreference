"""Command-line interface.

Example usage:

```
python -m coref --text "Der Hund jagt die Katze. Er bellt."
```
"""
from .coreference import main

if __name__ == "__main__":
    main()
