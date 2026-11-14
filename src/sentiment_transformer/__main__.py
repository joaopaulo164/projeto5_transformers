"""Main package execution entrypoint for python -m sentiment_transformer."""

import sys

from sentiment_transformer.cli import main

if __name__ == "__main__":
    sys.exit(main())
