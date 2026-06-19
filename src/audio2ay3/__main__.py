"""Enable ``python -m audio2ay3`` as an alias for the ``audio2ay3`` console script."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
