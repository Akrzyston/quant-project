"""Entry point for the Streamlit Codespaces template.

The devcontainer launches this filename. Layout lives in app/main.py.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _path in (str(_ROOT), str(_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from app.main import main  # noqa: E402

main()
