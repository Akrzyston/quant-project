"""Entry point for the Streamlit Codespaces template.

The devcontainer launches this filename. Layout lives in app/main.py.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _path in (str(_ROOT), str(_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Force pandas through Python's own import lock before any panel runs.
# Plotly's isinstance checks fetch it via sys.modules.get() rather than
# import, which skips that lock -- if a concurrent rerun thread is still
# mid-import the first time, that lookup can return a partially
# initialized module and crash with AttributeError.
import pandas  # noqa: E402, F401

from app.main import main  # noqa: E402

main()
