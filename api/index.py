"""Compatibility entry point for launching the Streamlit VornisAI dashboard."""

import runpy
from pathlib import Path


if __name__ == "__main__":
    app_path = Path(__file__).resolve().parents[1] / "ml" / "app.py"
    runpy.run_path(str(app_path), run_name="__main__")
