"""Convenience launcher for Streamlit Cloud and local demos."""

from pathlib import Path
import runpy


streamlit_app = Path(__file__).resolve().parent / "app" / "streamlit_app.py"
runpy.run_path(streamlit_app, run_name="__main__")
