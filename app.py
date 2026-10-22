"""Run locally: python -m streamlit run app.py."""

from pathlib import Path

from qualitypilot.ui.app import main

main(Path(__file__).resolve().parent)
