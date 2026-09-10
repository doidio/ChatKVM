from __future__ import annotations

from pathlib import Path

import streamlit as st


def main() -> None:
    st.App(Path(__file__).resolve().with_name("app.py")).run()


if __name__ == "__main__":
    main()
