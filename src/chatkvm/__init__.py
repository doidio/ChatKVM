from pathlib import Path

import streamlit as st


def main() -> None:
    st.App(Path(__file__).resolve().with_name('app.py')).run()
