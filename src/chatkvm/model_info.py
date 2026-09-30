"""Optional live Ollama context information for the chat UI."""

from __future__ import annotations

import json
from urllib.error import URLError
from urllib.request import Request, urlopen

import streamlit as st


def _ollama_api(base_url: str, endpoint: str, payload: dict | None = None) -> dict:
    root = base_url.rstrip('/').removesuffix('/v1')
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(f'{root}/api/{endpoint}', data=data, headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=3) as response:
        return json.load(response)


@st.cache_data(ttl=5, show_spinner=False)
def ollama_context_limits(model: dict) -> tuple[int | None, int | None]:
    """Return active runner window, then model metadata limit; neither implies the other."""
    if model['provider'] != 'ollama':
        return None, None

    runtime = maximum = None
    try:
        running = _ollama_api(model['base_url'], 'ps')
        match = next((item for item in running.get('models', [])
                      if model['name'] in (item.get('name'), item.get('model'))), None)
        if match:
            runtime = match.get('context_length')
    except (OSError, URLError, ValueError, KeyError):
        pass

    try:
        details = _ollama_api(model['base_url'], 'show', {'model': model['name']})
        maximum = details.get('model_info', {}).get(f"{details['details']['family']}.context_length")
    except (OSError, URLError, ValueError, KeyError):
        pass
    return runtime, maximum
