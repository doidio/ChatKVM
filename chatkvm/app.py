import argparse

import streamlit as st

from chatkvm import AwesunMcp

st.set_page_config(page_title="ChatKVM", layout="wide")

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="config.toml")
args = parser.parse_args()


@st.cache_resource(show_spinner="Connecting AweSun MCP")
def get_awesun_mcp(path: str) -> AwesunMcp:
    return AwesunMcp.connect(path)


mcp = get_awesun_mcp(args.config)

st.metric(
    label=f"{mcp.server_name} {mcp.server_version}",
    value="AweSun MCP",
    delta=f"{len(mcp.tools)} tools",
    delta_description=f"protocol {mcp.protocol_version}",
    delta_arrow="off",
)
