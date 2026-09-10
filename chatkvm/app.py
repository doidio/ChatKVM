import argparse
import tomllib
from pathlib import Path

import httpx
import streamlit as st
from ollama import Client, RequestError, ResponseError

from chatkvm import AwesunMcp

st.set_page_config(
    page_title="ChatKVM", layout="centered", initial_sidebar_state="expanded"
)
st.sidebar.title("ChatKVM")

# 配置

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="config.toml")
args = parser.parse_args()

cfg = tomllib.loads(Path(args.config).read_text())
ollama_vlm = cfg["ollama"]["vlm"]
ollama_host = cfg["ollama"]["host"]

# 全局资源


@st.cache_resource(show_spinner="Connecting AweSun MCP")
def get_awesun_mcp(path: str) -> AwesunMcp:
    return AwesunMcp.connect(path=path)


@st.cache_resource(show_spinner=False)
def get_ollama_client(host: str) -> Client:
    return Client(host=host, timeout=300.0)


awesun_mcp = get_awesun_mcp(args.config)
ollama_client = get_ollama_client(ollama_host)

# 聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，应该做点什么？"}
    ]

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])


# 设备发现
devices = awesun_mcp.search_devices().get("devices") or []

kvms = {
    int(device["remote_id"]): device
    for device in devices
    if device.get("os") == "oraykvm" and device.get("remote_id") is not None
}

if not kvms:
    st.sidebar.warning("设备列表为空")
    st.stop()

labels = {
    remote_id: device.get("name") or device.get("pc_name") or str(remote_id)
    for remote_id, device in kvms.items()
}
remote_id = st.sidebar.radio("设备列表", list(kvms), format_func=labels.__getitem__)

if not kvms[remote_id].get("online"):
    st.sidebar.warning("设备离线")
    st.stop()

# 屏幕快照
with st.sidebar.spinner("正在获取屏幕"):
    session_id = awesun_mcp.ensure_view_session(remote_id)
    screen = awesun_mcp.take_screenshot(session_id)

if screen:
    screen = Path(screen)
    st.sidebar.image(screen, caption=screen.name, width="stretch")
    for path in screen.parent.glob("awesun_*.jpg"):
        if path != screen:
            path.unlink(missing_ok=True)
else:
    st.sidebar.warning("无法获取屏幕")
    st.stop()

# 对话
if not st.session_state.messages or st.sidebar.button("新聊天", width="stretch"):
    del st.session_state.messages
    st.rerun()

prompt = st.chat_input("随心输入")
if prompt:
    with st.spinner("正在获取屏幕"):
        session_id = awesun_mcp.ensure_view_session(remote_id)
        screen = awesun_mcp.take_screenshot(session_id)
        st.session_state.screen = screen
        st.session_state.view_session_id = session_id

    history = st.session_state.messages
    for item in history:
        item.pop("images", None)  # 聊天记录不保存屏幕图像
    history.append({"role": "user", "content": prompt, "images": [screen]})

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("正在思考"):
                reply = st.write_stream(
                    text
                    for chunk in ollama_client.chat(
                        model=ollama_vlm,
                        messages=history,
                        stream=True,
                        think=False,
                        keep_alive="10m",
                    )
                    if (text := chunk.message.content)
                )
        except (RequestError, ResponseError, httpx.HTTPError) as exc:
            try:
                names = [item.model for item in ollama_client.list().models]
                available = ", ".join(names) or "无"
            except (RequestError, ResponseError, httpx.HTTPError):
                available = "无法查询"
            reply = f"调用 Ollama 失败：{exc}\n\n当前服务 {ollama_host} 已有模型：{available}"
            st.error(reply)
    history.append({"role": "assistant", "content": reply or ""})
    st.rerun()
