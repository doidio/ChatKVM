import argparse
import time
import tomllib
from pathlib import Path

import httpx
import ollama
import streamlit as st

from chatkvm.history import History
from chatkvm.mcp_awesun import McpAwesun

ROLES = {
    "user": {"label": "用户", "avatar": ":material/person:"},
    "assistant": {"label": "智能", "avatar": ":material/smart_toy:"},
    "tool": {"label": "工具", "avatar": ":material/build:"},
    "screen": {"label": "屏幕", "avatar": ":material/desktop_windows:"},
}

st.set_page_config(
    page_title="ChatKVM", layout="centered", initial_sidebar_state="expanded"
)
st.sidebar.title("ChatKVM")

# 配置

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="config.toml")
args = parser.parse_args()

cfg = tomllib.loads(Path(args.config).read_text())
ollama_host = cfg["ollama"]["host"]
ollama_vlm = cfg["ollama"]["vlm"]
coord_max = int(cfg["ollama"].get("coord_max", 1000))

SYSTEM = f"""
你在远程操作一台电脑，你做过的操作都在上文记录保留。
每轮只看最新全屏图。先读画面再行动；标题、菜单、对话框与目标不符就改策略。没有工具时用文字回答，不要假装操作。
坐标是 0 到 {coord_max} 的相对值，原点左上。x 是一个 JSON 整数，y 是另一个 JSON 整数。不要把两个值写进同一个字段，不要用逗号、方括号或 y>。
一次只做一步。调用工具时必须先写一句很短的话说明这一步做什么。单击、双击、右键、拖动、输入怎么用看工具说明。结果以最新截图为准。
目标界面一旦出现就停手，完整抄下需要的信息。
文本回答的语言与用户一致。
"""

# 全局资源


@st.cache_resource(show_spinner="Connecting AweSun MCP")
def get_mcp_awesun(path: str) -> McpAwesun:
    return McpAwesun.connect(path=path)


@st.cache_resource(show_spinner=False)
def get_ollama_client(host: str) -> ollama.Client:
    return ollama.Client(host=host, timeout=300.0)


mcp_awesun = get_mcp_awesun(args.config)
ollama_client = get_ollama_client(ollama_host)
history = History.current()


def _note(item: dict) -> str:
    at = str(item.get("at") or "")
    ms = item.get("ms")
    if ms is None:
        return at
    n = int(ms)
    if n < 1000:
        duration = f"{n}ms"
    else:
        seconds = n / 1000
        duration = f"{seconds:.1f}s" if seconds < 10 else f"{seconds:.0f}s"
    return f"{at} · {duration}" if at else duration


def show(item: dict) -> None:
    role = item.get("role") or ""
    spec = ROLES.get(role) or {}
    note = _note(item)
    label = spec.get("label") or role
    stamp = f"{label} {note}".strip() if note else label
    avatar = spec.get("avatar")
    if role == "tool":
        with st.chat_message("tool", avatar=avatar):
            st.caption(stamp)
            if item.get("content"):
                st.caption(item["content"])
            if item.get("image"):
                st.image(str(history.resolve(item["image"])), width=220)
        return
    with st.chat_message(role, avatar=avatar):
        st.caption(stamp)
        if item.get("content"):
            st.markdown(item["content"])
        for call in item.get("tool_calls") or []:
            function = call["function"]
            st.caption(f":material/build: {function['name']} {function['arguments']}")
        if item.get("image"):
            st.image(str(history.resolve(item["image"])))


# 聊天记录
rows = history.load()
if not rows:
    with st.chat_message("assistant", avatar=ROLES["assistant"]["avatar"]):
        st.markdown("你好，应该做点什么？")
else:
    for item in rows:
        show(item)

# 设备发现
devices = mcp_awesun.search_devices().get("devices") or []

kvms = {
    int(device["remote_id"]): device
    for device in devices
    if device.get("os") == "oraykvm" and device.get("remote_id") is not None
}

if not kvms:
    st.sidebar.caption("⚠️ 无可用设备")
    st.stop()

labels = {
    remote_id: device.get("name") or device.get("pc_name") or str(remote_id)
    for remote_id, device in kvms.items()
}
remote_id = st.sidebar.radio("可用设备", list(kvms), format_func=labels.__getitem__)

if not kvms[remote_id].get("online"):
    st.sidebar.caption("⚠️ 离线")
    st.stop()

st.sidebar.caption(f"📁 {history.folder.as_posix()}")

if st.sidebar.button("新聊天", width="stretch"):
    History.rotate()
    st.rerun()


with st.bottom:
    with st.container(
        horizontal=True,
        horizontal_alignment="center",
        vertical_alignment="center",
        wrap=False,
        gap="small",
    ):
        with_tools = st.checkbox(
            "允许操控",
            True,
            key="with_tools",
            help="自主操控循环上限",
        )
        max_steps = st.radio(
            "操控上限",
            [10, 25, 50],
            key="max_steps_tier",
            disabled=not with_tools,
            horizontal=True,
            label_visibility="collapsed",
        )
    prompt = st.chat_input("随心输入")

try:
    caps = " ".join(ollama_client.show(ollama_vlm).capabilities)
    st.sidebar.caption(f"✅ {ollama_vlm}\n\n{caps}")
except ollama.ResponseError:
    st.sidebar.caption(f"⚠️ {ollama_vlm} 无响应")
    st.stop()

if prompt:
    tools = {tool.__name__: tool for tool in mcp_awesun.agent_tools}
    show(history.append("user", content=prompt))

    with st.spinner("正在连接设备"):
        started = time.monotonic()
        mcp_awesun.use(remote_id)
        screen = mcp_awesun.screenshot()
    image = history.save_image("screen", screen)
    show(history.append("screen", image=image, started=started))

    for _ in range(1 if not with_tools else int(max_steps or 10)):
        messages = history.ollama_messages(SYSTEM)
        started = time.monotonic()
        try:
            with st.spinner("正在思考"):
                payload = {
                    "model": ollama_vlm,
                    "messages": messages,
                    "think": False,
                    "keep_alive": "10m",
                }
                if with_tools:
                    payload["tools"] = mcp_awesun.ollama_tools
                response = ollama_client.chat(**payload)
        except (ollama.RequestError, ollama.ResponseError, httpx.HTTPError) as exc:
            show(
                history.append(
                    "assistant",
                    content=f"调用 Ollama 失败：{exc}",
                    started=started,
                )
            )
            st.error(f"调用 Ollama 失败：{exc}")
            break

        dumped = response.message.model_dump(exclude_none=True)
        calls = dumped.get("tool_calls") or []
        show(
            history.append(
                "assistant",
                content=dumped.get("content") or "",
                started=started,
                tool_calls=calls or None,
            )
        )
        if not with_tools or not calls:
            break

        # 桌面操作必须串行，一步一截图，否则下一步是在旧画面上决策。
        for call in calls:
            name = call["function"]["name"]
            arguments = call["function"].get("arguments") or {}
            started = time.monotonic()
            peek_image = None
            if name in {
                "left_click",
                "right_click",
                "left_double_click",
                "left_drag",
                "scroll",
            }:
                px, py = arguments.get("x"), arguments.get("y")
                if px is not None:
                    try:
                        peek = mcp_awesun.peek(px, py)
                        peek_image = history.save_image("tool", peek)
                        peek.unlink(missing_ok=True)
                    except Exception:
                        pass
            with st.spinner(f"正在执行 {name}"):
                try:
                    result = str(tools[name](**arguments))
                except Exception as exc:
                    result = f"failed: {exc}"
            show(
                history.append(
                    "tool",
                    tool_name=name,
                    content=result,
                    started=started,
                    image=peek_image,
                )
            )

        with st.spinner("正在刷新画面"):
            started = time.monotonic()
            screen = mcp_awesun.screenshot()
        image = history.save_image("screen", screen)
        show(history.append("screen", image=image, started=started))
    st.rerun()
