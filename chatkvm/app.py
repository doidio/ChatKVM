import argparse
import time
import tomllib
from pathlib import Path
from typing import Any, NamedTuple

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
vlm = cfg["ollama"]["vlm"]
vlm_model = vlm["model"]
vlm_host = vlm["host"]
vlm_coord_max = int(vlm.get("coord_max", 1000))

SYSTEM = f"""
你在远程操作一台电脑，你做过的操作都在上文记录保留。
每轮只看最新全屏图。先读画面再行动；标题、菜单、对话框与目标不符就改策略。没有工具时用文字回答，不要假装操作。
坐标是 0 到 {vlm_coord_max} 的相对值，原点左上。x 是一个 JSON 整数，y 是另一个 JSON 整数。不要把两个值写进同一个字段，不要用逗号、方括号或 y>。
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
ollama_client = get_ollama_client(vlm_host)
history = History.current()

_PEEK = frozenset(
    {"left_click", "right_click", "left_double_click", "left_drag", "scroll"}
)


class Tokens(NamedTuple):
    prompt: int | None = None
    completion: int | None = None
    window: int | None = None

    def record(self) -> dict[str, int]:
        out: dict[str, int] = {}
        if self.prompt is not None:
            out["prompt_eval_count"] = self.prompt
        if self.completion is not None:
            out["eval_count"] = self.completion
        if self.window:
            out["num_ctx"] = self.window
        return out


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fmt_tokens(n: int) -> str:
    n = int(n)
    if n < 1024:
        return str(n)
    k = n / 1024
    if k < 10:
        return f"{k:.1f}".rstrip("0").rstrip(".") + "K"
    if k < 1024:
        return f"{k:.0f}K"
    m = n / (1024 * 1024)
    return f"{m:.1f}".rstrip("0").rstrip(".") + "M"


def _fmt_ms(ms: int) -> str:
    if ms < 1000:
        return f"{ms}ms"
    seconds = ms / 1000
    return f"{seconds:.1f}s" if seconds < 10 else f"{seconds:.0f}s"


def _steps_of(item: dict) -> str:
    step, cap = item.get("step"), item.get("max_steps")
    if step is None or cap is None:
        return ""
    return f"{int(step)}/{int(cap)}"


def _tokens_of(data: Any) -> Tokens:
    if isinstance(data, dict):
        return Tokens(
            _int(data.get("prompt_eval_count")),
            _int(data.get("eval_count")),
            _int(data.get("num_ctx")),
        )
    return Tokens(_int(data.prompt_eval_count), _int(data.eval_count), None)


def _last_tokens(rows: list[dict]) -> Tokens:
    for item in reversed(rows):
        tokens = _tokens_of(item)
        if tokens.prompt is not None:
            return tokens
    return Tokens()


def _context_window(shown: ollama.ShowResponse | None = None) -> int | None:
    try:
        for item in ollama_client.ps().models or []:
            name = item.model or item.name
            if name == vlm_model and item.context_length:
                return int(item.context_length)
    except Exception:
        pass
    info = (shown.modelinfo if shown is not None else None) or {}
    for key, value in info.items():
        if str(key).endswith(".context_length"):
            parsed = _int(value)
            if parsed:
                return parsed
    return None


def _show_steps(box, used: int) -> None:
    box.caption(f"{used} /", width=30, text_alignment="right")


def _show_tokens(box, tokens: Tokens) -> None:
    if not tokens.window:
        box.caption("上下文 —", width="content")
        return
    used = int(tokens.prompt or 0)
    pct = 100 * used / tokens.window
    color = "red" if pct >= 90 else "orange" if pct >= 70 else "green"
    tips = [f"提示 {used}", f"窗口 {int(tokens.window)}"]
    if tokens.completion is not None:
        tips.insert(1, f"生成 {int(tokens.completion)}")
    box.caption(f":{color}[**{pct:.2f}%**]", width=80, help=" · ".join(tips))


def _note(item: dict) -> str:
    parts: list[str] = []
    steps = _steps_of(item)
    if steps:
        parts.append(steps)
    tokens = _tokens_of(item)
    if tokens.prompt is not None and tokens.window:
        parts.append(f"{_fmt_tokens(tokens.prompt)}/{_fmt_tokens(tokens.window)}")
    elif tokens.prompt is not None:
        parts.append(_fmt_tokens(tokens.prompt))
    at = str(item.get("at") or "")
    if at:
        parts.append(at)
    ms = item.get("ms")
    if ms is not None:
        parts.append(_fmt_ms(int(ms)))
    return " · ".join(parts)


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

window = None
try:
    shown = ollama_client.show(vlm_model)
    caps = " ".join(shown.capabilities or [])
    st.sidebar.caption(f"✅ {vlm_model}\n\n{caps}")
    window = _context_window(shown)
except ollama.ResponseError:
    st.sidebar.caption(f"⚠️ {vlm_model} 无响应")
    st.stop()

tokens = _last_tokens(rows)
if window is None:
    window = tokens.window
tokens = tokens._replace(window=window)

with st.bottom:
    with st.container(
        horizontal=True,
        horizontal_alignment="center",
        vertical_alignment="center",
        gap="small",
    ):
        with_tools = st.checkbox(
            "允许操控",
            True,
            key="with_tools",
            help="自主操控循环上限",
        )
        quota_box = st.empty()
        max_steps = st.selectbox(
            "操控上限",
            [10, 25, 50, 100, 250, 500],
            key="max_steps_tier",
            disabled=not with_tools,
            label_visibility="collapsed",
            width=80,
        )
        _show_steps(quota_box, 0)
        ctx_box = st.empty()
        _show_tokens(ctx_box, tokens)
    prompt = st.chat_input("随心输入")

if prompt:
    tools = {tool.__name__: tool for tool in mcp_awesun.agent_tools}
    limit = 1 if not with_tools else int(max_steps or 10)
    if with_tools:
        _show_steps(quota_box, 0)
    show(history.append("user", content=prompt))

    with st.spinner("正在连接设备"):
        started = time.monotonic()
        mcp_awesun.use(remote_id)
        screen = mcp_awesun.screenshot()
    image = history.save_image("screen", screen)
    show(history.append("screen", image=image, started=started))

    for step in range(1, limit + 1):
        if with_tools:
            _show_steps(quota_box, step)
        messages = history.ollama_messages(SYSTEM)
        started = time.monotonic()
        thinking = f"正在思考 {step}/{limit}" if with_tools else "正在思考"
        try:
            with st.spinner(thinking):
                payload = {
                    "model": vlm_model,
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
        tokens = _tokens_of(response)._replace(window=window)
        extra = tokens.record()
        if with_tools and calls:
            extra["step"] = step
            extra["max_steps"] = limit
        _show_tokens(ctx_box, tokens)
        show(
            history.append(
                "assistant",
                content=dumped.get("content") or "",
                started=started,
                tool_calls=calls or None,
                **extra,
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
            if name in _PEEK:
                px, py = arguments.get("x"), arguments.get("y")
                if px is not None:
                    try:
                        peek = mcp_awesun.peek(px, py)
                        peek_image = history.save_image("tool", peek)
                        peek.unlink(missing_ok=True)
                    except Exception:
                        pass
            with st.spinner(f"正在执行 {name} {step}/{limit}"):
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
                    step=step,
                    max_steps=limit,
                )
            )

        with st.spinner("正在刷新画面"):
            started = time.monotonic()
            screen = mcp_awesun.screenshot()
        image = history.save_image("screen", screen)
        show(history.append("screen", image=image, started=started))
    st.rerun()
