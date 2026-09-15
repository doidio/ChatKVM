"""
向日葵 MCP 同步封装

MCP `Client` 必须停在 `async with` 里，stdio 子进程才不会被关掉。
Streamlit 脚本是同步瀑布流，因此在后台线程跑一条常驻事件循环，
主线程用 `run()` 把协程提交进去并阻塞取结果。

`use()` 绑定一台设备后，`agent_tools` 里的方法可以直接交给 Ollama 当 tools：
签名里只留模型该填的参数，会话与坐标换算都藏在实例里，返回值是短字符串。
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import sys
import threading
import time
import tomllib
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from mcp.client import Client, ClientRequestContext, IncomingMessage
from mcp.client.stdio import StdioServerParameters
from mcp.types import (
    CallToolResult,
    ElicitRequestParams,
    ElicitResult,
    Implementation,
    LoggingMessageNotificationParams,
    Tool,
)
from PIL import Image, ImageDraw

from chatkvm import __version__

T = TypeVar("T")
TIMEOUT = 60.0
THUMBNAIL = (48, 48)
SETTLE_MS = 500
_HID_KEY_ALIAS = {
    "esc": "ESCAPE",
    "escape": "ESCAPE",
    "backspace": "BACK",
    "back": "BACK",
    "return": "ENTER",
    "del": "DELETE",
    "spacebar": "SPACE",
    "ctrl": "control",
}

__all__ = ["McpAwesun", "ollama_tool_defs"]


# --- MCP 回调：stdio 噪音过滤 ---


async def _on_log(params: LoggingMessageNotificationParams) -> None:
    print(f"[mcp-log {params.level}] {params.data}", file=sys.stderr)


async def _on_transport_message(message: IncomingMessage) -> None:
    # [AweSun] 启动时往 stdout 打非 JSON 横幅，MCP 只能走 JSON-RPC，会变成传输异常。
    if isinstance(message, Exception):
        text = str(message)
        if "AweSun MCP Server" in text or "transport=stdio" in text:
            print("[awesun] AweSun MCP Server", file=sys.stderr)
            return
        print(f"[mcp-transport] {message}", file=sys.stderr)


async def _on_elicit(
    context: ClientRequestContext,
    params: ElicitRequestParams,
) -> ElicitResult:
    del context
    print(f"[elicitation] {params.message}", file=sys.stderr)
    return ElicitResult(action="decline")


# --- tools/call 结果 ---


def _tool_payload(result: CallToolResult) -> Any:
    texts = [text for block in result.content if (text := getattr(block, "text", None))]
    if result.is_error:
        raise RuntimeError("; ".join(texts) or "MCP tool error")
    if result.structured_content is not None:
        return result.structured_content
    for text in texts:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return None


def _session_id(payload: Any) -> str:
    return str(payload.get("session_id") or payload["sessionId"])


# --- 画面指纹：缩略灰度图，忽略 jpeg 压缩噪点 ---


def _fingerprint(path: Path) -> bytes:
    with Image.open(path) as image:
        return image.convert("L").resize(THUMBNAIL).tobytes()


def _differs(before: bytes, after: bytes) -> bool:
    pixels = sum(1 for a, b in zip(before, after) if abs(a - b) > 16)
    return pixels > 8


def ollama_tool_defs(
    tools: Sequence[Callable[..., Any]],
    coord_max: int,
) -> list[dict[str, Any]]:
    """Ollama 从函数生成 schema 时会丢掉 enum；坐标再标成 integer。"""
    from ollama._utils import convert_function_to_tool

    defs: list[dict[str, Any]] = []
    for fn in tools:
        dumped = convert_function_to_tool(fn).model_dump(exclude_none=True)
        props = (
            dumped.get("function", {}).get("parameters", {}).get("properties") or {}
        )
        for key in ("x", "y", "x2", "y2"):
            if key in props:
                props[key]["type"] = "integer"
                desc = str(props[key].get("description") or "").strip()
                if str(coord_max) not in desc:
                    props[key]["description"] = (
                        f"{desc} Range 0-{coord_max}.".strip()
                    )
        if "direction" in props:
            props["direction"]["type"] = "string"
            props["direction"]["enum"] = ["up", "down"]
        defs.append(dumped)
    return defs


@dataclass
class McpAwesun:
    """一条常驻 stdio MCP 会话"""

    client: Client
    mcp_tools: tuple[Tool, ...]
    protocol_version: str
    server_name: str
    server_version: str
    _loop: asyncio.AbstractEventLoop
    _thread: threading.Thread
    _stop: asyncio.Event
    coord_max: int = 1000
    _closed: bool = field(default=False, init=False)
    _session: str | None = field(default=None, init=False)
    _screen: Path | None = field(default=None, init=False)
    _size: tuple[int, int] = field(default=(0, 0), init=False)
    _frame: bytes = field(default=b"", init=False)

    def run(self, coro: Coroutine[Any, Any, T], timeout: float = TIMEOUT) -> T:
        """在后台事件循环里跑协程，并在当前线程等待结果。"""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(
            timeout=timeout
        )

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: float = TIMEOUT,
    ) -> Any:
        return _tool_payload(
            self.run(self.client.call_tool(name, arguments), timeout=timeout)
        )

    def search_devices(
        self,
        keyword: str = "",
        limit: int = 50,
        *,
        timeout: float = TIMEOUT,
    ) -> Any:
        arguments: dict[str, Any] = {"limit": limit}
        if keyword:
            arguments["keyword"] = keyword
        return self.call_tool("device_search", arguments, timeout=timeout)

    def list_sessions(
        self,
        session_type: str | None = None,
        *,
        timeout: float = TIMEOUT,
    ) -> Any:
        arguments = {"type": session_type} if session_type else None
        return self.call_tool("control_sessions", arguments, timeout=timeout)

    def find_desktop_session(self, remote_id: int) -> str | None:
        """复用已有的可键鼠桌面会话。t=desktopWatch 只能看，不能操作。"""
        for session in self.list_sessions().get("sessions") or []:
            try:
                if int(session["remote_id"]) != int(remote_id):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            session_id = str(session.get("session_id") or "")
            if "t=desktopWatch" in session_id:
                continue
            if session.get("type") == "desktop" or "p=desktop" in session_id:
                return session_id
        return None

    def wait_connected(self, session_id: str, *, timeout: float = 90.0) -> Any:
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            last = self.call_tool("control_connect_state", {"session_id": session_id})
            if isinstance(last, dict) and last.get("state") == "success":
                time.sleep(1)  # 避免远控连接后立刻截图的绿屏错误
                return last
            time.sleep(1)
        raise TimeoutError(f"远控会话未就绪: {last!r}")

    def disconnect(self, session_id: str | None = None) -> None:
        """关掉一条远控会话。不传则关掉当前绑定的那条。"""
        session_id = session_id or self._session
        if not session_id:
            return
        try:
            self.call_tool("control_disconnect", {"session_id": session_id})
        except Exception:
            pass
        if self._session == session_id:
            self._session = None

    def use(self, remote_id: int, *, timeout: float = 90.0) -> str:
        """绑定设备，之后的截图与键鼠都走这条 desktop 会话。"""
        if session_id := self.find_desktop_session(remote_id):
            try:
                self.wait_connected(session_id, timeout=5)
                self._session = session_id
                return session_id
            except Exception:
                self.disconnect(session_id)
        try:
            payload = self.call_tool(
                "control_connect",
                {"remote_id": int(remote_id), "type": "desktop"},
                timeout=timeout,
            )
            session_id = _session_id(payload)
            self.wait_connected(session_id, timeout=timeout)
        except Exception:
            if leftover := self.find_desktop_session(remote_id):
                self.disconnect(leftover)
            raise
        self._session = session_id
        return session_id

    @property
    def session(self) -> str:
        if self._session is None:
            raise RuntimeError("尚未绑定设备，先调用 use()")
        return self._session

    def screenshot(self, *, timeout: float = TIMEOUT) -> Path:
        """截当前画面，返回向日葵生成的文件；旧截图顺手清掉。"""
        payload = self.call_tool(
            "control_screenshot", {"session_id": self.session}, timeout=timeout
        )
        screen = Path(str(payload["image_path"]))
        with Image.open(screen) as image:
            self._size = image.size
        self._frame = _fingerprint(screen)
        self._screen = screen
        for path in screen.parent.glob("awesun_*.jpg"):
            if path != screen:
                path.unlink(missing_ok=True)
        return screen

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def _coordinates(self, x: int, y: int) -> list[float]:
        # [AweSun] 只要 0-1。x、y 已经是截图像素。
        width, height = self._size
        return [min(max(x / width, 0.0), 1.0), min(max(y / height, 0.0), 1.0)]

    def _settle(self) -> None:
        """鼠标动作后等菜单/窗口画完，再截一张结果图。"""
        self.call_tool("desktop_waiting", {"duration": SETTLE_MS})
        self.screenshot()

    def _parse_coord(self, name: str, value: Any) -> float:
        try:
            if isinstance(value, bool) or value is None:
                raise TypeError(name)
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{name} must be one JSON integer 0-{self.coord_max}, such as 59. "
                f"Got {value!r}. Put only this field's number here; "
                "the other axis is a separate field."
            ) from None

    def _to_pixels(
        self,
        x: Any,
        y: Any,
        *,
        x_name: str = "x",
        y_name: str = "y",
    ) -> tuple[int, int]:
        x = self._parse_coord(x_name, x)
        y = self._parse_coord(y_name, y)
        width, height = self._size
        scale = self.coord_max
        if 0 <= x <= scale and 0 <= y <= scale:
            x = x / scale * width
            y = y / scale * height
        return int(min(max(x, 0), width - 1)), int(min(max(y, 0), height - 1))

    def _from_pixels(self, x: int, y: int) -> tuple[int, int]:
        width, height = self._size
        scale = self.coord_max
        if width <= 0 or height <= 0:
            return 0, 0
        mx = round(x / width * scale)
        my = round(y / height * scale)
        return min(max(mx, 0), scale), min(max(my, 0), scale)

    def peek(self, x: Any, y: Any, *, size: int = 256) -> Path:
        """从当前全屏图裁出指针目标附近，动作用来前调用。"""
        x, y = self._to_pixels(x, y)
        if self._screen is None:
            raise RuntimeError("尚未截图")
        left, top = x - size // 2, y - size // 2
        with Image.open(self._screen) as image:
            rgb = image.convert("RGB")
            canvas = Image.new("RGB", (size, size), (255, 255, 255))
            src_left = max(left, 0)
            src_top = max(top, 0)
            src_right = min(left + size, rgb.width)
            src_bottom = min(top + size, rgb.height)
            if src_right > src_left and src_bottom > src_top:
                canvas.paste(
                    rgb.crop((src_left, src_top, src_right, src_bottom)),
                    (src_left - left, src_top - top),
                )
            mark = size // 2
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, size - 1, size - 1), outline=(200, 200, 200))
            draw.line((mark - 12, mark, mark + 12, mark), fill=(255, 0, 0), width=2)
            draw.line((mark, mark - 12, mark, mark + 12), fill=(255, 0, 0), width=2)
        dest = self._screen.with_name(f"peek_{x}_{y}.jpg")
        canvas.save(dest, quality=90)
        return dest

    # --- 交给 VLM 的工具 ---

    @property
    def agent_tools(self) -> tuple[Callable[..., str], ...]:
        """自主操作设备必需的动作。"""
        return (
            self.left_click,
            self.right_click,
            self.left_double_click,
            self.left_drag,
            self.type_text,
            self.press_keys,
            self.scroll,
            self.wait_for_change,
        )

    @property
    def ollama_tools(self) -> list[dict[str, Any]]:
        """Ollama 用的 JSON schema：坐标强制 integer，方向用 enum。"""
        return ollama_tool_defs(self.agent_tools, self.coord_max)

    def _pointer_click(self, x: Any, y: Any, *, button: str, clicks: int) -> str:
        x, y = self._to_pixels(x, y)
        self.call_tool(
            "desktop_click_mouse",
            {
                "session_id": self.session,
                "coordinates": self._coordinates(x, y),
                "button": button,
                "clicks": clicks,
            },
        )
        qx, qy = self._from_pixels(x, y)
        self._settle()
        if button == "right":
            action = "right_click"
        elif clicks == 2:
            action = "left_double_click"
        else:
            action = "left_click"
        return (
            f"{action} ({qx}, {qy}) on the 0-{self.coord_max} grid. "
            "Inspect the new screenshot."
        )

    def left_click(self, x: int, y: int) -> str:
        """Left-click a control using integer parameters x and y.

        Use for buttons, links, tabs, list rows, menu items, and focusing a text field.

        Args:
            x: One JSON integer 0-{coord_max} for this field only, such as 59.
            y: One JSON integer 0-{coord_max} for this field only, such as 900.
        """
        return self._pointer_click(x, y, button="left", clicks=1)

    def right_click(self, x: int, y: int) -> str:
        """Right-click using integer parameters x and y to open a context menu, then left_click the item.

        Args:
            x: One JSON integer 0-{coord_max} for this field only, such as 59.
            y: One JSON integer 0-{coord_max} for this field only, such as 900.
        """
        return self._pointer_click(x, y, button="right", clicks=1)

    def left_double_click(self, x: int, y: int) -> str:
        """Double left-click using integer parameters x and y to open a desktop icon, shortcut, or file.

        Args:
            x: One JSON integer 0-{coord_max} for this field only, such as 59.
            y: One JSON integer for this field only, such as 900.
        """
        return self._pointer_click(x, y, button="left", clicks=2)

    def left_drag(self, x: int, y: int, x2: int, y2: int) -> str:
        """Drag with the left button from integer x and y to integer x2 and y2.

        Use for sliders, selections, and window edges.

        Args:
            x: One JSON integer 0-{coord_max} for this field only, such as 59.
            y: One JSON integer 0-{coord_max} for this field only, such as 900.
            x2: One JSON integer 0-{coord_max} for this field only, such as 59.
            y2: One JSON integer 0-{coord_max} for this field only, such as 900.
        """
        x1, y1 = self._to_pixels(x, y)
        x2, y2 = self._to_pixels(x2, y2, x_name="x2", y_name="y2")
        self.call_tool(
            "desktop_drag_mouse",
            {
                "session_id": self.session,
                "button": "left",
                "paths": [
                    self._coordinates(x1, y1),
                    self._coordinates(x2, y2),
                ],
            },
        )
        self._settle()
        a = self._from_pixels(x1, y1)
        b = self._from_pixels(x2, y2)
        return (
            f"left_drag ({a[0]}, {a[1]}) -> ({b[0]}, {b[1]}) on the 0-{self.coord_max} grid. "
            "Inspect the new screenshot."
        )

    def type_text(self, text: str) -> str:
        """Type into the field that already has keyboard focus.

        left_click the field first if it is not focused. Characters are appended.
        Skip only when the field already shows exactly this text. If it shows
        something else, select all (for example press_keys(["control", "a"])) then type.

        Args:
            text: Characters to type.
        """
        for char in text:
            if char == " ":
                key = "space"
            elif char == "\n":
                key = "enter"
            elif char == "\t":
                key = "tab"
            else:
                key = char
            self.call_tool(
                "desktop_press_keys", {"session_id": self.session, "keys": [key]}
            )
        return (
            f"typed {len(text)} chars into the focused field. "
            "Inspect the new screenshot."
        )

    def press_keys(self, keys: list[str]) -> str:
        """Press a key or a shortcut.

        Use for Enter, Tab, Escape, Backspace, arrows, and chords like Ctrl+S.

        Args:
            keys: Keys pressed together. Spell ESCAPE, BACK, ENTER, TAB,
                DELETE, or control plus a letter. Examples: ["ENTER"],
                ["ESCAPE"], ["control", "a"]. Do not send esc or backspace.
        """
        keys = [_HID_KEY_ALIAS.get(key.lower(), key) for key in keys]
        self.call_tool(
            "desktop_typing_keys", {"session_id": self.session, "keys": keys}
        )
        return f"pressed {'+'.join(keys)}. Inspect the new screenshot."

    def scroll(self, x: int, y: int, direction: str = "down", amount: int | None = None) -> str:
        """Scroll the mouse wheel using integer parameters x and y.

        left_click the pane first if it is not focused.

        Args:
            x: One JSON integer 0-{coord_max} for this field only, such as 59.
            y: One JSON integer 0-{coord_max} for this field only, such as 900.
            direction: Wheel direction, up or down.
            amount: Wheel steps. Defaults to 3.
        """
        x, y = self._to_pixels(x, y)
        amount = amount or 3
        self.call_tool(
            "desktop_scroll_mouse",
            {
                "session_id": self.session,
                "coordinates": self._coordinates(x, y),
                "direction": direction,
                "scroll_count": amount,
            },
        )
        qx, qy = self._from_pixels(x, y)
        self._settle()
        return (
            f"scroll {direction} x{amount} at ({qx}, {qy}) on the 0-{self.coord_max} grid. "
            "Inspect the new screenshot."
        )

    def wait_for_change(self, timeout: int | None = None) -> str:
        """Wait until the screen pixels change a lot compared with the last screenshot.

        Use only after an action that should open a window or replace the page.
        Skip after typing or focusing a field. This does not read the image.

        Args:
            timeout: Milliseconds to wait at most, 500 to 10000. Defaults to 3000.
        """
        limit_ms = min(max(timeout or 3000, 500), 10000)
        before = self._frame
        start = time.monotonic()
        while True:
            time.sleep(0.3)
            self.screenshot()
            elapsed_ms = int((time.monotonic() - start) * 1000)
            if _differs(before, self._frame):
                return (
                    f"screen changed after {elapsed_ms}ms. "
                    "Read the new screenshot before the next action."
                )
            if elapsed_ms >= limit_ms:
                return (
                    f"screen unchanged after {limit_ms}ms. "
                    "Do not wait again; try a different action."
                )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=15)

    @classmethod
    def _bind_coord_docs(cls, scale: int) -> None:
        token = "{coord_max}"
        n = str(scale)
        for name in (
            "left_click",
            "right_click",
            "left_double_click",
            "left_drag",
            "scroll",
        ):
            fn = getattr(cls, name)
            if fn.__doc__ and token in fn.__doc__:
                fn.__doc__ = fn.__doc__.replace(token, n)

    @classmethod
    def connect(cls, path: str, *, timeout: float = TIMEOUT) -> McpAwesun:
        """读 config.toml，拉起 awesun-mcp-server stdio，并阻塞到 initialize 完成。"""
        # [AweSun] 0.0.2 启动横幅不是 JSON，SDK 会刷 Failed to parse JSONRPC。
        logging.getLogger("mcp.client.stdio").addFilter(
            lambda rec: (
                "Failed to parse JSONRPC message from server" not in rec.getMessage()
            )
        )

        raw = tomllib.loads(Path(path).read_text())
        cfg = raw["mcp_servers"]["awesun-mcp-server"]
        coord_max = int(raw.get("ollama", {}).get("coord_max", 1000))
        if coord_max < 1:
            raise ValueError("ollama.coord_max must be >= 1")
        ready = threading.Event()
        errors: list[BaseException] = []
        holder: dict[str, Any] = {}

        def thread_main() -> None:
            async def runner() -> None:
                holder["stop"] = stop = asyncio.Event()
                holder["loop"] = asyncio.get_running_loop()
                try:
                    # [MCP] Client 必须停在 async with 里。
                    # [AweSun] 0.0.2 只有 initialize，没有 2026 的 server/discover。
                    async with Client(
                        StdioServerParameters(
                            command=cfg["command"],
                            env={
                                "AWESUN_API_URL": cfg["env"]["AWESUN_API_URL"],
                                "AWESUN_API_TOKEN": cfg["env"]["AWESUN_API_TOKEN"],
                            },
                        ),
                        client_info=Implementation(name="chatkvm", version=__version__),
                        mode="legacy",
                        logging_callback=_on_log,
                        elicitation_callback=_on_elicit,
                        message_handler=_on_transport_message,
                        read_timeout_seconds=TIMEOUT,
                    ) as client:
                        listed = await client.list_tools()
                        info = client.server_info
                        holder["client"] = client
                        holder["mcp_tools"] = tuple(listed.tools)
                        holder["protocol_version"] = client.protocol_version
                        holder["server_name"] = info.name if info else "unknown"
                        holder["server_version"] = info.version if info else ""
                        ready.set()
                        await stop.wait()
                except BaseException as exc:
                    errors.append(exc)
                    ready.set()
                    raise

            asyncio.run(runner())

        thread = threading.Thread(target=thread_main, name="awesun-mcp", daemon=True)
        thread.start()
        if not ready.wait(timeout=timeout):
            raise TimeoutError("向日葵 MCP 初始化超时")
        if errors:
            raise errors[0]

        session = cls(
            client=holder["client"],
            mcp_tools=holder["mcp_tools"],
            protocol_version=holder["protocol_version"],
            server_name=holder["server_name"],
            server_version=holder["server_version"],
            _loop=holder["loop"],
            _thread=thread,
            _stop=holder["stop"],
            coord_max=coord_max,
        )
        cls._bind_coord_docs(coord_max)
        atexit.register(session.close)
        return session
