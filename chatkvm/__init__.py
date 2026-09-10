"""
向日葵 MCP 同步封装

MCP `Client` 必须停在 `async with` 里，stdio 子进程才不会被关掉。
Streamlit 脚本是同步瀑布流，因此在后台线程跑一条常驻事件循环，
主线程用 `run()` 把协程提交进去并阻塞取结果。
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
from collections.abc import Coroutine
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

T = TypeVar("T")
TIMEOUT = 60.0

__all__ = ["AwesunMcp"]


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


@dataclass
class AwesunMcp:
    """一条常驻 stdio MCP 会话"""

    client: Client
    tools: tuple[Tool, ...]
    protocol_version: str
    server_name: str
    server_version: str
    _loop: asyncio.AbstractEventLoop
    _thread: threading.Thread
    _stop: asyncio.Event
    _closed: bool = field(default=False, init=False)

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
        """复用已有桌面会话。优先 desktop_view（session_id 含 t=desktopWatch）。"""
        watch = desktop = None
        for session in self.list_sessions().get("sessions") or []:
            try:
                if int(session["remote_id"]) != int(remote_id):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            session_id = str(session.get("session_id") or "")
            if "t=desktopWatch" in session_id:
                watch = session_id
            elif session.get("type") == "desktop" or "t=desktop" in session_id:
                desktop = session_id
        return watch or desktop

    def wait_connected(self, session_id: str, *, timeout: float = 90.0) -> Any:
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            last = self.call_tool("control_connect_state", {"session_id": session_id})
            if isinstance(last, dict) and last.get("state") == "success":
                return last
            time.sleep(1)
        raise TimeoutError(f"远控会话未就绪: {last!r}")

    def ensure_view_session(self, remote_id: int, *, timeout: float = 90.0) -> str:
        """有桌面会话则复用，否则发起 desktop_view。"""
        if existing := self.find_desktop_session(remote_id):
            return existing
        try:
            payload = self.call_tool(
                "control_connect",
                {"remote_id": int(remote_id), "type": "desktop_view"},
                timeout=timeout,
            )
            session_id = _session_id(payload)
        except Exception:
            # connect 超时后会话有时已经出现在 control_sessions 里。
            if existing := self.find_desktop_session(remote_id):
                return existing
            raise
        self.wait_connected(session_id, timeout=timeout)
        return session_id

    def take_screenshot(
        self, session_id: str, *, timeout: float = TIMEOUT
    ) -> str | None:
        payload = self.call_tool(
            "control_screenshot", {"session_id": session_id}, timeout=timeout
        )
        if not isinstance(payload, dict):
            return None
        return str(payload.get("image_path") or "") or None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=15)

    @classmethod
    def connect(cls, path: str, *, timeout: float = TIMEOUT) -> AwesunMcp:
        """读 config.toml，拉起 awesun-mcp-server stdio，并阻塞到 initialize 完成。"""
        # [AweSun] 0.0.2 启动横幅不是 JSON，SDK 会刷 Failed to parse JSONRPC。
        logging.getLogger("mcp.client.stdio").addFilter(
            lambda rec: (
                "Failed to parse JSONRPC message from server" not in rec.getMessage()
            )
        )

        cfg = tomllib.loads(Path(path).read_text())["mcp_servers"]["awesun-mcp-server"]
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
                        client_info=Implementation(name="chatkvm", version="0.1.0"),
                        mode="legacy",
                        logging_callback=_on_log,
                        elicitation_callback=_on_elicit,
                        message_handler=_on_transport_message,
                        read_timeout_seconds=TIMEOUT,
                    ) as client:
                        listed = await client.list_tools()
                        info = client.server_info
                        holder["client"] = client
                        holder["tools"] = tuple(listed.tools)
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
            tools=holder["tools"],
            protocol_version=holder["protocol_version"],
            server_name=holder["server_name"],
            server_version=holder["server_version"],
            _loop=holder["loop"],
            _thread=thread,
            _stop=holder["stop"],
        )
        atexit.register(session.close)
        return session
