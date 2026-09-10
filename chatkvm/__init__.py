from __future__ import annotations

import asyncio
import atexit
import logging
import sys
import threading
import tomllib
from collections.abc import Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from mcp.client import Client, ClientRequestContext, IncomingMessage
from mcp.client.stdio import StdioServerParameters
from mcp.types import (
    ElicitRequestParams,
    ElicitResult,
    Implementation,
    LoggingMessageNotificationParams,
    Tool,
)

T = TypeVar("T")
config_file: str | None = None


def load_settings(path: str | None = None) -> tuple[str, str, str]:
    file = path or config_file
    if file is None:
        raise RuntimeError("未指定配置文件，请通过 uv run chatkvm 启动")
    path_obj = Path(file)
    if not path_obj.is_file():
        raise FileNotFoundError(f"找不到配置文件: {path_obj.resolve()}")
    raw = tomllib.loads(path_obj.read_text())
    missing = [key for key in ("command", "api_url", "api_token") if key not in raw]
    if missing:
        raise ValueError(f"{path_obj} 缺少字段: {', '.join(missing)}")
    return str(raw["command"]), str(raw["api_url"]), str(raw["api_token"])


async def _on_log(params: LoggingMessageNotificationParams) -> None:
    print(f"[mcp-log {params.level}] {params.data}", file=sys.stderr)


async def _on_transport_message(message: IncomingMessage) -> None:
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


@dataclass
class AwesunMcp:
    """一条常驻 MCP 会话：stdio 子进程 + 已拉取的 tools/list。"""

    client: Client
    tools: tuple[Tool, ...]
    protocol_version: str
    server_name: str
    server_version: str
    _loop: asyncio.AbstractEventLoop
    _thread: threading.Thread
    _stop: asyncio.Event
    _closed: bool = field(default=False, init=False)

    def run(self, coro: Coroutine[Any, Any, T], timeout: float = 60.0) -> T:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(
            timeout=timeout
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=15)

    @classmethod
    def connect(cls, path: str | None = None, *, timeout: float = 60.0) -> AwesunMcp:
        # [AweSun] 0.0.2 会往 stdout 打非 JSON 启动横幅；[MCP] stdout 只能是 JSON-RPC。
        logging.getLogger("mcp.client.stdio").addFilter(
            lambda rec: (
                "Failed to parse JSONRPC message from server" not in rec.getMessage()
            )
        )
        cfg = tomllib.loads(Path(path).read_text())
        command = cfg["mcp_servers"]["awesun-mcp-server"]["command"]
        api_url = cfg["mcp_servers"]["awesun-mcp-server"]["env"]["AWESUN_API_URL"]
        api_token = cfg["mcp_servers"]["awesun-mcp-server"]["env"]["AWESUN_API_TOKEN"]

        ready = threading.Event()
        errors: list[BaseException] = []
        holder: dict[str, Any] = {}

        def thread_main() -> None:
            async def runner() -> None:
                stop = asyncio.Event()
                holder["stop"] = stop
                holder["loop"] = asyncio.get_running_loop()
                try:
                    # [MCP] Client 必须停在 async with 里，stdio 子进程才不会被关掉。
                    # [AweSun] 0.0.2 只会 initialize，没有 2026 的 server/discover。
                    async with Client(
                        StdioServerParameters(
                            command=command,
                            env={
                                "AWESUN_API_URL": api_url,
                                "AWESUN_API_TOKEN": api_token,
                            },
                        ),
                        client_info=Implementation(name="chatkvm", version="0.1.0"),
                        mode="legacy",
                        logging_callback=_on_log,
                        elicitation_callback=_on_elicit,
                        message_handler=_on_transport_message,
                        read_timeout_seconds=60.0,
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
