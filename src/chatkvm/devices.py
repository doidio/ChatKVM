"""Route ChatKVM operations to the configured kvmcp servers."""

from __future__ import annotations

import base64
import struct
from dataclasses import dataclass
from typing import Any

from mcp import Client
from mcp.types import CallToolResult, ImageContent, TextContent


class DeviceError(Exception):
    """A device is unknown, unreachable, or rejected a tool call."""


@dataclass(frozen=True)
class Device:
    id: str
    url: str
    description: str
    text_input: str


@dataclass(frozen=True)
class Frame:
    device_id: str
    data: bytes
    mime_type: str
    description: str

    @property
    def size(self) -> tuple[int, int]:
        """Read the dimensions from the PNG itself, not from display scaling or prose."""
        if self.mime_type != 'image/png' or self.data[:16] != b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR' or len(self.data) < 24:
            raise DeviceError(f'{self.device_id} 的截图不是有效 PNG，无法确定原图像素坐标。')
        width, height = struct.unpack('>II', self.data[16:24])
        if not width or not height:
            raise DeviceError(f'{self.device_id} 的截图尺寸无效。')
        return width, height


class Devices:
    def __init__(self, config: dict[str, dict[str, str]]) -> None:
        self._devices = {}
        for device_id, values in config.items():
            method = values.get('text_input', 'ascii')
            if method not in ('ascii', 'windows_gbk'):
                raise ValueError(f'{device_id} 的 text_input 不支持: {method}')
            self._devices[device_id] = Device(device_id, values['url'], values.get('description', ''), method)
        self._read_timeout_seconds = 30

    def text_input_method(self, device_id: str) -> str:
        try:
            return self._devices[device_id].text_input
        except KeyError:
            raise DeviceError(f'未知设备 {device_id!r}；请先调用 list_devices。') from None

    def list(self) -> list[dict[str, str]]:
        """List configured devices; this does not imply that they are online."""
        return [{'device_id': device.id, 'description': device.description} for device in self._devices.values()]

    def ids(self) -> list[str]:
        return list(self._devices)

    async def call(self, device_id: str, tool_name: str, *, read_timeout_seconds: float | None = None, **arguments: Any) -> CallToolResult:
        device = self._devices.get(device_id)
        if device is None:
            raise DeviceError(f'未知设备 {device_id!r}；请先调用 list_devices。')

        try:
            async with Client(device.url) as client:
                result = await client.call_tool(tool_name, arguments or None, read_timeout_seconds=read_timeout_seconds or self._read_timeout_seconds)
        except Exception as exc:
            raise DeviceError(f'{device_id} 的 {tool_name} 请求失败：{exc}') from exc

        if result.is_error:
            message = '\n'.join(part.text for part in result.content if isinstance(part, TextContent))
            raise DeviceError(f'{device_id} 的 {tool_name} 执行失败：{message or "未知错误"}')
        return result

    async def capture(self, device_id: str) -> Frame:
        result = await self.call(device_id, 'latest_frame')
        image = next((part for part in result.content if isinstance(part, ImageContent)), None)
        if image is None:
            raise DeviceError(f'{device_id} 的 latest_frame 未返回图像。')

        description = '\n'.join(part.text for part in result.content if isinstance(part, TextContent))
        return Frame(device_id, base64.b64decode(image.data), image.mime_type, description)
