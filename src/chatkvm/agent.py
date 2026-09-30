"""The model sees one set of computer tools, regardless of device count."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Annotated, Literal

from pydantic import Field
from pydantic_ai import Agent, BinaryContent, RunContext, ToolReturn, UsageLimits
from pydantic_ai.capabilities import Hooks, WrapModelRequestHandler
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from chatkvm.devices import DeviceError, Devices, Frame

AGENT_INSTRUCTIONS = """你是 ChatKVM，通过工具观察和操作多台目标电脑。
使用 list_devices 了解设备，根据用户任务和设备说明选择目标；设备可能离线。
每轮首次操控某台设备前，先用 capture_screen 查看其当前画面；不要依赖上一轮的截图，因为期间设备状态可能已改变。连续操作时根据最新画面判断，并在需要时重新截图。
采集画面显示“无信号”时，目标主机可能熄屏、休眠、关机或断线。需要时可根据最新截图移动鼠标尝试唤醒，再等待并截图确认；指令发送成功不代表屏幕已恢复。
只执行用户授权的操作。操作指令发送成功后仍需观察结果；失败或状态不明时先确认当前状态，再决定下一步。"""


def resolve_model(config: dict) -> Model | str:
    """Each configured model owns its provider and provider-specific connection."""
    name = config['name']
    provider = config['provider']
    if provider == 'ollama':
        return OllamaModel(name, provider=OllamaProvider(base_url=config['base_url']))
    return f'{provider}:{name}'


@dataclass
class ChatDeps:
    devices: Devices
    allow_control: bool
    tool_limit: int
    frames: dict[str, Frame] = field(default_factory=dict)
    tool_count: int = 0
    events: list[ToolEvent] = field(default_factory=list)
    on_event: Callable[[ToolEvent], None] | None = None
    on_model_response: Callable[[ModelResponse], None] | None = None


@dataclass
class ToolEvent:
    name: str
    device_id: str | None
    arguments: dict[str, object]
    duration_ms: int
    result: str
    frame: Frame | None = None
    tool_call_id: str | None = None


@dataclass
class ChatResult:
    output: str
    messages: list[ModelMessage]
    new_messages_json: bytes
    input_tokens: int
    output_tokens: int


def compact_history(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Keep only the latest image without changing the complete conversation record."""
    latest = None
    for message_index, message in enumerate(messages):
        for part_index, part in enumerate(message.parts):
            content = getattr(part, 'content', None)
            if isinstance(content, list):
                for item_index, item in enumerate(content):
                    if isinstance(item, BinaryContent) and item.media_type.startswith('image/'):
                        latest = (message_index, part_index, item_index)
    compact: list[ModelMessage] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, ModelRequest):
            compact.append(message)
            continue
        parts = []
        for part_index, part in enumerate(message.parts):
            content = getattr(part, 'content', None)
            if isinstance(content, list) and any(isinstance(item, BinaryContent) for item in content):
                parts.append(
                    replace(
                        part,
                        content=[
                            '[此前截图已归档]'
                            if isinstance(item, BinaryContent)
                            and item.media_type.startswith('image/')
                            and (message_index, part_index, item_index) != latest
                            else item
                            for item_index, item in enumerate(content)
                        ],
                    )
                )
            else:
                parts.append(part)
        compact.append(replace(message, parts=parts))
    return compact


def _record(
    ctx: RunContext[ChatDeps], name: str, device_id: str | None, arguments: dict[str, object], started: float, result: str, frame: Frame | None = None
) -> None:
    event = ToolEvent(name, device_id, arguments, round((perf_counter() - started) * 1000), result, frame, ctx.tool_call_id)
    ctx.deps.events.append(event)
    if ctx.deps.on_event is not None:
        ctx.deps.on_event(event)


async def _latest_image_request(
    _ctx: RunContext[ChatDeps], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
) -> ModelResponse:
    return await handler(replace(request_context, messages=compact_history(request_context.messages)))


async def _model_response(ctx: RunContext[ChatDeps], *, request_context: ModelRequestContext, response: ModelResponse) -> ModelResponse:
    if ctx.deps.on_model_response is not None:
        ctx.deps.on_model_response(response)
    return response


def _tool_error(deps: ChatDeps) -> str | None:
    if deps.tool_count >= deps.tool_limit:
        return f'本轮工具调用已达到 {deps.tool_limit} 次上限；请总结当前状态，等待下一轮用户指令。'
    deps.tool_count += 1
    return None


def build_agent(model: Model | str, *, allow_control: bool, thinking: bool = False) -> Agent[ChatDeps, str]:
    model_settings = {'parallel_tool_calls': False}
    if isinstance(model, OllamaModel):
        model_settings['openai_reasoning_effort'] = 'medium' if thinking else 'none'
    else:
        model_settings['thinking'] = thinking
    agent = Agent(
        model,
        name='ChatKVM',
        instructions=AGENT_INSTRUCTIONS,
        deps_type=ChatDeps,
        model_settings=model_settings,
        capabilities=[Hooks(model_request=_latest_image_request, after_model_request=_model_response)],
    )

    @agent.tool
    async def list_devices(ctx: RunContext[ChatDeps]) -> list[dict[str, str]]:
        """列出 ChatKVM 配置的目标设备 ID 和说明；不代表设备当前在线。"""
        started = perf_counter()
        if error := _tool_error(ctx.deps):
            _record(ctx, 'list_devices', None, {}, started, error)
            return [{'error': error}]
        result = ctx.deps.devices.list()
        _record(ctx, 'list_devices', None, {}, started, f'{len(result)} 台已配置设备')
        return result

    @agent.tool
    async def capture_screen(
        ctx: RunContext[ChatDeps],
        device_id: Annotated[str, Field(description='list_devices 返回的设备 ID')],
        wait_seconds: Annotated[float, Field(ge=0, le=30, description='截图前等待的秒数；省略或设为 0 则立即取最新画面')] = 0,
    ) -> ToolReturn | str:
        """查看指定设备的原尺寸画面，可先等待一段时间再获取最新帧。"""
        started = perf_counter()
        arguments = {'wait_seconds': wait_seconds} if wait_seconds else {}
        if error := _tool_error(ctx.deps):
            _record(ctx, 'capture_screen', device_id, arguments, started, error)
            return error
        try:
            if wait_seconds:
                await asyncio.sleep(wait_seconds)
            frame = await ctx.deps.devices.capture(device_id)
            width, height = frame.size
        except DeviceError as exc:
            ctx.deps.frames.pop(device_id, None)
            _record(ctx, 'capture_screen', device_id, arguments, started, str(exc))
            return str(exc)
        ctx.deps.frames[device_id] = frame
        _record(ctx, 'capture_screen', device_id, arguments, started, frame.description, frame)
        return ToolReturn(
            return_value=f'设备 {device_id}，截图 {width}×{height} 像素。',
            content=[BinaryContent(data=frame.data, media_type=frame.mime_type)],
        )

    if not allow_control:
        return agent

    @agent.tool
    async def mouse_move(
        ctx: RunContext[ChatDeps],
        device_id: Annotated[str, Field(description='list_devices 返回的设备 ID')],
        x: Annotated[int, Field(ge=0, description='最近截图的原图水平像素坐标，左边缘为 0')],
        y: Annotated[int, Field(ge=0, description='最近截图的原图垂直像素坐标，上边缘为 0')],
    ) -> str:
        """按最近一次截图的原图像素坐标移动指定设备的鼠标。"""
        started = perf_counter()
        arguments = {'x': x, 'y': y}
        error = (
            _tool_error(ctx.deps) or ('本轮未允许操控。' if not ctx.deps.allow_control else None) or _screen_point_error(ctx.deps, device_id, x, y)
        )
        if error:
            _record(ctx, 'mouse_move', device_id, arguments, started, error)
            return error
        try:
            result = _text(await ctx.deps.devices.call(device_id, 'mouse_move', **arguments))
        except DeviceError as exc:
            result = f'{exc}；状态不确定时请先重新截图。'
        _record(ctx, 'mouse_move', device_id, arguments, started, result)
        return result

    @agent.tool
    async def mouse_click(
        ctx: RunContext[ChatDeps],
        device_id: Annotated[str, Field(description='list_devices 返回的设备 ID')],
        x: Annotated[int, Field(ge=0, description='最近截图的原图水平像素坐标，左边缘为 0')],
        y: Annotated[int, Field(ge=0, description='最近截图的原图垂直像素坐标，上边缘为 0')],
        button: Literal['left', 'right', 'middle'] = 'left',
    ) -> str:
        """先查看指定设备的画面，再按原图像素坐标单击一次。"""
        started = perf_counter()
        arguments = {'x': x, 'y': y, 'button': button}
        error = (
            _tool_error(ctx.deps) or ('本轮未允许操控。' if not ctx.deps.allow_control else None) or _screen_point_error(ctx.deps, device_id, x, y)
        )
        if error:
            _record(ctx, 'mouse_click', device_id, arguments, started, error)
            return error
        try:
            result = _text(await ctx.deps.devices.call(device_id, 'mouse_click', **arguments))
        except DeviceError as exc:
            result = f'{exc}；状态不确定时请先重新截图。'
        _record(ctx, 'mouse_click', device_id, arguments, started, result)
        return result

    @agent.tool
    async def key_press(
        ctx: RunContext[ChatDeps],
        device_id: Annotated[str, Field(description='list_devices 返回的设备 ID')],
        key: Annotated[str, Field(description="单个物理键名，例如 'enter'、'tab'、'escape'、'a'、'0'")],
        modifiers: list[Literal['ctrl', 'shift', 'alt', 'meta']] | None = None,
    ) -> str:
        """在指定设备按下并释放一个物理按键，可组合修饰键。"""
        started = perf_counter()
        arguments = {'key': key, 'modifiers': modifiers}
        error = (
            _tool_error(ctx.deps)
            or ('本轮未允许操控。' if not ctx.deps.allow_control else None)
            or _current_screen_error(ctx.deps, device_id)
        )
        if error:
            _record(ctx, 'key_press', device_id, arguments, started, error)
            return error
        try:
            result = _text(await ctx.deps.devices.call(device_id, 'key_press', **arguments))
        except DeviceError as exc:
            result = f'{exc}；状态不确定时请先重新截图。'
        _record(ctx, 'key_press', device_id, arguments, started, result)
        return result

    return agent


def _current_screen_error(deps: ChatDeps, device_id: str) -> str | None:
    if device_id not in deps.frames:
        return f'设备 {device_id} 尚无本轮截图；请先调用 capture_screen 查看当前画面。'
    return None


def _screen_point_error(deps: ChatDeps, device_id: str, x: int, y: int) -> str | None:
    if error := _current_screen_error(deps, device_id):
        return error
    frame = deps.frames.get(device_id)
    assert frame is not None
    try:
        width, height = frame.size
    except DeviceError as exc:
        return str(exc)
    if not (0 <= x < width and 0 <= y < height):
        return f'坐标 ({x}, {y}) 超出设备 {device_id} 原图 {width}×{height} 的范围；请重新截图并定位。'
    return None


def _text(result: object) -> str:
    from mcp.types import TextContent

    parts = getattr(result, 'content', [])
    return '\n'.join(part.text for part in parts if isinstance(part, TextContent)) or '操作指令已发送。'


async def chat(
    agent: Agent[ChatDeps, str],
    prompt: str,
    history: list[ModelMessage],
    deps: ChatDeps,
) -> ChatResult:
    result = await agent.run(
        prompt,
        message_history=history,
        deps=deps,
        usage_limits=UsageLimits(request_limit=max(12, deps.tool_limit + 3)),
    )
    usage = result.usage
    last_input_tokens = next(
        (message.usage.input_tokens for message in reversed(result.new_messages()) if isinstance(message, ModelResponse) and message.usage),
        0,
    )
    return ChatResult(result.output, result.all_messages(), result.new_messages_json(), last_input_tokens, usage.output_tokens or 0)
