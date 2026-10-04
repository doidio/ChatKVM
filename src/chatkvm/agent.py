"""The model sees one set of computer tools, regardless of device count."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass, field, replace
from io import BytesIO
from time import perf_counter
from typing import Annotated, Literal

from PIL import Image
from pydantic import Field
from pydantic_ai import Agent, BinaryContent, RunContext, ToolReturn, UsageLimits
from pydantic_ai.capabilities import Hooks, WrapModelRequestHandler
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ThinkingPart,
    ThinkingPartDelta,
)
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from chatkvm.devices import DeviceError, Devices, Frame

MODEL_IMAGE_SIZE = (3840, 2160)
MAX_TURN_IMAGES = 2

AGENT_INSTRUCTIONS = """你是 ChatKVM，通过工具观察和操作多台目标电脑。
使用 list_devices 了解设备，根据用户任务和设备说明选择目标；设备可能离线。
每轮首次查看或操作某台设备前，先用 capture_screen 获取当前画面；切换设备或画面可能变化时重新截图，不沿用旧图。
采集服务保证 HID 操作后获取新画面；点击、拖拽、按键和键入工具直接返回操作后画面。移动光标后若需观察，重新截图。wait_seconds 只用于等待目标界面响应；仍在加载时继续等待并截图，不重复发送已执行的操作。
采集画面显示“无信号”时，目标主机可能熄屏、休眠、关机或断线。需要时可根据最新截图移动鼠标尝试唤醒，再等待并截图确认；指令发送成功不代表屏幕已恢复。
依据工具返回的全屏截图观察和操作。放大图用于辨认画面；鼠标坐标始终使用截图注明的采集原图像素尺寸。光标遮挡目标时，先移开光标并重新截图。
识别界面文字时按画面原样判断；局部被截断或新画面与旧判断冲突时，补看完整画面，以新证据修正结论，不凭猜测补全。
每次键鼠操作前，依据当前画面确认目标状态、焦点等操作前提；操作后查看画面，确认落点和结果。使用带坐标的工具前，根据最新截图定位目标内部落点，不沿用界面变化前的坐标；光标未落在目标上时重新定位，落点正确但无预期变化时检查界面响应、焦点或交互方式，不在同一位置反复试错。静态画面未必能证明焦点；输入长文本时，若分段不会触发副作用，可先发送一小段原文，确认出现后再继续。
type_text 按设备配置键入，无法输入的字符会跳过并列明。按键已发送不代表文本进入了目标控件；未显示时先检查焦点和落点，再判断输入方式是否不支持。同一设备的操作应串行进行。
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
    on_thinking: Callable[[str | None], None] | None = None
    streamed_thinking: bool = False


@dataclass
class ToolEvent:
    name: str
    device_id: str | None
    arguments: dict[str, object]
    duration_ms: int
    result: str
    frame: Frame | None = None
    tool_call_id: str | None = None
    tool_count: int | None = None
    tool_limit: int | None = None


@dataclass
class ChatResult:
    output: str
    messages: list[ModelMessage]
    new_messages_json: bytes
    input_tokens: int
    output_tokens: int


def compact_history(messages: list[ModelMessage], *, keep_images: int = MAX_TURN_IMAGES) -> list[ModelMessage]:
    """Limit model-visible screenshots without changing the complete conversation record."""
    images = []
    for message_index, message in enumerate(messages):
        for part_index, part in enumerate(message.parts):
            content = getattr(part, 'content', None)
            if isinstance(content, list):
                for item_index, item in enumerate(content):
                    if isinstance(item, BinaryContent) and item.media_type.startswith('image/'):
                        images.append((message_index, part_index, item_index))
    retained = set(images[-keep_images:]) if keep_images else set()
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
                            and (message_index, part_index, item_index) not in retained
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
    event = ToolEvent(
        name,
        device_id,
        arguments,
        round((perf_counter() - started) * 1000),
        result,
        frame,
        ctx.tool_call_id,
        ctx.deps.tool_count,
        ctx.deps.tool_limit,
    )
    ctx.deps.events.append(event)
    if ctx.deps.on_event is not None:
        ctx.deps.on_event(event)


async def _latest_image_request(
    _ctx: RunContext[ChatDeps], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
) -> ModelResponse:
    return await handler(replace(request_context, messages=compact_history(request_context.messages)))


async def _model_response(ctx: RunContext[ChatDeps], *, request_context: ModelRequestContext, response: ModelResponse) -> ModelResponse:
    if not ctx.deps.streamed_thinking and ctx.deps.on_thinking is not None:
        for part in response.parts:
            if isinstance(part, ThinkingPart) and part.content:
                ctx.deps.on_thinking(part.content)
                ctx.deps.on_thinking(None)
    if ctx.deps.on_model_response is not None:
        ctx.deps.on_model_response(response)
    ctx.deps.streamed_thinking = False
    return response


async def _stream_events(ctx: RunContext[ChatDeps], events: AsyncIterable[AgentStreamEvent]) -> None:
    thinking_open = False
    try:
        async for event in events:
            if isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart):
                ctx.deps.streamed_thinking = True
                thinking_open = True
                if ctx.deps.on_thinking is not None:
                    ctx.deps.on_thinking(event.part.content)
            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, ThinkingPartDelta):
                if event.delta.content_delta and ctx.deps.on_thinking is not None:
                    ctx.deps.on_thinking(event.delta.content_delta)
            elif isinstance(event, PartEndEvent) and isinstance(event.part, ThinkingPart) and thinking_open:
                if ctx.deps.on_thinking is not None:
                    ctx.deps.on_thinking(None)
                thinking_open = False
    finally:
        if thinking_open and ctx.deps.on_thinking is not None:
            ctx.deps.on_thinking(None)


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

    @agent.instructions
    def tool_budget(ctx: RunContext[ChatDeps]) -> str:
        remaining = max(0, ctx.deps.tool_limit - ctx.deps.tool_count)
        return (
            f'本轮工具调用上限 {ctx.deps.tool_limit} 次，已用 {ctx.deps.tool_count} 次，剩余 {remaining} 次。'
            '额度每轮重新计算，历史轮次的上限提示不适用于本轮。'
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
        wait_seconds: Annotated[float, Field(ge=0, le=30, description='等待目标界面响应的秒数；省略或设为 0 则立即取最新画面')] = 0,
    ) -> ToolReturn | str:
        """采集指定设备的当前全屏画面；每轮首次查看该设备时先调用，可等待界面响应后再截图。"""
        started = perf_counter()
        arguments = {'wait_seconds': wait_seconds} if wait_seconds else {}
        if error := _tool_error(ctx.deps):
            _record(ctx, 'capture_screen', device_id, arguments, started, error)
            return error
        try:
            if wait_seconds:
                await asyncio.sleep(wait_seconds)
            frame = await ctx.deps.devices.capture(device_id)
            width, height = _display_size(frame.size)
            presentation = _present_frame(frame)
        except DeviceError as exc:
            ctx.deps.frames.pop(device_id, None)
            _record(ctx, 'capture_screen', device_id, arguments, started, str(exc))
            return str(exc)
        ctx.deps.frames[device_id] = frame
        _record(ctx, 'capture_screen', device_id, arguments, started, frame.description, frame)
        raw_width, raw_height = frame.size
        return ToolReturn(
            return_value=f'设备 {device_id}，采集原图 {raw_width}×{raw_height} 像素，显示图 {width}×{height} 像素；鼠标工具使用采集原图坐标。',
            content=[presentation],
        )

    if not allow_control:
        return agent

    @agent.tool
    async def mouse_move(
        ctx: RunContext[ChatDeps],
        device_id: Annotated[str, Field(description='list_devices 返回的设备 ID')],
        x: Annotated[int, Field(ge=0, description='采集原图的水平像素坐标，左边缘为 0')],
        y: Annotated[int, Field(ge=0, description='采集原图的垂直像素坐标，上边缘为 0')],
    ) -> str:
        """先看当前画面确认目标位置，再按采集原图像素坐标移动鼠标；需要核对移动结果时调用 capture_screen。"""
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
        x: Annotated[int, Field(ge=0, description='采集原图的水平像素坐标，左边缘为 0')],
        y: Annotated[int, Field(ge=0, description='采集原图的垂直像素坐标，上边缘为 0')],
        button: Literal['left', 'right', 'middle'] = 'left',
        count: Annotated[Literal[1, 2], Field(description='点击次数：1 为单击，2 为双击')] = 1,
    ) -> ToolReturn | str:
        """先看当前画面确认目标可操作，再按采集原图像素坐标单击或双击；核对返回画面的落点和结果。双击使用 count=2。"""
        started = perf_counter()
        arguments = {'x': x, 'y': y, 'button': button, 'count': count}
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
        return await _observed_result(ctx, 'mouse_click', device_id, arguments, started, result)

    @agent.tool
    async def mouse_drag(
        ctx: RunContext[ChatDeps],
        device_id: Annotated[str, Field(description='list_devices 返回的设备 ID')],
        start_x: Annotated[int, Field(ge=0, description='起点在采集原图中的水平像素坐标')],
        start_y: Annotated[int, Field(ge=0, description='起点在采集原图中的垂直像素坐标')],
        end_x: Annotated[int, Field(ge=0, description='终点在采集原图中的水平像素坐标')],
        end_y: Annotated[int, Field(ge=0, description='终点在采集原图中的垂直像素坐标')],
        button: Literal['left', 'right', 'middle'] = 'left',
        duration_seconds: Annotated[float, Field(ge=0.2, le=5, description='按住移动的时长，单位秒')] = 0.6,
    ) -> ToolReturn | str:
        """先看当前画面确认拖拽起点、终点及目标状态，再按采集原图像素坐标拖拽；核对返回画面的结果。"""
        started = perf_counter()
        arguments = {
            'start_x': start_x,
            'start_y': start_y,
            'end_x': end_x,
            'end_y': end_y,
            'button': button,
            'duration_seconds': duration_seconds,
        }
        error = (
            _tool_error(ctx.deps)
            or ('本轮未允许操控。' if not ctx.deps.allow_control else None)
            or _screen_point_error(ctx.deps, device_id, start_x, start_y)
            or _screen_point_error(ctx.deps, device_id, end_x, end_y)
        )
        if error:
            _record(ctx, 'mouse_drag', device_id, arguments, started, error)
            return error
        try:
            result = _text(await ctx.deps.devices.call(device_id, 'mouse_drag', **arguments))
        except DeviceError as exc:
            result = f'{exc}；状态不确定时请先重新截图。'
            _record(ctx, 'mouse_drag', device_id, arguments, started, result)
            return result
        return await _observed_result(ctx, 'mouse_drag', device_id, arguments, started, result)

    @agent.tool
    async def key_press(
        ctx: RunContext[ChatDeps],
        device_id: Annotated[str, Field(description='list_devices 返回的设备 ID')],
        key: Annotated[str, Field(description="单个物理键名，例如 'enter'、'tab'、'escape'、'a'、'0'")],
        modifiers: list[Literal['ctrl', 'shift', 'alt', 'meta']] | None = None,
    ) -> ToolReturn | str:
        """先看当前画面确认界面和焦点满足前提，再发送物理按键；核对返回画面的结果。可组合修饰键。"""
        started = perf_counter()
        arguments = {'key': key, 'modifiers': modifiers}
        error = _tool_error(ctx.deps) or ('本轮未允许操控。' if not ctx.deps.allow_control else None) or _current_screen_error(ctx.deps, device_id)
        if error:
            _record(ctx, 'key_press', device_id, arguments, started, error)
            return error
        try:
            result = _text(await ctx.deps.devices.call(device_id, 'key_press', **arguments))
        except DeviceError as exc:
            result = f'{exc}；状态不确定时请先重新截图。'
            _record(ctx, 'key_press', device_id, arguments, started, result)
            return result
        return await _observed_result(ctx, 'key_press', device_id, arguments, started, result)

    @agent.tool
    async def type_text(
        ctx: RunContext[ChatDeps],
        device_id: Annotated[str, Field(description='list_devices 返回的设备 ID')],
        text: Annotated[str, Field(min_length=1, description='要键入到当前焦点的文本；不支持的字符会跳过并在结果中报告')],
    ) -> ToolReturn | str:
        """先看当前画面确认目标输入控件可用且已获焦点，再向其键入文本；核对返回画面的实际输入。"""
        started = perf_counter()
        arguments = {'text': text}
        error = _tool_error(ctx.deps) or ('本轮未允许操控。' if not ctx.deps.allow_control else None) or _current_screen_error(ctx.deps, device_id)
        if error:
            _record(ctx, 'type_text', device_id, arguments, started, error)
            return error
        try:
            method = ctx.deps.devices.text_input_method(device_id)
            timeout = max(30, len(text) * (0.25 if method == 'windows_gbk' else 0.06) + 20)
            result = _text(await ctx.deps.devices.call(device_id, 'type_text', text=text, method=method, read_timeout_seconds=timeout))
        except DeviceError as exc:
            result = f'{exc}；部分字符可能已输入，请先截图核对。'
            _record(ctx, 'type_text', device_id, arguments, started, result)
            return result
        result += '\n按键发送结果不代表文字已进入目标控件；请核对操作后画面。'
        return await _observed_result(ctx, 'type_text', device_id, arguments, started, result)

    return agent


def _current_screen_error(deps: ChatDeps, device_id: str) -> str | None:
    if device_id not in deps.frames:
        return f'设备 {device_id} 尚无本轮截图；请先调用 capture_screen 查看当前画面。'
    return None


def _display_size(size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    if width < 640 or height < 360:
        return size
    scale = min(MODEL_IMAGE_SIZE[0] / width, MODEL_IMAGE_SIZE[1] / height)
    return round(width * scale), round(height * scale)


def _present_frame(frame: Frame) -> BinaryContent:
    if _display_size(frame.size) == frame.size:
        return BinaryContent(data=frame.data, media_type=frame.mime_type)
    try:
        with Image.open(BytesIO(frame.data)) as source:
            image = source.resize(_display_size(frame.size), Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format='PNG')
    except OSError as exc:
        raise DeviceError(f'{frame.device_id} 的截图无法解码：{exc}') from exc
    return BinaryContent(data=output.getvalue(), media_type='image/png')


async def _observed_result(
    ctx: RunContext[ChatDeps],
    name: str,
    device_id: str,
    arguments: dict[str, object],
    started: float,
    result: str,
) -> ToolReturn | str:
    try:
        frame = await ctx.deps.devices.capture(device_id)
        presentation = _present_frame(frame)
    except DeviceError as exc:
        ctx.deps.frames.pop(device_id, None)
        result += f'\n操作后画面获取失败：{exc}；结果尚未核实。'
        _record(ctx, name, device_id, arguments, started, result)
        return result
    ctx.deps.frames[device_id] = frame
    _record(ctx, name, device_id, arguments, started, result, frame)
    return ToolReturn(return_value=result, content=[presentation])


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
        return f'坐标 ({x}, {y}) 超出设备 {device_id} 采集原图 {width}×{height} 的范围；请重新截图并定位。'
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
        message_history=compact_history(history, keep_images=0),
        deps=deps,
        usage_limits=UsageLimits(request_limit=max(12, deps.tool_limit + 3)),
        event_stream_handler=_stream_events if deps.on_thinking is not None else None,
    )
    usage = result.usage
    last_input_tokens = next(
        (message.usage.input_tokens for message in reversed(result.new_messages()) if isinstance(message, ModelResponse) and message.usage),
        0,
    )
    return ChatResult(result.output, result.all_messages(), result.new_messages_json(), last_input_tokens, usage.output_tokens or 0)
