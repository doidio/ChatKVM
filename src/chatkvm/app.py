"""Persistent multi-device ChatKVM chatbox."""

import argparse
import asyncio
import time
from pathlib import Path

import streamlit as st
from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart, ToolCallPart

from chatkvm.agent import ChatDeps, ToolEvent, build_agent, chat, resolve_model
from chatkvm.config import read_config
from chatkvm.devices import DeviceError, Devices
from chatkvm.history import History
from chatkvm.model_info import ollama_context_limits

VIDEO_SIZES = ('1920x1080', '2560x1440')
INPUT_FORMATS = ('yuyv422', 'mjpeg')
TOOL_LIMITS = (10, 25, 50, 100, 250, 500)
DEFAULT_ALLOW_CONTROL = True
DEFAULT_TOOL_LIMIT = 10
DEFAULT_THINKING = False

TOOL_LABELS = {
    'list_devices': '查看设备',
    'capture_screen': '查看屏幕',
    'wake_display': '旧版唤醒尝试',
    'mouse_move': '移动鼠标',
    'mouse_click': '单击鼠标',
    'key_press': '按键',
}


def show_tool(name: str, device_id: str | None, arguments: dict, duration_ms: int, result: str, screen=None) -> None:
    label = TOOL_LABELS.get(name, name)
    if device_id:
        label = f'{device_id} {label}'
    with st.expander(label, type='step'):
        if arguments:
            st.json(arguments, expanded=False)
        st.write(result)
        st.caption(f'{duration_ms / 1000:.1f} 秒')
    if screen is not None:
        st.image(screen, caption=device_id)


def show_event(event: ToolEvent) -> None:
    show_tool(event.name, event.device_id, event.arguments, event.duration_ms, event.result, event.frame.data if event.frame else None)


def show_thinking(content: str) -> None:
    with st.expander('思考', type='compact'):
        st.markdown(content)


def show_model_response(response: ModelResponse) -> None:
    has_tool_call = any(isinstance(part, ToolCallPart) for part in response.parts)
    for part in response.parts:
        if isinstance(part, ThinkingPart) and part.content:
            show_thinking(part.content)
        elif isinstance(part, TextPart) and has_tool_call and part.content:
            st.markdown(part.content)


def show_timeline(history: History, conversation_id: str, turn_number: int, turn: dict) -> None:
    for item in history.timeline(conversation_id, turn_number, turn):
        if item['kind'] == 'thinking':
            show_thinking(item['text'])
        elif item['kind'] == 'text':
            st.markdown(item['text'])
        else:
            event = item['event']
            screen = str(history.screen_path(conversation_id, event['screen'])) if 'screen' in event else None
            show_tool(event['name'], event['device_id'], event['arguments'], event['duration_ms'], event['result'], screen)


@st.dialog('删除对话')
def confirm_delete(history: History, conversation_id: str, title: str) -> None:
    st.write(f'确定删除“{title}”吗？对话和截图将从列表中移除。')
    if st.button('确认删除', type='primary'):
        try:
            history.delete(conversation_id)
        except OSError as exc:
            st.error(f'删除失败：{exc}')
        else:
            st.rerun()


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default='config.toml')
args = parser.parse_args()

st.set_page_config(page_title='ChatKVM', layout='wide')

try:
    config_path = Path(args.config).resolve()
    config = read_config(config_path)
    model_config = config['agent']['model']
    model = resolve_model(model_config)
    devices = Devices(config['kvmcp'])
    history = History(config_path.parent / 'history')
    st.session_state.setdefault('allow_control', DEFAULT_ALLOW_CONTROL)
    st.session_state.setdefault('tool_limit', DEFAULT_TOOL_LIMIT)
    st.session_state.setdefault('thinking', DEFAULT_THINKING)
except (OSError, KeyError, TypeError, ValueError) as exc:
    st.error(f'读取配置失败：{exc}')
    st.stop()

records = history.list()
if not records:
    history.create()
    records = history.list()
ids = [record['id'] for record in records]
if st.session_state.get('active_chat') not in ids:
    st.session_state['active_chat'] = ids[0]

with st.sidebar:
    st.header('ChatKVM')
    conversations_tab, settings_tab = st.tabs(['对话', '设置'], key='sidebar_tabs')
    with conversations_tab:
        if st.button('新对话', icon=':material/add:', width='stretch'):
            st.session_state['active_chat'] = history.create()
            st.rerun()
        st.radio(
            '对话记录',
            ids,
            format_func=lambda item: next(record['title'] for record in records if record['id'] == item),
            key='active_chat',
            label_visibility='collapsed',
        )
        if st.button('删除对话', icon=':material/delete:', width='stretch'):
            current_id = st.session_state['active_chat']
            title = next(record['title'] for record in records if record['id'] == current_id)
            confirm_delete(history, current_id, title)
    with settings_tab:
        st.caption(f'模型：{model_config["name"]}')
        selected_device = st.selectbox('设备', devices.ids())
        st.caption(next(device['description'] for device in devices.list() if device['device_id'] == selected_device))
        if st.button('查看画面', icon=':material/screenshot_monitor:', width='stretch'):
            try:
                frame = asyncio.run(devices.capture(selected_device))
            except DeviceError as exc:
                st.error(str(exc))
            else:
                st.image(frame.data, caption=selected_device)

        with st.container(horizontal=True, vertical_alignment='bottom'):
            video_size = st.selectbox('截图尺寸', VIDEO_SIZES)
            set_size = st.button('应用', key='apply_video_size')
        if set_size:
            try:
                asyncio.run(devices.call(selected_device, 'set_video_size', video_size=video_size))
            except DeviceError as exc:
                st.error(str(exc))
            else:
                st.success('截图尺寸已更新')

        with st.container(horizontal=True, vertical_alignment='bottom'):
            input_format = st.selectbox('输入格式', INPUT_FORMATS)
            set_format = st.button('应用', key='apply_input_format')
        if set_format:
            try:
                asyncio.run(devices.call(selected_device, 'set_input_format', input_format=input_format))
            except DeviceError as exc:
                st.error(str(exc))
            else:
                st.success('输入格式已更新')

conversation_id = st.session_state['active_chat']
turns = history.turns(conversation_id)
runtime_context, model_context_max = ollama_context_limits(model_config)

for turn_number, turn in enumerate(turns, start=1):
    with st.chat_message('user'):
        st.markdown(turn['prompt'])
    with st.chat_message('assistant'):
        show_timeline(history, conversation_id, turn_number, turn)
        if turn['error']:
            st.error(turn['response'])
        else:
            st.markdown(turn['response'])
        st.caption(f'用时 {turn["duration_ms"] / 1000:.1f} 秒')

with st.bottom:
    prompt = st.chat_input('发送消息', submit_mode='disable')
    with st.container(horizontal=True, horizontal_alignment='center', vertical_alignment='center', gap='small', key='chat_controls'):
        allow_control = st.toggle('操控', key='allow_control', width='content')
        thinking = st.toggle('思考', key='thinking', width='content')
        st.caption('上限', width='content')
        tool_limit = st.selectbox('每轮工具调用上限', sorted(TOOL_LIMITS, reverse=True), key='tool_limit', width=90, label_visibility='collapsed')
        usage_turn = next((turn for turn in reversed(turns) if not turn['error']), None)
        input_tokens = usage_turn['input_tokens'] if usage_turn else 0
        context_window = (usage_turn.get('context_window') if usage_turn else None) or runtime_context
        estimate = context_window is None and model_context_max is not None
        context_window = context_window or model_context_max
        usage = f'上下文{"约 " if estimate else " "}{input_tokens / context_window:.1%}' if context_window else '上下文 —'
        usage_help = f'{input_tokens:,} / {context_window:,} tokens' if context_window else f'{input_tokens:,} / — tokens'
        st.caption(usage, help=usage_help, width='content')

if prompt:
    with st.chat_message('user'):
        st.markdown(prompt)
    deps = ChatDeps(devices, allow_control=allow_control, tool_limit=int(tool_limit), on_event=show_event, on_model_response=show_model_response)
    started = time.perf_counter()
    with st.chat_message('assistant'), st.spinner('正在处理…'):
        try:
            agent = build_agent(model, allow_control=allow_control, thinking=thinking)
            result = asyncio.run(chat(agent, prompt, history.context(conversation_id), deps))
        except Exception as exc:
            response = f'模型或工具调用失败：{exc}'
            history.append(
                conversation_id,
                prompt=prompt,
                response=response,
                duration_ms=round((time.perf_counter() - started) * 1000),
                events=deps.events,
                error=True,
            )
            st.error(response)
        else:
            if runtime_context is None:
                ollama_context_limits.clear()
            context_window, _ = ollama_context_limits(model_config)
            history.append(
                conversation_id,
                prompt=prompt,
                response=result.output,
                duration_ms=round((time.perf_counter() - started) * 1000),
                events=deps.events,
                messages=result.messages,
                new_messages_json=result.new_messages_json,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                context_window=context_window,
            )
            st.markdown(result.output)
    st.rerun()
