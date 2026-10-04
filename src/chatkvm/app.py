"""Persistent multi-device ChatKVM chatbox."""

import argparse
import asyncio
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import streamlit as st
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from chatkvm.agent import ChatDeps, ToolEvent, build_agent, chat, resolve_model
from chatkvm.config import read_config
from chatkvm.devices import DeviceError, Devices
from chatkvm.history import History
from chatkvm.model_info import ollama_context_limits
from chatkvm.speech import SpeechRecognizer

VIDEO_SIZES = ('1920x1080', '2560x1440')
INPUT_FORMATS = ('yuyv422', 'mjpeg')
TOOL_LIMITS = (25, 50, 100, 250, 500)
DEFAULT_ALLOW_CONTROL = True
DEFAULT_TOOL_LIMIT = 25
DEFAULT_THINKING = False


TOOL_LABELS = {
    'list_devices': '查看设备',
    'capture_screen': '查看屏幕',
    'mouse_move': '移动鼠标',
    'mouse_click': '点击鼠标',
    'mouse_drag': '拖拽鼠标',
    'key_press': '按键',
    'type_text': '输入文本',
}


@st.cache_resource(max_entries=1, show_spinner=False)
def speech_recognizer(model: str, vad_model: str) -> Future[SpeechRecognizer]:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='speech-load')
    future = executor.submit(SpeechRecognizer, model, vad_model)
    executor.shutdown(wait=False)
    return future


class ThinkingDisplay:
    def __init__(self) -> None:
        self._status = None
        self._text = None
        self._content = ''
        self._updated_at = 0.0

    def append(self, chunk: str | None) -> None:
        if chunk is None:
            if self._status is not None:
                self._text.caption(self._content)
                self._status.update(label='思考', state='complete', expanded=False)
                self._status = None
            return
        if self._status is None:
            self._status = st.status('思考中', expanded=True)
            self._text = self._status.empty()
            self._content = ''
            self._updated_at = 0.0
        self._content += chunk
        now = time.perf_counter()
        if self._content and now - self._updated_at >= 0.1:
            self._text.caption(self._content)
            self._updated_at = now


def handle_submission(input_key: str, recognizer: Future[SpeechRecognizer] | None, config_error: Exception | None) -> None:
    submission = st.session_state[input_key]
    if not submission or not getattr(submission, 'audio', None):
        if submission if isinstance(submission, str) else getattr(submission, 'text', ''):
            st.session_state['agent_busy'] = True
        return
    text = submission.text or ''
    try:
        with st.spinner('正在识别语音…'):
            if recognizer is None:
                raise ValueError(f'语音配置无效：{config_error}')
            transcript = recognizer.result().transcribe(submission.audio.getvalue())
        text = '\n'.join(part for part in (text, transcript) if part)
    except Exception as exc:
        st.session_state[f'{input_key}_error'] = f'语音识别失败：{exc}'
    st.session_state[input_key] = text


def show_tool(
    name: str,
    device_id: str | None,
    arguments: dict,
    duration_ms: int,
    result: str,
    screen=None,
    *,
    tool_count: int | None = None,
    tool_limit: int | None = None,
) -> None:
    label = TOOL_LABELS.get(name, name)
    if device_id:
        label = f'{device_id} {label}'
    if tool_count is not None and tool_limit is not None:
        label = f'{tool_count}/{tool_limit} {label}'
    with st.expander(label, type='step'):
        if arguments:
            st.json(arguments, expanded=False)
        st.write(result)
        st.caption(f'{duration_ms / 1000:.1f} 秒')
    if screen is not None:
        st.image(screen, caption=device_id)


def show_event(event: ToolEvent) -> None:
    show_tool(
        event.name,
        event.device_id,
        event.arguments,
        event.duration_ms,
        event.result,
        event.frame.data if event.frame else None,
        tool_count=event.tool_count,
        tool_limit=event.tool_limit,
    )


def show_thinking(content: str) -> None:
    with st.status('思考', state='complete'):
        st.caption(content)


def show_model_response(response: ModelResponse) -> None:
    has_tool_call = any(isinstance(part, ToolCallPart) for part in response.parts)
    for part in response.parts:
        if isinstance(part, TextPart) and has_tool_call and part.content:
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
            show_tool(
                event['name'],
                event['device_id'],
                event['arguments'],
                event['duration_ms'],
                event['result'],
                screen,
                tool_count=event.get('tool_count'),
                tool_limit=event.get('tool_limit'),
            )


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


def select_conversation() -> None:
    if not st.session_state.get('agent_busy', False):
        st.session_state['active_chat'] = st.session_state['conversation_picker']


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
    st.session_state.setdefault('include_history', False)
    st.session_state.setdefault('agent_busy', False)
except (OSError, KeyError, TypeError, ValueError) as exc:
    st.session_state['agent_busy'] = False
    st.error(f'读取配置失败：{exc}')
    st.stop()

busy = st.session_state['agent_busy']
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
        if busy and st.button('恢复操作', icon=':material/refresh:', width='stretch'):
            st.session_state['agent_busy'] = False
            st.rerun()
        if st.button('新对话', icon=':material/add:', width='stretch', disabled=busy):
            st.session_state['active_chat'] = history.create()
            st.rerun()
        st.session_state['conversation_picker'] = st.session_state['active_chat']
        st.radio(
            '对话记录',
            ids,
            format_func=lambda item: next(record['title'] for record in records if record['id'] == item),
            key='conversation_picker',
            on_change=select_conversation,
            disabled=busy,
            label_visibility='collapsed',
        )
        if st.button('删除对话', icon=':material/delete:', width='stretch', disabled=busy):
            current_id = st.session_state['active_chat']
            title = next(record['title'] for record in records if record['id'] == current_id)
            confirm_delete(history, current_id, title)
    with settings_tab:
        st.caption(f'模型：{model_config["name"]}')
        selected_device = st.selectbox('设备', devices.ids(), disabled=busy)
        st.caption(next(device['description'] for device in devices.list() if device['device_id'] == selected_device))
        if st.button('查看画面', icon=':material/screenshot_monitor:', width='stretch', disabled=busy):
            try:
                frame = asyncio.run(devices.capture(selected_device))
            except DeviceError as exc:
                st.error(str(exc))
            else:
                st.image(frame.data, caption=selected_device)

        with st.container(horizontal=True, vertical_alignment='bottom'):
            video_size = st.selectbox('截图尺寸', VIDEO_SIZES, disabled=busy)
            set_size = st.button('应用', key='apply_video_size', disabled=busy)
        if set_size:
            try:
                asyncio.run(devices.call(selected_device, 'set_video_size', video_size=video_size))
            except DeviceError as exc:
                st.error(str(exc))
            else:
                st.success('截图尺寸已更新')

        with st.container(horizontal=True, vertical_alignment='bottom'):
            input_format = st.selectbox('输入格式', INPUT_FORMATS, disabled=busy)
            set_format = st.button('应用', key='apply_input_format', disabled=busy)
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
    speech_config = config.get('speech')
    speech_future = None
    speech_error = None
    if speech_config:
        try:
            model_path = str((config_path.parent / speech_config['model']).resolve())
            vad_path = str((config_path.parent / speech_config['vad_model']).resolve())
            speech_future = speech_recognizer(model_path, vad_path)
        except (KeyError, TypeError, ValueError) as exc:
            speech_error = exc
    input_key = f'chat_input_{conversation_id}'
    if error := st.session_state.pop(f'{input_key}_error', None):
        st.error(error)
    submission = st.chat_input(
        '发送消息',
        key=input_key,
        accept_audio=bool(speech_config),
        max_upload_size=4,
        submit_mode='disable',
        on_submit=handle_submission,
        args=(input_key, speech_future, speech_error),
    )
    prompt = submission.text if speech_config and submission else submission
    with st.container(horizontal=True, horizontal_alignment='center', vertical_alignment='center', gap='small', key='chat_controls'):
        allow_control = st.toggle('操控', key='allow_control', width='content', disabled=busy)
        st.caption('上限', width='content')
        tool_limit = st.selectbox(
            '操控上限', sorted(TOOL_LIMITS, reverse=True), key='tool_limit', width=90, label_visibility='collapsed', disabled=busy
        )
        thinking = st.toggle('思考', key='thinking', width='content', disabled=busy)
        include_history = st.toggle('前文', key='include_history', width='content', disabled=busy)
        last_success = next((turn for turn in reversed(turns) if not turn['error']), None)
        usage_turn = last_success if last_success and last_success.get('include_history', True) == include_history else None
        input_tokens = usage_turn['input_tokens'] if usage_turn else 0
        context_window = (usage_turn.get('context_window') if usage_turn else None) or runtime_context
        context_window = context_window or model_context_max
        usage = f'{input_tokens / context_window:.1%}' if usage_turn and context_window else ('0.0%' if not turns else '—')
        usage_help = f'{input_tokens:,} / {context_window:,} tokens' if usage_turn and context_window else None
        st.caption(usage, help=usage_help, width='content')

if busy and not prompt:
    st.session_state['agent_busy'] = False
    st.rerun()

if prompt:
    with st.chat_message('user'):
        st.markdown(prompt)
    deps = ChatDeps(devices, allow_control=allow_control, tool_limit=int(tool_limit), on_event=show_event, on_model_response=show_model_response)
    started = time.perf_counter()
    response = '本轮操作已中断；已完成的指令不会撤销，请先查看当前屏幕再继续。'
    failed = True
    result_metadata = {}
    with st.chat_message('assistant'), st.spinner('正在处理…'):
        try:
            if thinking:
                deps.on_thinking = ThinkingDisplay().append
            previous_context = history.context(conversation_id)
            agent = build_agent(model, allow_control=allow_control, thinking=thinking)
            result = asyncio.run(chat(agent, prompt, previous_context if include_history else [], deps))
        except Exception as exc:
            response = f'模型或工具调用失败：{exc}'
            st.error(response)
        else:
            if runtime_context is None:
                ollama_context_limits.clear()
            context_window, _ = ollama_context_limits(model_config)
            response = result.output
            failed = False
            result_metadata = {
                'messages': result.messages if include_history else previous_context + result.messages,
                'new_messages_json': result.new_messages_json,
                'input_tokens': result.input_tokens,
                'output_tokens': result.output_tokens,
                'context_window': context_window,
                'include_history': include_history,
            }
            st.markdown(result.output)
        finally:
            try:
                history.append(
                    conversation_id,
                    prompt=prompt,
                    response=response,
                    duration_ms=round((time.perf_counter() - started) * 1000),
                    events=deps.events,
                    error=failed,
                    **result_metadata,
                )
            finally:
                st.session_state['agent_busy'] = False
    st.rerun()
