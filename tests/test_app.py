import tempfile
import unittest
import io
import json
from types import SimpleNamespace
from pathlib import Path
from threading import Event
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest
from streamlit.runtime.scriptrunner_utils.exceptions import StopException
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, ModelResponse, TextPart, ThinkingPart, ToolCallPart, UserPromptPart

from chatkvm.history import History
from chatkvm.agent import ToolEvent
from chatkvm.devices import Frame
from test_agent import PNG


def ollama_urlopen(runtime_context: int | None = 8192, maximum_context: int = 262144):
    def open_url(request, timeout=3):
        del timeout
        if request.full_url.endswith('/api/ps'):
            models = [] if runtime_context is None else [{'name': 'test', 'context_length': runtime_context}]
            payload = {'models': models}
        else:
            payload = {'details': {'family': 'test'}, 'model_info': {'test.context_length': maximum_context}}
        return io.BytesIO(json.dumps(payload).encode())

    return open_url


class AppTests(unittest.TestCase):
    def test_agent_locks_controls_and_saves_results_on_completion_failure_or_stop(self):
        config = {
            'agent': {'model': {'name': 'test', 'provider': 'ollama', 'base_url': 'http://127.0.0.1:11434/v1'}},
            'kvmcp': {'pc': {'url': 'http://127.0.0.1:9527/mcp', 'description': '测试电脑'}},
        }
        for outcome in ('success', 'failure', 'stop'):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory:
                history = History(Path(directory))
                conversation = history.create()
                event = ToolEvent('list_devices', None, {}, 1, '1 台设备', tool_count=1, tool_limit=25)

                async def run_agent(_agent, _prompt, _context, deps):
                    self.assertTrue(st.session_state['agent_busy'])
                    self.assertTrue(radio.call_args.kwargs['disabled'])
                    self.assertTrue(all(call.kwargs.get('disabled') for call in buttons.call_args_list[-5:]))
                    self.assertTrue(all(call.kwargs.get('disabled') for call in toggles.call_args_list[-2:]))
                    deps.events.append(event)
                    deps.on_event(event)
                    if outcome == 'failure':
                        raise ValueError('测试失败')
                    if outcome == 'stop':
                        raise StopException()
                    return SimpleNamespace(output='已完成', messages=[], new_messages_json=b'[]', input_tokens=12, output_tokens=3)

                with (
                    patch('sys.argv', ['app.py']),
                    patch('chatkvm.config.read_config', return_value=config),
                    patch('chatkvm.history.History', return_value=history),
                    patch('urllib.request.urlopen', side_effect=ollama_urlopen()),
                    patch('chatkvm.agent.chat', side_effect=run_agent),
                    patch('streamlit.radio', wraps=st.radio) as radio,
                    patch('streamlit.button', wraps=st.button) as buttons,
                    patch('streamlit.toggle', wraps=st.toggle) as toggles,
                ):
                    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'src/chatkvm/app.py')).run(timeout=20)
                    app.chat_input[0].set_value('测试').run(timeout=20)
                    self.assertFalse(app.exception)
                    self.assertFalse(app.session_state['agent_busy'])
                    turns = history.turns(conversation)
                    self.assertEqual(len(turns), 1)
                    self.assertEqual(len(turns[0]['events']), 1)
                    self.assertEqual(turns[0]['events'][0]['tool_count'], 1)
                    self.assertEqual(turns[0]['events'][0]['tool_limit'], 25)
                    self.assertEqual(turns[0]['error'], outcome != 'success')
                    if outcome == 'stop':
                        self.assertIn('中断', turns[0]['response'])
                        self.assertTrue(app.radio(key='conversation_picker').disabled)
                        next(button for button in app.button if button.label == '恢复操作').click().run()
                        self.assertFalse(app.exception)
                    self.assertIn('1/25 查看设备', [item.label for item in app.expander])
                    self.assertFalse(app.radio(key='conversation_picker').disabled)
                    self.assertTrue(all(not button.disabled for button in app.sidebar.button))

    def test_stale_busy_state_recovers_on_next_run(self):
        config = {
            'agent': {'model': {'name': 'test', 'provider': 'ollama', 'base_url': 'http://127.0.0.1:11434/v1'}},
            'kvmcp': {'pc': {'url': 'http://127.0.0.1:9527/mcp', 'description': '测试电脑'}},
        }
        with tempfile.TemporaryDirectory() as directory:
            history = History(Path(directory))
            history.create()
            with (
                patch('sys.argv', ['app.py']),
                patch('chatkvm.config.read_config', return_value=config),
                patch('chatkvm.history.History', return_value=history),
                patch('urllib.request.urlopen', side_effect=ollama_urlopen()),
            ):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'src/chatkvm/app.py'))
                app.session_state['agent_busy'] = True
                app.run(timeout=20)
                self.assertFalse(app.exception)
                self.assertFalse(app.session_state['agent_busy'])
                self.assertFalse(app.radio(key='conversation_picker').disabled)

    def test_conversation_selection_survives_title_and_order_changes(self):
        config = {
            'agent': {'model': {'name': 'test', 'provider': 'ollama', 'base_url': 'http://127.0.0.1:11434/v1'}},
            'kvmcp': {'pc': {'url': 'http://127.0.0.1:9527/mcp', 'description': '测试电脑'}},
        }
        with tempfile.TemporaryDirectory() as directory:
            history = History(Path(directory))
            first, second = history.create(), history.create()
            with (
                patch('sys.argv', ['app.py']),
                patch('chatkvm.config.read_config', return_value=config),
                patch('chatkvm.history.History', return_value=history),
                patch('urllib.request.urlopen', side_effect=ollama_urlopen()),
            ):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'src/chatkvm/app.py'))
                app.session_state['active_chat'] = first
                app.run(timeout=20)
                history.append(first, prompt='更新后的标题', response='完成', duration_ms=1, events=[])
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(app.radio(key='conversation_picker').value, first)
                self.assertEqual(app.radio(key='conversation_picker').proto.raw_value, '更新后的标题')
                self.assertTrue(app.radio(key='conversation_picker').proto.set_value)
                history.append(second, prompt='另一对话', response='完成', duration_ms=1, events=[])
                app.run()
                self.assertEqual(app.radio(key='conversation_picker').value, first)
                app.radio(key='conversation_picker').set_value(second).run()
                self.assertEqual(app.session_state['active_chat'], second)
                self.assertEqual(app.radio(key='conversation_picker').value, second)
                history.delete(second)
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state['active_chat'], first)

    def test_voice_preloads_once_without_blocking_page(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {
                'agent': {'model': {'name': 'test', 'provider': 'ollama', 'base_url': 'http://127.0.0.1:11434/v1'}},
                'kvmcp': {'pc': {'url': 'http://127.0.0.1:9527/mcp', 'description': '测试电脑'}},
                'voice': {'model': directory, 'vad_model': directory},
            }
            history = History(Path(directory))
            started, release, finished = Event(), Event(), Event()

            def load(*args):
                started.set()
                release.wait(20)
                finished.set()
                raise ValueError('模型加载失败')

            with (
                patch('sys.argv', ['app.py']),
                patch('chatkvm.config.read_config', return_value=config),
                patch('chatkvm.history.History', return_value=history),
                patch('urllib.request.urlopen', side_effect=ollama_urlopen()),
                patch('chatkvm.speech.SpeechRecognizer', side_effect=load) as recognizer,
            ):
                try:
                    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'src/chatkvm/app.py')).run(timeout=10)
                    self.assertFalse(app.exception)
                    self.assertTrue(started.wait(2))
                    self.assertFalse(finished.is_set())
                    app.run(timeout=10)
                    self.assertFalse(app.exception)
                    recognizer.assert_called_once()
                    release.set()
                    self.assertTrue(finished.wait(2))
                    app.run(timeout=10)
                    self.assertFalse(app.exception)
                    self.assertEqual(len(app.chat_input), 1)
                finally:
                    release.set()

    def test_voice_fills_draft_without_running_agent_or_saving_audio(self):
        config = {
            'agent': {'model': {'name': 'test', 'provider': 'ollama', 'base_url': 'http://127.0.0.1:11434/v1'}},
            'kvmcp': {'pc': {'url': 'http://127.0.0.1:9527/mcp', 'description': '测试电脑'}},
            'voice': {'model': 'models/SenseVoiceSmall', 'vad_model': 'models/fsmn-vad'},
        }
        with tempfile.TemporaryDirectory() as directory:
            config['voice']['model'] = str(Path(directory) / 'model')
            config['voice']['vad_model'] = str(Path(directory) / 'vad')
            history = History(Path(directory))
            conversation = history.create()
            chat_input = st.chat_input
            def recorded_input(*args, **kwargs):
                callback, callback_args = kwargs['on_submit'], kwargs['args']

                def submit_audio():
                    st.session_state[kwargs['key']] = SimpleNamespace(text='请', audio=io.BytesIO(b'wav'))
                    callback(*callback_args)

                return chat_input(*args, **{**kwargs, 'on_submit': submit_audio, 'args': ()})

            with (
                patch('sys.argv', ['app.py']),
                patch('chatkvm.config.read_config', return_value=config),
                patch('chatkvm.history.History', return_value=history),
                patch('urllib.request.urlopen', side_effect=ollama_urlopen()),
                patch('streamlit.chat_input', side_effect=recorded_input),
                patch('chatkvm.speech.SpeechRecognizer') as recognizer,
                patch('chatkvm.agent.chat') as agent_chat,
            ):
                recognizer.return_value.transcribe.return_value = '查看屏幕'
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'src/chatkvm/app.py')).run(timeout=20)
                app.chat_input[0].set_value('recording').run(timeout=20)
                self.assertFalse(app.exception)
                self.assertEqual(app.chat_input[0].proto.value, '请\n查看屏幕')
                self.assertTrue(app.chat_input[0].proto.set_value)
                self.assertEqual(history.turns(conversation), [])
                agent_chat.assert_not_called()

    def test_voice_failure_preserves_typed_text_and_never_runs_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {
                'agent': {'model': {'name': 'test', 'provider': 'ollama', 'base_url': 'http://127.0.0.1:11434/v1'}},
                'kvmcp': {'pc': {'url': 'http://127.0.0.1:9527/mcp', 'description': '测试电脑'}},
                'voice': {'model': directory, 'vad_model': directory},
            }
            history = History(Path(directory))
            conversation = history.create()
            chat_input = st.chat_input
            def recorded_input(*args, **kwargs):
                callback, callback_args = kwargs['on_submit'], kwargs['args']

                def submit_audio():
                    st.session_state[kwargs['key']] = SimpleNamespace(text='草稿', audio=io.BytesIO(b'wav'))
                    callback(*callback_args)

                return chat_input(*args, **{**kwargs, 'on_submit': submit_audio, 'args': ()})

            with (
                patch('sys.argv', ['app.py']),
                patch('chatkvm.config.read_config', return_value=config),
                patch('chatkvm.history.History', return_value=history),
                patch('urllib.request.urlopen', side_effect=ollama_urlopen()),
                patch('streamlit.chat_input', side_effect=recorded_input),
                patch('chatkvm.speech.SpeechRecognizer') as recognizer,
                patch('chatkvm.agent.chat') as agent_chat,
            ):
                recognizer.return_value.transcribe.side_effect = ValueError('未识别到文字')
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'src/chatkvm/app.py')).run(timeout=20)
                app.chat_input[0].set_value('recording').run(timeout=20)
                self.assertFalse(app.exception)
                self.assertEqual(app.chat_input[0].proto.value, '草稿')
                self.assertIn('语音识别失败', app.error[0].value)
                self.assertEqual(history.turns(conversation), [])
                agent_chat.assert_not_called()

    def test_replays_intermediate_text_and_collapsed_thinking(self):
        config = {
            'agent': {'model': {'name': 'test', 'provider': 'ollama', 'base_url': 'http://127.0.0.1:11434/v1'}},
            'kvmcp': {'pc': {'url': 'http://127.0.0.1:9527/mcp', 'description': '测试电脑'}},
        }
        with tempfile.TemporaryDirectory() as directory:
            history = History(Path(directory))
            conversation = history.create()
            messages = [
                ModelResponse(parts=[ThinkingPart('先看设备。'), TextPart('我先查看设备。'), ToolCallPart('list_devices', {})]),
                ModelResponse(parts=[TextPart('已完成。')]),
            ]
            history.append(
                conversation,
                prompt='测试',
                response='已完成。',
                duration_ms=100,
                events=[ToolEvent('list_devices', None, {}, 0, '1 台已配置设备')],
                new_messages_json=ModelMessagesTypeAdapter.dump_json(messages),
            )
            with (
                patch('sys.argv', ['app.py']),
                patch('chatkvm.config.read_config', return_value=config),
                patch('chatkvm.history.History', return_value=history),
                patch('urllib.request.urlopen', side_effect=ollama_urlopen()),
            ):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'src/chatkvm/app.py')).run(timeout=20)
                self.assertFalse(app.exception)
                self.assertIn('我先查看设备。', [item.value for item in app.markdown])
                self.assertIn('已完成。', [item.value for item in app.markdown])
                self.assertIn('思考', [item.label for item in app.status])
                self.assertIn('先看设备。', [item.value for item in app.caption])
                self.assertNotIn('先看设备。', [item.value for item in app.markdown])

    def test_sidebar_tabs_and_shared_control_row_preserve_session_options(self):
        config = {
            'agent': {'model': {'name': 'test', 'provider': 'ollama', 'base_url': 'http://127.0.0.1:11434/v1'}},
            'kvmcp': {'pc': {'url': 'http://127.0.0.1:9527/mcp', 'description': '测试电脑'}},
        }
        with tempfile.TemporaryDirectory() as directory:
            history = History(Path(directory))
            conversation = history.create()
            events = [ToolEvent('capture_screen', 'pc', {}, 50, '截图', Frame('pc', PNG, 'image/png', '截图')) for _ in range(2)]
            events.append(ToolEvent('key_press', 'pc', {'key': 'enter'}, 50, '已发送', tool_count=3, tool_limit=100))
            history.append(
                conversation, prompt='你好', response='你好', duration_ms=1500, events=events,
                input_tokens=512, output_tokens=64, include_history=False,
            )
            with (
                patch('sys.argv', ['app.py']),
                patch('chatkvm.config.read_config', return_value=config),
                patch('chatkvm.history.History', return_value=history),
                patch('urllib.request.urlopen', side_effect=ollama_urlopen()),
            ):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'src/chatkvm/app.py')).run(timeout=20)
                self.assertFalse(app.exception)
                self.assertEqual([tab.label for tab in app.sidebar.tabs], ['对话', '设置'])
                self.assertIn('删除对话', [button.label for button in app.sidebar.button])
                controls = app.get_by_key('chat_controls')
                self.assertEqual([toggle.label for toggle in controls.toggle], ['操控', '思考', '上文'])
                self.assertIn('3/100 pc 按键', [item.label for item in app.expander])
                self.assertEqual(len(controls.selectbox), 1)
                self.assertEqual(controls.selectbox[0].options, ['500', '250', '100', '50', '25'])
                self.assertEqual(len(controls.caption), 1)
                self.assertEqual(controls.caption[0].value, '上限')
                self.assertFalse(controls.caption[0].proto.help)
                self.assertIn('用时 1.5 秒  上下文 6.2%  512 / 8,192 tokens', [caption.value for caption in app.caption])
                self.assertFalse(controls.toggle[0].proto.help)
                self.assertFalse(app.session_state['thinking'])
                self.assertFalse(app.session_state['include_history'])
                self.assertEqual(len(app.get('image')), 2)
                app.toggle(key='allow_control').set_value(False)
                app.toggle(key='thinking').set_value(True)
                app.selectbox(key='tool_limit').set_value(250)
                app.run()
                self.assertFalse(app.exception)
                self.assertFalse(app.session_state['allow_control'])
                self.assertTrue(app.session_state['thinking'])
                self.assertEqual(app.session_state['tool_limit'], 250)
                self.assertEqual(app.session_state['active_chat'], conversation)

    def test_switching_conversations_updates_turn_usage_and_preserves_recorded_window(self):
        config = {
            'agent': {'model': {'name': 'test', 'provider': 'ollama', 'base_url': 'http://127.0.0.1:11499/v1'}},
            'kvmcp': {'pc': {'url': 'http://127.0.0.1:9527/mcp', 'description': '测试电脑'}},
        }
        with tempfile.TemporaryDirectory() as directory:
            history = History(Path(directory))
            first, second, empty = [history.create() for _ in range(3)]
            for conversation, tokens in [(first, 5322), (second, 6276)]:
                history.append(
                    conversation, prompt=f'测试 {tokens}', response='完成', duration_ms=100, events=[],
                    input_tokens=tokens, context_window=262144, include_history=False,
                )
            history.append(first, prompt='失败请求', response='错误', duration_ms=100, events=[], error=True)
            with (
                patch('sys.argv', ['app.py']),
                patch('chatkvm.config.read_config', return_value=config),
                patch('chatkvm.history.History', return_value=history),
                patch('urllib.request.urlopen', side_effect=ollama_urlopen(runtime_context=None)),
            ):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'src/chatkvm/app.py'))
                app.session_state['active_chat'] = first
                app.run(timeout=20)
                for conversation, percentage, tokens in [(first, '2.0%', '5,322'), (second, '2.4%', '6,276'), (first, '2.0%', '5,322')]:
                    app.radio(key='conversation_picker').set_value(conversation).run()
                    self.assertFalse(app.exception)
                    expected = f'用时 0.1 秒  上下文 {percentage}  {tokens} / 262,144 tokens'
                    self.assertIn(expected, [caption.value for caption in app.caption])
                app.run()
                self.assertIn('用时 0.1 秒  上下文 2.0%  5,322 / 262,144 tokens', [caption.value for caption in app.caption])
                app.radio(key='conversation_picker').set_value(empty).run()
                self.assertEqual(app.get_by_key('chat_controls').caption[0].value, '上限')

                history.append(empty, prompt='旧记录', response='完成', duration_ms=100, events=[], input_tokens=6276, include_history=False)
                app.run()
                self.assertIn('用时 0.1 秒  上下文 2.4%  6,276 / 262,144 tokens', [caption.value for caption in app.caption])

    def test_previous_context_toggle_only_changes_model_input_not_saved_history(self):
        config = {
            'agent': {'model': {'name': 'test', 'provider': 'ollama', 'base_url': 'http://127.0.0.1:11434/v1'}},
            'kvmcp': {'pc': {'url': 'http://127.0.0.1:9527/mcp', 'description': '测试电脑'}},
        }
        with tempfile.TemporaryDirectory() as directory:
            history = History(Path(directory))
            conversation = history.create()
            seen_contexts = []

            async def run_agent(_agent, prompt, context, _deps):
                seen_contexts.append(len(context))
                new_messages = [ModelRequest(parts=[UserPromptPart(prompt)]), ModelResponse(parts=[TextPart('完成')])]
                return SimpleNamespace(
                    output='完成', messages=context + new_messages,
                    new_messages_json=ModelMessagesTypeAdapter.dump_json(new_messages),
                    input_tokens=400 if context else 100, output_tokens=10,
                )

            with (
                patch('sys.argv', ['app.py']),
                patch('chatkvm.config.read_config', return_value=config),
                patch('chatkvm.history.History', return_value=history),
                patch('urllib.request.urlopen', side_effect=ollama_urlopen()),
                patch('chatkvm.agent.chat', side_effect=run_agent),
            ):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'src/chatkvm/app.py')).run(timeout=20)
                self.assertFalse(app.session_state['include_history'])
                for prompt in ('第一轮', '第二轮'):
                    app.chat_input[0].set_value(prompt).run(timeout=20)
                    self.assertFalse(app.exception)
                self.assertEqual(seen_contexts, [0, 0])
                self.assertEqual(len(history.context(conversation)), 4)
                self.assertTrue(any('上下文 1.2%  100 / 8,192 tokens' in caption.value for caption in app.caption))

                app.toggle(key='include_history').set_value(True).run()
                app.chat_input[0].set_value('第三轮').run(timeout=20)
                self.assertFalse(app.exception)
                self.assertEqual(seen_contexts, [0, 0, 4])
                self.assertTrue(any('上下文 4.9%  400 / 8,192 tokens' in caption.value for caption in app.caption))

                app.toggle(key='include_history').set_value(False).run()
                app.chat_input[0].set_value('第四轮').run(timeout=20)
                self.assertFalse(app.exception)
                self.assertEqual(seen_contexts, [0, 0, 4, 0])
                self.assertEqual(len(history.context(conversation)), 8)
                self.assertTrue(any('上下文 1.2%  100 / 8,192 tokens' in caption.value for caption in app.caption))
                self.assertEqual([turn['include_history'] for turn in history.turns(conversation)], [False, False, True, False])


if __name__ == '__main__':
    unittest.main()
