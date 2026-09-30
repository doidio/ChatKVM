import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelResponse, TextPart, ThinkingPart, ToolCallPart

from chatkvm.history import History
from chatkvm.agent import ToolEvent
from chatkvm.devices import Frame
from test_agent import PNG


class AppTests(unittest.TestCase):
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
                patch('chatkvm.model_info.ollama_context_limits', return_value=(8192, 262144)),
            ):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'src/chatkvm/app.py')).run(timeout=20)
                self.assertFalse(app.exception)
                self.assertIn('我先查看设备。', [item.value for item in app.markdown])
                self.assertIn('已完成。', [item.value for item in app.markdown])
                self.assertIn('思考', [item.label for item in app.expander])

    def test_sidebar_tabs_and_shared_control_row_preserve_session_options(self):
        config = {
            'agent': {'model': {'name': 'test', 'provider': 'ollama', 'base_url': 'http://127.0.0.1:11434/v1'}},
            'kvmcp': {'pc': {'url': 'http://127.0.0.1:9527/mcp', 'description': '测试电脑'}},
        }
        with tempfile.TemporaryDirectory() as directory:
            history = History(Path(directory))
            conversation = history.create()
            events = [ToolEvent('capture_screen', 'pc', {}, 50, '截图', Frame('pc', PNG, 'image/png', '截图')) for _ in range(2)]
            events.append(ToolEvent('key_press', 'pc', {'key': 'enter'}, 50, '已发送'))
            history.append(conversation, prompt='你好', response='你好', duration_ms=1500, events=events, input_tokens=512, output_tokens=64)
            with (
                patch('sys.argv', ['app.py']),
                patch('chatkvm.config.read_config', return_value=config),
                patch('chatkvm.history.History', return_value=history),
                patch('chatkvm.model_info.ollama_context_limits', return_value=(8192, 262144)),
            ):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'src/chatkvm/app.py')).run(timeout=20)
                self.assertFalse(app.exception)
                self.assertEqual([tab.label for tab in app.sidebar.tabs], ['对话', '设置'])
                self.assertIn('删除对话', [button.label for button in app.sidebar.button])
                controls = app.get_by_key('chat_controls')
                self.assertEqual([toggle.label for toggle in controls.toggle], ['操控', '思考'])
                self.assertIn('pc 按键', [item.label for item in app.expander])
                self.assertEqual(len(controls.selectbox), 1)
                self.assertEqual(controls.selectbox[0].options, ['500', '250', '100', '50', '25', '10'])
                self.assertEqual(len(controls.caption), 2)
                self.assertEqual(controls.caption[0].value, '上限')
                self.assertFalse(controls.caption[0].proto.help)
                self.assertEqual(controls.caption[1].value, '上下文 6.2%')
                self.assertEqual(controls.caption[1].proto.help, '512 / 8,192 tokens')
                self.assertFalse(controls.toggle[0].proto.help)
                self.assertFalse(app.session_state['thinking'])
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

    def test_switching_conversations_updates_usage_and_preserves_recorded_window(self):
        config = {
            'agent': {'model': {'name': 'test', 'provider': 'ollama', 'base_url': 'http://127.0.0.1:11434/v1'}},
            'kvmcp': {'pc': {'url': 'http://127.0.0.1:9527/mcp', 'description': '测试电脑'}},
        }
        with tempfile.TemporaryDirectory() as directory:
            history = History(Path(directory))
            first, second, empty = [history.create() for _ in range(3)]
            for conversation, tokens in [(first, 5322), (second, 6276)]:
                history.append(
                    conversation, prompt=f'测试 {tokens}', response='完成', duration_ms=100, events=[], input_tokens=tokens, context_window=262144
                )
            history.append(first, prompt='失败请求', response='错误', duration_ms=100, events=[], error=True)
            with (
                patch('sys.argv', ['app.py']),
                patch('chatkvm.config.read_config', return_value=config),
                patch('chatkvm.history.History', return_value=history),
                patch('chatkvm.model_info.ollama_context_limits', return_value=(131072, 262144)) as limits,
            ):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'src/chatkvm/app.py'))
                app.session_state['active_chat'] = first
                app.run(timeout=20)
                for conversation, percentage, tokens in [(first, '2.0%', '5,322'), (second, '2.4%', '6,276'), (first, '2.0%', '5,322')]:
                    app.radio(key='active_chat').set_value(conversation).run()
                    self.assertFalse(app.exception)
                    usage = app.get_by_key('chat_controls').caption[1]
                    self.assertEqual(usage.value, f'上下文 {percentage}')
                    self.assertEqual(usage.proto.help, f'{tokens} / 262,144 tokens')
                limits.return_value = (None, 262144)
                app.run()
                self.assertEqual(app.get_by_key('chat_controls').caption[1].value, '上下文 2.0%')
                limits.return_value = (131072, 262144)
                app.radio(key='active_chat').set_value(empty).run()
                self.assertEqual(app.get_by_key('chat_controls').caption[1].value, '上下文 0.0%')

                history.append(empty, prompt='旧记录', response='完成', duration_ms=100, events=[], input_tokens=6276)
                limits.return_value = (None, 262144)
                app.run()
                usage = app.get_by_key('chat_controls').caption[1]
                self.assertEqual(usage.value, '上下文约 2.4%')
                self.assertEqual(usage.proto.help, '6,276 / 262,144 tokens')


if __name__ == '__main__':
    unittest.main()
