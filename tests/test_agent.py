import asyncio
import base64
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from mcp.types import CallToolResult, ImageContent, TextContent
from PIL import Image
from pydantic_ai import BinaryContent, ModelResponse, TextPart, ThinkingPart, ToolCallPart
from pydantic_ai.models.function import DeltaThinkingPart, DeltaToolCall, FunctionModel
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from chatkvm.agent import ChatDeps, _display_size, _present_frame, _screen_point_error, build_agent, chat
from chatkvm.devices import CAPTURE_TIMEOUT_SECONDS, DeviceError, Devices, Frame
from chatkvm.history import History

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/lXcAAAAASUVORK5CYII=')


class FakeClient:
    urls: ClassVar[list[str]] = []

    def __init__(self, url: str) -> None:
        self.url = url

    async def __aenter__(self):
        self.urls.append(self.url)
        return self

    async def __aexit__(self, *_args):
        return None

    async def call_tool(self, name: str, _arguments, read_timeout_seconds: int):
        assert name == 'latest_frame'
        assert read_timeout_seconds == CAPTURE_TIMEOUT_SECONDS
        return CallToolResult(
            content=[
                TextContent(type='text', text='截图尺寸 1x1 像素'),
                ImageContent(type='image', data=base64.b64encode(PNG).decode(), mimeType='image/png'),
            ]
        )


class DeviceTests(unittest.IsolatedAsyncioTestCase):
    async def test_routes_capture_to_selected_device(self):
        devices = Devices(
            {'pc_a': {'url': 'http://a/mcp', 'description': 'A'}, 'pc_b': {'url': 'http://b/mcp', 'description': 'B'}},
        )
        FakeClient.urls = []
        with patch('chatkvm.devices.Client', FakeClient):
            frame = await devices.capture('pc_b')
        self.assertEqual(FakeClient.urls, ['http://b/mcp'])
        self.assertEqual(frame.data, PNG)
        self.assertEqual(frame.mime_type, 'image/png')
        self.assertEqual(frame.size, (1, 1))

    async def test_unknown_device_never_connects(self):
        devices = Devices({'pc_a': {'url': 'http://a/mcp', 'description': 'A'}})
        FakeClient.urls = []
        with patch('chatkvm.devices.Client', FakeClient), self.assertRaises(DeviceError):
            await devices.capture('not_configured')
        self.assertEqual(FakeClient.urls, [])

    async def test_capture_timeout_reports_device_may_be_offline(self):
        devices = Devices({'pc_a': {'url': 'http://a/mcp', 'description': 'A'}})

        async def slow_call(*_args, **_kwargs):
            await asyncio.sleep(1)

        with patch.object(devices, 'call', side_effect=slow_call), patch('chatkvm.devices.CAPTURE_TIMEOUT_SECONDS', 0.001):
            with self.assertRaisesRegex(DeviceError, '请求失败，设备可能离线'):
                await devices.capture('pc_a')

    async def test_capture_connection_failure_reports_device_may_be_offline(self):
        devices = Devices({'pc_a': {'url': 'http://a/mcp', 'description': 'A'}})
        with patch.object(devices, 'call', side_effect=DeviceError('连接被拒绝')):
            with self.assertRaisesRegex(DeviceError, '请求失败，设备可能离线'):
                await devices.capture('pc_a')


class FakeDevices:
    def __init__(self):
        self.calls = []

    def list(self):
        return [{'device_id': 'pc_a', 'description': '测试电脑'}]

    async def capture(self, device_id: str):
        return Frame(device_id, PNG, 'image/png', '截图尺寸 1x1 像素')

    async def call(self, device_id: str, tool_name: str, **arguments):
        self.calls.append((device_id, tool_name, arguments))
        return CallToolResult(content=[TextContent(type='text', text='已发送')])


class AgentTests(unittest.TestCase):
    def make_agent(self, *, allow_control=True):
        model = OllamaModel('qwen3.8:27b', provider=OllamaProvider(base_url='http://127.0.0.1:11435/v1'))
        return build_agent(model, allow_control=allow_control)

    def make_deps(self, devices, *, allow_control=True, tool_limit=10, frames=None):
        return ChatDeps(devices, allow_control=allow_control, tool_limit=tool_limit, frames=frames or {})

    def run_chat(self, agent, prompt, deps):
        return asyncio.run(chat(agent, prompt, [], deps))

    def test_thinking_setting_and_intermediate_response_timeline(self):
        agent = self.make_agent()
        self.assertEqual(agent.model_settings['openai_reasoning_effort'], 'none')
        self.assertNotIn('temperature', agent.model_settings)
        model = OllamaModel('qwen3.8:27b', provider=OllamaProvider(base_url='http://127.0.0.1:11435/v1'))
        self.assertEqual(build_agent(model, allow_control=True, thinking=True).model_settings['openai_reasoning_effort'], 'medium')
        deps = self.make_deps(FakeDevices())
        responses = []
        deps.on_model_response = responses.append
        calls = 0

        def model_response(_messages, _info):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(parts=[ThinkingPart('先检查设备。'), TextPart('我先查看设备。'), ToolCallPart('list_devices', {})])
            return ModelResponse(parts=[ThinkingPart('设备已找到。'), TextPart('找到设备。')])

        with agent.override(model=FunctionModel(model_response)):
            result = self.run_chat(agent, '查设备', deps)
        self.assertEqual(len(responses), 2)
        self.assertEqual(deps.events[0].tool_call_id, responses[0].parts[2].tool_call_id)
        with tempfile.TemporaryDirectory() as directory:
            history = History(Path(directory))
            conversation = history.create()
            turn = history.append(
                conversation, prompt='查设备', response=result.output, duration_ms=100, events=deps.events, new_messages_json=result.new_messages_json
            )
            timeline = history.timeline(conversation, 1, turn)
            self.assertEqual([item['kind'] for item in timeline], ['thinking', 'text', 'tool', 'thinking'])
            self.assertEqual(timeline[1]['text'], '我先查看设备。')
            self.assertEqual(timeline[2]['event']['name'], 'list_devices')

    def test_thinking_streams_before_final_response(self):
        agent = self.make_agent()
        deps = self.make_deps(FakeDevices())
        live = []
        deps.on_thinking = lambda chunk: live.append(('thinking', chunk))
        deps.on_model_response = lambda response: live.append(('response', response))
        calls = 0

        async def stream(_messages, _info):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield {0: DeltaThinkingPart(content='先检查')}
                yield {0: DeltaThinkingPart(content='设备。')}
                yield {1: DeltaToolCall(name='list_devices', json_args='{}')}
            else:
                yield {0: DeltaThinkingPart(content='设备已找到。')}
                yield '已完成。'

        with agent.override(model=FunctionModel(stream_function=stream)):
            result = self.run_chat(agent, '测试思考', deps)
        self.assertEqual(result.output, '已完成。')
        self.assertEqual([event.name for event in deps.events], ['list_devices'])
        self.assertEqual(''.join(chunk for kind, chunk in live if kind == 'thinking' and chunk), '先检查设备。设备已找到。')
        self.assertLess(next(i for i, item in enumerate(live) if item[0] == 'thinking'), next(i for i, item in enumerate(live) if item[0] == 'response'))

    def test_tool_limit_counts_read_only_calls_too(self):
        agent = self.make_agent()
        devices = FakeDevices()
        deps = self.make_deps(devices, tool_limit=2)
        calls = 0

        def model(_messages, _info):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(parts=[ToolCallPart('list_devices', {})])
            if calls == 2:
                return ModelResponse(parts=[ToolCallPart('capture_screen', {'device_id': 'pc_a'})])
            if calls == 3:
                return ModelResponse(parts=[ToolCallPart('key_press', {'device_id': 'pc_a', 'key': 'enter'})])
            return ModelResponse(parts=[TextPart('上限已达到。')])

        with agent.override(model=FunctionModel(model)):
            result = self.run_chat(agent, '测试调用次数', deps)
        self.assertEqual(result.output, '上限已达到。')
        self.assertEqual(deps.tool_count, 2)
        self.assertEqual([event.name for event in deps.events], ['list_devices', 'capture_screen', 'key_press'])
        self.assertEqual([(event.tool_count, event.tool_limit) for event in deps.events], [(1, 2), (2, 2), (2, 2)])
        self.assertIn('上限', deps.events[-1].result)
        self.assertEqual(devices.calls, [])

    def test_tool_budget_updates_each_request_and_resets_next_turn(self):
        agent = self.make_agent()
        history = []
        for _ in range(2):
            deps = self.make_deps(FakeDevices(), tool_limit=100)
            calls = 0

            def model(_messages, info):
                nonlocal calls
                self.assertIn(f'本轮工具调用上限 100 次，已用 {calls} 次，剩余 {100 - calls} 次。', info.instructions)
                calls += 1
                if calls <= 2:
                    return ModelResponse(parts=[ToolCallPart('list_devices', {})])
                return ModelResponse(parts=[TextPart('完成。')])

            with agent.override(model=FunctionModel(model)):
                result = asyncio.run(chat(agent, '查看设备', history, deps))
            history = result.messages
            self.assertEqual(deps.tool_count, 2)
            self.assertEqual([(event.tool_count, event.tool_limit) for event in deps.events], [(1, 100), (2, 100)])

    def test_disabling_control_hides_hid_tools(self):
        agent = self.make_agent(allow_control=False)
        deps = self.make_deps(FakeDevices(), allow_control=False)

        def model(_messages, info):
            self.assertEqual([tool.name for tool in info.function_tools], ['list_devices', 'capture_screen'])
            return ModelResponse(parts=[TextPart('只能查看。')])

        with agent.override(model=FunctionModel(model)):
            result = self.run_chat(agent, '查看', deps)
        self.assertEqual(result.output, '只能查看。')

    def test_persists_full_run_and_compact_context(self):
        agent = self.make_agent()
        deps = self.make_deps(FakeDevices())
        calls = 0

        def model(_messages, _info):
            nonlocal calls
            calls += 1
            if calls <= 2:
                return ModelResponse(parts=[ToolCallPart('capture_screen', {'device_id': 'pc_a'})])
            return ModelResponse(parts=[TextPart('看到画面。')])

        with agent.override(model=FunctionModel(model)):
            result = self.run_chat(agent, '截图', deps)
        with tempfile.TemporaryDirectory() as directory:
            history = History(Path(directory))
            conversation_id = history.create()
            history.append(
                conversation_id,
                prompt='截图',
                response=result.output,
                duration_ms=123,
                events=deps.events,
                messages=result.messages,
                new_messages_json=result.new_messages_json,
            )
            self.assertEqual(history.list()[0]['title'], '截图')
            self.assertEqual(history.turns(conversation_id)[0]['duration_ms'], 123)
            self.assertEqual(history.screen_path(conversation_id, 'screens/0001-001.png').read_bytes(), PNG)
            self.assertEqual(history.screen_path(conversation_id, 'screens/0001-002.png').read_bytes(), PNG)
            self.assertIn(b'iVBOR', (Path(directory) / conversation_id / 'runs' / '0001.json').read_bytes())
            context = history.context(conversation_id)
            self.assertTrue(context)
            self.assertEqual(
                sum(
                    isinstance(item, BinaryContent)
                    for message in context
                    for part in message.parts
                    for item in (part.content if isinstance(getattr(part, 'content', None), list) else [])
                ),
                0,
            )
            observed = []

            def next_turn(messages, _info):
                observed.extend(
                    item for message in messages for part in message.parts
                    for item in (part.content if isinstance(getattr(part, 'content', None), list) else [])
                    if isinstance(item, BinaryContent)
                )
                return ModelResponse(parts=[TextPart('下一轮。')])

            with agent.override(model=FunctionModel(next_turn)):
                asyncio.run(chat(agent, '继续', context, self.make_deps(FakeDevices())))
            self.assertEqual(observed, [])

    def test_each_model_request_keeps_two_recent_images_and_emits_all_events(self):
        agent = self.make_agent()
        deps = self.make_deps(FakeDevices())
        events = []
        deps.on_event = events.append
        calls = 0

        async def capture(device_id):
            return Frame(device_id, PNG + bytes([calls]), 'image/png', '截图')

        def model(messages, _info):
            nonlocal calls
            calls += 1
            images = [
                item
                for message in messages
                for part in message.parts
                for item in (part.content if isinstance(getattr(part, 'content', None), list) else [])
                if isinstance(item, BinaryContent)
            ]
            self.assertEqual(len(images), min(2, calls - 1))
            if images:
                self.assertEqual([image.data[-1] for image in images], list(range(max(1, calls - 2), calls)))
            self.assertEqual(len(events), calls - 1)
            if calls <= 7:
                return ModelResponse(parts=[ToolCallPart('capture_screen', {'device_id': 'pc_a'})])
            return ModelResponse(parts=[TextPart('完成。')])

        with patch.object(deps.devices, 'capture', capture), agent.override(model=FunctionModel(model)):
            result = self.run_chat(agent, '依次查看画面', deps)
        self.assertEqual(len(events), 7)
        self.assertEqual(events, deps.events)
        self.assertEqual(result.new_messages_json.count(b'iVBOR'), 7)

    def test_pixel_coordinates_require_current_frame_and_stay_unconverted(self):
        devices = FakeDevices()
        deps = self.make_deps(devices)
        self.assertIn('capture_screen', _screen_point_error(deps, 'pc_a', 0, 0))

        deps.frames['pc_a'] = Frame('pc_a', PNG, 'image/png', '截图尺寸 1000x1000 像素')
        self.assertIsNone(_screen_point_error(deps, 'pc_a', 0, 0))
        self.assertIn('超出', _screen_point_error(deps, 'pc_a', 1, 0))
        self.assertIn('超出', _screen_point_error(deps, 'pc_a', 0, 1))

    def test_full_screen_presentation_preserves_capture_coordinate_bounds(self):
        for source_size in ((1920, 1080), (2560, 1440)):
            with self.subTest(source_size=source_size):
                output = BytesIO()
                Image.new('RGB', source_size, 'white').save(output, format='PNG')
                frame = Frame('pc_a', output.getvalue(), 'image/png', '截图')
                self.assertEqual(_display_size(frame.size), (3840, 2160))
                with Image.open(BytesIO(_present_frame(frame).data)) as presented:
                    self.assertEqual(presented.size, (3840, 2160))
                deps = self.make_deps(FakeDevices(), frames={'pc_a': frame})
                self.assertIsNone(_screen_point_error(deps, 'pc_a', source_size[0] - 1, source_size[1] - 1))
                self.assertIn('超出', _screen_point_error(deps, 'pc_a', source_size[0], source_size[1]))

    def test_mouse_tool_passes_capture_coordinates_without_rescaling(self):
        output = BytesIO()
        Image.new('RGB', (2560, 1440), 'white').save(output, format='PNG')
        devices = FakeDevices()
        deps = self.make_deps(devices, frames={'pc_a': Frame('pc_a', output.getvalue(), 'image/png', '截图')})
        agent = self.make_agent()
        calls = 0

        def model(_messages, _info):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(parts=[ToolCallPart('mouse_click', {'device_id': 'pc_a', 'x': 1280, 'y': 720})])
            return ModelResponse(parts=[TextPart('已点击。')])

        with agent.override(model=FunctionModel(model)):
            self.run_chat(agent, '点击中心', deps)
        self.assertEqual(devices.calls[0], ('pc_a', 'mouse_click', {'x': 1280, 'y': 720, 'button': 'left', 'count': 1}))

    def test_name_field_and_query_button_coordinates_do_not_halve(self):
        output = BytesIO()
        Image.new('RGB', (1920, 1080), 'white').save(output, format='PNG')
        frame = Frame('pc_a', output.getvalue(), 'image/png', '截图')
        devices = FakeDevices()
        deps = self.make_deps(devices, frames={'pc_a': frame})
        agent = self.make_agent()
        calls = 0

        def model(_messages, info):
            nonlocal calls
            calls += 1
            click = next(tool for tool in info.function_tools if tool.name == 'mouse_click')
            self.assertIn('采集原图像素坐标', click.description)
            if calls == 1:
                return ModelResponse(parts=[ToolCallPart('mouse_click', {'device_id': 'pc_a', 'x': 650, 'y': 169})])
            if calls == 2:
                deps.frames['pc_a'] = frame
                return ModelResponse(parts=[ToolCallPart('mouse_click', {'device_id': 'pc_a', 'x': 1664, 'y': 169})])
            return ModelResponse(parts=[TextPart('完成。')])

        with agent.override(model=FunctionModel(model)):
            self.run_chat(agent, '操作输入框和查询按钮', deps)
        self.assertEqual([(call[2]['x'], call[2]['y']) for call in devices.calls], [(650, 169), (1664, 169)])

    def test_mouse_tool_rejects_out_of_bounds_without_calling_device(self):
        agent = self.make_agent()
        devices = FakeDevices()
        deps = self.make_deps(devices)
        calls = 0

        def model(_messages, _info):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(parts=[ToolCallPart('capture_screen', {'device_id': 'pc_a'})])
            if calls == 2:
                return ModelResponse(parts=[ToolCallPart('mouse_click', {'device_id': 'pc_a', 'x': 700, 'y': 0})])
            return ModelResponse(parts=[TextPart('坐标越界，未点击。')])

        with agent.override(model=FunctionModel(model)):
            result = self.run_chat(agent, '测试越界坐标', deps)
        self.assertEqual(result.output, '坐标越界，未点击。')
        self.assertEqual(devices.calls, [])

    def test_mouse_tool_passes_pixel_coordinates_unchanged(self):
        agent = self.make_agent()
        devices = FakeDevices()
        deps = self.make_deps(devices, frames={'pc_a': Frame('pc_a', PNG, 'image/png', '截图尺寸 1x1 像素')})
        calls = 0

        def model(_messages, _info):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(parts=[ToolCallPart('mouse_move', {'device_id': 'pc_a', 'x': 0, 'y': 0})])
            return ModelResponse(parts=[TextPart('已移动。')])

        with agent.override(model=FunctionModel(model)):
            self.run_chat(agent, '移动鼠标', deps)
        self.assertEqual(devices.calls, [('pc_a', 'mouse_move', {'x': 0, 'y': 0})])

    def test_double_click_is_one_device_call(self):
        agent = self.make_agent()
        devices = FakeDevices()
        deps = self.make_deps(devices, frames={'pc_a': Frame('pc_a', PNG, 'image/png', '截图')})
        calls = 0

        def model(_messages, info):
            nonlocal calls
            calls += 1
            tool = next(tool for tool in info.function_tools if tool.name == 'mouse_click')
            self.assertEqual(tool.parameters_json_schema['properties']['count']['enum'], [1, 2])
            if calls == 1:
                return ModelResponse(parts=[ToolCallPart('mouse_click', {'device_id': 'pc_a', 'x': 0, 'y': 0, 'count': 2})])
            return ModelResponse(parts=[TextPart('已双击。')])

        with agent.override(model=FunctionModel(model)):
            self.run_chat(agent, '双击', deps)
        self.assertEqual(devices.calls, [('pc_a', 'mouse_click', {'x': 0, 'y': 0, 'button': 'left', 'count': 2})])
        self.assertEqual(deps.tool_count, 1)

    def test_drag_is_one_device_call_and_validates_both_endpoints(self):
        agent = self.make_agent()
        devices = FakeDevices()
        deps = self.make_deps(devices, frames={'pc_a': Frame('pc_a', PNG, 'image/png', '截图尺寸 1x1 像素')})
        calls = 0

        def model(_messages, info):
            nonlocal calls
            calls += 1
            tool = next(tool for tool in info.function_tools if tool.name == 'mouse_drag')
            self.assertEqual(tool.parameters_json_schema['properties']['duration_seconds']['default'], 0.6)
            if calls == 1:
                return ModelResponse(parts=[ToolCallPart('mouse_drag', {'device_id': 'pc_a', 'start_x': 0, 'start_y': 0, 'end_x': 1, 'end_y': 0})])
            if calls == 2:
                return ModelResponse(parts=[ToolCallPart('mouse_drag', {'device_id': 'pc_a', 'start_x': 0, 'start_y': 0, 'end_x': 0, 'end_y': 0})])
            return ModelResponse(parts=[TextPart('完成。')])

        with agent.override(model=FunctionModel(model)):
            self.run_chat(agent, '拖拽', deps)
        self.assertIn('超出', deps.events[0].result)
        self.assertEqual(devices.calls, [('pc_a', 'mouse_drag', {
            'start_x': 0, 'start_y': 0, 'end_x': 0, 'end_y': 0, 'button': 'left', 'duration_seconds': 0.6,
        })])
        self.assertEqual(deps.tool_count, 2)

    def test_key_press_requires_a_capture_in_the_current_turn(self):
        agent = self.make_agent()
        devices = FakeDevices()
        deps = self.make_deps(devices)
        calls = 0

        def model(_messages, _info):
            nonlocal calls
            calls += 1
            if calls in (1, 3):
                return ModelResponse(parts=[ToolCallPart('key_press', {'device_id': 'pc_a', 'key': 'enter'})])
            if calls == 2:
                return ModelResponse(parts=[ToolCallPart('capture_screen', {'device_id': 'pc_a'})])
            return ModelResponse(parts=[TextPart('完成。')])

        with agent.override(model=FunctionModel(model)):
            self.run_chat(agent, '按回车', deps)
        self.assertIn('本轮截图', deps.events[0].result)
        self.assertEqual([event.name for event in deps.events], ['key_press', 'capture_screen', 'key_press'])
        self.assertEqual(devices.calls, [('pc_a', 'key_press', {'key': 'enter', 'modifiers': None})])

    def test_agent_lists_devices_through_one_tool(self):
        agent = self.make_agent()
        deps = self.make_deps(FakeDevices())
        calls = 0

        def model(messages, info):
            nonlocal calls
            calls += 1
            self.assertIn('每次键鼠操作前，依据当前画面确认目标状态、焦点等操作前提', info.instructions)
            assert [tool.name for tool in info.function_tools] == [
                'list_devices',
                'capture_screen',
                'mouse_move',
                'mouse_click',
                'mouse_drag',
                'key_press',
                'type_text',
            ]
            if calls == 1:
                return ModelResponse(parts=[ToolCallPart('list_devices', {})])
            assert any(part.part_kind == 'tool-return' for message in messages for part in message.parts)
            return ModelResponse(parts=[TextPart('共配置了 1 台设备：pc_a。')])

        with agent.override(model=FunctionModel(model)):
            result = self.run_chat(agent, '有哪些设备？', deps)
        self.assertEqual(result.output, '共配置了 1 台设备：pc_a。')

    def test_text_tool_leaves_input_method_to_device(self):
        agent = self.make_agent()
        devices = FakeDevices()
        deps = self.make_deps(devices, frames={'pc_a': Frame('pc_a', PNG, 'image/png', '截图')})
        calls = 0

        def model(_messages, _info):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(parts=[ToolCallPart('type_text', {'device_id': 'pc_a', 'text': '中文 ABC'})])
            return ModelResponse(parts=[TextPart('已发送。')])

        with agent.override(model=FunctionModel(model)):
            self.run_chat(agent, '输入文本', deps)
        self.assertEqual(len(devices.calls), 1)
        device_id, tool_name, arguments = devices.calls[0]
        self.assertEqual((device_id, tool_name), ('pc_a', 'type_text'))
        self.assertEqual(arguments['text'], '中文 ABC')
        self.assertNotIn('method', arguments)
        self.assertGreaterEqual(arguments['read_timeout_seconds'], 30)
        self.assertIn('不代表文字已进入', deps.events[0].result)

    def test_state_changing_tools_return_fresh_frame_without_extra_tool_call(self):
        cases = [
            ('mouse_click', {'device_id': 'pc_a', 'x': 0, 'y': 0}),
            ('mouse_drag', {'device_id': 'pc_a', 'start_x': 0, 'start_y': 0, 'end_x': 0, 'end_y': 0}),
            ('key_press', {'device_id': 'pc_a', 'key': 'enter'}),
            ('type_text', {'device_id': 'pc_a', 'text': 'A'}),
        ]
        for name, arguments in cases:
            with self.subTest(tool=name):
                agent = self.make_agent()
                devices = FakeDevices()
                deps = self.make_deps(devices, frames={'pc_a': Frame('pc_a', PNG, 'image/png', '旧画面')})
                calls = 0

                def model(messages, _info, tool_name=name, tool_arguments=arguments):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        return ModelResponse(parts=[ToolCallPart(tool_name, tool_arguments)])
                    images = [
                        item for message in messages for part in message.parts
                        for item in (part.content if isinstance(getattr(part, 'content', None), list) else [])
                        if isinstance(item, BinaryContent)
                    ]
                    self.assertEqual([item.data for item in images], [PNG])
                    self.assertFalse(any(
                        '显示图' in part.content
                        for message in messages for part in message.parts
                        if isinstance(getattr(part, 'content', None), str)
                    ))
                    return ModelResponse(parts=[TextPart('已核对。')])

                with patch.object(devices, 'capture', wraps=devices.capture) as capture, agent.override(model=FunctionModel(model)):
                    self.run_chat(agent, '测试操作后画面', deps)
                self.assertEqual(capture.call_count, 1)
                self.assertEqual(deps.tool_count, 1)
                self.assertEqual(deps.events[0].frame, deps.frames['pc_a'])
                self.assertEqual(devices.calls[0][1], name)

    def test_action_capture_failure_does_not_claim_result_verified(self):
        agent = self.make_agent()
        devices = FakeDevices()
        deps = self.make_deps(devices, frames={'pc_a': Frame('pc_a', PNG, 'image/png', '旧画面')})
        calls = 0

        def model(_messages, _info):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(parts=[ToolCallPart('mouse_click', {'device_id': 'pc_a', 'x': 0, 'y': 0})])
            return ModelResponse(parts=[TextPart('结果尚未核实。')])

        with patch.object(devices, 'capture', side_effect=DeviceError('截图离线')), agent.override(model=FunctionModel(model)):
            self.run_chat(agent, '测试截图失败', deps)
        self.assertEqual(len(devices.calls), 1)
        self.assertEqual(deps.frames, {})
        self.assertIn('结果尚未核实', deps.events[0].result)
        self.assertIsNone(deps.events[0].frame)

    def test_delayed_capture_waits_before_fetching_frame(self):
        agent = self.make_agent()
        devices = FakeDevices()
        deps = self.make_deps(devices)
        calls = 0
        order = []
        original_capture = devices.capture

        async def wait(seconds):
            order.append(('wait', seconds))

        async def capture(device_id):
            order.append(('capture', device_id))
            return await original_capture(device_id)

        def model(_messages, _info):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(parts=[ToolCallPart('capture_screen', {'device_id': 'pc_a', 'wait_seconds': 2.5})])
            return ModelResponse(parts=[TextPart('已查看。')])

        with patch('chatkvm.agent.asyncio.sleep', wait), patch.object(devices, 'capture', capture), agent.override(model=FunctionModel(model)):
            self.run_chat(agent, '稍后截图', deps)
        self.assertEqual(order, [('wait', 2.5), ('capture', 'pc_a')])
        self.assertEqual(deps.events[0].arguments, {'wait_seconds': 2.5})

    def test_agent_receives_screen_image(self):
        agent = self.make_agent()
        deps = self.make_deps(FakeDevices())
        calls = 0

        def model(messages, _info):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(parts=[ToolCallPart('capture_screen', {'device_id': 'pc_a'})])
            assert any(
                isinstance(content, BinaryContent)
                for message in messages
                for part in message.parts
                for content in (part.content if isinstance(getattr(part, 'content', None), list) else [])
            )
            return ModelResponse(parts=[TextPart('看到了 pc_a 的画面。')])

        with agent.override(model=FunctionModel(model)):
            result = self.run_chat(agent, '查看 pc_a', deps)

        self.assertEqual(result.output, '看到了 pc_a 的画面。')
        self.assertEqual(deps.frames['pc_a'].data, PNG)
        self.assertTrue(result.messages)

    def test_capture_reports_image_dimensions_without_device_instructions(self):
        agent = self.make_agent()
        devices = FakeDevices()
        deps = self.make_deps(devices)
        calls = 0

        async def capture(device_id):
            return Frame(device_id, PNG, 'image/png', 'backend-only-metadata')

        def model(messages, _info):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(parts=[ToolCallPart('capture_screen', {'device_id': 'pc_a'})])
            returns = [part.content for message in messages for part in message.parts if part.part_kind == 'tool-return']
            self.assertIn('1×1', str(returns))
            self.assertNotIn('backend-only-metadata', str(messages))
            return ModelResponse(parts=[TextPart('看到画面。')])

        with patch.object(devices, 'capture', capture), agent.override(model=FunctionModel(model)):
            self.run_chat(agent, '查看 pc_a', deps)
        self.assertEqual(deps.events[0].result, 'backend-only-metadata')


if __name__ == '__main__':
    unittest.main()
