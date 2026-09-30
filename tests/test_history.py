import tempfile
import unittest
from pathlib import Path

from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart, ToolCallPart

from chatkvm.history import History, _timeline


class HistoryTests(unittest.TestCase):
    def test_legacy_timeline_matches_parallel_tool_results_by_device(self):
        messages = [
            ModelResponse(
                parts=[
                    ThinkingPart('检查两台设备。'),
                    TextPart('我先查看画面。'),
                    ToolCallPart('capture_screen', {'device_id': 'a'}),
                    ToolCallPart('capture_screen', {'device_id': 'b'}),
                ]
            ),
            ModelResponse(parts=[TextPart('已完成。')]),
        ]
        events = [
            {'name': 'capture_screen', 'device_id': 'b'},
            {'name': 'capture_screen', 'device_id': 'a'},
        ]
        timeline = _timeline(messages, events)
        self.assertEqual([item['kind'] for item in timeline], ['thinking', 'text', 'tool', 'tool'])
        self.assertEqual([item['event']['device_id'] for item in timeline[2:]], ['a', 'b'])

    def test_delete_moves_only_selected_conversation_to_trash(self):
        with tempfile.TemporaryDirectory() as directory:
            history = History(Path(directory))
            selected = history.create()
            retained = history.create()
            screenshot = Path(directory, selected, 'screens', 'capture.png')
            screenshot.parent.mkdir()
            screenshot.write_bytes(b'capture')

            history.delete(selected)

            self.assertEqual([record['id'] for record in history.list()], [retained])
            self.assertEqual(Path(directory, '.trash', selected, 'screens', 'capture.png').read_bytes(), b'capture')
            self.assertTrue(Path(directory, retained, 'meta.json').is_file())

    def test_delete_rejects_invalid_id_and_missing_conversation(self):
        with tempfile.TemporaryDirectory() as directory:
            history = History(Path(directory))
            with self.assertRaises(ValueError):
                history.delete('../other')
            with self.assertRaises(FileNotFoundError):
                history.delete('00000000-0000-0000-0000-000000000001')


if __name__ == '__main__':
    unittest.main()
