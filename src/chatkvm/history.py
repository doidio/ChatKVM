"""Local, persistent conversation record and separate model continuation context."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelResponse, TextPart, ThinkingPart, ToolCallPart

from chatkvm.agent import ToolEvent, compact_history


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _timeline(messages: list[ModelMessage], events: list[dict]) -> list[dict]:
    remaining = list(events)
    timeline = []
    responses = [message for message in messages if isinstance(message, ModelResponse)]
    final_response = responses[-1] if responses else None
    for message in responses:
        for part in message.parts:
            if isinstance(part, ThinkingPart) and part.content:
                timeline.append({'kind': 'thinking', 'text': part.content})
            elif isinstance(part, TextPart) and message is not final_response and part.content:
                timeline.append({'kind': 'text', 'text': part.content})
            elif isinstance(part, ToolCallPart):
                arguments = part.args_as_dict()
                index = next((i for i, event in enumerate(remaining) if event.get('tool_call_id') == part.tool_call_id and part.tool_call_id), None)
                if index is None:
                    index = next(
                        (i for i, event in enumerate(remaining) if event['name'] == part.tool_name and event['device_id'] == arguments.get('device_id')),
                        None,
                    )
                if index is not None:
                    timeline.append({'kind': 'tool', 'event': remaining.pop(index)})
    timeline.extend({'kind': 'tool', 'event': event} for event in remaining)
    return timeline


class History:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _directory(self, conversation_id: str) -> Path:
        try:
            parsed = UUID(conversation_id)
        except ValueError as exc:
            raise ValueError('无效的对话 ID') from exc
        if str(parsed) != conversation_id:
            raise ValueError('无效的对话 ID')
        return self.root / conversation_id

    def create(self) -> str:
        conversation_id = str(uuid4())
        directory = self._directory(conversation_id)
        directory.mkdir()
        self._write_metadata(directory, {'id': conversation_id, 'title': '新对话', 'created_at': _now(), 'updated_at': _now()})
        return conversation_id

    def delete(self, conversation_id: str) -> None:
        directory = self._directory(conversation_id)
        if not directory.is_dir() or directory.is_symlink():
            raise FileNotFoundError('对话不存在')
        trash = self.root / '.trash'
        trash.mkdir(exist_ok=True)
        destination = trash / conversation_id
        if destination.exists():
            raise FileExistsError('回收目录中已有同名对话')
        directory.rename(destination)

    def list(self) -> list[dict]:
        records = []
        for path in self.root.iterdir():
            if path.is_dir():
                try:
                    records.append(json.loads((path / 'meta.json').read_text(encoding='utf-8')))
                except (OSError, ValueError):
                    continue
        return sorted(records, key=lambda item: item['updated_at'], reverse=True)

    def _write_metadata(self, directory: Path, metadata: dict) -> None:
        _atomic_write(directory / 'meta.json', json.dumps(metadata, ensure_ascii=False).encode('utf-8'))

    def turns(self, conversation_id: str) -> list[dict]:
        path = self._directory(conversation_id) / 'turns.jsonl'
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]

    def context(self, conversation_id: str) -> list[ModelMessage]:
        path = self._directory(conversation_id) / 'context.json'
        return ModelMessagesTypeAdapter.validate_json(path.read_bytes()) if path.exists() else []

    def timeline(self, conversation_id: str, turn_number: int, turn: dict) -> list[dict]:
        if 'timeline' in turn:
            return turn['timeline']
        path = self._directory(conversation_id) / 'runs' / f'{turn_number:04d}.json'
        if path.exists():
            try:
                return _timeline(ModelMessagesTypeAdapter.validate_json(path.read_bytes()), turn['events'])
            except (OSError, ValueError):
                pass
        return [{'kind': 'tool', 'event': event} for event in turn['events']]

    def append(
        self,
        conversation_id: str,
        *,
        prompt: str,
        response: str,
        duration_ms: int,
        events: list[ToolEvent],
        messages: list[ModelMessage] | None = None,
        new_messages_json: bytes | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        context_window: int | None = None,
        error: bool = False,
    ) -> dict:
        directory = self._directory(conversation_id)
        metadata = json.loads((directory / 'meta.json').read_text(encoding='utf-8'))
        index = len(self.turns(conversation_id)) + 1
        event_records = []
        for event_number, event in enumerate(events, start=1):
            record = {
                'name': event.name,
                'device_id': event.device_id,
                'arguments': event.arguments,
                'duration_ms': event.duration_ms,
                'result': event.result,
            }
            if event.tool_call_id:
                record['tool_call_id'] = event.tool_call_id
            if event.frame:
                screen_dir = directory / 'screens'
                screen_dir.mkdir(exist_ok=True)
                relative = f'screens/{index:04d}-{event_number:03d}.png'
                (directory / relative).write_bytes(event.frame.data)
                record['screen'] = relative
            event_records.append(record)
        turn = {
            'at': _now(),
            'prompt': prompt,
            'response': response,
            'duration_ms': duration_ms,
            'events': event_records,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'error': error,
            'context_window': context_window,
        }
        if new_messages_json is not None:
            turn['timeline'] = _timeline(ModelMessagesTypeAdapter.validate_json(new_messages_json), event_records)
        if new_messages_json is not None:
            runs = directory / 'runs'
            runs.mkdir(exist_ok=True)
            _atomic_write(runs / f'{index:04d}.json', new_messages_json)
        with (directory / 'turns.jsonl').open('a', encoding='utf-8') as file:
            file.write(json.dumps(turn, ensure_ascii=False) + '\n')
        if messages is not None:
            _atomic_write(directory / 'context.json', ModelMessagesTypeAdapter.dump_json(compact_history(messages)))
        if metadata['title'] == '新对话':
            metadata['title'] = prompt.replace('\n', ' ').strip()[:40] or '新对话'
        metadata['updated_at'] = _now()
        self._write_metadata(directory, metadata)
        return turn

    def screen_path(self, conversation_id: str, relative: str) -> Path:
        if not relative.startswith('screens/') or Path(relative).name != relative.removeprefix('screens/'):
            raise ValueError('无效的截图路径')
        return self._directory(conversation_id) / relative
