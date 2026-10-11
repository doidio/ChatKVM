"""Prepare and distribute resumable batch tasks across KVMCP devices."""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import sqlite3
import unicodedata
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chatkvm.config import read_config


class FlowError(ValueError):
    """A flow configuration or input is invalid."""


class TaskError(Exception):
    """An expected failure while processing one task."""


@dataclass(frozen=True)
class FlowConfig:
    chatkvm_path: Path
    task_path: Path
    input_path: Path
    save_path: Path
    column: int
    graph_name: str
    graph: dict[str, Any]
    devices: tuple[FlowDevice, ...]
    chatkvm: dict[str, Any]


@dataclass(frozen=True)
class ImportResult:
    files: int
    values: int
    added: int
    total: int


@dataclass(frozen=True)
class FlowDevice:
    name: str
    url: str


@dataclass(frozen=True)
class ClaimedTask:
    value: str
    attempts: int


@dataclass(frozen=True)
class RunResult:
    claimed: int
    succeeded: int
    failed: int


@dataclass(frozen=True)
class FlowResult:
    imported: ImportResult
    executed: RunResult


TaskRunner = Callable[[FlowConfig, FlowDevice, ClaimedTask], Awaitable[None]]


def _resolve(base: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise FlowError(f'{field} must be a non-empty path')
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_flow_config(chatkvm_path: Path, task_path: Path) -> FlowConfig:
    """Load shared ChatKVM settings and one task configuration."""
    chatkvm_path = chatkvm_path.expanduser().resolve()
    task_path = task_path.expanduser().resolve()
    chatkvm = read_config(chatkvm_path)
    task = read_config(task_path)

    column = task.get('column')
    if type(column) is not int or column < 1:
        raise FlowError('column must be a positive, one-based integer')
    graph_name = task.get('graph')
    if not isinstance(graph_name, str) or not graph_name.strip():
        raise FlowError('graph must be a non-empty name')
    graph = chatkvm.get('graph', {}).get(graph_name)
    if not isinstance(graph, dict):
        raise FlowError(f'graph {graph_name!r} is not defined in {chatkvm_path}')
    devices = _graph_devices(chatkvm.get('kvmcp'), graph_name)

    base = task_path.parent
    input_path = _resolve(base, task.get('input'), 'input')
    if not input_path.exists():
        raise FlowError(f'input does not exist: {input_path}')
    if input_path.is_file() and input_path.suffix.lower() != '.csv':
        raise FlowError(f'input file must be CSV: {input_path}')

    return FlowConfig(
        chatkvm_path=chatkvm_path,
        task_path=task_path,
        input_path=input_path,
        save_path=_resolve(base, task.get('save'), 'save'),
        column=column,
        graph_name=graph_name,
        graph=graph,
        devices=devices,
        chatkvm=chatkvm,
    )


def _graph_devices(value: object, graph_name: str) -> tuple[FlowDevice, ...]:
    if not isinstance(value, dict):
        raise FlowError('chatkvm configuration must define kvmcp devices')
    devices = []
    for name, settings in value.items():
        if not isinstance(name, str) or not isinstance(settings, dict):
            raise FlowError('each kvmcp device must be a named table')
        graphs = settings.get('graphs', [])
        if not isinstance(graphs, list) or any(not isinstance(item, str) or not item for item in graphs):
            raise FlowError(f'kvmcp device {name!r} graphs must be an array of names')
        if graph_name not in graphs:
            continue
        url = settings.get('url')
        if not isinstance(url, str) or not url.strip():
            raise FlowError(f'kvmcp device {name!r} url must be a non-empty string')
        devices.append(FlowDevice(name, url))
    if not devices:
        raise FlowError(f'no kvmcp device supports graph {graph_name!r}')
    return tuple(devices)


def _csv_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted((candidate for candidate in path.rglob('*') if candidate.is_file() and candidate.suffix.lower() == '.csv'), key=str)
    if not files:
        raise FlowError(f'no CSV files found in {path}')
    return files


def _encoding(path: Path) -> str:
    sample = path.read_bytes()[:65536]
    if sample.startswith(b'\xef\xbb\xbf'):
        return 'utf-8-sig'
    try:
        sample.decode('utf-8')
    except UnicodeDecodeError:
        return 'gb18030'
    return 'utf-8'


def _fallback_header(sample: str, dialect: type[csv.Dialect] | csv.Dialect, index: int) -> bool:
    rows = [row for row in csv.reader(io.StringIO(sample), dialect) if any(cell.strip() for cell in row)][:21]
    if len(rows) < 2 or index >= len(rows[0]):
        return False
    first = rows[0][index].strip()
    following = [row[index].strip() for row in rows[1:] if index < len(row) and row[index].strip()]
    numeric = lambda value: value.replace('.', '', 1).isdigit()
    return bool(first and following and not numeric(first) and sum(numeric(value) for value in following) > len(following) / 2)


def _dialect_and_header(path: Path, encoding: str, index: int) -> tuple[type[csv.Dialect] | csv.Dialect, bool]:
    with path.open(encoding=encoding, newline='') as source:
        sample = source.read(65536)
    sniff = csv.Sniffer()
    try:
        dialect = sniff.sniff(sample, delimiters=',\t;|')
    except csv.Error:
        dialect = csv.excel
    try:
        has_header = sniff.has_header(sample)
    except (csv.Error, ValueError):
        has_header = _fallback_header(sample, dialect, index)
    return dialect, has_header


def read_task_values(path: Path, column: int) -> tuple[set[str], int]:
    """Read and deduplicate one one-based column from one CSV or a directory."""
    values: set[str] = set()
    files = _csv_files(path)
    index = column - 1
    for csv_path in files:
        encoding = _encoding(csv_path)
        dialect, has_header = _dialect_and_header(csv_path, encoding, index)
        with csv_path.open(encoding=encoding, newline='') as source:
            rows = csv.reader(source, dialect)
            first_nonempty = True
            for record, row in enumerate(rows, 1):
                if not any(cell.strip() for cell in row):
                    continue
                if first_nonempty:
                    first_nonempty = False
                    if has_header:
                        continue
                if index >= len(row):
                    raise FlowError(f'{csv_path}:{record} has only {len(row)} columns; requested column {column}')
                value = unicodedata.normalize('NFKC', row[index]).strip()
                if value:
                    values.add(value)
    return values, len(files)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) STRICT;
CREATE TABLE IF NOT EXISTS tasks (
    value TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'succeeded', 'failed', 'paused', 'skipped')),
    node TEXT,
    device TEXT,
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    error TEXT,
    started_at TEXT,
    finished_at TEXT,
    updated_at TEXT
) STRICT;
CREATE INDEX IF NOT EXISTS tasks_status ON tasks(status);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=5)
    connection.execute('PRAGMA journal_mode = WAL')
    connection.execute('PRAGMA synchronous = NORMAL')
    connection.execute('PRAGMA busy_timeout = 5000')
    return connection


def _migrate_tasks(connection: sqlite3.Connection) -> None:
    columns = {row[1] for row in connection.execute('PRAGMA table_info(tasks)')}
    for name in ('device', 'started_at', 'finished_at'):
        if name not in columns:
            connection.execute(f'ALTER TABLE tasks ADD COLUMN {name} TEXT')


def sync_tasks(config: FlowConfig) -> ImportResult:
    """Add newly discovered values without changing existing task state."""
    config.save_path.parent.mkdir(parents=True, exist_ok=True)
    connection = _connect(config.save_path)
    try:
        connection.executescript(_SCHEMA)
        _migrate_tasks(connection)
        expected = {'graph': config.graph_name, 'column': str(config.column)}
        existing = dict(connection.execute("SELECT key, value FROM metadata WHERE key IN ('graph', 'column')"))
        for key, value in expected.items():
            if key in existing and existing[key] != value:
                raise FlowError(f'{config.save_path} belongs to {key}={existing[key]!r}, not {value!r}')
            connection.execute('INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)', (key, value))
        values, file_count = read_task_values(config.input_path, config.column)
        before = connection.execute('SELECT count(*) FROM tasks').fetchone()[0]
        connection.executemany('INSERT OR IGNORE INTO tasks(value) VALUES (?)', ((value,) for value in sorted(values)))
        total = connection.execute('SELECT count(*) FROM tasks').fetchone()[0]
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return ImportResult(files=file_count, values=len(values), added=total - before, total=total)


class TaskQueue:
    """A short-transaction SQLite queue shared by all device workers."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def recover(self) -> int:
        """Return tasks left running by an interrupted execution to the queue."""
        connection = _connect(self.path)
        try:
            cursor = connection.execute(
                """UPDATE tasks SET status = 'pending', device = NULL, started_at = NULL,
                   updated_at = ? WHERE status = 'running' """,
                (_now(),),
            )
            connection.commit()
            return cursor.rowcount
        finally:
            connection.close()

    def claim(self, device: str) -> ClaimedTask | None:
        connection = _connect(self.path)
        try:
            now = _now()
            row = connection.execute(
                """
                WITH next AS (
                    SELECT value FROM tasks WHERE status = 'pending' ORDER BY rowid LIMIT 1
                )
                UPDATE tasks
                SET status = 'running', device = ?, attempts = attempts + 1,
                    error = NULL, started_at = ?, finished_at = NULL, updated_at = ?
                WHERE value = (SELECT value FROM next) AND status = 'pending'
                RETURNING value, attempts
                """,
                (device, now, now),
            ).fetchone()
            connection.commit()
            return ClaimedTask(row[0], row[1]) if row else None
        finally:
            connection.close()

    def succeed(self, task: ClaimedTask) -> None:
        self._finish(task, 'succeeded', None)

    def fail(self, task: ClaimedTask, error: BaseException) -> None:
        self._finish(task, 'failed', str(error))

    def release(self, task: ClaimedTask) -> None:
        connection = _connect(self.path)
        try:
            connection.execute(
                """UPDATE tasks SET status = 'pending', device = NULL, started_at = NULL,
                   updated_at = ? WHERE value = ? AND status = 'running' """,
                (_now(), task.value),
            )
            connection.commit()
        finally:
            connection.close()

    def _finish(self, task: ClaimedTask, status: str, error: str | None) -> None:
        connection = _connect(self.path)
        try:
            now = _now()
            connection.execute(
                """UPDATE tasks SET status = ?, error = ?, finished_at = ?, updated_at = ?
                   WHERE value = ? AND status = 'running' """,
                (status, error, now, now, task.value),
            )
            connection.commit()
        finally:
            connection.close()


async def run_workers(config: FlowConfig, runner: TaskRunner) -> RunResult:
    """Run one worker per compatible device until the shared queue is empty."""
    queue = TaskQueue(config.save_path)
    counts = {'claimed': 0, 'succeeded': 0, 'failed': 0}
    lock = asyncio.Lock()

    async def worker(device: FlowDevice) -> None:
        while task := queue.claim(device.name):
            async with lock:
                counts['claimed'] += 1
            try:
                await runner(config, device, task)
            except asyncio.CancelledError:
                queue.release(task)
                raise
            except TaskError as exc:
                queue.fail(task, exc)
                async with lock:
                    counts['failed'] += 1
            else:
                queue.succeed(task)
                async with lock:
                    counts['succeeded'] += 1

    async with asyncio.TaskGroup() as group:
        for device in config.devices:
            group.create_task(worker(device), name=device.name)
    return RunResult(**counts)


async def execute_flow(chatkvm_path: Path, task_path: Path, runner: TaskRunner) -> FlowResult:
    """Synchronize inputs, recover interruptions, and run all compatible workers."""
    config = load_flow_config(chatkvm_path, task_path)
    imported = sync_tasks(config)
    TaskQueue(config.save_path).recover()
    return FlowResult(imported, await run_workers(config, runner))


def run(chatkvm_path: Path, task_path: Path) -> ImportResult:
    return sync_tasks(load_flow_config(chatkvm_path, task_path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Prepare a resumable ChatKVM batch task list.')
    parser.add_argument('chatkvm', type=Path, help='shared chatkvm.toml')
    parser.add_argument('task', type=Path, help='task TOML whose directory is the task directory')
    args = parser.parse_args(argv)
    try:
        result = run(args.chatkvm, args.task)
    except (FlowError, OSError, sqlite3.Error) as exc:
        parser.error(str(exc))
    config = load_flow_config(args.chatkvm, args.task)
    device_names = ', '.join(device.name for device in config.devices)
    print(f'{result.files} CSV files, {result.values} unique values, {result.added} added, {result.total} total')
    print(f'graph {config.graph_name}: {device_names}')
    return 0


__all__ = [
    'ClaimedTask',
    'FlowConfig',
    'FlowDevice',
    'FlowError',
    'FlowResult',
    'ImportResult',
    'RunResult',
    'TaskError',
    'TaskQueue',
    'execute_flow',
    'load_flow_config',
    'main',
    'read_task_values',
    'run',
    'run_workers',
    'sync_tasks',
]
