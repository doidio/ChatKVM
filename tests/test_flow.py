import asyncio
import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

import chatkvm.flow
from chatkvm.flow import FlowError, TaskError, load_flow_config, run, run_workers


class FlowPackageTests(unittest.TestCase):
    def test_flow_namespace_is_importable(self):
        self.assertIsNotNone(chatkvm.flow)

    def test_imports_one_based_column_and_preserves_existing_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root / 'chatkvm.toml'
            shared.write_text('[kvmcp.one]\nurl = "http://one/mcp"\ngraphs = ["example"]\n[graph.example]\n', encoding='utf-8')
            task_dir = root / 'task'
            task_dir.mkdir()
            task = task_dir / 'example.toml'
            task.write_text('input = "input.csv"\ncolumn = 2\ngraph = "example"\nsave = "state.db"\n', encoding='utf-8')
            source = task_dir / 'input.csv'
            with source.open('w', encoding='utf-8-sig', newline='') as stream:
                csv.writer(stream).writerows([['姓名', '身份证号'], ['甲', ' 123 '], ['乙', ''], ['丙', '123'], ['丁', '456']])

            first = run(shared, task)
            self.assertEqual((first.files, first.values, first.added, first.total), (1, 2, 2, 2))
            database = sqlite3.connect(task_dir / 'state.db')
            database.execute("UPDATE tasks SET status = 'succeeded' WHERE value = '123'")
            database.commit()
            database.close()
            with source.open('a', encoding='utf-8', newline='') as stream:
                csv.writer(stream).writerow(['戊', '789'])

            second = run(shared, task)
            self.assertEqual((second.values, second.added, second.total), (3, 1, 3))
            database = sqlite3.connect(task_dir / 'state.db')
            self.assertEqual(database.execute("SELECT status FROM tasks WHERE value = '123'").fetchone()[0], 'succeeded')
            database.close()

    def test_recursively_reads_headerless_csv_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / 'inputs'
            (inputs / 'more').mkdir(parents=True)
            (inputs / 'first.csv').write_text('1,100\n2,200\n', encoding='utf-8')
            (inputs / 'more' / 'second.CSV').write_text('3,200\n4,300\n', encoding='utf-8')
            values, files = chatkvm.flow.read_task_values(inputs, 2)
            self.assertEqual(files, 2)
            self.assertEqual(values, {'100', '200', '300'})

    def test_rejects_unknown_graph_and_database_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root / 'chatkvm.toml'
            shared.write_text('[kvmcp.one]\nurl = "http://one/mcp"\ngraphs = ["known"]\n[graph.known]\n', encoding='utf-8')
            source = root / 'input.csv'
            source.write_text('value\n123\n', encoding='utf-8')
            task = root / 'task.toml'
            task.write_text('input = "input.csv"\ncolumn = 1\ngraph = "missing"\nsave = "state.db"\n', encoding='utf-8')
            with self.assertRaisesRegex(FlowError, 'not defined'):
                load_flow_config(shared, task)

            task.write_text('input = "input.csv"\ncolumn = 1\ngraph = "known"\nsave = "state.db"\n', encoding='utf-8')
            run(shared, task)
            task.write_text('input = "input.csv"\ncolumn = 2\ngraph = "known"\nsave = "state.db"\n', encoding='utf-8')
            with self.assertRaisesRegex(FlowError, 'belongs to column'):
                run(shared, task)

    def test_selects_every_device_supporting_the_task_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root / 'chatkvm.toml'
            shared.write_text(
                '[kvmcp.one]\nurl = "http://one/mcp"\ngraphs = ["example"]\n'
                '[kvmcp.two]\nurl = "http://two/mcp"\ngraphs = ["other", "example"]\n'
                '[kvmcp.three]\nurl = "http://three/mcp"\ngraphs = ["other"]\n'
                '[graph.example]\n[graph.other]\n',
                encoding='utf-8',
            )
            (root / 'input.csv').write_text('value\n1\n', encoding='utf-8')
            task = root / 'task.toml'
            task.write_text('input = "input.csv"\ncolumn = 1\ngraph = "example"\nsave = "state.db"\n', encoding='utf-8')

            config = load_flow_config(shared, task)
            self.assertEqual([device.name for device in config.devices], ['one', 'two'])

    def test_async_workers_share_tasks_and_persist_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root / 'chatkvm.toml'
            shared.write_text(
                '[kvmcp.one]\nurl = "http://one/mcp"\ngraphs = ["example"]\n'
                '[kvmcp.two]\nurl = "http://two/mcp"\ngraphs = ["example"]\n'
                '[graph.example]\n',
                encoding='utf-8',
            )
            (root / 'input.csv').write_text('value\n1\n2\n3\n4\n', encoding='utf-8')
            task = root / 'task.toml'
            task.write_text('input = "input.csv"\ncolumn = 1\ngraph = "example"\nsave = "state.db"\n', encoding='utf-8')
            run(shared, task)
            config = load_flow_config(shared, task)

            async def runner(_config, _device, claimed):
                await asyncio.sleep(0)
                if claimed.value == '3':
                    raise TaskError('test failure')

            result = asyncio.run(run_workers(config, runner))
            self.assertEqual((result.claimed, result.succeeded, result.failed), (4, 3, 1))
            database = sqlite3.connect(root / 'state.db')
            rows = database.execute('SELECT value, status, device, attempts FROM tasks ORDER BY value').fetchall()
            database.close()
            self.assertEqual([row[1] for row in rows], ['succeeded', 'succeeded', 'failed', 'succeeded'])
            self.assertEqual({row[2] for row in rows}, {'one', 'two'})
            self.assertTrue(all(row[3] == 1 for row in rows))


if __name__ == '__main__':
    unittest.main()
