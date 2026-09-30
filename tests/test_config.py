import tempfile
import unittest
from pathlib import Path

from pydantic_ai.models.ollama import OllamaModel

from chatkvm.agent import resolve_model
from chatkvm.config import read_config


class ConfigTests(unittest.TestCase):
    def test_reads_nested_model_provider_without_writing_config(self):
        source = '''[agent.model]
name = "qwen3.8:27b"
provider = "ollama"
base_url = "http://127.0.0.1:11435/v1"
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.toml'
            path.write_text(source, encoding='utf-8')
            config = read_config(path)
            self.assertEqual(path.read_text(encoding='utf-8'), source)
        model = resolve_model(config['agent']['model'])
        self.assertIsInstance(model, OllamaModel)
        self.assertEqual(model.model_name, 'qwen3.8:27b')

    def test_other_model_uses_its_own_provider(self):
        self.assertEqual(resolve_model({'name': 'gpt-4o', 'provider': 'openai'}), 'openai:gpt-4o')

    def test_missing_model_provider_is_an_error(self):
        with self.assertRaises(KeyError):
            resolve_model({'name': 'qwen3.8:27b'})


if __name__ == '__main__':
    unittest.main()
