import unittest
from unittest.mock import patch

from chatkvm.model_info import ollama_context_limits


class ModelInfoTests(unittest.TestCase):
    def tearDown(self):
        ollama_context_limits.clear()

    def test_distinguishes_running_window_from_model_limit(self):
        model = {'name': 'qwen3.8:27b', 'provider': 'ollama', 'base_url': 'http://localhost:11435/v1'}

        def response(_base, endpoint, _payload=None):
            if endpoint == 'ps':
                return {'models': [{'name': 'qwen3.8:27b', 'context_length': 32768}]}
            return {'details': {'family': 'qwen35'}, 'model_info': {'qwen35.context_length': 262144}}

        with patch('chatkvm.model_info._ollama_api', side_effect=response):
            self.assertEqual(ollama_context_limits(model), (32768, 262144))

    def test_unloaded_model_has_no_runtime_window(self):
        model = {'name': 'qwen3.8:27b', 'provider': 'ollama', 'base_url': 'http://localhost:11435/v1'}

        def response(_base, endpoint, _payload=None):
            if endpoint == 'ps':
                return {'models': []}
            return {'details': {'family': 'qwen35'}, 'model_info': {'qwen35.context_length': 262144}}

        with patch('chatkvm.model_info._ollama_api', side_effect=response):
            self.assertEqual(ollama_context_limits(model), (None, 262144))
