import io
import sys
import tempfile
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from chatkvm.speech import SpeechRecognizer, decode_recording


def recording(seconds=1, rate=16000, channels=1, silent=False):
    output = io.BytesIO()
    with wave.open(output, 'wb') as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes((b'\0\0' if silent else b'\1\0') * int(seconds * rate) * channels)
    return output.getvalue()


class SpeechTests(unittest.TestCase):
    def test_recording_validates_format_duration_and_truncation(self):
        self.assertEqual(len(decode_recording(recording())), 32000)
        for data in (b'not wav', recording(0), recording(121), recording(rate=48000), recording(channels=2), recording()[:-2]):
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                decode_recording(data)

    def test_cpu_inference_uses_local_models_and_removes_metadata(self):
        model = MagicMock()
        model.generate.return_value = [{'text': '<|zh|><|NEUTRAL|><|Speech|>打开电脑。'}, {'text': '<|en|>Hello!'}]
        factory = MagicMock(return_value=model)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'funasr': SimpleNamespace(AutoModel=factory)}):
            recognizer = SpeechRecognizer(directory, directory)
            self.assertEqual(factory.call_args.kwargs['device'], 'cpu')
            self.assertTrue(factory.call_args.kwargs['disable_update'])
            self.assertEqual(recognizer.transcribe(recording()), '打开电脑。Hello!')
            self.assertEqual(len(model.generate.call_args.kwargs['input']), 16000)
            self.assertEqual(model.generate.call_args.kwargs['cache'], {})
            model.generate.reset_mock()
            with self.assertRaises(ValueError):
                recognizer.transcribe(recording(silent=True))
            model.generate.assert_not_called()
            model.generate.return_value = [{'text': '<|nospeech|>'}]
            with self.assertRaises(ValueError):
                recognizer.transcribe(recording())


if __name__ == '__main__':
    unittest.main()
