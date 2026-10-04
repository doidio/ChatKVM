"""Offline CPU speech recognition for recorded chat input."""

import io
import re
import wave
from pathlib import Path
from threading import Lock

SAMPLE_RATE = 16000
MAX_SECONDS = 120


def decode_recording(data: bytes) -> bytes:
    if len(data) > SAMPLE_RATE * 2 * MAX_SECONDS + 4096:
        raise ValueError('录音不能超过 120 秒')
    try:
        with wave.open(io.BytesIO(data)) as audio:
            if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (SAMPLE_RATE, 1, 2):
                raise ValueError('录音需要为 16 kHz 单声道 16 位 WAV')
            frames = audio.getnframes()
            if not 0 < frames <= SAMPLE_RATE * MAX_SECONDS:
                raise ValueError('录音不能为空或超过 120 秒')
            samples = audio.readframes(frames)
            if len(samples) != frames * 2:
                raise ValueError('录音文件不完整')
            return samples
    except (wave.Error, EOFError) as exc:
        raise ValueError('无法读取 WAV 录音') from exc


class SpeechRecognizer:
    def __init__(self, model: str, vad_model: str):
        from funasr import AutoModel

        for directory in (model, vad_model):
            if not Path(directory).is_dir():
                raise ValueError(f'语音模型目录不存在：{directory}')
        self._lock = Lock()
        self._model = AutoModel(
            model=model,
            vad_model=vad_model,
            vad_kwargs={'max_single_segment_time': 30000},
            device='cpu',
            ncpu=4,
            disable_update=True,
            disable_pbar=True,
        )

    def transcribe(self, data: bytes) -> str:
        import numpy as np

        samples = np.frombuffer(decode_recording(data), dtype='<i2').astype(np.float32) / 32768
        if not np.any(samples):
            raise ValueError('未检测到语音，请重新录音')
        with self._lock:
            results = self._model.generate(input=samples, cache={}, language='auto', use_itn=True, batch_size_s=30)
        text = ''.join(re.sub(r'<\|[^|]*\|>', '', result.get('text', '')) for result in results).strip()
        if not text:
            raise ValueError('未识别到文字，请重新录音')
        return text
