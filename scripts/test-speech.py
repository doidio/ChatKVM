"""Smoke-test offline recognition with the downloaded model's sample audio."""

import io
import time
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from chatkvm.speech import SpeechRecognizer

directory = Path('models')
started = time.perf_counter()
recognizer = SpeechRecognizer(str(directory / 'SenseVoiceSmall'), str(directory / 'fsmn-vad'))
print(f'Model load: {time.perf_counter() - started:.2f}s', flush=True)
print(f'Device: {next(recognizer._model.model.parameters()).device}', flush=True)
for language in ('zh', 'en'):
    samples, rate = sf.read(directory / 'SenseVoiceSmall' / 'example' / f'{language}.mp3', dtype='float32')
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if rate != 16000:
        from math import gcd

        divisor = gcd(rate, 16000)
        samples = resample_poly(samples, 16000 // divisor, rate // divisor)
    data = io.BytesIO()
    with wave.open(data, 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes((np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes())
    for attempt in range(2):
        started = time.perf_counter()
        text = recognizer.transcribe(data.getvalue())
        assert text
        print(f'{language} {len(samples) / 16000:.2f}s audio, inference {time.perf_counter() - started:.2f}s: {text}', flush=True)
