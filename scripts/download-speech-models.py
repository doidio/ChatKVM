"""Download the complete offline speech models before starting ChatKVM."""

import argparse
from pathlib import Path

from modelscope import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=Path, default=Path('models'))
args = parser.parse_args()
for name, model_id in (
    ('SenseVoiceSmall', 'iic/SenseVoiceSmall'),
    ('fsmn-vad', 'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch'),
):
    snapshot_download(model_id, local_dir=str(args.directory / name))
