"""
Export the VoiceSplit checkpoint to voicesplit.onnx.

VoiceSplit (https://github.com/Edresson/VoiceSplit) publishes trained
checkpoints of the VoiceFilter architecture adapted from
https://github.com/maum-ai/voicefilter (the same implementation the
existing model.onnx is based on). The "Power-Law" checkpoint uses the
original paper's power-law compressed loss, the same audio frontend
(n_fft=1200, hop=160, win=400, num_freq=601, 16kHz) and the same GE2E
embedder by Seungwon Park, so the exported model is a drop-in
replacement for model.onnx.

Usage:
    git clone https://github.com/Edresson/VoiceSplit.git
    wget "https://github.com/Edresson/VoiceSplit/releases/download/checkpoints/voiceSplit-trained-with-Power-Law-compressed_Loss-GE2E-Seungwonpark-best_checkpoint.pt.pt" -O voicesplit_powerlaw.pt
    python3 export_voicesplit.py --repo VoiceSplit --checkpoint voicesplit_powerlaw.pt
"""
import argparse
import ast
import sys
from types import SimpleNamespace

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--repo', default='VoiceSplit', help='path to a clone of Edresson/VoiceSplit')
parser.add_argument('--checkpoint', default='voicesplit_powerlaw.pt')
parser.add_argument('--output', default='../voicesplit.onnx')
args = parser.parse_args()

sys.path.append(args.repo)
from models.voicefilter.model import VoiceFilter  # noqa: E402

ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
config = ast.literal_eval(ckpt['config_str'])
print('checkpoint step:', ckpt['step'])
print('audio backend:', config['audio']['backend'])
print('model config:', config['model'])

model = VoiceFilter(SimpleNamespace(audio=config['audio'], model=config['model']))
model.load_state_dict(ckpt['model'])
model.eval()

L = 301  # 3 sec dummy, the time axis is exported as dynamic
mag = torch.randn(1, L, config['model']['fc2_dim'])
dvec = torch.randn(1, config['model']['emb_dim'])

torch.onnx.export(
    model, (mag, dvec), args.output,
    input_names=['mag', 'dvec'],
    output_names=['mask'],
    dynamic_axes={'mag': {1: 'L'}, 'mask': {1: 'L'}},
    opset_version=11, dynamo=False)
print('saved', args.output)

# verify against onnxruntime, including a length not seen at export time
import onnxruntime as ort  # noqa: E402
sess = ort.InferenceSession(args.output)
for L in [301, 173]:
    mag = torch.randn(1, L, config['model']['fc2_dim'])
    dvec = torch.randn(1, config['model']['emb_dim'])
    with torch.no_grad():
        expected = model(mag, dvec).numpy()
    actual = sess.run(None, {'mag': mag.numpy(), 'dvec': dvec.numpy()})[0]
    print(f'L={L}: max diff vs torch = {np.abs(expected - actual).max():.3e}')
