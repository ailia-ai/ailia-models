import sys
import time
from typing import Union
import json

import ailia  # noqa: E402
import numpy as np
import onnxruntime
import torch
from einops import rearrange
from narabas_util import load_audio

sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from symbols import BOS, EOS, PAD, id_to_phoneme, phoneme_to_id

logger = getLogger(__name__)

NARABAS_WEIGHT_PASS = "narabas-v0.onnx"
NARABAS_MODEL_PATH = "narabas-v0.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/narabas/"
AUDIO_PATH = "input.wav"
HOP_LENGTH_SEC = 0.02

parser = get_base_parser('narabas', AUDIO_PATH, None, fp16_support=False)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='By default, the ailia SDK is used, but with this option, you can switch to using ONNX Runtime'
)
parser.add_argument(
    "--phonemes",
    type=str,
    help="phoneme (splitted by space)",
    default="a n e m u s u m e n o ts u i k o w a"
)
parser.add_argument(
    "--sample_rate",
    type=int,
    help="sample rate",
    default=16000,
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)


def create_instance(weight_path, model_path, ):
    if not args.onnx:
        env_id = args.env_id
        session = ailia.Net(model_path, weight_path, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        session = onnxruntime.InferenceSession(weight_path)

    return session


def execute_session(session, wav):
    wav_np = wav.numpy()
    if not args.onnx:
        result = session.run(wav_np)[0]
        a, b = result.shape[1:]
        result = result.reshape((1, 1, a, b))
    else:
        result = session.run(
            ["output"],
            {"input": wav_np}
        )

    return result


def infer(net: Union[ailia.Net, onnxruntime.InferenceSession]):
    input_audio_filename = args.input[0]
    sample_rate = args.sample_rate
    phonemes = args.phonemes
    wav = load_audio(input_audio_filename, sample_rate)
    phn_ids = [phoneme_to_id[phn] for phn in phonemes.split()]
    phn_ids = [BOS, *phn_ids, EOS]
    y_hat = execute_session(net, wav)
    y_hat = torch.tensor(y_hat)
    emission = torch.log_softmax(y_hat.squeeze(0), dim=-1)[0]
    num_frames = emission.size()[0]
    num_tokens = len(phn_ids)

    likelihood = np.full((num_tokens + 1,), -np.inf)
    likelihood[0] = 0
    path = np.zeros((num_frames, num_tokens + 1), dtype=np.int32)

    for t in range(num_frames):
        for i in range(1, num_tokens + 1):
            stay = likelihood[i] + emission[t, PAD]
            move = likelihood[i-1] + emission[t, phn_ids[i - 1]]
            if stay > move:
                path[t][i] = 0
            else:
                path[t][i] = 1

            likelihood[i] = np.max([stay, move])

    alignment = []
    t = num_frames - 1
    i = num_tokens

    while t >= 0:
        if path[t][i] == 1:
            i -= 1
            alignment.append((t, i))

        t -= 1

    alignment = alignment[-2::-1]

    segments = []
    for(t, i), (t_next, _) in zip(alignment, alignment[1:]):
        start = t * HOP_LENGTH_SEC
        end = t_next * HOP_LENGTH_SEC
        token = id_to_phoneme[phn_ids[i]]
        segments.append((start, end, token))


    for (start, end, phoneme) in segments:
        logger.info(f"{start:.3f} {end:.3f} {phoneme}")

    if args.write_json:
        result = []
        for (start, end, phoneme) in segments:
            result.append({"start": float(start), "end": float(end), "phoneme": phoneme})
        with open("output.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    check_and_download_models(NARABAS_WEIGHT_PASS, NARABAS_MODEL_PATH, REMOTE_PATH)
    net = create_instance(NARABAS_WEIGHT_PASS, NARABAS_MODEL_PATH)
    infer(net)
