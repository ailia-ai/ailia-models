import time
import sys

import numpy as np

import ailia  # noqa: E402
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models, check_and_download_file  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# for clap
import librosa
from clap_utils import *


# ======================
# Arguemnt Parser Config
# ======================
AUDIO_PATH = 'input.wav'
parser = get_base_parser('CLAP', AUDIO_PATH, None)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='By default, the ailia SDK is used, but with this option, you can switch to using ONNX Runtime'
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
parser.add_argument(
    '--disable_ailia_audio',
    action='store_true',
    help='disable ailia audio and use librosa to get spectrogram feature'
)
args = update_parser(parser)

# ======================
# PARAMETERS
# ======================
CLAP_AUDIO_WEIGHT_PATH = "CLAP_audio_LAION-Audio-630K_with_fusion.onnx"
CLAP_AUDIO_MODEL_PATH  = "CLAP_audio_LAION-Audio-630K_with_fusion.onnx.prototxt"
CLAP_TEXT_ROBERTAMODEL_WEIGHT_PATH = "CLAP_text_text_branch_RobertaModel_roberta-base.onnx"
CLAP_TEXT_ROBERTAMODEL_MODEL_PATH  = "CLAP_text_text_branch_RobertaModel_roberta-base.onnx.prototxt"
CLAP_TEXT_PROJECTION_WEIGHT_PATH   = "CLAP_text_projection_LAION-Audio-630K_with_fusion.onnx"
CLAP_TEXT_PROJECTION_MODEL_PATH    = "CLAP_text_projection_LAION-Audio-630K_with_fusion.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/clap/"


# ======================
# Utils
# ======================
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# ======================
# Main function
# ======================
def infer_text(net_text_branch, net_text_projection, text_data):
    # tokenizer
    if args.disable_ailia_tokenizer:
        from transformers import RobertaTokenizer
        tokenize = RobertaTokenizer.from_pretrained('roberta-base')
        result = tokenize(
            text_data,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        data = {k: v.squeeze(0) for k, v in result.items()}
        data["input_ids"] = data["input_ids"].to('cpu').detach().numpy().copy()
        data["attention_mask"] = data["attention_mask"].to('cpu').detach().numpy().copy()
    else:
        from ailia_tokenizer import RobertaTokenizer
        tokenize = RobertaTokenizer.from_pretrained("./tokenizer/")
        result = tokenize(
            text_data,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="np",
        )
        data = {k: v for k, v in result.items()}

    #print("input_ids", data["input_ids"])
    #print("attention_mask", data["attention_mask"])

    # predict
    input_data = {
        'input_ids': data["input_ids"],
        'attention_mask': data["attention_mask"]
    }
    if not args.onnx:
        output = net_text_branch.predict(input_data) # text_branch
        _, x = output[0], output[1] # last_hidden_state, pooler_output
        text_embeds = net_text_projection.predict(x) # projection
    else:
        output = net_text_branch.run(None, input_data) # text_branch
        _, x = output[0], output[1] # last_hidden_state, pooler_output
        text_embeds = net_text_projection.run(None, {'x': x})[0] # projection

    return text_embeds


def infer_audio(net_audio, audio_src):
    # load the waveform of the shape (T,), should resample to 48000
    audio_waveform, sr = librosa.load(audio_src, sr=48000) 

    # quantize
    audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))

    # get audio features
    _, mel_fusion, _ = get_audio_features(
        {}, audio_waveform, 480000, 
        data_truncating='fusion', 
        data_filling='repeatpad',
        audio_cfg={
            'audio_length': 1024, 
            'clip_samples': 480000, 
            'mel_bins': 64, 
            'sample_rate': 48000, 
            'window_size': 1024, 
            'hop_size': 480, 
            'fmin': 50, 
            'fmax': 14000, 
            'class_num': 527, 
            'model_type': 'HTSAT', 
            'model_name': 'tiny'
        },
        b_use_ailia=not args.disable_ailia_audio
    )
    input_dict = {
        'longer': [[True]], # Error occers when longer value is "False".
        'mel_fusion': mel_fusion[np.newaxis, :, :, :]
    }

    # predict
    if not args.onnx:
        input_dict["longer"] = np.array(input_dict["longer"])
        audio_embed = net_audio.predict(input_dict)[0]
    else:
        audio_embed = net_audio.run(None, input_dict)[0]

    return audio_embed


def main():
    # model files check and download
    check_and_download_models(CLAP_AUDIO_WEIGHT_PATH, CLAP_AUDIO_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(CLAP_TEXT_PROJECTION_WEIGHT_PATH, CLAP_TEXT_PROJECTION_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(CLAP_TEXT_ROBERTAMODEL_WEIGHT_PATH, CLAP_TEXT_ROBERTAMODEL_MODEL_PATH, REMOTE_PATH)

    # net initialize
    if not args.onnx:
        net_text_branch = \
            ailia.Net(CLAP_TEXT_ROBERTAMODEL_MODEL_PATH, CLAP_TEXT_ROBERTAMODEL_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
        net_text_projection = \
            ailia.Net(CLAP_TEXT_PROJECTION_MODEL_PATH, CLAP_TEXT_PROJECTION_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
        net_audio = \
            ailia.Net(CLAP_AUDIO_MODEL_PATH, CLAP_AUDIO_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        net_text_branch = \
            onnxruntime.InferenceSession(CLAP_TEXT_ROBERTAMODEL_WEIGHT_PATH)
        net_text_projection = \
            onnxruntime.InferenceSession(CLAP_TEXT_PROJECTION_WEIGHT_PATH)
        net_audio = \
            onnxruntime.InferenceSession(CLAP_AUDIO_WEIGHT_PATH)

    # text predict
    text_inputs = [
        "applause applaud clap", 
        "The crowd is clapping.",
        "I love the contrastive learning", 
        "bell", 
        "soccer", 
        "open the door.",
        "applause",
        "dog",
        "dog barking"
    ]
    text_embedding = infer_text(net_text_branch, net_text_projection, text_inputs)

    # audio predict
    for audio_path in args.input:
        audio_embedding = infer_audio(net_audio, audio_path)
        # show result
        print('===== cosine similality between text and audio =====')
        print('audio: {}'.format(audio_path))
        for i in range(text_embedding.shape[0]):
            print('cossim={:.04f}, word={}'.format(cos_sim(text_embedding[i], audio_embedding[0]), text_inputs[i]))

    logger.info('Script finished successfully.')


if __name__ == "__main__":
    main()
