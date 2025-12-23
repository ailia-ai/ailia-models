import ailia
# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

WEIGHT_ENC_PATH = "encoder-epoch-75-avg-11-chunk-16-left-128.int8.onnx"
WEIGHT_DEC_PATH = "decoder-epoch-75-avg-11-chunk-16-left-128.onnx"
JOINER_PATH = "joiner-epoch-75-avg-11-chunk-16-left-128.int8.onnx"
TOKEN_PATH = "tokens.txt"
USE_AILIANET = False

if args.ailia_audio:
    from ailia_audio_utils import (
        CHUNK_LENGTH,
        HOP_LENGTH,
        N_FRAMES,
        N_SAMPLES,
        SAMPLE_RATE,
        load_audio,
        log_mel_spectrogram,
        pad_or_trim,
    )
else:
    from audio_utils import (
        CHUNK_LENGTH,
        HOP_LENGTH,
        N_FRAMES,
        N_SAMPLES,
        SAMPLE_RATE,
        load_audio,
        log_mel_spectrogram,
        pad_or_trim,
    )

def recognize_from_wave():
    if USE_AILIANET:
        enc_net = ailia.Net()
    else:
        import sherpa_onnx
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=TOKEN_PATH,
            encoder=WEIGHT_ENC_PATH,
            decoder=WEIGHT_DEC_PATH,
            joiner=JOINER_PATH,
            num_threads=1,
            provider="cpu", # "gpu",
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search", # "modified_beam_search"
            max_active_paths=4,
            lm="",
            lm_scale=0.1,
            lodr_fst="",
            lodr_scale=0.1, 
            hotwords_file="",
            hotwords_score=1.5,
            modeling_unit="", 
            bpe_vocab="",
            blank_penalty=0.0,
        )

        
