import sys
import time
from logging import getLogger

import numpy as np
from scipy.special import log_softmax
import librosa
import soundfile as sf

use_torch = True
try:
    import torchaudio.compliance.kaldi as kaldi
except ImportError:
    use_torch = False

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath
from model_utils import check_and_download_models, check_and_download_file
from math_utils import softmax

from audio_utils import load_wav, mel_spectrogram, log_mel_spectrogram, compute_fbank
from cosyvoice2_utils import text_normalize


logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_SPEECH_PATH = "CosyVoice2-0.5B_speech_tokenizer_v2.onnx"
WEIGHT_CAMP_PATH = "campplus.onnx"
WEIGHT_EMB_TKN_PATH = "CosyVoice2-0.5B_embed_tokens.onnx"
WEIGHT_LLM_PATH = "CosyVoice2-0.5B_llm.onnx"
WEIGHT_FLOW_ENC_PATH = "CosyVoice2-0.5B_flow_encoder.onnx"
WEIGHT_FLOW_DEC_PATH = "CosyVoice2-0.5B_flow.decoder.estimator.fp32.onnx"
WEIGHT_HIFT_PATH = "CosyVoice2-0.5B_hift.onnx"
MODEL_SPEECH_PATH = "CosyVoice2-0.5B_speech_tokenizer_v2.onnx.prototxt"
MODEL_CAMP_PATH = "campplus.onnx.prototxt"
MODEL_EMB_TKN_PATH = "CosyVoice2-0.5B_embed_tokens.onnx.prototxt"
MODEL_LLM_PATH = "CosyVoice2-0.5B_llm.onnx.prototxt"
MODEL_FLOW_ENC_PATH = "CosyVoice2-0.5B_flow_encoder.onnx.prototxt"
MODEL_FLOW_DEC_PATH = "CosyVoice2-0.5B_flow.decoder.estimator.fp32.onnx.prototxt"
MODEL_HIFT_PATH = "CosyVoice2-0.5B_hift.onnx.prototxt"
NPY_SPEECH_EMB_PATH = "CosyVoice2-0.5B_speech_embedding.npy"
NPY_LLM_EMB_PATH = "CosyVoice2-0.5B_llm_embedding.npy"
NPY_LLM_DEC_WEIGHT_PATH = "CosyVoice2-0.5B_llm_decoder_weight.npy"
NPY_LLM_DEC_BIAS_PATH = "CosyVoice2-0.5B_llm_decoder_bias.npy"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/cosyvoice2/"

SAMPLE_RATE = 24000

WAV_PATH = "zero_shot_prompt.wav"
SAVE_WAV_PATH = "output.wav"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("CosyVoice2", WAV_PATH, SAVE_WAV_PATH, fp16_support = False)
parser.add_argument(
    "-t",
    "--tts_text",
    type=str,
    default="收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
    help="tts_text",
)
parser.add_argument(
    "-p",
    "--prompt_text",
    type=str,
    default="希望你以后能够做的比我还好呦。",
    help="prompt_text",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="random seed",
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)


# ======================
# Secondary Functions
# ======================


def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    prob, indices = [], []
    cum_prob = 0.0

    score = softmax(weighted_scores, axis=0)

    # Sort scores and indices
    sorted_idx = np.argsort(score)[::-1]
    sorted_value = score[sorted_idx]

    for i in range(len(sorted_idx)):
        # Sampling both top-p and numbers.
        if cum_prob < top_p and len(prob) < top_k:
            cum_prob += sorted_value[i]
            prob.append(sorted_value[i])
            indices.append(sorted_idx[i])
        else:
            break

    prob = np.array(prob)
    indices = np.array(indices, dtype=np.int64)

    # Multinomial sampling
    top_ids = np.random.choice(indices, size=1, p=prob / prob.sum())

    return top_ids


def istft(magnitude, phase):
    magnitude = np.clip(magnitude, a_min=None, a_max=1e2)
    real = magnitude * np.cos(phase)
    img = magnitude * np.sin(phase)

    complex_stft = real + 1j * img
    inverse_transform = librosa.istft(
        complex_stft,
        hop_length=4,
        win_length=16,
        window="hann",
    )
    return inverse_transform


# ======================
# Main functions
# ======================


def extract_speech_token(speech_tokenizer, speech):
    assert (
        speech.shape[1] / 16000 <= 30
    ), "do not support extract speech token for audio longer than 30s"

    feat = log_mel_spectrogram(
        speech,
        n_mels=128,
    )

    if not args.onnx:
        output = speech_tokenizer.predict(
            [feat, np.array([feat.shape[2]], dtype=np.int32)]
        )
    else:
        output = speech_tokenizer.run(
            None,
            {
                "feats": feat,
                "feats_length": np.array([feat.shape[2]], dtype=np.int32),
            },
        )
    speech_token = output[0]

    speech_token_len = np.array([speech_token.shape[1]], dtype=int)
    return speech_token, speech_token_len


def extract_spk_embedding(campplus, speech):
    if use_torch:
        import torch

        x = torch.from_numpy(speech)
        feat = kaldi.fbank(x, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        feat = feat.numpy()
    else:
        feat = compute_fbank(speech[0], sr=16000)

    feat = np.expand_dims(feat, axis=0)
    if not args.onnx:
        output = campplus.predict([feat])
    else:
        output = campplus.run(
            None,
            {"input": feat},
        )
    embedding = output[0]

    return embedding


def extract_speech_feat(speech):
    speech_feat = mel_spectrogram(
        speech,
        n_fft=1920,
        num_mels=80,
        sampling_rate=SAMPLE_RATE,
        hop_size=480,
        win_size=1920,
        fmin=0,
        fmax=8000,
        center=False,
    )
    speech_feat = np.squeeze(speech_feat, axis=0).transpose(1, 0)
    speech_feat = np.expand_dims(speech_feat, axis=0)
    speech_feat_len = np.array([speech_feat.shape[1]], dtype=int)

    return speech_feat, speech_feat_len


def llm_forward(llm, xs, cache):
    masks = np.expand_dims(np.tril(np.ones((xs.shape[1], xs.shape[1]))), axis=0)
    masks = masks.astype(np.bool_)
    if not args.onnx:
        output = llm.predict([xs, masks, *cache])
    else:
        key_cache = {"key_cache%d" % i: cache[i * 2] for i in range(24)}
        value_cache = {"value_cache%d" % i: cache[i * 2 + 1] for i in range(24)}
        output = llm.run(None, {"xs": xs, "masks": masks, **key_cache, **value_cache})
    y_pred, *cache = output
    return y_pred, cache


def llm_inference(
    models,
    text: np.ndarray,
    text_len: np.ndarray,
    prompt_text: np.ndarray,
    prompt_text_len: np.ndarray,
    prompt_speech_token: np.ndarray,
    prompt_speech_token_len: np.ndarray,
    embedding: np.ndarray,
    max_token_text_ratio: float = 20,
    min_token_text_ratio: float = 2,
):
    speech_token_size = 6561

    text = np.concatenate([prompt_text, text], axis=1).astype(np.int32)
    text_len += prompt_text_len

    net = models["embed_tokens"]
    if not args.onnx:
        output = net.predict([text])
    else:
        output = net.run(None, {"input": text})
    text = output[0]

    # 2. encode embedding
    llm_input_size = 896
    embedding = np.zeros((1, 0, llm_input_size), dtype=text.dtype)

    # 3. concat llm_input
    sos_eos = 0
    task_id = 1
    speech_embedding = models["speech_embedding"]
    llm_embedding = models["llm_embedding"]
    sos_eos_emb = llm_embedding[sos_eos].reshape(1, 1, -1)
    task_id_emb = llm_embedding[task_id].reshape(1, 1, -1)
    if prompt_speech_token_len != 0:
        prompt_speech_token_emb = speech_embedding[prompt_speech_token]
    else:
        prompt_speech_token_emb = np.zeros((1, 0, llm_input_size), dtype=text.dtype)
    lm_input = np.concatenate(
        [sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], axis=1
    )

    # 4. cal min/max_length
    min_len = int((text_len - prompt_text_len).item() * min_token_text_ratio)
    max_len = int((text_len - prompt_text_len).item() * max_token_text_ratio)

    cache = [np.zeros((1, 2, 0, 64), dtype=np.float32)] * (2 * 24)

    # 5. step by step decode
    decoded_tokens = []
    for i in range(max_len):
        net = models["llm"]
        y_pred, cache = llm_forward(net, lm_input, cache)

        weight = models["llm_decoder_weight"]
        bias = models["llm_decoder_bias"]
        x = (y_pred[:, -1] @ weight.T) + bias
        logp = log_softmax(x, axis=-1)

        # sampling_ids
        weighted_scores = np.squeeze(logp, axis=0)
        num_trials, max_trials = 0, 100
        ignore_eos = True if i < min_len else False
        top_p, top_k, win_size, tau_r = 0.8, 25, 10, 0.1
        while True:
            top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
            rep_num = np.sum(
                np.array(decoded_tokens[-win_size:]).reshape(-1) == top_ids
            )
            if rep_num >= win_size * tau_r:
                score = softmax(weighted_scores, axis=0)
                top_ids = np.random.choice(len(score), size=1, p=score)
            if (not ignore_eos) or (speech_token_size not in top_ids):
                break
            num_trials += 1
            if num_trials > max_trials:
                raise RuntimeError(
                    "sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!".format(
                        max_trials
                    )
                )

        if top_ids == speech_token_size:
            break
        if top_ids > speech_token_size:
            continue
        decoded_tokens.append(top_ids)
        lm_input = speech_embedding[top_ids].reshape(1, 1, -1)

    tts_speech_token = np.concatenate(decoded_tokens)
    return tts_speech_token


def solve_euler(models, x, t_span, mu, mask, spks, cond):
    """
    Fixed euler solver for ODEs.
    Args:
        x (np.ndarray): random noise
        t_span (np.ndarray): n_timesteps interpolated
            shape: (n_timesteps + 1,)
        mu (np.ndarray): output of encoder
            shape: (batch_size, n_feats, mel_timesteps)
        mask (np.ndarray): output_mask
            shape: (batch_size, 1, mel_timesteps)
        spks (np.ndarray, optional): speaker ids. Defaults to None.
            shape: (batch_size, spk_emb_dim)
        cond: Not used but kept for future purposes
    """
    t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
    t = t[None, ...]

    # I am storing this because I can later plot it by putting a debugger here and saving it to a file
    # Or in future might add like a return_all_steps flag
    sol = []

    # Do not use concat, it may cause memory format changed and trt infer with wrong results!
    x_in = np.zeros([2, 80, x.shape[2]], dtype=x.dtype)
    mask_in = np.zeros([2, 1, x.shape[2]], dtype=x.dtype)
    mu_in = np.zeros([2, 80, x.shape[2]], dtype=x.dtype)
    t_in = np.zeros([2], dtype=x.dtype)
    spks_in = np.zeros([2, 80], dtype=x.dtype)
    cond_in = np.zeros([2, 80, x.shape[2]], dtype=x.dtype)

    for step in range(1, len(t_span)):
        # Classifier-Free Guidance inference introduced in VoiceBox
        x_in[:] = x
        mask_in[:] = mask
        mu_in[0] = mu
        t_in[:] = t[None, ...]
        spks_in[0] = spks
        cond_in[0] = cond

        net = models["flow_dec"]
        if not args.onnx:
            output = net.predict([x_in, mask_in, mu_in, t_in, spks_in, cond_in])
        else:
            output = net.run(
                None,
                {
                    "x": x_in,
                    "mask": mask_in,
                    "mu": mu_in,
                    "t": t_in,
                    "spks": spks_in,
                    "cond": cond_in,
                },
            )
        dphi_dt = output[0]

        inference_cfg_rate = 0.7
        cumsum_indices = np.cumsum([x.shape[0], x.shape[0]])[:-1]
        dphi_dt, cfg_dphi_dt = np.split(dphi_dt, cumsum_indices, axis=0)
        dphi_dt = (
            1.0 + inference_cfg_rate
        ) * dphi_dt - inference_cfg_rate * cfg_dphi_dt

        x = x + dt * dphi_dt
        t = t + dt
        sol.append(x)
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t

    return sol[-1]


def token2wav(models, token, prompt_token, prompt_feat, embedding, token_offset):
    token_len = np.array([token.shape[1]], dtype=np.int32)
    prompt_token_len = np.array([prompt_token.shape[1]], dtype=np.int32)

    net = models["flow_enc"]
    if not args.onnx:
        output = net.predict(
            [
                token,
                token_len,
                prompt_token,
                prompt_token_len,
                prompt_feat,
                embedding,
            ]
        )
    else:
        output = net.run(
            None,
            {
                "token": token,
                "token_len": token_len,
                "prompt_token": prompt_token,
                "prompt_token_len": prompt_token_len,
                "prompt_feat": prompt_feat,
                "embedding": embedding,
            },
        )
    mu, mask, spks, cond = output

    temperature = 1.0
    n_timesteps = 10
    rand_noise = np.random.randn(1, 80, 50 * 300).astype(np.float32)
    z = rand_noise[:, :, : mu.shape[2]] * temperature
    # fix prompt and overlap part mu and z
    t_span = np.linspace(0, 1, n_timesteps + 1, dtype=mu.dtype)
    t_span = 1 - np.cos(t_span * 0.5 * np.pi)

    feat = solve_euler(models, z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)

    mel_len1 = prompt_feat.shape[1]
    tts_mel = feat[:, :, mel_len1:]

    token_mel_ratio = 2
    tts_mel = tts_mel[:, :, token_offset * token_mel_ratio :]

    net = models["hift"]
    if not args.onnx:
        output = net.predict([tts_mel])
    else:
        output = net.run(None, {"speech_feat": tts_mel})
    magnitude, phase = output

    x = istft(magnitude, phase)

    audio_limit = 0.99
    tts_speech = np.clip(x, -audio_limit, audio_limit)

    return tts_speech


def inference(models, tts_text, prompt_text, prompt_speech_16k):
    start_time = time.time()

    tokenizer = models["tokenizer"]
    tokens = tokenizer([tts_text], return_tensors="np")
    tts_text_token = tokens["input_ids"]

    tokens = tokenizer([prompt_text], return_tensors="np")
    prompt_text_token = tokens["input_ids"]

    prompt_speech_resample = librosa.resample(
        prompt_speech_16k, orig_sr=16000, target_sr=SAMPLE_RATE
    )

    speech_feat, speech_feat_len = extract_speech_feat(prompt_speech_resample)

    speech_tokenizer = models["speech_tokenizer"]
    speech_token, speech_token_len = extract_speech_token(
        speech_tokenizer, prompt_speech_16k
    )

    # cosyvoice2, force speech_feat % speech_token = 2
    token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
    speech_feat, speech_feat_len[:] = speech_feat[:, : 2 * token_len], 2 * token_len
    speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len

    campplus = models["campplus"]
    embedding = extract_spk_embedding(campplus, prompt_speech_16k)

    text = tts_text_token
    prompt_text = prompt_text_token
    llm_prompt_speech_token = speech_token
    flow_prompt_speech_token = speech_token
    prompt_speech_feat = speech_feat
    llm_embedding = embedding
    flow_embedding = embedding

    tts_speech_token = llm_inference(
        models,
        text=text,
        text_len=np.array([text.shape[1]], dtype=int),
        prompt_text=prompt_text,
        prompt_text_len=np.array([prompt_text.shape[1]], dtype=int),
        prompt_speech_token=llm_prompt_speech_token,
        prompt_speech_token_len=np.array([llm_prompt_speech_token.shape[1]], dtype=int),
        embedding=llm_embedding,
    )
    tts_speech_token = np.expand_dims(tts_speech_token, axis=0)

    tts_speech = token2wav(
        models,
        token=tts_speech_token,
        prompt_token=flow_prompt_speech_token,
        prompt_feat=prompt_speech_feat,
        embedding=flow_embedding,
        token_offset=0,
    )

    speech_len = tts_speech.shape[1] / SAMPLE_RATE
    logger.info(
        "speech len {}, rtf {}".format(
            speech_len, (time.time() - start_time) / speech_len
        )
    )

    return tts_speech


def inference_zero_shot(models):
    audio_path = args.input[0]
    tts_text = args.tts_text
    prompt_text = args.prompt_text

    speech = load_wav(audio_path, 16000)
    speech = speech[None, :]

    top_db = 60
    hop_length = 220
    win_length = 440
    max_val = 0.8
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db, frame_length=win_length, hop_length=hop_length
    )
    if np.max(np.abs(speech)) > max_val:
        speech = speech / np.max(np.abs(speech)) * max_val
    speech = np.concatenate(
        [speech, np.zeros([1, int(SAMPLE_RATE * 0.2)], dtype=np.float32)], axis=1
    )

    logger.info("tts_text: %s" % tts_text)
    logger.info("prompt_text: %s" % prompt_text)

    logger.info("Start inference...")

    tokenizer = models["tokenizer"]
    prompt_text = text_normalize(tokenizer, prompt_text, split=False)
    a = []
    for s in text_normalize(tokenizer, tts_text, split=True):
        if len(s) < 0.5 * len(prompt_text):
            logger.warning(
                "synthesis text {} too short than prompt text {}, this may lead to bad performance".format(
                    s, prompt_text
                )
            )

        tts_speech = inference(models, s, prompt_text, speech)
        a.append(tts_speech)

    tts_speech = np.concatenate(a, axis=1)
    tts_speech = tts_speech.squeeze()

    # save result
    savepath = get_savepath(args.savepath, audio_path, ext=".wav")
    logger.info(f"saved at : {savepath}")
    sf.write(savepath, tts_speech, SAMPLE_RATE)

    logger.info("Script finished successfully.")


def main():
    check_and_download_models(WEIGHT_SPEECH_PATH, MODEL_SPEECH_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_CAMP_PATH, MODEL_CAMP_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_EMB_TKN_PATH, MODEL_EMB_TKN_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_LLM_PATH, MODEL_LLM_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_FLOW_ENC_PATH, MODEL_FLOW_ENC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_FLOW_DEC_PATH, MODEL_FLOW_DEC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_HIFT_PATH, MODEL_HIFT_PATH, REMOTE_PATH)
    check_and_download_file(NPY_SPEECH_EMB_PATH, REMOTE_PATH)
    check_and_download_file(NPY_LLM_EMB_PATH, REMOTE_PATH)
    check_and_download_file(NPY_LLM_DEC_WEIGHT_PATH, REMOTE_PATH)
    check_and_download_file(NPY_LLM_DEC_BIAS_PATH, REMOTE_PATH)

    seed = args.seed
    if seed is not None:
        np.random.seed(seed)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
        speech_tokenizer = ailia.Net(
            MODEL_SPEECH_PATH,
            WEIGHT_SPEECH_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        campplus = ailia.Net(
            MODEL_CAMP_PATH, WEIGHT_CAMP_PATH, env_id=env_id, memory_mode=memory_mode
        )
        embed_tokens = ailia.Net(
            MODEL_EMB_TKN_PATH,
            WEIGHT_EMB_TKN_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        llm = ailia.Net(
            MODEL_LLM_PATH, WEIGHT_LLM_PATH, env_id=env_id, memory_mode=memory_mode
        )
        flow_enc = ailia.Net(
            MODEL_FLOW_ENC_PATH,
            WEIGHT_FLOW_ENC_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        flow_dec = ailia.Net(
            MODEL_FLOW_DEC_PATH,
            WEIGHT_FLOW_DEC_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        hift = ailia.Net(
            MODEL_HIFT_PATH, WEIGHT_HIFT_PATH, env_id=env_id, memory_mode=memory_mode
        )
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        speech_tokenizer = onnxruntime.InferenceSession(
            WEIGHT_SPEECH_PATH, providers=providers
        )
        campplus = onnxruntime.InferenceSession(WEIGHT_CAMP_PATH, providers=providers)
        embed_tokens = onnxruntime.InferenceSession(
            WEIGHT_EMB_TKN_PATH, providers=providers
        )
        llm = onnxruntime.InferenceSession(WEIGHT_LLM_PATH, providers=providers)
        flow_enc = onnxruntime.InferenceSession(
            WEIGHT_FLOW_ENC_PATH, providers=providers
        )
        flow_dec = onnxruntime.InferenceSession(
            WEIGHT_FLOW_DEC_PATH, providers=providers
        )
        hift = onnxruntime.InferenceSession(WEIGHT_HIFT_PATH, providers=providers)

    if args.disable_ailia_tokenizer:
        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained("./tokenizer")

        special_tokens = {
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            # fmt: off
            'additional_special_tokens': [
                '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
                '[breath]', '<strong>', '</strong>', '[noise]',
                '[laughter]', '[cough]', '[clucking]', '[accent]',
                '[quick_breath]',
                "<laughter>", "</laughter>",
                "[hissing]", "[sigh]", "[vocalized-noise]",
                "[lipsmack]", "[mn]"
            ]
            # fmt: on
        }
        tokenizer.add_special_tokens(special_tokens)
    else:
        from ailia_tokenizer import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer")
        tokenizer.add_special_tokens({"additional_special_tokens":['<|im_start|>', '<|im_end|>', '<|endofprompt|>',
                '[breath]', '<strong>', '</strong>', '[noise]',
                '[laughter]', '[cough]', '[clucking]', '[accent]',
                '[quick_breath]',
                "<laughter>", "</laughter>",
                "[hissing]", "[sigh]", "[vocalized-noise]",
                "[lipsmack]", "[mn]"]})

    speech_embedding = np.load(NPY_SPEECH_EMB_PATH)
    llm_embedding = np.load(NPY_LLM_EMB_PATH)
    llm_decoder_weight = np.load(NPY_LLM_DEC_WEIGHT_PATH)
    llm_decoder_bias = np.load(NPY_LLM_DEC_BIAS_PATH)

    models = dict(
        tokenizer=tokenizer,
        speech_tokenizer=speech_tokenizer,
        campplus=campplus,
        embed_tokens=embed_tokens,
        speech_embedding=speech_embedding,
        llm_embedding=llm_embedding,
        llm=llm,
        llm_decoder_weight=llm_decoder_weight,
        llm_decoder_bias=llm_decoder_bias,
        flow_enc=flow_enc,
        flow_dec=flow_dec,
        hift=hift,
    )

    # generate
    inference_zero_shot(models)


if __name__ == "__main__":
    main()
