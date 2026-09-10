# LLM: MioTTS-0.1B (FalconH1ForCausalLM)
# MioCodec: Aratako/MioCodec-25Hz-44.1kHz-v2

from __future__ import annotations

import multiprocessing as mp
import queue
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# PARAMETERS
# ======================

SAVE_WAV_PATH = 'output.wav'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/mio-tts/'

WEIGHT_LLM = 'miotts_llm_prefill.onnx'
MODEL_LLM = 'miotts_llm_prefill.onnx.prototxt'
WEIGHT_DECODER = 'miocodec_decoder.onnx'
MODEL_DECODER = 'miocodec_decoder.onnx.prototxt'
WEIGHT_GLOBAL_ENC = 'miocodec_global_encoder.onnx'
MODEL_GLOBAL_ENC = 'miocodec_global_encoder.onnx.prototxt'

PRESET_DIR = Path('presets')
PRESET_EXT = '.npy'
TOKEN_RATE_HZ = 25.0
MODEL_ID = 'Aratako/MioTTS-0.1B'

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser('MioTTS', None, SAVE_WAV_PATH)
parser.add_argument(
    '-i', '--input', metavar='TEXT', default='Hello, World.',
    help='Input text to synthesize'
)
parser.add_argument(
    '--preset_id', default=None,
    help='Voice preset ID: jp_female / jp_male / en_female / en_male (default: auto)'
)
parser.add_argument(
    '--temperature', type=float, default=0.8,
    help='Sampling temperature (default: 0.8)'
)
parser.add_argument(
    '--top_p', type=float, default=1.0,
    help='Top-p nucleus sampling (default: 1.0)'
)
parser.add_argument(
    '--max_new_tokens', type=int, default=700,
    help='Max generated tokens (default: 700)'
)
parser.add_argument(
    '--repetition_penalty', type=float, default=1.0,
    help='Repetition penalty (default: 1.0)'
)
parser.add_argument(
    '--seed', type=int, default=None,
    help='Random seed (default: random)'
)
parser.add_argument(
    '--greedy', action='store_true',
    help='Use greedy decoding instead of sampling'
)
args = update_parser(parser, check_input_type=False)

# ======================
# Text Preprocessing
# ======================

REPLACE_MAP: dict[str, str] = {
    r"\t": "",
    r"\[n\]": "",
    r" ": "",
    r"　": "",
    r"[;▼♀♂《》≪≫①②③④⑤⑥]": "",
    r"[\u02d7\u2010-\u2015\u2043\u2212\u23af\u23e4\u2500\u2501\u2e3a\u2e3b]": "",
    r"[\uff5e\u301C]": "ー",
    r"？": "?",
    r"！": "!",
    r"[●◯〇]": "○",
    r"♥": "♡",
}

FULLWIDTH_ALPHA_TO_HALFWIDTH = str.maketrans(
    {
        chr(full): chr(half)
        for full, half in zip(
            list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B)),
            list(range(0x41, 0x5B)) + list(range(0x61, 0x7B)),
        )
    }
)
_HALFWIDTH_KATAKANA_CHARS = "ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ"
_FULLWIDTH_KATAKANA_CHARS = "ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン"
HALFWIDTH_KATAKANA_TO_FULLWIDTH = str.maketrans(
    _HALFWIDTH_KATAKANA_CHARS, _FULLWIDTH_KATAKANA_CHARS
)
FULLWIDTH_DIGITS_TO_HALFWIDTH = str.maketrans(
    {
        chr(full): chr(half)
        for full, half in zip(range(0xFF10, 0xFF1A), range(0x30, 0x3A))
    }
)

TOKEN_PATTERN = re.compile(r"<\|s_(\d+)\|>")


def normalize_text(text: str) -> str:
    for pattern, replacement in REPLACE_MAP.items():
        text = re.sub(pattern, replacement, text)
    text = text.translate(FULLWIDTH_ALPHA_TO_HALFWIDTH)
    text = text.translate(FULLWIDTH_DIGITS_TO_HALFWIDTH)
    text = text.translate(HALFWIDTH_KATAKANA_TO_FULLWIDTH)
    text = re.sub(r"…{3,}", "……", text)
    if text.startswith("「") and text.endswith("」"):
        text = text[1:-1]
    if text.startswith("『") and text.endswith("』"):
        text = text[1:-1]
    if text.startswith("（") and text.endswith("）"):
        text = text[1:-1]
    if text.startswith("【") and text.endswith("】"):
        text = text[1:-1]
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    if text.endswith(("。", "、")):
        text = text.rstrip("。、")
    return text


def parse_speech_tokens(text: str) -> list[int]:
    tokens = [int(v) for v in TOKEN_PATTERN.findall(text)]
    if not tokens:
        raise ValueError("No speech tokens found in LLM output.")
    return tokens


def _is_ascii_alpha(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def _is_japanese_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3040 <= code <= 0x309F
        or 0x30A0 <= code <= 0x30FF
        or 0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
    )


def detect_language(text: str) -> str:
    total = sum(1 for ch in text if not ch.isspace())
    if total == 0:
        return "auto"
    ja_count = sum(1 for ch in text if _is_japanese_char(ch))
    en_count = sum(1 for ch in text if _is_ascii_alpha(ch))
    if ja_count / total >= 0.2:
        return "ja"
    if en_count / total >= 0.5:
        return "en"
    return "auto"


def preprocess_text(text: str) -> tuple[str, str]:
    language = detect_language(text)
    if language == "ja":
        normalized = normalize_text(text)
    else:
        normalized = text.strip()
    return normalized, language


# ======================
# Sampling
# ======================

def sample_next_token(
    last_logits: np.ndarray,
    generated_ids: list[int],
    rng: np.random.Generator,
    temperature: float,
    top_p: float,
    do_sample: bool,
    repetition_penalty: float,
) -> int:
    last_logits = last_logits.copy()

    if repetition_penalty != 1.0 and len(generated_ids) > 0:
        for token_id in set(generated_ids):
            if last_logits[token_id] > 0:
                last_logits[token_id] /= repetition_penalty
            else:
                last_logits[token_id] *= repetition_penalty

    if not do_sample or temperature <= 0:
        return int(np.argmax(last_logits))

    scaled = last_logits.astype(np.float64) / max(temperature, 1e-5)
    scaled -= scaled.max()
    probs = np.exp(scaled)
    probs_sum = probs.sum()
    if probs_sum <= 0 or not np.isfinite(probs_sum):
        return int(np.argmax(last_logits))
    probs /= probs_sum

    top_p = float(np.clip(top_p, 1e-6, 1.0))
    if top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        keep_mask = cumulative_probs <= top_p
        keep_mask[0] = True
        candidate_indices = sorted_indices[keep_mask]
        candidate_probs = sorted_probs[keep_mask]
        candidate_probs /= candidate_probs.sum()
        return int(rng.choice(candidate_indices, p=candidate_probs))

    return int(rng.choice(np.arange(probs.shape[0]), p=probs))


# ======================
# ailia LLM Runner (subprocess)
# ======================

def _ailia_llm_run_worker(onnx_path: str, request_queue, response_queue):
    try:
        import ailia
        net = ailia.Net(stream=None, weight=onnx_path)
        response_queue.put(("ready", None))
    except Exception as exc:
        response_queue.put(("err", f"ailia.Net load failed: {exc}"))
        return

    while True:
        request = request_queue.get()
        if request is None:
            return
        input_ids_np = request
        try:
            logits = net.run([input_ids_np])[0]
            last_logits = np.ascontiguousarray(logits[0, -1])
            response_queue.put(("ok", last_logits))
        except Exception as exc:
            response_queue.put(("err", str(exc)))
            return


class _AiliaLLMRunner:
    def __init__(self, onnx_path: str, timeout_sec: int):
        self._timeout_sec = timeout_sec
        self._onnx_path = onnx_path
        start_method = "fork" if sys.platform.startswith("linux") else "spawn"
        self._ctx = mp.get_context(start_method)
        self._request_queue = self._ctx.Queue()
        self._response_queue = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_ailia_llm_run_worker,
            args=(onnx_path, self._request_queue, self._response_queue),
        )

    def __enter__(self):
        self._proc.start()
        status, payload = self._receive()
        if status != "ready":
            self.close()
            raise RuntimeError(payload)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        if self._proc.is_alive():
            self._request_queue.put(None)
            self._proc.join(timeout=1)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join()

    def run_last(self, input_ids_np: np.ndarray) -> np.ndarray:
        input_ids_np = np.ascontiguousarray(input_ids_np.astype(np.int64, copy=False))
        self._request_queue.put(input_ids_np)
        status, payload = self._receive()
        if status == "ok":
            return payload
        self.close()
        raise RuntimeError(f"ailia runtime error: {payload}")

    def _receive(self):
        try:
            return self._response_queue.get(timeout=self._timeout_sec)
        except queue.Empty as exc:
            self.close()
            raise RuntimeError(f"ailia run timeout ({self._timeout_sec}s)") from exc


# ======================
# Generation
# ======================

def generate_speech_tokens(
    runner: _AiliaLLMRunner,
    tokenizer,
    text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    repetition_penalty: float,
    seed: int | None,
) -> list[int]:
    normalized_text, language = preprocess_text(text)
    if not normalized_text:
        raise ValueError("Normalized text is empty.")

    messages = [{"role": "user", "content": normalized_text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokenized = tokenizer(prompt, return_tensors="np")
    tokenized.pop("token_type_ids", None)
    input_ids_np = np.ascontiguousarray(tokenized.input_ids.astype(np.int64))

    logger.info(f"text: {text}")
    logger.info(f"normalized: {normalized_text}, language: {language}")
    logger.info(f"input_ids shape: {input_ids_np.shape}")

    rng = np.random.default_rng(seed)
    eos_token_id = tokenizer.eos_token_id
    current_ids = input_ids_np
    generated_ids: list[int] = []

    for step in range(max_new_tokens):
        last_logits = runner.run_last(current_ids)
        next_token = sample_next_token(
            last_logits, generated_ids, rng, temperature, top_p, do_sample, repetition_penalty
        )
        generated_ids.append(next_token)
        current_ids = np.concatenate(
            [current_ids, np.array([[next_token]], dtype=np.int64)], axis=1
        )

        if eos_token_id is not None and next_token == eos_token_id:
            break

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        if "<|im_end|>" in generated_text:
            break

        if (step + 1) % 25 == 0:
            logger.info(f"  generated: {step + 1} tokens")

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    speech_tokens = parse_speech_tokens(generated_text)
    logger.info(f"speech tokens: {len(speech_tokens)}")
    return speech_tokens, language


# ======================
# Main
# ======================

def main():
    check_and_download_models(WEIGHT_LLM, MODEL_LLM, REMOTE_PATH)
    check_and_download_models(WEIGHT_DECODER, MODEL_DECODER, REMOTE_PATH)

    import ailia

    memory_mode = ailia.get_memory_mode(
        reduce_constant=True,
        ignore_input_with_initializer=True,
        reduce_interstage=False,
        reuse_interstage=True,
    )
    net_dec = ailia.Net(
        weight=WEIGHT_DECODER,
        stream=None,
        env_id=args.env_id,
        memory_mode=memory_mode,
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    text = args.input
    do_sample = not args.greedy

    with _AiliaLLMRunner(WEIGHT_LLM, timeout_sec=300) as runner:
        speech_tokens, language = generate_speech_tokens(
            runner=runner,
            tokenizer=tokenizer,
            text=text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=do_sample,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        )

    preset_id = args.preset_id or ("jp_female" if language == "ja" else "en_male")
    preset_path = PRESET_DIR / f"{preset_id}{PRESET_EXT}"
    preset_emb_np = np.ascontiguousarray(np.load(str(preset_path)).astype(np.float32))

    tokens_np = np.ascontiguousarray(np.array(speech_tokens, dtype=np.int64))
    target_audio_length = round(len(tokens_np) / TOKEN_RATE_HZ * 44100)
    target_audio_length_np = np.array([target_audio_length], dtype=np.int64)

    logger.info(f"decoding {len(tokens_np)} tokens -> {target_audio_length / 44100:.2f}s")
    waveform = net_dec.run([tokens_np, preset_emb_np, target_audio_length_np])[0]

    sf.write(args.savepath, waveform.flatten(), samplerate=44100)
    logger.info(f'saved at : {args.savepath}')


if __name__ == '__main__':
    main()
