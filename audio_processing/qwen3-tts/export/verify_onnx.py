"""
Compare the exported Qwen3-TTS ONNX models against the PyTorch reference.

Tested versions:
    torch 2.10.0 (CPU wheel)
    transformers 4.57.3
    qwen-tts 0.1.1
    onnxruntime 1.28.0

Usage:
    python3 verify_onnx.py --parameter_num 1.7B --onnx_dir .

Each module is checked in its own process, because the reference model and the
ONNX session cannot comfortably share memory for the 1.7B talker.
"""

import argparse
import os
import subprocess
import sys

import librosa
import numpy as np
import onnxruntime
import torch
from transformers.cache_utils import DynamicCache

import export_onnx as ex

TARGETS = [
    "speaker_encoder",
    "tokenizer_encoder",
    "tokenizer_decoder",
    "talker_io_units",
    "subtalker_decoder",
    "talker_decoder",
]

REF_AUDIO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clone_2.wav")


def session(path):
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return onnxruntime.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def run(sess, inputs):
    names = [i.name for i in sess.get_inputs()]
    return sess.run(None, dict(zip(names, inputs)))


def report(name, reference, actual, tolerance=2e-3):
    reference = np.asarray(reference)
    actual = np.asarray(actual)
    if reference.shape != actual.shape:
        print(f"  {name}: SHAPE MISMATCH torch={reference.shape} onnx={actual.shape}")
        return False
    if reference.dtype.kind in "iu":
        mismatch = int((reference != actual).sum())
        ok = mismatch == 0
        print(f"  {name}: {reference.shape} mismatching elements {mismatch} -> {'OK' if ok else 'NG'}")
        return ok
    diff = float(np.abs(reference - actual).max())
    ok = diff <= tolerance
    print(f"  {name}: {reference.shape} max abs diff {diff:.3e} -> {'OK' if ok else 'NG'}")
    return ok


def load_reference_audio(seconds=None):
    wav, _ = librosa.load(REF_AUDIO, sr=24000, mono=True)
    if seconds is not None:
        wav = wav[: int(24000 * seconds)]
    return wav.astype(np.float32)


def mel_spectrogram(wav):
    """extract_speaker_embedding()'s mel front end, as used by qwen3-tts.py."""
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram as ref_mel

    mels = ref_mel(
        torch.from_numpy(wav).unsqueeze(0),
        n_fft=1024,
        num_mels=128,
        sampling_rate=24000,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12000,
    ).transpose(1, 2)
    return mels


def causal_mask(seq_len, past_len=0):
    return ex.causal_mask(seq_len, past_len)


# ======================================================================
# per module checks
# ======================================================================
def check_speaker_encoder(model_dir, onnx_path):
    model = ex.load_tts_model(model_dir)
    mels = mel_spectrogram(load_reference_audio())
    with torch.no_grad():
        reference = model.speaker_encoder(mels)
    actual = run(session(onnx_path), [mels.numpy()])[0]
    return report("embedding", reference, actual)


def check_tokenizer_encoder(model_dir, onnx_path):
    ex.patch_codebook_quantize()
    model = ex.load_speech_tokenizer(model_dir)
    sess = session(onnx_path)
    ok = True
    # two different lengths: the exported graph has to stay length agnostic
    for seconds in (5.0, None):
        wav = load_reference_audio(seconds)
        audio = torch.from_numpy(wav)[None, None]
        with torch.no_grad():
            reference = model.encoder.encode(input_values=audio, return_dict=True).audio_codes
        actual = run(sess, [audio.numpy()])[0]
        ok &= report(f"audio_codes ({wav.shape[0]} samples)", reference.numpy(), actual)
    return ok


def check_tokenizer_decoder(model_dir, onnx_path):
    model = ex.load_speech_tokenizer(model_dir)
    num_quantizers = model.config.decoder_config.num_quantizers
    sess = session(onnx_path)
    ok = True
    rng = np.random.default_rng(0)
    for num_tokens in (60, 137):
        codes = torch.from_numpy(
            rng.integers(0, 2048, size=(1, num_quantizers, num_tokens)).astype(np.int64)
        )
        with torch.no_grad():
            reference = model.decoder(codes)
        actual = run(sess, [codes.numpy()])[0]
        ok &= report(f"waveform ({num_tokens} tokens)", reference.numpy(), actual)
    return ok


def check_talker_io_units(model_dir, onnx_path):
    model = ex.load_tts_model(model_dir)
    talker_config = model.config.talker_config
    rng = np.random.default_rng(0)
    text_features = rng.standard_normal((1, 7, talker_config.text_hidden_size)).astype(np.float32)
    hidden = rng.standard_normal((1, 1, talker_config.hidden_size)).astype(np.float32)
    with torch.no_grad():
        projected = model.talker.text_projection(torch.from_numpy(text_features))
        logits = model.talker.codec_head(torch.from_numpy(hidden))
    actual = run(session(onnx_path), [text_features, hidden])
    ok = report("projected_text", projected.numpy(), actual[0])
    ok &= report("logits", logits.numpy(), actual[1])
    return ok


def check_transformer(reference_model, hidden_size, config, onnx_path, project=None):
    """Prefill + one decode step, against the reference module with a Cache."""
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    prefill_len = 6

    rng = np.random.default_rng(0)
    embeds = rng.standard_normal((1, prefill_len, hidden_size)).astype(np.float32)
    step_embeds = rng.standard_normal((1, 1, hidden_size)).astype(np.float32)

    with torch.no_grad():
        cache = DynamicCache()
        inputs = torch.from_numpy(embeds)
        if project is not None:
            inputs = project(inputs)
        reference = reference_model(
            inputs_embeds=inputs,
            attention_mask=torch.ones(1, prefill_len, dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
        )
        reference_prefill = reference.last_hidden_state.numpy()

        step_inputs = torch.from_numpy(step_embeds)
        if project is not None:
            step_inputs = project(step_inputs)
        reference_step = reference_model(
            inputs_embeds=step_inputs,
            attention_mask=torch.ones(1, prefill_len + 1, dtype=torch.long),
            position_ids=torch.tensor([[prefill_len]]),
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.tensor([prefill_len]),
        ).last_hidden_state.numpy()

    sess = session(onnx_path)
    empty = [
        np.zeros((1, num_kv_heads, 0, head_dim), dtype=np.float32) for _ in range(num_layers * 2)
    ]
    outputs = run(
        sess,
        [
            embeds,
            causal_mask(prefill_len).numpy(),
            np.arange(prefill_len, dtype=np.int64)[None],
            *empty,
        ],
    )
    ok = report("last_hidden_state (prefill)", reference_prefill, outputs[0])

    outputs = run(
        sess,
        [
            step_embeds,
            causal_mask(1, prefill_len).numpy(),
            np.array([[prefill_len]], dtype=np.int64),
            *outputs[1:],
        ],
    )
    ok &= report("last_hidden_state (decode)", reference_step, outputs[0])
    return ok


def check_talker_decoder(model_dir, onnx_path):
    model = ex.load_tts_model(model_dir)
    talker_config = model.config.talker_config
    return check_transformer(
        model.talker.model, talker_config.hidden_size, talker_config, onnx_path
    )


def check_subtalker_decoder(model_dir, onnx_path):
    model = ex.load_tts_model(model_dir)
    talker_config = model.config.talker_config
    code_predictor = model.talker.code_predictor
    # the ONNX takes the talker hidden size and projects inside the graph
    return check_transformer(
        code_predictor.model,
        talker_config.hidden_size,
        talker_config.code_predictor_config,
        onnx_path,
        project=code_predictor.small_to_mtp_projection,
    )


CHECKS = {
    "speaker_encoder": check_speaker_encoder,
    "tokenizer_encoder": check_tokenizer_encoder,
    "tokenizer_decoder": check_tokenizer_decoder,
    "talker_io_units": check_talker_io_units,
    "subtalker_decoder": check_subtalker_decoder,
    "talker_decoder": check_talker_decoder,
}


def main():
    parser = argparse.ArgumentParser(description="Verify the exported Qwen3-TTS ONNX models")
    parser.add_argument(
        "-p", "--parameter_num", default="0.6B", choices=sorted(ex.MODEL_ID.keys())
    )
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--onnx_dir", default=".")
    parser.add_argument("--only", default=None, choices=TARGETS)
    args = parser.parse_args()

    model_dir = args.model_dir
    if model_dir is None:
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(ex.MODEL_ID[args.parameter_num])

    if args.only is None:
        failed = []
        for target in TARGETS:
            code = subprocess.call(
                [
                    sys.executable, os.path.abspath(__file__),
                    "--parameter_num", args.parameter_num,
                    "--model_dir", model_dir,
                    "--onnx_dir", args.onnx_dir,
                    "--only", target,
                ]
            )
            if code != 0:
                failed.append(target)
        if failed:
            print("FAILED: " + ", ".join(failed))
            sys.exit(1)
        print("all modules match the reference implementation")
        return

    onnx_path = os.path.join(
        args.onnx_dir, f"qwen3_tts_{args.only}_{args.parameter_num}.onnx"
    )
    print(f"[{args.only}]")
    if not CHECKS[args.only](model_dir, onnx_path):
        sys.exit(1)


if __name__ == "__main__":
    main()
