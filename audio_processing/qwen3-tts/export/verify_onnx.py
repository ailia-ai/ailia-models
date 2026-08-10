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
import gc
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
    "codec_tables",
    "encoder",
    "tokenizer_decoder",
    "prompt",
    "code_predictor",
    "talker",
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
    return ex.causal_mask(seq_len, past_len).numpy()


def empty_cache(config, num_layers=None):
    num_layers = config.num_hidden_layers if num_layers is None else num_layers
    return [
        np.zeros((1, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32)
        for _ in range(num_layers * 2)
    ]


# ======================================================================
# per module checks
# ======================================================================
def check_encoder(model_dir, onnx_path):
    """Speech tokenizer encoder and speaker encoder, which share one graph."""
    ex.patch_codebook_quantize()
    tokenizer = ex.load_speech_tokenizer(model_dir)
    model = ex.load_tts_model(model_dir)

    # two different lengths: the exported graph has to stay length agnostic
    cases = []
    for seconds in (5.0, None):
        wav = load_reference_audio(seconds)
        audio = torch.from_numpy(wav)[None, None]
        mels = mel_spectrogram(wav)
        with torch.no_grad():
            codes = tokenizer.encoder.encode(input_values=audio, return_dict=True).audio_codes
            embedding = model.speaker_encoder(mels)
        cases.append((audio.numpy(), mels.numpy(), codes.numpy(), embedding.numpy()))

    del model, tokenizer
    gc.collect()

    sess = session(onnx_path)
    ok = True
    for audio, mels, codes, embedding in cases:
        actual = run(sess, [audio, mels])
        label = f"{audio.shape[2]} samples"
        ok &= report(f"audio_codes ({label})", codes, actual[0])
        ok &= report(f"speaker_embedding ({label})", embedding, actual[1])
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


def check_prompt(model_dir, onnx_path):
    """The embedding tables and text_projection used to build the talker prompt."""
    model = ex.load_tts_model(model_dir)
    talker = model.talker
    code_predictor = talker.code_predictor
    num_code_groups = model.config.talker_config.num_code_groups

    rng = np.random.default_rng(0)
    text_tokens = rng.integers(0, 151000, size=(1, 9)).astype(np.int64)
    codec_ids = rng.integers(0, 2048, size=(1, 5)).astype(np.int64)
    ref_codes = rng.integers(0, 2048, size=(1, num_code_groups, 7)).astype(np.int64)

    with torch.no_grad():
        text_embeds = talker.model.text_embedding(torch.from_numpy(text_tokens))
        projected_text = talker.text_projection(text_embeds)
        codec_embeds = talker.model.codec_embedding(torch.from_numpy(codec_ids))
        codes = torch.from_numpy(ref_codes)
        ref_codec_sum = talker.model.codec_embedding(codes[:, 0])
        for group in range(num_code_groups - 1):
            ref_codec_sum = ref_codec_sum + code_predictor.model.codec_embedding[group](
                codes[:, group + 1]
            )
    reference = [projected_text.numpy(), codec_embeds.numpy(), ref_codec_sum.numpy()]

    del model, talker, code_predictor
    gc.collect()

    actual = run(session(onnx_path), [text_tokens, codec_ids, ref_codes])
    ok = report("projected_text", reference[0], actual[0])
    ok &= report("codec_embeds", reference[1], actual[1])
    ok &= report("ref_codec_sum", reference[2], actual[2])
    return ok


def check_codec_tables(model_dir, onnx_path):
    """The npy the runtime looks codec embeddings up in."""
    tables = np.load(onnx_path)
    model = ex.load_tts_model(model_dir)
    talker = model.talker
    code_predictor = talker.code_predictor
    num_code_groups = model.config.talker_config.num_code_groups
    talker_vocab = talker.model.codec_embedding.weight.shape[0]
    group_vocab = code_predictor.model.codec_embedding[0].weight.shape[0]

    ok = report(
        "group 0 table", talker.model.codec_embedding.weight.detach().numpy(),
        tables[:talker_vocab], tolerance=0.0,
    )
    for group in range(num_code_groups - 1):
        start = talker_vocab + group * group_vocab
        ok &= report(
            f"group {group + 1} table",
            code_predictor.model.codec_embedding[group].weight.detach().numpy(),
            tables[start:start + group_vocab], tolerance=0.0,
        )
    return ok


def check_talker(model_dir, onnx_path):
    """Prefill and one decode step, with the output head read back too."""
    model = ex.load_tts_model(model_dir)
    talker_config = model.config.talker_config
    talker = model.talker
    prefill_len = 6

    rng = np.random.default_rng(0)
    embeds = rng.standard_normal((1, prefill_len, talker_config.hidden_size)).astype(np.float32)
    step_embeds = rng.standard_normal((1, 1, talker_config.hidden_size)).astype(np.float32)

    with torch.no_grad():
        cache = DynamicCache()
        reference_hidden = talker.model(
            inputs_embeds=torch.from_numpy(embeds),
            attention_mask=torch.ones(1, prefill_len, dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
        ).last_hidden_state[:, -1:, :]
        reference_logits = talker.codec_head(reference_hidden)

        reference_step = talker.model(
            inputs_embeds=torch.from_numpy(step_embeds),
            attention_mask=torch.ones(1, prefill_len + 1, dtype=torch.long),
            position_ids=torch.tensor([[prefill_len]]),
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.tensor([prefill_len]),
        ).last_hidden_state
        reference_step_logits = talker.codec_head(reference_step)

    reference = [
        reference_logits.numpy(), reference_hidden.numpy(),
        reference_step_logits.numpy(), reference_step.numpy(),
    ]
    del model, talker, cache
    gc.collect()

    sess = session(onnx_path)
    outputs = run(
        sess,
        [
            embeds,
            causal_mask(prefill_len),
            np.arange(prefill_len, dtype=np.int64)[None],
            *empty_cache(talker_config),
        ],
    )
    ok = report("logits (prefill)", reference[0], outputs[0])
    ok &= report("last_hidden (prefill)", reference[1], outputs[1])

    outputs = run(
        sess,
        [
            step_embeds,
            causal_mask(1, prefill_len),
            np.array([[prefill_len]], dtype=np.int64),
            *outputs[2:],
        ],
    )
    ok &= report("logits (decode)", reference[2], outputs[0])
    ok &= report("last_hidden (decode)", reference[3], outputs[1])
    return ok


def check_code_predictor(model_dir, onnx_path):
    """All 15 code groups of one frame, against the reference module."""
    model = ex.load_tts_model(model_dir)
    talker_config = model.config.talker_config
    config = talker_config.code_predictor_config
    talker = model.talker
    code_predictor = talker.code_predictor
    num_code_groups = talker_config.num_code_groups
    group_vocab = config.vocab_size

    rng = np.random.default_rng(0)
    talker_hidden = rng.standard_normal((1, 1, talker_config.hidden_size)).astype(np.float32)
    # group 0 comes from the talker, groups 1..14 are fed back one at a time
    tokens = rng.integers(0, group_vocab, size=num_code_groups - 1).astype(np.int64)

    with torch.no_grad():
        cache = DynamicCache()
        group0 = talker.model.codec_embedding(torch.tensor([[int(tokens[0])]]))
        outputs = code_predictor(
            inputs_embeds=torch.cat([torch.from_numpy(talker_hidden), group0], dim=1),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
        )
        reference_logits = [outputs.logits[:, -1:, :].numpy()]
        embeds = [torch.cat([torch.from_numpy(talker_hidden), group0], dim=1).numpy()]
        for k in range(1, num_code_groups - 1):
            step = code_predictor.model.codec_embedding[k - 1](
                torch.tensor([[int(tokens[k])]]))
            embeds.append(step.numpy())
            outputs = code_predictor(
                input_ids=torch.tensor([[int(tokens[k])]]),
                attention_mask=torch.ones(1, k + 2, dtype=torch.long),
                position_ids=torch.tensor([[k + 1]]),
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.tensor([k + 1]),
                generation_steps=k,
            )
            reference_logits.append(outputs.logits[:, -1:, :].numpy())

    del model, talker, code_predictor, cache
    gc.collect()

    sess = session(onnx_path)
    head_rows = [
        np.arange(k * group_vocab, (k + 1) * group_vocab, dtype=np.int64)
        for k in range(num_code_groups - 1)
    ]
    actual = run(
        sess,
        [embeds[0], head_rows[0], causal_mask(2), np.array([[0, 1]], dtype=np.int64),
         *empty_cache(config)],
    )
    ok = report("logits (group 0 -> 1)", reference_logits[0], actual[0])
    for k in range(1, num_code_groups - 1):
        actual = run(
            sess,
            [embeds[k], head_rows[k], causal_mask(1, k + 1),
             np.array([[k + 1]], dtype=np.int64), *actual[1:]],
        )
        ok &= report(f"logits (group {k} -> {k + 1})", reference_logits[k], actual[0])
    return ok


CHECKS = {
    "codec_tables": check_codec_tables,
    "encoder": check_encoder,
    "tokenizer_decoder": check_tokenizer_decoder,
    "prompt": check_prompt,
    "code_predictor": check_code_predictor,
    "talker": check_talker,
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

    suffix = "npy" if args.only == "codec_tables" else "onnx"
    onnx_path = os.path.join(
        args.onnx_dir, f"qwen3_tts_{args.only}_{args.parameter_num}.{suffix}"
    )
    print(f"[{args.only}]")
    if not CHECKS[args.only](model_dir, onnx_path):
        sys.exit(1)


if __name__ == "__main__":
    main()
