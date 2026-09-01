import os
import re

import logging
import numpy as np
import tqdm

from math_utils import softmax

CONTEXT_WINDOW_SIZE = 1024

SEMANTIC_RATE_HZ = 49.9
SEMANTIC_VOCAB_SIZE = 10_000

CODEBOOK_SIZE = 1024
N_COARSE_CODEBOOKS = 2
N_FINE_CODEBOOKS = 8
COARSE_RATE_HZ = 75

SAMPLE_RATE = 24_000

logger = logging.getLogger(__name__)

onnx = False

####
# Generation Functionality
####


TEXT_ENCODING_OFFSET = 10_048
SEMANTIC_PAD_TOKEN = 10_000
TEXT_PAD_TOKEN = 129_595
SEMANTIC_INFER_TOKEN = 129_599

COARSE_SEMANTIC_PAD_TOKEN = 12_048
COARSE_INFER_TOKEN = 12_050


def _normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def _flatten_codebooks(arr, offset_size=CODEBOOK_SIZE):
    arr = arr.copy()
    if offset_size is not None:
        for n in range(1, arr.shape[0]):
            arr[n, :] += offset_size * n
    flat_arr = arr.ravel("F")
    return flat_arr


def generate_text_semantic(
        models,
        text,
        temp=0.7,
        silent=False,
        min_eos_p=0.2,
        max_gen_duration_s=None,
        allow_early_stop=True):
    """Generate semantic tokens from text."""
    text = _normalize_whitespace(text)
    assert len(text.strip()) > 0

    tokenizer = models["tokenizer"]
    encoded_text = np.array(tokenizer.encode(text, add_special_tokens=False))
    encoded_text = encoded_text + TEXT_ENCODING_OFFSET

    if len(encoded_text) > 256:
        p = round((len(encoded_text) - 256) / len(encoded_text) * 100, 1)
        logger.warning(f"warning, text too long, lopping of last {p}%")
        encoded_text = encoded_text[:256]

    encoded_text = np.pad(
        encoded_text,
        (0, 256 - len(encoded_text)),
        constant_values=TEXT_PAD_TOKEN,
        mode="constant",
    )

    semantic_history = np.array([SEMANTIC_PAD_TOKEN] * 256)
    x = np.hstack([
        encoded_text, semantic_history, np.array([SEMANTIC_INFER_TOKEN])
    ]).astype(np.int64)
    x = np.expand_dims(x, axis=0)
    assert x.shape[1] == 256 + 256 + 1

    net = models["net"]
    kv_cache = None

    pbar_state = 0
    tot_generated_duration_s = 0
    n_tot_steps = 768
    pbar = tqdm.tqdm(disable=silent, total=n_tot_steps)
    for n in range(n_tot_steps):
        if kv_cache is not None:
            x_input = x[:, [-1]]
        else:
            x_input = x
            kv_cache = np.zeros((48, 16, 0, 64), dtype=np.float32)

        # feedforward
        if not onnx:
            output = net.predict([
                x_input, kv_cache,
            ])
        else:
            output = net.run(None, {
                'x_input': x_input, 'past_kv': kv_cache,
            })
        logits, kv_cache = output

        relevant_logits = logits[0, 0, :SEMANTIC_VOCAB_SIZE]
        if allow_early_stop:
            relevant_logits = np.hstack(
                (relevant_logits, logits[0, 0, [SEMANTIC_PAD_TOKEN]])  # eos
            )

        probs = softmax(relevant_logits / temp, axis=-1).astype(np.float64)
        item_next = np.random.multinomial(1, probs / sum(probs))
        item_next = np.argsort(-item_next)[:1]

        if allow_early_stop and (
                item_next == SEMANTIC_VOCAB_SIZE
                or (min_eos_p is not None and probs[-1] >= min_eos_p)
        ):
            # eos found, so break
            pbar.update(n - pbar_state)
            break

        x = np.concatenate((x, item_next[None]), axis=1)
        tot_generated_duration_s += 1 / SEMANTIC_RATE_HZ
        if max_gen_duration_s is not None \
                and tot_generated_duration_s > max_gen_duration_s:
            pbar.update(n - pbar_state)
            break
        if n == n_tot_steps - 1:
            pbar.update(n - pbar_state)
            break

        if n > pbar_state:
            if n > pbar.total:
                pbar.total = n
            pbar.update(n - pbar_state)

        pbar_state = n

    pbar.total = n
    pbar.refresh()
    pbar.close()
    out = x.squeeze()[256 + 256 + 1:]

    return out


def generate_coarse(
        models,
        x_semantic,
        temp=0.7,
        silent=False,
        max_coarse_history=630,  # min 60 (faster), max 630 (more context)
        sliding_window_len=60):
    """Generate coarse audio codes from semantic tokens."""
    semantic_to_coarse_ratio = COARSE_RATE_HZ / SEMANTIC_RATE_HZ * N_COARSE_CODEBOOKS
    max_semantic_history = int(np.floor(max_coarse_history / semantic_to_coarse_ratio))

    x_semantic_history = np.array([], dtype=np.int32)
    x_coarse_history = np.array([], dtype=np.int32)

    net = models["coarse"]

    # start loop
    n_steps = int(
        round(
            np.floor(len(x_semantic) * semantic_to_coarse_ratio / N_COARSE_CODEBOOKS)
            * N_COARSE_CODEBOOKS
        )
    )
    x_semantic = np.hstack([x_semantic_history, x_semantic]).astype(np.int32)
    x_coarse = x_coarse_history.astype(np.int32)
    base_semantic_idx = len(x_semantic_history)

    x_semantic_in = x_semantic[None]
    x_coarse_in = x_coarse[None]
    n_window_steps = int(np.ceil(n_steps / sliding_window_len))
    n_step = 0
    for _ in tqdm.tqdm(range(n_window_steps), total=n_window_steps, disable=silent):
        semantic_idx = base_semantic_idx + int(round(n_step / semantic_to_coarse_ratio))
        # pad from right side
        x_in = x_semantic_in[:, np.max([0, semantic_idx - max_semantic_history]):]
        x_in = x_in[:, :256]
        x_in = np.pad(
            x_in,
            [(0, 0), (0, 256 - x_in.shape[-1])],
            "constant",
            constant_values=COARSE_SEMANTIC_PAD_TOKEN,
        )
        x_in = np.hstack([
            x_in,
            np.array([COARSE_INFER_TOKEN])[None],
            x_coarse_in[:, -max_coarse_history:],
        ])

        kv_cache = None
        for _ in range(sliding_window_len):
            if n_step >= n_steps:
                continue
            is_major_step = n_step % N_COARSE_CODEBOOKS == 0

            if kv_cache is not None:
                x_input = x_in[:, [-1]]
            else:
                x_input = x_in
                kv_cache = np.zeros((48, 16, 0, 64), dtype=np.float32)

            # feedforward
            if not onnx:
                output = net.predict([
                    x_input, kv_cache,
                ])
            else:
                output = net.run(None, {
                    'x_input': x_input, 'past_kv': kv_cache,
                })
            logits, kv_cache = output

            logit_start_idx = \
                SEMANTIC_VOCAB_SIZE + (1 - int(is_major_step)) * CODEBOOK_SIZE
            logit_end_idx = \
                SEMANTIC_VOCAB_SIZE + (2 - int(is_major_step)) * CODEBOOK_SIZE

            relevant_logits = logits[0, 0, logit_start_idx:logit_end_idx]

            probs = softmax(relevant_logits / temp, axis=-1).astype(np.float64)
            item_next = np.random.multinomial(1, probs / sum(probs))
            item_next = np.argsort(-item_next)[:1]
            item_next += logit_start_idx

            x_coarse_in = np.concatenate((x_coarse_in, item_next[None]), axis=1)
            x_in = np.concatenate((x_in, item_next[None]), axis=1)

            n_step += 1

    gen_coarse_arr = x_coarse_in[len(x_coarse_history):]
    gen_coarse_audio_arr = gen_coarse_arr.reshape(-1, N_COARSE_CODEBOOKS).T - SEMANTIC_VOCAB_SIZE
    for n in range(1, N_COARSE_CODEBOOKS):
        gen_coarse_audio_arr[n, :] -= n * CODEBOOK_SIZE

    return gen_coarse_audio_arr


def generate_fine(
        models,
        x_coarse_gen,
        temp=0.5,
        silent=False):
    """Generate full audio codes from coarse audio codes."""
    x_fine_history = None
    n_coarse = x_coarse_gen.shape[0]

    net = models["fine"]

    use_torch = False
    if hasattr(net, "parameters"):
        device = next(net.parameters()).device
        use_torch = True

    # make input arr
    in_arr = np.vstack(
        [
            x_coarse_gen,
            np.zeros((N_FINE_CODEBOOKS - n_coarse, x_coarse_gen.shape[1]))
            + CODEBOOK_SIZE,  # padding
        ]
    ).astype(np.int32)
    # prepend history if available (max 512)
    if x_fine_history is not None:
        x_fine_history = x_fine_history.astype(np.int32)
        in_arr = np.hstack(
            [
                x_fine_history[:, -512:].astype(np.int32),
                in_arr,
            ]
        )
        n_history = x_fine_history[:, -512:].shape[1]
    else:
        n_history = 0
    n_remove_from_end = 0
    # need to pad if too short (since non-causal model)
    if in_arr.shape[1] < 1024:
        n_remove_from_end = 1024 - in_arr.shape[1]
        in_arr = np.hstack(
            [
                in_arr,
                np.zeros((N_FINE_CODEBOOKS, n_remove_from_end), dtype=np.int32) + CODEBOOK_SIZE,
            ]
        )
    # we can be lazy about fractional loop and just keep overwriting codebooks
    n_loops = np.max([0, int(np.ceil((x_coarse_gen.shape[1] - (1024 - n_history)) / 512))]) + 1

    in_arr = in_arr.T
    for n in tqdm.tqdm(range(n_loops), disable=silent):
        start_idx = np.min([n * 512, in_arr.shape[0] - 1024])
        start_fill_idx = np.min([n_history + n * 512, in_arr.shape[0] - 512])
        rel_start_fill_idx = start_fill_idx - start_idx
        in_buffer = in_arr[start_idx: start_idx + 1024, :][None]
        for nn in range(n_coarse, N_FINE_CODEBOOKS):
            # feedforward
            if use_torch:
                import torch
                logits = net(nn, torch.tensor(in_buffer).to(device))
                logits = logits.detach().cpu().numpy()
            else:
                if not onnx:
                    output = net.predict([
                        np.array(nn), in_buffer
                    ])
                else:
                    output = net.run(None, {
                        'pred_idx': np.array(nn), 'idx': in_buffer,
                    })
                logits = output[0]

            if temp is None:
                relevant_logits = logits[0, rel_start_fill_idx:, :CODEBOOK_SIZE]
                codebook_preds = np.argmax(relevant_logits, axis=-1)
            else:
                relevant_logits = logits[0, :, :CODEBOOK_SIZE] / temp
                probs = softmax(relevant_logits, axis=-1).astype(np.float64)
                a = [
                    np.random.multinomial(1, p / sum(p))
                    for p in probs[rel_start_fill_idx:1024]
                ]
                codebook_preds = np.array([np.argsort(-x)[0] for x in a])
            codebook_preds = codebook_preds.astype(int)
            in_buffer[0, rel_start_fill_idx:, nn] = codebook_preds

        # transfer over info into model_in and convert to numpy
        for nn in range(n_coarse, N_FINE_CODEBOOKS):
            in_arr[
            start_fill_idx: start_fill_idx + (1024 - rel_start_fill_idx), nn
            ] = in_buffer[0, rel_start_fill_idx:, nn]

    gen_fine_arr = in_arr.squeeze().T

    gen_fine_arr = gen_fine_arr[:, n_history:]
    if n_remove_from_end > 0:
        gen_fine_arr = gen_fine_arr[:, :-n_remove_from_end]

    return gen_fine_arr


def codec_decode(models, fine_tokens):
    """Turn quantized audio codes into audio array using encodec."""
    net = models["codec"]
    device = next(net.parameters()).device

    arr = fine_tokens[None]
    arr = arr.transpose(1, 0, 2)

    import torch
    arr = torch.from_numpy(arr).to(device)
    emb = net.quantizer.decode(arr)
    out = net.decoder(emb)
    audio_arr = out.detach().cpu().numpy().squeeze()

    return audio_arr
