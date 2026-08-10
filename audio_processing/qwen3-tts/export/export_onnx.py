"""
Export Qwen3-TTS 12Hz Base (0.6B / 1.7B) to ONNX for ailia SDK.

Tested versions:
    torch 2.10.0 (CPU wheel)
    transformers 4.57.3
    qwen-tts 0.1.1
    onnx 1.22.0

Requirements:
    pip install -r requirements.txt

Usage:
    python3 export_onnx.py --parameter_num 0.6B
    python3 export_onnx.py --parameter_num 1.7B

    # export a single module (each module is exported in its own process by
    # default, so this is mainly useful to resume an interrupted run)
    python3 export_onnx.py --parameter_num 1.7B --only talker_decoder

The model is split so that the auto regressive loop can be driven from Python
(see ../qwen3-tts.py). ``<p>`` is the parameter_num (0.6B or 1.7B), ``H`` the
talker hidden size (1024 for 0.6B, 2048 for 1.7B) and ``Hs`` the sub talker
(code predictor) hidden size (1024 for both).

    qwen3_tts_speaker_encoder_<p>.onnx
        mel [B, frames, 128]                    -> speaker embedding [B, H]
    qwen3_tts_tokenizer_encoder_<p>.onnx
        waveform [1, 1, L]                      -> audio codes [1, 32, T]
    qwen3_tts_tokenizer_decoder_<p>.onnx
        audio codes [B, 16, T]                  -> waveform [1, 1, L]
    qwen3_tts_talker_io_units_<p>.onnx
        (text features [B, seq, 2048],          -> (projected text [B, seq, H],
         talker hidden [B, 1, H])                   codec logits [B, 1, 3072])
    qwen3_tts_talker_decoder_<p>.onnx
        talker transformer (28 layers) with a KV cache passed in/out
    qwen3_tts_subtalker_decoder_<p>.onnx
        code predictor (5 layers) with a KV cache passed in/out
    qwen3_tts_text_embedding_<p>.npy        [text_vocab_size, 2048]
    qwen3_tts_codec_embeddings_<p>.npy      [1, 3072, H]
    qwen3_tts_subtalker_lm_heads_<p>.npy    [15, 2048, Hs]
    qwen3_tts_subtalker_codec_emb_<p>.npy   [15, 2048, H]

The speech tokenizer weights are identical for 0.6B and 1.7B, so
qwen3_tts_tokenizer_{encoder,decoder}_0.6B.onnx and the 1.7B ones have the same
content. They are exported per size to keep the runtime file names uniform.
"""

import argparse
import gc
import importlib.machinery
import os
import subprocess
import sys
import types
import urllib.request

import numpy as np
import onnx
import torch
from torch import nn

# qwen_tts pulls in its 25Hz tokenizer through __init__, which imports pysox.
# pysox is not needed for the 12Hz models exported here and does not build on
# every platform, so register a stub before importing anything from qwen_tts.
if "sox" not in sys.modules:
    _sox = types.ModuleType("sox")
    _sox.__spec__ = importlib.machinery.ModuleSpec("sox", None)
    sys.modules["sox"] = _sox

import transformers.masking_utils  # noqa: E402
import transformers.models.mimi.modeling_mimi  # noqa: E402

from qwen_tts.core.models.modeling_qwen3_tts import (  # noqa: E402
    Qwen3TTSForConditionalGeneration,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from qwen_tts.core.tokenizer_12hz import modeling_qwen3_tts_tokenizer_v2  # noqa: E402
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (  # noqa: E402
    Qwen3TTSTokenizerV2Model,
)

OPSET = 17

MODEL_ID = {
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

TARGETS = [
    "npy",
    "speaker_encoder",
    "tokenizer_encoder",
    "tokenizer_decoder",
    "talker_io_units",
    "subtalker_decoder",
    # exported last: it is by far the largest module
    "talker_decoder",
]

ONNX2PROTOTXT_URL = (
    "https://raw.githubusercontent.com/ailia-ai/export-to-onnx/master/onnx2prototxt.py"
)

# Weights only have to live in a separate .onnx.data file when the model does not
# fit in the 2GB protobuf limit, which is the case for the 1.7B talker only. The
# published 0.6B files additionally use external data for these two modules, and
# ../qwen3-tts.py downloads exactly that set, so a re-export has to keep it.
PUBLISHED_EXTERNAL_DATA = {
    "0.6B": {"speaker_encoder": True, "talker_io_units": True},
    "1.7B": {"speaker_encoder": False, "talker_io_units": False},
}


# ======================================================================
# monkey patches needed to make the reference modules exportable
# ======================================================================
def traceable_causal_mask(
    config=None, input_embeds=None, attention_mask=None, cache_position=None, **kwargs
):
    """create_causal_mask() built from plain tensor ops.

    transformers builds its attention masks with torch.vmap, which the
    TorchScript ONNX exporter cannot trace. The speech tokenizer only ever runs
    on a single unpadded sequence without a KV cache, so the mask is just the
    additive causal mask; deriving it from cache_position keeps the sequence
    length dynamic in the exported graph.
    """
    dtype = input_embeds.dtype
    positions = cache_position[:, None]
    keys = cache_position[None, :]
    mask = torch.zeros(1, 1, 1, 1, dtype=dtype) + torch.where(
        keys <= positions, 0.0, torch.finfo(dtype).min
    ).to(dtype)
    return mask


def patch_causal_mask():
    """Make the speech tokenizer traceable by the TorchScript ONNX exporter."""
    transformers.masking_utils.create_causal_mask = traceable_causal_mask
    transformers.models.mimi.modeling_mimi.create_causal_mask = traceable_causal_mask
    modeling_qwen3_tts_tokenizer_v2.create_causal_mask = traceable_causal_mask


def traceable_quantize(self, hidden_states):
    """MimiEuclideanCodebook.quantize() without torch.cdist.

    The opset 9 symbolic function for cdist needs a statically known number of
    rows, which would pin the exported encoder to a single audio length.
    Expanding ||x - e||^2 into x^2 - 2*x*e + e^2 keeps the length dynamic;
    argmin over the squared distance picks the same centroid.
    """
    embed = self.embed.float()
    hidden_states = hidden_states.float()
    dists = (
        hidden_states.pow(2).sum(-1, keepdim=True)
        - 2 * hidden_states @ embed.t()
        + embed.pow(2).sum(-1)[None]
    )
    return dists.argmin(dim=-1)


def patch_codebook_quantize():
    transformers.models.mimi.modeling_mimi.MimiEuclideanCodebook.quantize = traceable_quantize


# ======================================================================
# rotary embedding
# ======================================================================
def rope_cos_sin(inv_freq, position_ids, dtype):
    """Qwen3TTSRotaryEmbedding.forward for 2D position_ids.

    The talker uses an mRoPE with 3 sections, but Qwen3-TTS-Base feeds the same
    position id to all 3 of them (get_rope_index returns 3 identical rows for a
    text only sequence). With identical rows the interleaved mRoPE collapses to
    the plain rotary embedding, so the exported graph only takes [B, seq]
    position ids. This is checked against the reference implementation by
    verify_onnx.py.
    """
    inv_freq = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    positions = position_ids[:, None, :].float()
    freqs = (inv_freq @ positions).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def decoder_layer_forward(layer, hidden_states, attention_mask, cos, sin, past_key, past_value):
    """Qwen3TTS{,Talker}DecoderLayer.forward with an explicit KV cache."""
    attn = layer.self_attn
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)

    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)

    query = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    key = torch.cat([past_key, key], dim=2)
    value = torch.cat([past_value, value], dim=2)

    attn_output, _ = eager_attention_forward(
        attn, query, key, value, attention_mask, scaling=attn.scaling, dropout=0.0
    )
    attn_output = attn_output.reshape(*input_shape, -1)
    hidden_states = residual + attn.o_proj(attn_output)

    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = residual + layer.mlp(hidden_states)

    return hidden_states, key, value


# ======================================================================
# wrappers
# ======================================================================
class SpeakerEncoder(nn.Module):
    """ECAPA-TDNN speaker encoder: mel spectrogram -> speaker embedding."""

    def __init__(self, speaker_encoder):
        super().__init__()
        self.speaker_encoder = speaker_encoder

    def forward(self, hidden_states):
        return self.speaker_encoder(hidden_states)


class TokenizerEncoder(nn.Module):
    """Qwen3-TTS-Tokenizer-12Hz encoder: waveform -> audio codes."""

    def __init__(self, tokenizer_model):
        super().__init__()
        self.encoder = tokenizer_model.encoder

    def forward(self, audio_values):
        return self.encoder.encode(input_values=audio_values, return_dict=True).audio_codes


class TokenizerDecoder(nn.Module):
    """Qwen3-TTS-Tokenizer-12Hz decoder: audio codes -> waveform."""

    def __init__(self, tokenizer_model):
        super().__init__()
        self.decoder = tokenizer_model.decoder

    def forward(self, codes):
        return self.decoder(codes)


class TalkerIOUnits(nn.Module):
    """The two small talker units that surround the transformer.

    text_projection maps a text embedding to the talker hidden size and
    codec_head maps the talker hidden state to codec logits. They are bundled
    into a single ONNX file because both are tiny.
    """

    def __init__(self, talker):
        super().__init__()
        self.text_proj = talker.text_projection
        self.codec_head = talker.codec_head

    def forward(self, text_features, last_hidden_states):
        return self.text_proj(text_features), self.codec_head(last_hidden_states)


class TalkerDecoder(nn.Module):
    """Talker transformer with the KV cache as plain inputs / outputs."""

    def __init__(self, talker_model):
        super().__init__()
        self.layers = talker_model.layers
        self.norm = talker_model.norm
        self.register_buffer("inv_freq", talker_model.rotary_emb.inv_freq, persistent=False)

    def forward(self, inputs_embeds, attention_mask, position_ids, *past_key_values):
        cos, sin = rope_cos_sin(self.inv_freq, position_ids, inputs_embeds.dtype)

        hidden_states = inputs_embeds
        present = []
        for i, layer in enumerate(self.layers):
            hidden_states, key, value = decoder_layer_forward(
                layer,
                hidden_states,
                attention_mask,
                cos,
                sin,
                past_key_values[2 * i],
                past_key_values[2 * i + 1],
            )
            present += [key, value]

        return (self.norm(hidden_states), *present)


class SubTalkerDecoder(nn.Module):
    """Code predictor transformer with the KV cache as plain inputs / outputs.

    inputs_embeds is in the talker hidden size; small_to_mtp_projection brings
    it down to the code predictor hidden size (an identity for 0.6B).
    """

    def __init__(self, code_predictor):
        super().__init__()
        self.small_to_mtp_projection = code_predictor.small_to_mtp_projection
        self.layers = code_predictor.model.layers
        self.norm = code_predictor.model.norm
        self.register_buffer(
            "inv_freq", code_predictor.model.rotary_emb.inv_freq, persistent=False
        )

    def forward(self, inputs_embeds, attention_mask, position_ids, *past_key_values):
        inputs_embeds = self.small_to_mtp_projection(inputs_embeds)
        cos, sin = rope_cos_sin(self.inv_freq, position_ids, inputs_embeds.dtype)

        hidden_states = inputs_embeds
        present = []
        for i, layer in enumerate(self.layers):
            hidden_states, key, value = decoder_layer_forward(
                layer,
                hidden_states,
                attention_mask,
                cos,
                sin,
                past_key_values[2 * i],
                past_key_values[2 * i + 1],
            )
            present += [key, value]

        return (self.norm(hidden_states), *present)


# ======================================================================
# helpers
# ======================================================================
def causal_mask(seq_len, past_len=0):
    total_len = seq_len + past_len
    mask = torch.full((seq_len, total_len), float("-inf"))
    for i in range(seq_len):
        mask[i, : past_len + i + 1] = 0.0
    return mask[None, None, :, :]


def export(
    model, args, path, input_names, output_names, dynamic_axes,
    dynamo=False, external_data=False,
):
    print(f"exporting {os.path.basename(path)} ...")
    model.eval()
    torch.onnx.export(
        model,
        args,
        path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=dynamo,
    )
    consolidate_external_data(path, force=external_data)
    generate_prototxt(path)


def consolidate_external_data(path, force=False):
    """Merge external weights into a single <name>.onnx.data file.

    A model over the 2GB protobuf limit cannot store its weights inline. The
    TorchScript exporter then writes one file per tensor, which is unwieldy to
    upload, so everything is rewritten into a single data file next to the
    model. This is a no-op for models that fit inline, unless force is set.
    """
    location = os.path.basename(path) + ".data"
    model = onnx.load(path, load_external_data=False)
    externals = {
        entry.value
        for tensor in model.graph.initializer
        if tensor.data_location == onnx.TensorProto.EXTERNAL
        for entry in tensor.external_data
        if entry.key == "location"
    }
    if not externals and not force:
        return

    print(f"  merging {len(externals)} weight files into {location} ...")
    model = onnx.load(path)
    # the weights are in memory now, and onnx appends to an existing data file
    for name in externals:
        os.remove(os.path.join(os.path.dirname(path), name))
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=location,
        # small tensors stay inline: onnx shape inference cannot read external
        # data, and some ops (Slice) need their operand values to infer shapes
        size_threshold=1024,
        convert_attribute=False,
    )
    # onnx writes the data file with the process umask, make it world readable
    # like the model itself so it can be uploaded as is
    os.chmod(os.path.join(os.path.dirname(path), location), 0o644)


def generate_prototxt(onnx_path):
    """Generate the ailia prototxt from an ONNX model using onnx2prototxt.py."""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx2prototxt.py")
    if not os.path.exists(script_path):
        print("  downloading onnx2prototxt.py ...")
        urllib.request.urlretrieve(ONNX2PROTOTXT_URL, script_path)
    print(f"  generating {os.path.basename(onnx_path)}.prototxt ...")
    subprocess.check_call([sys.executable, script_path, onnx_path])


def kv_cache_names(num_layers):
    past = [f"past_pkv_{i}" for i in range(num_layers * 2)]
    present = [f"present_pkv_{i}" for i in range(num_layers * 2)]
    return past, present


def load_tts_model(model_dir):
    print(f"loading {model_dir} ...")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    return model


def load_speech_tokenizer(model_dir):
    speech_tokenizer_dir = os.path.join(model_dir, "speech_tokenizer")
    print(f"loading {speech_tokenizer_dir} ...")
    model = Qwen3TTSTokenizerV2Model.from_pretrained(
        speech_tokenizer_dir, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    return model


# ======================================================================
# per module export
# ======================================================================
def export_npy(model_dir, out, parameter_num):
    model = load_tts_model(model_dir)
    talker = model.talker
    code_predictor = talker.code_predictor

    def save(name, array):
        path = out(name, "npy")
        print(f"saving {os.path.basename(path)} {array.shape} ...")
        np.save(path, array)

    with torch.no_grad():
        # The talker text embedding is only used as a lookup table on the
        # Python side, so it is stored as a plain npy instead of an ONNX graph.
        save("text_embedding", talker.model.text_embedding.weight.numpy())
        # [1, 3072, H]. The leading axis is kept so that the runtime can index
        # the group 0 table as codec_embeddings[0].
        save("codec_embeddings", talker.model.codec_embedding.weight.numpy()[None])
        save(
            "subtalker_lm_heads",
            torch.stack([head.weight for head in code_predictor.lm_head]).numpy(),
        )
        save(
            "subtalker_codec_emb",
            torch.stack(
                [emb.weight for emb in code_predictor.model.codec_embedding]
            ).numpy(),
        )


def export_speaker_encoder(model_dir, out, parameter_num):
    model = load_tts_model(model_dir)
    wrapper = SpeakerEncoder(model.speaker_encoder)
    mel = torch.randn(1, 100, model.config.speaker_encoder_config.mel_dim)

    export(
        wrapper,
        (mel,),
        out("speaker_encoder", "onnx"),
        ["hidden_states"],
        ["embedding"],
        {
            "hidden_states": {0: "batch_size", 1: "num_frames"},
            "embedding": {0: "batch_size"},
        },
        external_data=PUBLISHED_EXTERNAL_DATA[parameter_num]["speaker_encoder"],
    )


def export_tokenizer_encoder(model_dir, out, parameter_num):
    patch_causal_mask()
    patch_codebook_quantize()
    model = load_speech_tokenizer(model_dir)
    wrapper = TokenizerEncoder(model)
    audio = torch.randn(1, 1, 24000 * 5)

    export(
        wrapper,
        (audio,),
        out("tokenizer_encoder", "onnx"),
        ["audio_values"],
        ["audio_codes"],
        {
            "audio_values": {0: "batch_size", 2: "num_samples"},
            "audio_codes": {0: "batch_size", 2: "num_frames"},
        },
    )


def export_tokenizer_decoder(model_dir, out, parameter_num):
    model = load_speech_tokenizer(model_dir)
    wrapper = TokenizerDecoder(model)
    num_quantizers = model.config.decoder_config.num_quantizers
    codes = torch.randint(0, 2048, (1, num_quantizers, 60))

    export(
        wrapper,
        (codes,),
        out("tokenizer_decoder", "onnx"),
        ["codes"],
        ["waveform"],
        {
            "codes": {0: "batch", 2: "num_tokens"},
            "waveform": {0: "batch", 2: "num_samples"},
        },
        # the causal convolutions compute their padding with math.ceil, which
        # the TorchScript exporter would bake in as a constant
        dynamo=True,
    )


def export_talker_io_units(model_dir, out, parameter_num):
    model = load_tts_model(model_dir)
    talker_config = model.config.talker_config
    wrapper = TalkerIOUnits(model.talker)
    text_features = torch.randn(1, 8, talker_config.text_hidden_size)
    last_hidden_states = torch.randn(1, 1, talker_config.hidden_size)

    export(
        wrapper,
        (text_features, last_hidden_states),
        out("talker_io_units", "onnx"),
        ["text_features", "last_hidden_states"],
        ["projected_text", "logits"],
        {
            "text_features": {0: "batch", 1: "seq"},
            "last_hidden_states": {0: "batch", 1: "hidden_seq"},
            "projected_text": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "hidden_seq"},
        },
        external_data=PUBLISHED_EXTERNAL_DATA[parameter_num]["talker_io_units"],
    )


def export_transformer(wrapper, config, hidden_size, path):
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    seq_len = 4
    past_len = 2
    inputs_embeds = torch.randn(1, seq_len, hidden_size)
    attention_mask = causal_mask(seq_len, past_len)
    position_ids = torch.arange(past_len, past_len + seq_len)[None]
    past = [
        torch.randn(1, num_kv_heads, past_len, head_dim) for _ in range(num_layers * 2)
    ]

    past_names, present_names = kv_cache_names(num_layers)
    dynamic_axes = {
        "inputs_embeds": {1: "seq_len"},
        "attention_mask": {2: "seq_len", 3: "total_seq_len"},
        "position_ids": {1: "seq_len"},
        "last_hidden_state": {1: "seq_len"},
    }
    for name in past_names:
        dynamic_axes[name] = {2: "past_seq_len"}
    for name in present_names:
        dynamic_axes[name] = {2: "total_seq_len"}

    export(
        wrapper,
        (inputs_embeds, attention_mask, position_ids, *past),
        path,
        ["inputs_embeds", "attention_mask", "position_ids"] + past_names,
        ["last_hidden_state"] + present_names,
        dynamic_axes,
    )


def export_talker_decoder(model_dir, out, parameter_num):
    model = load_tts_model(model_dir)
    talker_config = model.config.talker_config
    wrapper = TalkerDecoder(model.talker.model)

    # free everything the talker transformer does not need before the export,
    # the ONNX serialization needs a second copy of the weights in memory
    del model
    gc.collect()

    export_transformer(
        wrapper, talker_config, talker_config.hidden_size, out("talker_decoder", "onnx")
    )


def export_subtalker_decoder(model_dir, out, parameter_num):
    model = load_tts_model(model_dir)
    talker_config = model.config.talker_config
    code_predictor_config = talker_config.code_predictor_config
    wrapper = SubTalkerDecoder(model.talker.code_predictor)

    del model
    gc.collect()

    # inputs_embeds is in the talker hidden size (small_to_mtp_projection is
    # part of the graph)
    export_transformer(
        wrapper,
        code_predictor_config,
        talker_config.hidden_size,
        out("subtalker_decoder", "onnx"),
    )


EXPORTERS = {
    "npy": export_npy,
    "speaker_encoder": export_speaker_encoder,
    "tokenizer_encoder": export_tokenizer_encoder,
    "tokenizer_decoder": export_tokenizer_decoder,
    "talker_io_units": export_talker_io_units,
    "subtalker_decoder": export_subtalker_decoder,
    "talker_decoder": export_talker_decoder,
}


# ======================================================================
# main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-TTS to ONNX")
    parser.add_argument(
        "-p", "--parameter_num", default="0.6B", choices=sorted(MODEL_ID.keys()),
        help="model size to export"
    )
    parser.add_argument(
        "--model_dir", default=None,
        help="local directory of the Qwen3-TTS snapshot (downloaded from the Hub if omitted)"
    )
    parser.add_argument("--output_dir", default=".", help="directory to write the ONNX files to")
    parser.add_argument(
        "--only", default=None, choices=TARGETS,
        help="export a single module in this process instead of spawning one process per module"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_dir = args.model_dir
    if model_dir is None:
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(MODEL_ID[args.parameter_num])

    if args.only is None:
        # Each module is exported in a fresh process: the talker needs ~2x its
        # weights in RAM while ONNX serializes it, and the modules exported
        # before it must not still be resident.
        for target in TARGETS:
            subprocess.check_call(
                [
                    sys.executable, os.path.abspath(__file__),
                    "--parameter_num", args.parameter_num,
                    "--model_dir", model_dir,
                    "--output_dir", args.output_dir,
                    "--only", target,
                ]
            )
        print("done")
        return

    def out(name, ext):
        return os.path.join(
            args.output_dir, f"qwen3_tts_{name}_{args.parameter_num}.{ext}"
        )

    with torch.no_grad():
        EXPORTERS[args.only](model_dir, out, args.parameter_num)


if __name__ == "__main__":
    main()
