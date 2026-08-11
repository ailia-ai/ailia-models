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
    python3 export_onnx.py --parameter_num 1.7B --only talker

    # the fixed KV cache variants of the talker and the code predictor
    python3 export_onnx.py --parameter_num 0.6B --static --max_seq_len 512

The model is split so that the auto regressive loop can be driven from Python
(see ../qwen3-tts.py). Every weight is in a graph and there are no npy files, so
the runtime only reshapes arrays and samples tokens. ``<p>`` is the parameter_num
(0.6B or 1.7B) and ``H`` the talker hidden size (1024 for 0.6B, 2048 for 1.7B).

    qwen3_tts_encoder_<p>.onnx
        (waveform [1, 1, L], mel [1, frames, 128])
            -> (audio codes [1, 32, T], speaker embedding [1, H])
    qwen3_tts_decoder_<p>.onnx
        audio codes [B, 16, T]                  -> waveform [1, 1, L]
    qwen3_tts_prompt_<p>.onnx
        text token ids [1, n]                   -> projected text [1, n, H]
    qwen3_tts_codec_embedding_<p>.onnx
        codec table rows [n, 16]                -> their sums [1, n, H]
    qwen3_tts_talker_<p>.onnx
        talker transformer (28 layers) with a KV cache passed in/out and its
        output head included, so a step returns logits and the hidden state
    qwen3_tts_code_predictor_<p>.onnx
        code predictor (5 layers) with a KV cache passed in/out and its 15 output
        heads included, so a step takes the rows of the head to use and returns
        logits

--static rebuilds the two modules that carry a KV cache as
qwen3_tts_talker_<p>_static.onnx and qwen3_tts_code_predictor_<p>_static.onnx,
where the cache is a fixed length buffer written at cache_position instead of one
that grows by a step each call. Nothing else about them changes; see StaticTalker
for why, and cache_write() for why the write is a matmul rather than the one
ScatterElements node index_copy would give.

The codec embedding tables are a model of their own rather than part of the two
decode loop graphs, where they belong and briefly were: ailia stops following a
gather's index a few calls into a decode loop. ailia_gather_repro.py is a 90KB
reproduction, and export/README.md has the details.

The speech tokenizer weights are identical for 0.6B and 1.7B, so the two
qwen3_tts_decoder_*.onnx have the same content. They are exported per size to keep
the runtime file names uniform.
"""

import argparse
import gc
import importlib.machinery
import os
import subprocess
import sys
import types

import torch
from torch import nn

from onnx_utils import consolidate_external_data, generate_prototxt

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
    "codec_embedding",
    "encoder",
    "decoder",
    "prompt",
    "code_predictor",
    # exported last: it is by far the largest module
    "talker",
]

# --static only rebuilds the two modules that carry a KV cache
STATIC_TARGETS = ["code_predictor", "talker"]

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


def cache_write(past, new, cache_position):
    """Write new into the fixed length buffer past at cache_position.

    index_copy would say this in one ScatterElements node, and ailia gets that
    right for a single position but not for more than one, which the prompt needs
    (ailia_scatter_repro.py is a 2KB reproduction). A one hot of the positions
    spreads new over the buffer and clears the slots being overwritten, which is
    a matmul and two elementwise ops instead, and ailia agrees with onnxruntime
    on it for every length.
    """
    positions = torch.arange(past.shape[2], device=new.device)
    onehot = (positions[None, :] == cache_position[:, None]).to(new.dtype)
    spread = torch.einsum("bhsd,sm->bhmd", new, onehot)
    keep = (1.0 - onehot.sum(0))[None, None, :, None]
    return past * keep + spread


def decoder_layer_forward(layer, hidden_states, attention_mask, cos, sin, past_key,
                          past_value, cache_position=None):
    """Qwen3TTS{,Talker}DecoderLayer.forward with an explicit KV cache.

    Without cache_position the cache grows: the new key and value are appended
    and the returned cache is one step longer than the one passed in. With it
    the cache is a fixed length buffer and the new key and value are written at
    cache_position, so the returned cache has the shape of the one passed in and
    no shape in the graph depends on how many steps have run.
    """
    attn = layer.self_attn
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)

    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)

    query = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    if cache_position is None:
        key = torch.cat([past_key, key], dim=2)
        value = torch.cat([past_value, value], dim=2)
    else:
        key = cache_write(past_key, key, cache_position)
        value = cache_write(past_value, value, cache_position)

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
class Encoder(nn.Module):
    """Speech tokenizer encoder and ECAPA-TDNN speaker encoder in one graph.

    The reference audio always goes through both, so merging them costs nothing
    and saves a file. The mel spectrogram is computed on the Python side and
    passed in; only the speaker encoder needs it.
    """

    def __init__(self, tokenizer_model, speaker_encoder):
        super().__init__()
        self.encoder = tokenizer_model.encoder
        self.speaker_encoder = speaker_encoder

    def forward(self, audio_values, mel):
        audio_codes = self.encoder.encode(
            input_values=audio_values, return_dict=True
        ).audio_codes
        return audio_codes, self.speaker_encoder(mel)


class Decoder(nn.Module):
    """Qwen3-TTS-Tokenizer-12Hz decoder: audio codes -> waveform."""

    def __init__(self, tokenizer_model):
        super().__init__()
        self.decoder = tokenizer_model.decoder

    def forward(self, codes):
        return self.decoder(codes)


class CodecEmbedding(nn.Module):
    """The 16 codec embedding tables, and nothing else.

    Group 0 of a codec frame is embedded with the talker's table (vocabulary 3072)
    and groups 1..15 with the code predictor's tables (vocabulary 2048). All 16 sit
    in one table so a lookup is a single gather:

        row t                       group 0, token t
        row 3072 + (g-1)*2048 + t   group g, token t
        row 3072 + 15*2048          all zeros, for a group with nothing to add

    A call takes 16 row indices per position and returns their sum, which is what
    both callers want: the talker's decode input is a whole frame's 16 groups
    summed, and a code predictor step is one group with the other 15 pointing at
    the zero row.

    This is a model of its own rather than part of the two decode loop graphs
    because ailia stops following a gather's index a few calls into a decode loop;
    see ailia_gather_repro.py. A graph that only gathers stays correct.
    """

    def __init__(self, talker, code_predictor):
        super().__init__()
        talker_codec = talker.model.codec_embedding.weight.detach()
        group_codec = torch.stack(
            [emb.weight.detach() for emb in code_predictor.model.codec_embedding]
        ).flatten(0, 1)
        self.register_buffer(
            "table",
            torch.cat(
                [talker_codec, group_codec, talker_codec.new_zeros(1, talker_codec.shape[1])]
            ),
            persistent=False,
        )

    def forward(self, codec_rows):
        """codec_rows [n, 16] -> [1, n, H], each position's 16 rows summed"""
        return self.table[codec_rows].sum(dim=1)[None]


class Talker(nn.Module):
    """Talker transformer, with its output head inside.

    The prompt and the decode steps both arrive as inputs_embeds, so one graph
    serves both. Only the last position is read out, which is all the decode loop
    needs: logits to sample the next group 0 token, and the hidden state to seed
    the code predictor.
    """

    def __init__(self, talker):
        super().__init__()
        self.layers = talker.model.layers
        self.norm = talker.model.norm
        self.register_buffer(
            "inv_freq", talker.model.rotary_emb.inv_freq, persistent=False
        )
        self.codec_head = talker.codec_head

    def run_layers(self, hidden_states, attention_mask, position_ids, past_key_values,
                   cache_position=None):
        cos, sin = rope_cos_sin(self.inv_freq, position_ids, hidden_states.dtype)
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
                cache_position,
            )
            present += [key, value]
        return hidden_states, present

    def head(self, hidden_states):
        last_hidden = self.norm(hidden_states)[:, -1:, :]
        return self.codec_head(last_hidden), last_hidden

    def forward(self, inputs_embeds, attention_mask, position_ids, *past_key_values):
        hidden_states, present = self.run_layers(
            inputs_embeds, attention_mask, position_ids, past_key_values
        )
        return (*self.head(hidden_states), *present)


class StaticTalker(Talker):
    """Talker with a fixed length KV cache instead of a growing one.

    The graph is the same except that the cache is a [1, kv_heads, max_seq_len,
    head_dim] buffer and cache_position says where the step writes into it, so
    the cache shape no longer changes from one step to the next. attention_mask
    covers the whole buffer and masks the slots that have not been written yet.

    The point is the runtime, not the graph: ailia re-infers the shape of the
    whole network on every set_input_blob_shape, and with a growing cache the
    decode loop has to set 2 * num_layers of them per step. Here nothing changes
    after the first decode step. The cost is that attention reads the whole
    buffer whatever the current length is, so max_seq_len should not be set much
    higher than the longest sequence that will be generated.
    """

    def forward(self, inputs_embeds, attention_mask, position_ids, cache_position,
                *past_key_values):
        hidden_states, present = self.run_layers(
            inputs_embeds, attention_mask, position_ids, past_key_values, cache_position
        )
        return (*self.head(hidden_states), *present)


class TalkerPrompt(nn.Module):
    """The text side of the talker's prompt: the text embedding and its projection.

    The prompt needs codec embeddings as well, but those come from
    qwen3_tts_codec_embedding_<p>.onnx, so one text token id sequence in and its
    projection out is all this is.
    """

    def __init__(self, talker):
        super().__init__()
        self.register_buffer(
            "text_embedding", talker.model.text_embedding.weight.detach(), persistent=False
        )
        self.text_projection = talker.text_projection

    def forward(self, text_tokens):
        return self.text_projection(self.text_embedding[text_tokens])


class CodePredictor(nn.Module):
    """Code predictor: codec token in, logits for the next code group out.

    The 15 output heads are part of the graph, so the runtime never runs a matmul
    of its own. That matters beyond tidiness: numpy's BLAS holds onto its threads
    after a matmul and starves ailia on the next call.

Every position arrives as inputs_embeds: position 0 is the talker hidden state
    and every later position is a codec embedding the runtime looked up.

    The 15 output heads are concatenated into one [15 * 2048, Hs] matrix and a
    step passes head_rows, the 2048 rows of the head it needs. Naming the rows
    from the runtime rather than deriving them from position_ids keeps the lookup
    to a gather whose indices come straight from an input.

    inputs_embeds is in the talker hidden size; small_to_mtp_projection brings it
    down to the code predictor hidden size (an identity for 0.6B).
    """

    def __init__(self, code_predictor):
        super().__init__()
        self.small_to_mtp_projection = code_predictor.small_to_mtp_projection
        self.layers = code_predictor.model.layers
        self.norm = code_predictor.model.norm
        self.register_buffer(
            "inv_freq", code_predictor.model.rotary_emb.inv_freq, persistent=False
        )

        self.register_buffer(
            "lm_heads",
            torch.stack(
                [head.weight.detach() for head in code_predictor.lm_head]
            ).flatten(0, 1),
            persistent=False,
        )

    def run_layers(self, hidden_states, attention_mask, position_ids, past_key_values,
                   cache_position=None):
        hidden_states = self.small_to_mtp_projection(hidden_states)
        cos, sin = rope_cos_sin(self.inv_freq, position_ids, hidden_states.dtype)

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
                cache_position,
            )
            present += [key, value]
        return hidden_states, present

    def head(self, hidden_states, head_rows):
        hidden_states = self.norm(hidden_states)
        return hidden_states[:, -1:, :] @ self.lm_heads[head_rows].t()

    def forward(self, inputs_embeds, head_rows, attention_mask, position_ids,
                *past_key_values):
        hidden_states, present = self.run_layers(
            inputs_embeds, attention_mask, position_ids, past_key_values
        )
        return (self.head(hidden_states, head_rows), *present)


class StaticCodePredictor(CodePredictor):
    """Code predictor with a fixed length KV cache, see StaticTalker.

    A frame is always 16 positions (the talker hidden state and the 15 code group
    embeddings), so max_seq_len is not a choice here: the buffer is exactly
    num_code_groups long and is overwritten frame by frame.
    """

    def forward(self, inputs_embeds, head_rows, attention_mask, position_ids,
                cache_position, *past_key_values):
        hidden_states, present = self.run_layers(
            inputs_embeds, attention_mask, position_ids, past_key_values, cache_position
        )
        return (self.head(hidden_states, head_rows), *present)


# ======================================================================
# helpers
# ======================================================================
def causal_mask(seq_len, past_len=0, total_len=None):
    """Additive attention mask [1, 1, seq_len, total_len].

    total_len defaults to the length of a cache that grew by seq_len; a fixed
    length cache passes its whole length and the slots past the current step are
    masked out along with the ones a position may not attend to.
    """
    if total_len is None:
        total_len = seq_len + past_len
    mask = torch.full((seq_len, total_len), float("-inf"))
    for i in range(seq_len):
        mask[i, : past_len + i + 1] = 0.0
    return mask[None, None, :, :]


def export(
    model, args, path, input_names, output_names, dynamic_axes, dynamo=False,
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
    consolidate_external_data(path)
    generate_prototxt(path)


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
def export_encoder(model_dir, out, parameter_num):
    """Speech tokenizer encoder and speaker encoder in one graph.

    Both only ever run on the reference audio, and always together, so nothing is
    computed needlessly by merging them. The mel spectrogram stays on the Python
    side and is passed in.
    """
    patch_causal_mask()
    patch_codebook_quantize()
    tokenizer = load_speech_tokenizer(model_dir)
    model = load_tts_model(model_dir)
    wrapper = Encoder(tokenizer, model.speaker_encoder)
    mel_dim = model.config.speaker_encoder_config.mel_dim

    del model, tokenizer
    gc.collect()

    audio = torch.randn(1, 1, 24000 * 5)
    mel = torch.randn(1, 100, mel_dim)

    export(
        wrapper,
        (audio, mel),
        out("encoder", "onnx"),
        ["audio_values", "mel"],
        ["audio_codes", "speaker_embedding"],
        {
            "audio_values": {0: "batch_size", 2: "num_samples"},
            "mel": {0: "batch_size", 1: "num_frames"},
            "audio_codes": {0: "batch_size", 2: "num_frames"},
            "speaker_embedding": {0: "batch_size"},
        },
    )


def export_decoder(model_dir, out, parameter_num):
    model = load_speech_tokenizer(model_dir)
    wrapper = Decoder(model)
    num_quantizers = model.config.decoder_config.num_quantizers
    codes = torch.randint(0, 2048, (1, num_quantizers, 60))

    export(
        wrapper,
        (codes,),
        out("decoder", "onnx"),
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


def export_prompt(model_dir, out, parameter_num):
    model = load_tts_model(model_dir)
    wrapper = TalkerPrompt(model.talker)

    del model
    gc.collect()

    text_tokens = torch.zeros(1, 8, dtype=torch.int64)

    export(
        wrapper,
        (text_tokens,),
        out("prompt", "onnx"),
        ["text_tokens"],
        ["projected_text"],
        {
            "text_tokens": {1: "text_len"},
            "projected_text": {1: "text_len"},
        },
        # the text embedding table alone is over 1GB, and the TorchScript
        # exporter cannot serialize a model this large
        dynamo=True,
    )


def export_codec_embedding(model_dir, out, parameter_num):
    model = load_tts_model(model_dir)
    num_code_groups = model.config.talker_config.num_code_groups
    wrapper = CodecEmbedding(model.talker, model.talker.code_predictor)

    del model
    gc.collect()

    codec_rows = torch.zeros(5, num_code_groups, dtype=torch.int64)

    export(
        wrapper,
        (codec_rows,),
        out("codec_embedding", "onnx"),
        ["codec_rows"],
        ["codec_embeds"],
        {"codec_rows": {0: "num_positions"}, "codec_embeds": {1: "num_positions"}},
    )


def export_talker(model_dir, out, parameter_num, static=False, max_seq_len=None):
    model = load_tts_model(model_dir)
    talker_config = model.config.talker_config
    wrapper = StaticTalker(model.talker) if static else Talker(model.talker)

    # free everything the talker does not need before the export, the ONNX
    # serialization needs a second copy of the weights in memory
    del model
    gc.collect()

    num_layers = talker_config.num_hidden_layers
    num_kv_heads = talker_config.num_key_value_heads
    head_dim = talker_config.head_dim
    hidden_size = talker_config.hidden_size

    seq_len = 4
    past_len = 2
    cache_len = max_seq_len if static else past_len
    inputs_embeds = torch.randn(1, seq_len, hidden_size)
    attention_mask = causal_mask(
        seq_len, past_len, max_seq_len if static else None
    )
    position_ids = torch.arange(past_len, past_len + seq_len)[None]
    cache_position = torch.arange(past_len, past_len + seq_len)
    past = [
        torch.randn(1, num_kv_heads, cache_len, head_dim) for _ in range(num_layers * 2)
    ]

    past_names, present_names = kv_cache_names(num_layers)
    dynamic_axes = {
        "inputs_embeds": {1: "seq_len"},
        "attention_mask": {2: "seq_len"},
        "position_ids": {1: "seq_len"},
    }
    if static:
        # only the sequence axis is left dynamic: the cache is a fixed length
        # buffer and the mask spans all of it
        dynamic_axes["cache_position"] = {0: "seq_len"}
    else:
        dynamic_axes["attention_mask"][3] = "total_seq_len"
        for name in past_names:
            dynamic_axes[name] = {2: "past_seq_len"}
        for name in present_names:
            dynamic_axes[name] = {2: "total_seq_len"}

    args = (inputs_embeds, attention_mask, position_ids)
    names = ["inputs_embeds", "attention_mask", "position_ids"]
    if static:
        args += (cache_position,)
        names += ["cache_position"]

    export(
        wrapper,
        args + tuple(past),
        out("talker", "onnx"),
        names + past_names,
        ["logits", "last_hidden"] + present_names,
        dynamic_axes,
    )


def export_code_predictor(model_dir, out, parameter_num, static=False, max_seq_len=None):
    model = load_tts_model(model_dir)
    talker_config = model.config.talker_config
    code_predictor_config = talker_config.code_predictor_config
    wrapper = (StaticCodePredictor if static else CodePredictor)(
        model.talker.code_predictor
    )

    del model
    gc.collect()

    num_layers = code_predictor_config.num_hidden_layers
    num_kv_heads = code_predictor_config.num_key_value_heads
    head_dim = code_predictor_config.head_dim

    seq_len = 2
    past_len = 0
    # a frame is always num_code_groups positions long, so that is the buffer
    cache_len = talker_config.num_code_groups if static else past_len
    # inputs_embeds is in the talker hidden size (small_to_mtp_projection is
    # part of the graph) and only position 0 of it is read
    inputs_embeds = torch.randn(1, seq_len, talker_config.hidden_size)
    head_rows = torch.arange(code_predictor_config.vocab_size)
    attention_mask = causal_mask(seq_len, past_len, cache_len if static else None)
    position_ids = torch.arange(past_len, past_len + seq_len)[None]
    cache_position = torch.arange(past_len, past_len + seq_len)
    past = [
        torch.randn(1, num_kv_heads, cache_len, head_dim) for _ in range(num_layers * 2)
    ]

    past_names, present_names = kv_cache_names(num_layers)
    dynamic_axes = {
        "inputs_embeds": {1: "seq_len"},
        "attention_mask": {2: "seq_len"},
        "position_ids": {1: "seq_len"},
    }
    if static:
        dynamic_axes["cache_position"] = {0: "seq_len"}
    else:
        dynamic_axes["attention_mask"][3] = "total_seq_len"
        for name in past_names:
            dynamic_axes[name] = {2: "past_seq_len"}
        for name in present_names:
            dynamic_axes[name] = {2: "total_seq_len"}

    args = (inputs_embeds, head_rows, attention_mask, position_ids)
    names = ["inputs_embeds", "head_rows", "attention_mask", "position_ids"]
    if static:
        args += (cache_position,)
        names += ["cache_position"]

    export(
        wrapper,
        args + tuple(past),
        out("code_predictor", "onnx"),
        names + past_names,
        ["logits"] + present_names,
        dynamic_axes,
    )


EXPORTERS = {
    "codec_embedding": export_codec_embedding,
    "encoder": export_encoder,
    "decoder": export_decoder,
    "prompt": export_prompt,
    "code_predictor": export_code_predictor,
    "talker": export_talker,
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
    parser.add_argument(
        "--static", action="store_true",
        help="export the fixed KV cache variants of the two decode loop modules, "
             "named qwen3_tts_<name>_<p>_static.onnx"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=512,
        help="length of the talker's fixed KV cache, --static only. Attention reads "
             "all of it every step, so it should not be much larger than the longest "
             "sequence to generate (prompt included)."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_dir = args.model_dir
    if model_dir is None:
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(MODEL_ID[args.parameter_num])

    # only the two decode loop modules have a KV cache, the rest are unchanged
    targets = STATIC_TARGETS if args.static else TARGETS

    if args.only is None:
        # Each module is exported in a fresh process: the talker needs ~2x its
        # weights in RAM while ONNX serializes it, and the modules exported
        # before it must not still be resident.
        for target in targets:
            subprocess.check_call(
                [
                    sys.executable, os.path.abspath(__file__),
                    "--parameter_num", args.parameter_num,
                    "--model_dir", model_dir,
                    "--output_dir", args.output_dir,
                    "--max_seq_len", str(args.max_seq_len),
                    "--only", target,
                ] + (["--static"] if args.static else [])
            )
        print("done")
        return

    if args.only not in targets:
        print(f"{args.only} has no --static variant")
        return

    suffix = "_static" if args.static else ""

    def out(name, ext):
        return os.path.join(
            args.output_dir, f"qwen3_tts_{name}_{args.parameter_num}{suffix}.{ext}"
        )

    kwargs = {"static": True, "max_seq_len": args.max_seq_len} if args.static else {}
    with torch.no_grad():
        EXPORTERS[args.only](model_dir, out, args.parameter_num, **kwargs)


if __name__ == "__main__":
    main()
