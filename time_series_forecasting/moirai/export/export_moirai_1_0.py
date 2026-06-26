"""Export Moirai (uni2ts) to ONNX.

Wraps `uni2ts.model.moirai.MoiraiModule` so that its forward returns the raw
mixture-distribution parameter tensors (along with the scaler's loc / scale).
The actual sampling and post-processing is performed in the inference script.

Supported sizes: small, base, large (Moirai 1.0-R, Apache-2.0 era weights).

The Moirai-1.0-R weights on Hugging Face were originally released under
Apache-2.0. Salesforce relicensed them to CC-BY-NC-4.0 on 2024-03-28; the
`Update license` commit, however, did not modify the weights themselves.
This exporter pins each model to the **last revision before the
relicense**, where the README still declared `license: apache-2.0`, and
downloads the legacy `model.ckpt` (PyTorch Lightning checkpoint) so that
the weights flowing into the exported ONNX file remain unambiguously
Apache-2.0.
"""

import argparse
import math
import os

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn

from uni2ts.model.moirai import MoiraiModule


def _manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, scale=None, is_causal=False
):
    """ONNX-export-friendly replacement for F.scaled_dot_product_attention.

    PyTorch's ONNX exporter (torchscript path, opset 17) has a bug when the
    `scale` argument is a Python float (TypeError on z_(...)). We avoid the
    builtin op entirely.
    """
    scale_factor = (1.0 / math.sqrt(query.size(-1))) if scale is None else scale
    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = torch.zeros_like(attn_weight)
            attn_bias = attn_bias.masked_fill(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask
        attn_weight = attn_weight + attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return torch.matmul(attn_weight, value)


# Last commit before the 2024-03-28 license change for each Moirai-1.0-R size.
# At these revisions the README still declares `license: apache-2.0`.
APACHE_REVISIONS = {
    "small": "4a950dea3b2c38b9675082959109e1b36d40ab16",
    "base":  "03e0d0f88ea7dee295d398d102fb582494b549e1",
    "large": "bc5caba1947b76c9efd513ada3675b8d5006f09a",
}

REPO_TEMPLATE = "Salesforce/moirai-1.0-R-{size}"


class MoiraiExportWrapper(nn.Module):
    """Wraps MoiraiModule to return distribution parameters as a flat tuple."""

    def __init__(self, module: MoiraiModule):
        super().__init__()
        self.module = module

    @staticmethod
    def _packed_attention_mask(sample_id: torch.Tensor) -> torch.Tensor:
        # Replacement for uni2ts.common.torch_util.packed_attention_mask which
        # uses Tensor.mT (matrix transpose); aten::mT is not supported by the
        # ONNX exporter.
        s = sample_id.unsqueeze(-1)
        return s.eq(s.transpose(-2, -1))

    def forward(
        self,
        target,
        observed_mask,
        sample_id,
        time_id,
        variate_id,
        prediction_mask,
        patch_size,
    ):
        m = self.module

        loc, scale = m.scaler(
            target,
            observed_mask * ~prediction_mask.unsqueeze(-1),
            sample_id,
            variate_id,
        )
        # The einops `reduce(..., "... seq1 seq2 -> ... seq1 1", "sum")` inside
        # PackedStdScaler emits an ONNX subgraph whose output dim_param matches
        # `seq_len` instead of being a literal 1, causing broadcast errors at
        # runtime. Slice to force the trailing axis to be statically 1.
        loc = loc[..., :1]
        scale = scale[..., :1]
        scaled_target = (target - loc) / scale
        reprs = m.in_proj(scaled_target, patch_size)
        from uni2ts.common.torch_util import mask_fill

        masked_reprs = mask_fill(reprs, prediction_mask, m.mask_encoding.weight)
        reprs = m.encoder(
            masked_reprs,
            self._packed_attention_mask(sample_id),
            time_id=time_id,
            var_id=variate_id,
        )
        distr_param = m.param_proj(reprs, patch_size)

        # Flatten the mixture parameter pytree into a fixed list of tensors.
        # Order matches uni2ts MixtureOutput components for Moirai-1.x-R:
        #   [StudentT, NormalFixedScale, NegativeBinomial, LogNormal]
        weights_logits = distr_param["weights_logits"]
        comps = distr_param["components"]
        student_t = comps[0]
        normal_fs = comps[1]
        neg_bin = comps[2]
        log_normal = comps[3]

        return (
            weights_logits,
            student_t["df"],
            student_t["loc"],
            student_t["scale"],
            normal_fs["loc"],
            neg_bin["total_count"],
            neg_bin["logits"],
            log_normal["loc"],
            log_normal["scale"],
            loc,
            scale,
        )


def _load_apache_module(size: str) -> MoiraiModule:
    """Download the Apache-2.0 era `model.ckpt` and rebuild the MoiraiModule.

    The legacy Lightning checkpoint stores both the module's hyperparameters
    (under ``hyper_parameters['module_kwargs']``) and the weights (under
    ``state_dict`` with a ``module.`` prefix), so we can construct a fresh
    ``MoiraiModule`` without touching the current Hugging Face revision.
    """
    revision = APACHE_REVISIONS[size]
    repo = REPO_TEMPLATE.format(size=size)
    print(f"Loading {repo} @ {revision[:10]} (Apache-2.0) ...", flush=True)
    ckpt_path = hf_hub_download(
        repo_id=repo, filename="model.ckpt", revision=revision
    )
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    module_kwargs = ck["hyper_parameters"]["module_kwargs"]
    state_dict = {
        k.removeprefix("module."): v
        for k, v in ck["state_dict"].items()
        if k.startswith("module.")
    }
    module = MoiraiModule(**module_kwargs)
    missing, unexpected = module.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected state_dict mismatch: missing={missing} "
            f"unexpected={unexpected}"
        )
    return module


def export(size: str, opset: int, output_dir: str, seq_len: int):
    # Patch SDPA before loading the module to avoid the ONNX exporter bug.
    F.scaled_dot_product_attention = _manual_scaled_dot_product_attention

    # Moirai's QueryKeyProjection.forward calls ``tensor.split(split_sizes,
    # dim=-1)`` where ``split_sizes`` has a leading 0 (because
    # ``partial_factor=(0.0, 0.5)`` makes the first slice empty). The ONNX
    # exporter dutifully emits a 3-output Split, which ailia rejects with
    # "Unexpected mixed empty and non-empty outputs". Patch the forward to
    # skip the size-0 slices, which is functionally identical.
    from uni2ts.module.position.attn_projection import QueryKeyProjection

    def _qk_forward(self, query, key, query_id, kv_id):
        if self.partial_factor is None:
            return (
                self.query_proj(query, seq_id=query_id),
                self.key_proj(key, seq_id=kv_id),
            )
        sizes = self.split_sizes  # (left, mid, right)
        head_dim = self.head_dim
        # Slice with explicit indices to avoid emitting size-0 splits.
        left = sizes[0]
        right_start = head_dim - sizes[2]

        q_left = query[..., :left]
        q_mid = query[..., left:right_start]
        q_right = query[..., right_start:]
        k_left = key[..., :left]
        k_mid = key[..., left:right_start]
        k_right = key[..., right_start:]

        q_mid = self.query_proj(q_mid, seq_id=query_id)
        k_mid = self.key_proj(k_mid, seq_id=kv_id)

        # Drop empty slices before concatenating.
        q_parts = [t for t, s in [(q_left, sizes[0]), (q_mid, sizes[1]), (q_right, sizes[2])] if s > 0]
        k_parts = [t for t, s in [(k_left, sizes[0]), (k_mid, sizes[1]), (k_right, sizes[2])] if s > 0]
        return torch.cat(q_parts, dim=-1), torch.cat(k_parts, dim=-1)

    QueryKeyProjection.forward = _qk_forward

    module = _load_apache_module(size)
    module.eval()

    wrapper = MoiraiExportWrapper(module).eval()

    max_patch = max(module.patch_sizes)
    patch_value = max_patch  # any valid patch size for tracing

    B, S, P = 1, seq_len, max_patch
    target = torch.zeros(B, S, P, dtype=torch.float32)
    observed_mask = torch.ones(B, S, P, dtype=torch.bool)
    sample_id = torch.zeros(B, S, dtype=torch.long)
    time_id = torch.arange(S, dtype=torch.long).unsqueeze(0).repeat(B, 1)
    variate_id = torch.zeros(B, S, dtype=torch.long)
    prediction_mask = torch.zeros(B, S, dtype=torch.bool)
    prediction_mask[:, -1:] = True
    patch_size = torch.full((B, S), patch_value, dtype=torch.long)

    with torch.no_grad():
        wrapper(
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
            patch_size,
        )

    onnx_name = f"moirai-1.0-R-{size}.onnx"
    onnx_path = os.path.join(output_dir, onnx_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Exporting to {onnx_path} (opset={opset}) ...", flush=True)

    input_names = [
        "target",
        "observed_mask",
        "sample_id",
        "time_id",
        "variate_id",
        "prediction_mask",
        "patch_size",
    ]
    output_names = [
        "weights_logits",
        "student_t_df",
        "student_t_loc",
        "student_t_scale",
        "normal_loc",
        "nb_total_count",
        "nb_logits",
        "lognormal_loc",
        "lognormal_scale",
        "loc",
        "scale",
    ]

    dynamic_axes = {n: {0: "batch", 1: "seq_len"} for n in input_names}
    for n in output_names:
        dynamic_axes[n] = {0: "batch", 1: "seq_len"}

    torch.onnx.export(
        wrapper,
        (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
            patch_size,
        ),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )

    print(f"Saved: {onnx_path}", flush=True)
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="Export Moirai-1.0-R to ONNX")
    parser.add_argument(
        "--size",
        type=str,
        default="small",
        choices=["small", "base", "large"],
        help="Model size",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Where to write the .onnx file",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=64,
        help="Trace sequence length (output is dynamic)",
    )
    args = parser.parse_args()

    export(
        size=args.size,
        opset=args.opset,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    main()
