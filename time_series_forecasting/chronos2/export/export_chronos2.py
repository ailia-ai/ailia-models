"""Export Chronos-2 (amazon/chronos-2) to ONNX.

Wraps `chronos.chronos2.model.Chronos2Model` so its forward returns the
quantile predictions tensor as a single output. The wrapper applies the
ONNX-friendly fixes that the upstream PyTorch operators require:

- ``F.scaled_dot_product_attention`` is replaced with a manual
  matmul/softmax implementation (the torchscript ONNX exporter has a
  Python-float bug on the ``scale`` argument).
- ``torch.nanmean`` is replaced with a NaN-safe mean that uses standard
  ONNX-supported ops (mask + sum / count). The legacy opset-17 exporter
  does not support ``aten::nanmean``.

The export pins ``num_output_patches`` at trace time. With Chronos-2's
``output_patch_size = 16``, the prediction horizon equals
``num_output_patches * 16``. Re-export with a different
``--num_output_patches`` if you need a longer horizon (max 64 → 1024
steps).

License: amazon/chronos-2 weights are Apache-2.0.
"""

import argparse
import math
import os

import torch
import torch.nn.functional as F
from torch import nn


def _manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, scale=None, is_causal=False
):
    scale_factor = (1.0 / math.sqrt(query.size(-1))) if scale is None else scale
    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = torch.zeros_like(attn_weight).masked_fill(
                attn_mask.logical_not(), float("-inf")
            )
        else:
            attn_bias = attn_mask
        attn_weight = attn_weight + attn_bias
    return torch.matmul(torch.softmax(attn_weight, dim=-1), value)


def _nan_safe_mean(x, dim=None, keepdim=False):
    """ONNX-friendly drop-in replacement for torch.nanmean."""
    mask = ~torch.isnan(x)
    x0 = torch.where(mask, x, torch.zeros_like(x))
    if dim is None:
        return x0.sum() / mask.to(x.dtype).sum().clamp_min(1.0)
    n = mask.to(x.dtype).sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return x0.sum(dim=dim, keepdim=keepdim) / n


class Chronos2ExportWrapper(nn.Module):
    """Wraps Chronos2Model to expose only the quantile_preds output and a
    fixed ``num_output_patches`` value."""

    def __init__(self, model, num_output_patches: int):
        super().__init__()
        self.model = model
        self.num_output_patches = num_output_patches

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        group_ids: torch.Tensor,
        future_covariates: torch.Tensor,
        future_covariates_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.model(
            context=context,
            context_mask=context_mask,
            group_ids=group_ids,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            num_output_patches=self.num_output_patches,
        )
        return out.quantile_preds


def export(
    output_dir: str,
    opset: int,
    num_output_patches: int,
    trace_batch: int,
    trace_ctx_len: int,
):
    # Patch torch ops before loading the pipeline so the model's submodules
    # bind to the ONNX-friendly versions. Chronos calls both ``torch.nanmean``
    # (free function) and ``tensor.nanmean()`` (method on Tensor), so we need
    # to monkey-patch ``InstanceNorm`` directly to avoid the unsupported op.
    F.scaled_dot_product_attention = _manual_scaled_dot_product_attention

    from chronos.chronos_bolt import InstanceNorm, Patch

    def _patch_forward(self, x):
        # ``tensor.unfold`` exports as ``aten::Unfold`` which the ONNX
        # exporter rejects when the size dimension is dynamic. With
        # ``patch_size == patch_stride`` (which is true for Chronos-2),
        # unfold is equivalent to a reshape, which exports cleanly.
        assert self.patch_size == self.patch_stride, (
            "Export wrapper only supports patch_size == patch_stride"
        )
        length = x.shape[-1]
        if length % self.patch_size != 0:
            pad = (*x.shape[:-1], self.patch_size - (length % self.patch_size))
            padding = torch.full(pad, float("nan"), dtype=x.dtype, device=x.device)
            x = torch.cat((padding, x), dim=-1)
        new_shape = list(x.shape[:-1]) + [-1, self.patch_size]
        return x.reshape(*new_shape)

    Patch.forward = _patch_forward

    # Chronos2Model builds a [REG]-token id tensor with ``torch.full`` (which
    # the ONNX exporter traces as ConstantOfShape with float dtype, breaking
    # the downstream Embedding/Gather op). Force the embedding lookup to
    # cast its indices to long so the export stays valid.
    _orig_embed_forward = nn.Embedding.forward

    def _embed_forward(self, ids):
        if not torch.is_floating_point(ids):
            return _orig_embed_forward(self, ids)
        return _orig_embed_forward(self, ids.to(torch.long))

    nn.Embedding.forward = _embed_forward

    def _arcsinh_compat(z):
        # ``aten::asinh`` is not supported by the torchscript ONNX exporter,
        # but the closed-form identity is.
        return torch.log(z + torch.sqrt(z * z + 1.0))

    def _sinh_compat(z):
        # Same story for ``aten::sinh``.
        return 0.5 * (torch.exp(z) - torch.exp(-z))

    def _instancenorm_forward(self, x, loc_scale=None):
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if loc_scale is None:
            loc = torch.nan_to_num(_nan_safe_mean(x, dim=-1, keepdim=True), nan=0.0)
            sq = (x - loc).square()
            scale = torch.nan_to_num(
                _nan_safe_mean(sq, dim=-1, keepdim=True).sqrt(), nan=1.0
            )
            scale = torch.where(scale == 0, self.eps, scale)
        else:
            loc, scale = loc_scale
        scaled_x = (x - loc) / scale
        if self.use_arcsinh:
            scaled_x = _arcsinh_compat(scaled_x)
        return scaled_x.to(orig_dtype), (loc, scale)

    def _instancenorm_inverse(self, x, loc_scale):
        loc, scale = loc_scale
        if self.use_arcsinh:
            x = _sinh_compat(x)
        return x * scale + loc

    InstanceNorm.forward = _instancenorm_forward
    InstanceNorm.inverse = _instancenorm_inverse

    from chronos import Chronos2Pipeline

    print("Loading amazon/chronos-2 ...", flush=True)
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    model = pipeline.model.eval()
    config = pipeline.model.chronos_config
    output_patch_size = config.output_patch_size  # 16

    wrapper = Chronos2ExportWrapper(model, num_output_patches).eval()

    pred_len = num_output_patches * output_patch_size
    B, L = trace_batch, trace_ctx_len
    torch.manual_seed(0)
    context = torch.randn(B, L, dtype=torch.float32)
    context_mask = torch.ones(B, L, dtype=torch.bool)
    group_ids = torch.zeros(B, dtype=torch.long)
    future_covariates = torch.zeros(B, pred_len, dtype=torch.float32)
    future_covariates_mask = torch.zeros(B, pred_len, dtype=torch.bool)
    if B >= 2:
        future_covariates_mask[1] = True  # second series treated as a known
                                          # future covariate during the trace.

    with torch.no_grad():
        wrapper(
            context, context_mask, group_ids, future_covariates, future_covariates_mask
        )

    onnx_name = f"chronos-2-p{num_output_patches}.onnx"
    onnx_path = os.path.join(output_dir, onnx_name)
    os.makedirs(output_dir, exist_ok=True)
    print(
        f"Exporting to {onnx_path} (opset={opset}, num_output_patches="
        f"{num_output_patches} → prediction_length={pred_len}) ...",
        flush=True,
    )

    input_names = [
        "context",
        "context_mask",
        "group_ids",
        "future_covariates",
        "future_covariates_mask",
    ]
    output_names = ["quantile_preds"]

    dynamic_axes = {
        "context": {0: "batch", 1: "ctx_len"},
        "context_mask": {0: "batch", 1: "ctx_len"},
        "group_ids": {0: "batch"},
        "future_covariates": {0: "batch"},
        "future_covariates_mask": {0: "batch"},
        "quantile_preds": {0: "batch"},
    }

    torch.onnx.export(
        wrapper,
        (
            context,
            context_mask,
            group_ids,
            future_covariates,
            future_covariates_mask,
        ),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )

    # Post-process: the torchscript exporter emits ``ConstantOfShape`` with
    # FLOAT dtype for the reg-token id, even though Chronos2Model uses an
    # int reg_token_id with the default integer ``torch.full``. The
    # downstream ``Gather`` (Embedding lookup) then fails ONNX type checks.
    # We rewrite every offending ConstantOfShape -> Gather-indices chain to
    # use INT64.
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(onnx_path)
    cos_outputs_to_fix = set()
    for node in model.graph.node:
        if node.op_type == "Gather":
            # Find the producer of indices (input[1]). If it's a
            # ConstantOfShape with FLOAT dtype, switch it to INT64.
            ind_name = node.input[1]
            for src in model.graph.node:
                if ind_name in src.output and src.op_type == "ConstantOfShape":
                    for attr in src.attribute:
                        if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT:
                            arr = numpy_helper.to_array(attr.t)
                            new_t = numpy_helper.from_array(
                                arr.astype("int64"), name=attr.t.name
                            )
                            attr.t.CopyFrom(new_t)
                            cos_outputs_to_fix.add(src.output[0])
    if cos_outputs_to_fix:
        onnx.save(model, onnx_path)
        print(
            f"Patched {len(cos_outputs_to_fix)} ConstantOfShape outputs to INT64 "
            f"so Gather/Embedding stays ONNX-valid.",
            flush=True,
        )

    print(f"Saved: {onnx_path}", flush=True)
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="Export Chronos-2 to ONNX")
    parser.add_argument(
        "--num_output_patches",
        type=int,
        default=4,
        help="Fixed number of output patches per call. With patch_size=16 "
        "this becomes the forecast horizon (e.g. 4 → 64 steps). Max 64.",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--trace_batch", type=int, default=2)
    parser.add_argument("--trace_ctx_len", type=int, default=192)
    args = parser.parse_args()

    export(
        output_dir=args.output_dir,
        opset=args.opset,
        num_output_patches=args.num_output_patches,
        trace_batch=args.trace_batch,
        trace_ctx_len=args.trace_ctx_len,
    )


if __name__ == "__main__":
    main()
