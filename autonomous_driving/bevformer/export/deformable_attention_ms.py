"""
Multi-Scale Deformable Attention using com.microsoft::MultiScaleDeformableAttention
ONNX op.

When exported to ONNX, this emits the MS extension custom op instead of the
F.grid_sample-based implementation. At runtime in PyTorch, it produces the
same results as the standard implementation.

Reference:
    Deformable DETR: Deformable Transformers for End-to-End Object Detection
    https://arxiv.org/abs/2010.04159
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDeformableAttentionOp(torch.autograd.Function):
    """Custom autograd Function that emits com.microsoft::MultiScaleDeformableAttention
    during ONNX export while computing correct results in PyTorch."""

    @staticmethod
    def forward(ctx, value, spatial_shapes, level_start_index,
                sampling_locations, attention_weights):
        """
        Args:
            value: (N, sum(Hi*Wi), M, D)
            spatial_shapes: (L, 2) int64 tensor
            level_start_index: (L,) int64 tensor
            sampling_locations: (N, Lq, M, L, P, 2) in [0, 1]
            attention_weights: (N, Lq, M, L, P)

        Returns:
            output: (N, Lq, M*D)
        """
        N, _, M, D = value.shape
        _, Lq, _, L, P, _ = sampling_locations.shape

        shapes_list = [
            (int(spatial_shapes[i, 0]), int(spatial_shapes[i, 1]))
            for i in range(L)
        ]
        split_sizes = [H * W for H, W in shapes_list]
        value_list = value.split(split_sizes, dim=1)

        sampling_grids = 2 * sampling_locations - 1

        sampling_value_list = []
        for lid in range(L):
            H, W = shapes_list[lid]
            value_l = (
                value_list[lid]
                .permute(0, 2, 3, 1)
                .reshape(N * M, D, H, W)
            )
            sampling_grid_l = (
                sampling_grids[:, :, :, lid]
                .permute(0, 2, 1, 3, 4)
                .reshape(N * M, Lq, P, 2)
            )
            sampling_value_l = F.grid_sample(
                value_l, sampling_grid_l,
                mode='bilinear', padding_mode='zeros', align_corners=False
            )
            sampling_value_list.append(sampling_value_l)

        attention_weights_flat = (
            attention_weights
            .permute(0, 2, 1, 3, 4)
            .reshape(N * M, 1, Lq, L * P)
        )
        sampling_values = (
            torch.stack(sampling_value_list, dim=-1)
            .reshape(N * M, D, Lq, L * P)
        )
        output = (sampling_values * attention_weights_flat).sum(-1)
        output = output.reshape(N, M * D, Lq).permute(0, 2, 1)

        return output.contiguous()

    @staticmethod
    def symbolic(g, value, spatial_shapes, level_start_index,
                 sampling_locations, attention_weights):
        return g.op(
            "com.microsoft::MultiScaleDeformableAttention",
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights)


def multi_scale_deformable_attn_ms(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """Multi-scale deformable attention using MS ONNX op.

    Same interface as multi_scale_deformable_attn_pytorch but emits
    com.microsoft::MultiScaleDeformableAttention during ONNX export.

    Args:
        value: (N, sum(Hi*Wi), M, D)
        value_spatial_shapes: list of (H, W) tuples
        sampling_locations: (N, Lq, M, L, P, 2) in [0, 1]
        attention_weights: (N, Lq, M, L, P)

    Returns:
        output: (N, Lq, M*D)
    """
    device = value.device
    L = len(value_spatial_shapes)

    spatial_shapes = torch.tensor(
        value_spatial_shapes, dtype=torch.int64, device=device)
    level_start_index = torch.zeros(L, dtype=torch.int64, device=device)
    for i in range(1, L):
        level_start_index[i] = (
            level_start_index[i - 1]
            + spatial_shapes[i - 1, 0] * spatial_shapes[i - 1, 1]
        )

    return MultiScaleDeformableAttentionOp.apply(
        value, spatial_shapes, level_start_index,
        sampling_locations, attention_weights)


class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention using MS ONNX extension op.

    Same interface as the standard MSDeformAttn but uses
    com.microsoft::MultiScaleDeformableAttention for ONNX export.

    Args:
        d_model: hidden dimension
        n_levels: number of feature levels
        n_heads: number of attention heads
        n_points: number of sampling points per head per level
    """

    def __init__(self, d_model=256, n_levels=1, n_heads=8, n_points=4):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        self.sampling_offsets = nn.Linear(
            d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(
            d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)

        thetas = torch.arange(
            self.n_heads, dtype=torch.float32
        ) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        ).view(self.n_heads, 1, 1, 2).repeat(
            1, self.n_levels, self.n_points, 1
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(self, query, reference_points, input_flatten,
                input_spatial_shapes, input_padding_mask=None):
        """
        Args:
            query: (N, Lq, C)
            reference_points: (N, Lq, n_levels, 2) in [0, 1]
            input_flatten: (N, sum(Hi*Wi), C)
            input_spatial_shapes: list of (H, W) tuples
            input_padding_mask: (N, sum(Hi*Wi)), optional

        Returns:
            output: (N, Lq, C)
        """
        N, Lq, _ = query.shape

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(N, -1, self.n_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).view(
            N, Lq, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            N, Lq, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Lq, self.n_heads, self.n_levels, self.n_points
        )

        spatial_shapes_t = torch.tensor(
            input_spatial_shapes, dtype=torch.float32,
            device=query.device)
        offset_normalizer = spatial_shapes_t.flip(-1)[
            None, None, None, :, None, :]
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer
        )

        output = multi_scale_deformable_attn_ms(
            value, input_spatial_shapes,
            sampling_locations, attention_weights
        )
        output = self.output_proj(output)

        return output
