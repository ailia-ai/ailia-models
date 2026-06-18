"""
BEVFormer-tiny model in pure PyTorch (no mmcv/mmdet3d dependency).

Produces state_dict keys matching the original BEVFormer-tiny checkpoint
(bevformer_tiny_epoch_24.pth) from mmdetection3d, enabling direct weight
loading. All operators are standard PyTorch ops for ONNX export.

Architecture:
  - img_backbone: ResNet-50 (mmcv naming: conv1, bn1, layer1-4)
  - img_neck: FPN with 1 output level (stride 32, 2048 -> 256)
  - pts_bbox_head.transformer.encoder: 3 BEV encoder layers
      - attentions.0: Temporal self-attention (MSDeformAttn, d_model=512)
      - attentions.1: Spatial cross-attention (MSDeformAttn wrapped)
      - ffns.0: Feed-forward network
  - pts_bbox_head.transformer.decoder: 6 DETR decoder layers
      - attentions.0: Self-attention (nn.MultiheadAttention)
      - attentions.1: Cross-attention (MSDeformAttn)
      - ffns.0: Feed-forward network
  - pts_bbox_head.cls_branches / reg_branches: Per-layer detection heads

Reference:
    BEVFormer: Learning Bird's-Eye-View Representation from
    Multi-Camera Images via Spatiotemporal Transformers (ECCV 2022)
    https://arxiv.org/abs/2203.17270
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from deformable_attention import MSDeformAttn


# ============================================================
# Backbone: ResNet-50 (mmcv-compatible naming)
# ============================================================

class ResNet50Backbone(nn.Module):
    """ResNet-50 backbone with mmcv-compatible parameter naming.

    Produces keys like:
        conv1.weight, bn1.weight, bn1.bias, bn1.running_mean, ...
        layer1.0.conv1.weight, layer1.0.bn1.weight, ...
        layer2.*, layer3.*, layer4.*

    Only layer4 output (stride 32, 2048ch) is used by BEVFormer-tiny.
    """

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=None)
        # Expose conv1, bn1, relu, maxpool as direct attributes so that
        # state_dict keys become img_backbone.conv1.weight, img_backbone.bn1.*
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # stride 4, 256ch
        self.layer2 = resnet.layer2  # stride 8, 512ch
        self.layer3 = resnet.layer3  # stride 16, 1024ch
        self.layer4 = resnet.layer4  # stride 32, 2048ch

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        c5 = self.layer4(x)
        return c5  # (B, 2048, H/32, W/32)


# ============================================================
# Neck: FPN with 1 level (mmcv ConvModule naming)
# ============================================================

class ConvModule(nn.Module):
    """Mimics mmcv ConvModule: wraps nn.Conv2d under a .conv attribute.

    This produces state_dict keys like:
        lateral_convs.0.conv.weight, lateral_convs.0.conv.bias
    instead of:
        lateral_convs.0.weight, lateral_convs.0.bias
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


class FPN(nn.Module):
    """Feature Pyramid Network with a single output level.

    BEVFormer-tiny uses only 1 FPN level: the stride-32 feature map
    from layer4 (2048ch) is projected to 256ch.

    State_dict keys:
        lateral_convs.0.conv.weight: [256, 2048, 1, 1]
        lateral_convs.0.conv.bias: [256]
        fpn_convs.0.conv.weight: [256, 256, 3, 3]
        fpn_convs.0.conv.bias: [256]
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            ConvModule(in_channels, out_channels, 1)
        ])
        self.fpn_convs = nn.ModuleList([
            ConvModule(out_channels, out_channels, 3, padding=1)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, 2048, H, W) feature map from layer4.

        Returns:
            List with single element: (B, 256, H, W)
        """
        lat = self.lateral_convs[0](x)
        out = self.fpn_convs[0](lat)
        return [out]


# ============================================================
# Learned Positional Encoding (row/col embed)
# ============================================================

class LearnedPositionalEncoding(nn.Module):
    """Learned 2D positional encoding for BEV grid.

    State_dict keys:
        row_embed.weight: [num_feats_y, embed_dim // 2]
        col_embed.weight: [num_feats_x, embed_dim // 2]
    """

    def __init__(self, num_feats=128, row_num=50, col_num=50):
        super().__init__()
        self.row_embed = nn.Embedding(row_num, num_feats)
        self.col_embed = nn.Embedding(col_num, num_feats)
        self.num_feats = num_feats
        self.row_num = row_num
        self.col_num = col_num

    def forward(self, bev_h, bev_w, device):
        """Generate positional encoding of shape (1, bev_h*bev_w, embed_dim).

        Args:
            bev_h: height of BEV grid
            bev_w: width of BEV grid
            device: torch device

        Returns:
            pos: (1, bev_h * bev_w, num_feats * 2)
        """
        row_idx = torch.arange(bev_h, device=device)
        col_idx = torch.arange(bev_w, device=device)
        row_emb = self.row_embed(row_idx)  # (bev_h, num_feats)
        col_emb = self.col_embed(col_idx)  # (bev_w, num_feats)
        # Combine: (bev_h, bev_w, 2*num_feats)
        pos = torch.cat([
            col_emb[None, :, :].expand(bev_h, -1, -1),
            row_emb[:, None, :].expand(-1, bev_w, -1),
        ], dim=-1)
        pos = pos.reshape(1, bev_h * bev_w, self.num_feats * 2)
        return pos


# ============================================================
# FFN (mmdet-style: ffns.0.layers.0.0.* / ffns.0.layers.1.*)
# ============================================================

class FFN(nn.Module):
    """Feed-forward network matching mmdet's FFN state_dict layout.

    State_dict keys (under ffns.0):
        layers.0.0.weight: [d_ffn, d_model]
        layers.0.0.bias: [d_ffn]
        layers.1.weight: [d_model, d_ffn]
        layers.1.bias: [d_model]

    The structure uses nn.ModuleList with:
        layers[0] = nn.Sequential(nn.Linear, nn.ReLU, nn.Dropout)
        layers[1] = nn.Linear
    This gives keys: layers.0.0.weight (Linear in Sequential), layers.1.weight
    """

    def __init__(self, d_model=256, d_ffn=512, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ),
            nn.Linear(d_ffn, d_model),
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.layers[0](x)
        out = self.layers[1](out)
        out = self.dropout(out)
        return out


# ============================================================
# CAN Bus MLP
# ============================================================

class CanBusMLP(nn.Module):
    """MLP for CAN bus features with named LayerNorm.

    State_dict keys (under can_bus_mlp):
        0.weight: [128, 18],  0.bias: [128]
        2.weight: [256, 128], 2.bias: [256]
        norm.weight: [256],   norm.bias: [256]

    Submodules are registered with numeric string names ('0', '1', '2') via
    add_module to produce the same keys as mmcv's Sequential, plus a named
    'norm' LayerNorm.
    """

    def __init__(self, in_dim=18, hidden_dim=128, out_dim=256):
        super().__init__()
        # Register with numeric string keys to match checkpoint naming
        self.add_module('0', nn.Linear(in_dim, hidden_dim))
        self.add_module('1', nn.ReLU(inplace=True))
        self.add_module('2', nn.Linear(hidden_dim, out_dim))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = getattr(self, '0')(x)
        x = getattr(self, '1')(x)
        x = getattr(self, '2')(x)
        x = self.norm(x)
        return x


# ============================================================
# Encoder: Temporal Self-Attention (MSDeformAttn with d_model=512)
# ============================================================

class TemporalSelfAttention(MSDeformAttn):
    """Temporal self-attention for BEV encoder.

    This is MSDeformAttn with d_model=512 to handle the concatenation of
    current BEV features with previous BEV features (temporal fusion).
    For single-frame inference, previous BEV is replaced with zeros.

    The value_proj and output_proj operate on 256-dim (the actual BEV dim),
    while sampling_offsets and attention_weights take 512-dim input.

    State_dict keys match MSDeformAttn:
        sampling_offsets.weight: [128, 512]  (8 heads * 1 level * 8 points * 2)
        sampling_offsets.bias: [128]
        attention_weights.weight: [64, 512]  (8 heads * 1 level * 8 points)
        attention_weights.bias: [64]
        value_proj.weight: [256, 256]
        value_proj.bias: [256]
        output_proj.weight: [256, 256]
        output_proj.bias: [256]
    """

    def __init__(self, d_model=256, n_heads=8, n_levels=1, n_points=8):
        # Initialize with d_model=256 for value_proj/output_proj
        super().__init__(
            d_model=d_model, n_levels=n_levels,
            n_heads=n_heads, n_points=n_points)
        # Override sampling_offsets and attention_weights to accept 512-dim input
        # (the temporal concatenation dimension)
        self.temporal_dim = d_model * 2  # 512
        self.sampling_offsets = nn.Linear(
            self.temporal_dim, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(
            self.temporal_dim, n_heads * n_levels * n_points)
        self._reset_parameters()

    def forward(self, query, reference_points, value,
                spatial_shapes, query_temporal=None):
        """
        Args:
            query: (B, Lq, 256) - current BEV queries
            reference_points: (B, Lq, n_levels, 2) in [0, 1]
            value: (B, Lq, 256) - BEV features to attend over
            spatial_shapes: list of (H, W) tuples
            query_temporal: (B, Lq, 512) - concat of current + prev BEV
                           If None, creates by concatenating query with zeros.

        Returns:
            output: (B, Lq, 256)
        """
        if query_temporal is None:
            # No previous BEV: concat with zeros
            query_temporal = torch.cat([
                query,
                torch.zeros_like(query)
            ], dim=-1)  # (B, Lq, 512)

        N, Lq, _ = query_temporal.shape

        # Project values (256-dim)
        val = self.value_proj(value)
        val = val.view(N, -1, self.n_heads, self.head_dim)

        # Compute offsets and weights from 512-dim temporal query
        sampling_offsets = self.sampling_offsets(query_temporal).view(
            N, Lq, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query_temporal).view(
            N, Lq, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Lq, self.n_heads, self.n_levels, self.n_points)

        # Compute sampling locations
        spatial_shapes_t = torch.tensor(
            spatial_shapes, dtype=torch.float32, device=query.device)
        offset_normalizer = spatial_shapes_t.flip(-1)[None, None, None, :, None, :]
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer
        )

        from deformable_attention import multi_scale_deformable_attn_pytorch
        output = multi_scale_deformable_attn_pytorch(
            val, spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output


# ============================================================
# Encoder: Spatial Cross-Attention (wrapper around MSDeformAttn)
# ============================================================

class MSDeformAttnNoOutputProj(nn.Module):
    """MSDeformAttn without output_proj.

    Used inside SpatialCrossAttention where the output_proj lives at the
    wrapper level instead. Contains only sampling_offsets, attention_weights,
    and value_proj.

    State_dict keys:
        sampling_offsets.weight/bias
        attention_weights.weight/bias
        value_proj.weight/bias
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=2):
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
            1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)

    def forward(self, query, reference_points, input_flatten,
                input_spatial_shapes):
        """
        Args:
            query: (N, Lq, C)
            reference_points: (N, Lq, D, 2) where D = num_Z_anchors
            input_flatten: (N, sum(Hi*Wi), C)
            input_spatial_shapes: list of (H, W)

        Returns:
            output: (N, Lq, C) -- without output_proj applied
        """
        from deformable_attention import multi_scale_deformable_attn_pytorch

        N, Lq, _ = query.shape
        num_Z_anchors = reference_points.shape[2]

        value = self.value_proj(input_flatten)
        value = value.view(N, -1, self.n_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).view(
            N, Lq, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Lq, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Lq, self.n_heads, self.n_levels, self.n_points)

        spatial_shapes_t = torch.tensor(
            input_spatial_shapes, dtype=torch.float32,
            device=query.device)
        offset_normalizer = spatial_shapes_t.flip(-1)[
            None, None, None, :, None, :]

        # Distribute sampling offsets across Z-anchors (matching original
        # MSDeformableAttention3D): n_points offsets are split into
        # (n_points // D) per Z-anchor.
        # reference_points: (N, Lq, D, 2) -> (N, Lq, 1, 1, 1, D, 2)
        ref_pts = reference_points[:, :, None, None, None, :, :]

        # sampling_offsets: (N, Lq, H, L, P, 2) -> (N, Lq, H, L, P//D, D, 2)
        sampling_offsets = sampling_offsets / offset_normalizer
        sampling_offsets = sampling_offsets.view(
            N, Lq, self.n_heads, self.n_levels,
            self.n_points // num_Z_anchors, num_Z_anchors, 2)

        sampling_locations = ref_pts + sampling_offsets
        # -> (N, Lq, H, L, P//D, D, 2) -> flatten to (N, Lq, H, L, P, 2)
        sampling_locations = sampling_locations.view(
            N, Lq, self.n_heads, self.n_levels, self.n_points, 2)

        output = multi_scale_deformable_attn_pytorch(
            value, input_spatial_shapes,
            sampling_locations, attention_weights)
        return output


class SpatialCrossAttention(nn.Module):
    """Spatial cross-attention for BEV encoder.

    Wraps MSDeformAttnNoOutputProj in a structure matching the checkpoint:
        attentions.1.deformable_attention.sampling_offsets/attention_weights/value_proj.*
        attentions.1.output_proj.*

    For each camera, deformable attention is applied using projected reference
    points and the results are averaged across cameras.
    """

    def __init__(self, d_model=256, n_heads=8, n_levels=1, n_points=8,
                 num_cams=6):
        super().__init__()
        self.deformable_attention = MSDeformAttnNoOutputProj(
            d_model=d_model, n_levels=n_levels,
            n_heads=n_heads, n_points=n_points)
        self.output_proj = nn.Linear(d_model, d_model)
        self.num_cams = num_cams
        self.n_levels = n_levels

    def forward(self, query, reference_points_cam, bev_mask, value,
                spatial_shapes, num_cams=6):
        """
        Args:
            query: (B, num_queries, C) - BEV queries
            reference_points_cam: (num_cams, B, num_queries, D, 2)
                2D reference points for each camera, normalized to [0, 1]
                D = num_points_in_pillar (e.g. 4)
            bev_mask: (num_cams, B, num_queries, D) - visibility mask
            value: (B*num_cams, sum(Hi*Wi), C) - multi-cam flattened features
            spatial_shapes: list of (H, W) tuples
            num_cams: number of cameras

        Returns:
            output: (B, num_queries, C)
        """
        B, num_queries, C = query.shape

        # Per-camera deformable attention and accumulation
        slots = torch.zeros_like(query)  # (B, num_queries, C)

        # Compute per-query visibility: a query is visible in a camera if
        # ANY of its D Z-anchors are visible
        # bev_mask: (num_cams, B, nq, D)
        vis = bev_mask.sum(-1) > 0  # (num_cams, B, nq)

        for cam_idx in range(num_cams):
            # Get this camera's features: (B, L, C)
            cam_value = value[cam_idx::num_cams]  # (B, L, C)

            # Get this camera's reference points: (B, num_queries, D, 2)
            ref_pts = reference_points_cam[cam_idx]  # (B, nq, D, 2)

            # Apply deformable attention with D Z-anchors
            out = self.deformable_attention(
                query, ref_pts, cam_value, spatial_shapes)

            # Mask out invisible queries
            cam_vis = vis[cam_idx]  # (B, nq)
            out = out * cam_vis.unsqueeze(-1).float()
            slots = slots + out

        # Average across visible cameras per query
        count = vis.permute(1, 2, 0).sum(-1)  # (B, nq)
        count = count.clamp(min=1.0)
        slots = slots / count.unsqueeze(-1)
        slots = self.output_proj(slots)
        return slots


# ============================================================
# Decoder Self-Attention wrapper (nn.MultiheadAttention)
# ============================================================

class DecoderSelfAttention(nn.Module):
    """Self-attention for decoder, wrapping nn.MultiheadAttention.

    State_dict keys (under attentions.0):
        attn.in_proj_weight: [768, 256]
        attn.in_proj_bias: [768]
        attn.out_proj.weight: [256, 256]
        attn.out_proj.bias: [256]
    """

    def __init__(self, d_model=256, n_heads=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, query, key=None, value=None, attn_mask=None,
                key_padding_mask=None):
        if key is None:
            key = query
        if value is None:
            value = query
        out, _ = self.attn(query, key, value,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask)
        return out


# ============================================================
# BEV Encoder Layer
# ============================================================

class BEVEncoderLayer(nn.Module):
    """Single BEV encoder layer.

    State_dict keys (under encoder.layers.{i}):
        attentions.0.* : TemporalSelfAttention (MSDeformAttn, d_model_in=512)
        attentions.1.deformable_attention.* : SpatialCrossAttention inner
        attentions.1.output_proj.* : SpatialCrossAttention outer
        ffns.0.layers.0.0.* : FFN first linear
        ffns.0.layers.1.* : FFN second linear
        norms.0/1/2.* : LayerNorm
    """

    def __init__(self, d_model=256, n_heads=8, d_ffn=512, dropout=0.1,
                 self_attn_n_levels=1, self_attn_n_points=8,
                 cross_attn_n_levels=1, cross_attn_n_points=8,
                 num_cams=6):
        super().__init__()
        # attentions: ModuleList with 2 elements
        self.attentions = nn.ModuleList([
            TemporalSelfAttention(
                d_model=d_model, n_heads=n_heads,
                n_levels=self_attn_n_levels, n_points=self_attn_n_points),
            SpatialCrossAttention(
                d_model=d_model, n_heads=n_heads,
                n_levels=cross_attn_n_levels, n_points=cross_attn_n_points,
                num_cams=num_cams),
        ])

        # ffns: ModuleList with 1 element
        self.ffns = nn.ModuleList([
            FFN(d_model, d_ffn, dropout),
        ])

        # norms: 3 LayerNorms (after self-attn, after cross-attn, after FFN)
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model),
            nn.LayerNorm(d_model),
            nn.LayerNorm(d_model),
        ])

    def forward(self, bev_queries, img_feats_flat, bev_ref_points,
                reference_points_cam, bev_mask, bev_spatial_shapes,
                cross_spatial_shapes, bev_pos=None, num_cams=6):
        """
        Args:
            bev_queries: (B, bev_h*bev_w, C)
            img_feats_flat: (B*num_cams, sum(Hi*Wi), C)
            bev_ref_points: (B, bev_h*bev_w, 1, 2) ref points for self-attn
            reference_points_cam: (num_cams, B, bev_h*bev_w, D, 2)
                projected reference points per camera
            bev_mask: (num_cams, B, bev_h*bev_w, D) visibility mask
            bev_spatial_shapes: [(bev_h, bev_w)] for self-attn
            cross_spatial_shapes: list of (H, W) for cross-attn levels
            bev_pos: (B, bev_h*bev_w, C) positional encoding
            num_cams: number of cameras

        Returns:
            bev_queries: (B, bev_h*bev_w, C)
        """
        # --- Self-attention (temporal) ---
        residual = bev_queries
        query_input = bev_queries
        if bev_pos is not None:
            query_input = query_input + bev_pos
        # Concat with zeros for temporal dim (no previous BEV)
        query_temporal = torch.cat([
            query_input,
            torch.zeros_like(query_input)
        ], dim=-1)  # (B, Lq, 512)
        bev_queries = self.attentions[0](
            bev_queries, bev_ref_points, bev_queries,
            bev_spatial_shapes, query_temporal=query_temporal)
        bev_queries = self.norms[0](residual + bev_queries)

        # --- Cross-attention (spatial) ---
        residual = bev_queries
        query_input = bev_queries
        if bev_pos is not None:
            query_input = query_input + bev_pos
        # Use camera-aware cross-attention with image features
        bev_queries_out = self.attentions[1](
            query_input, reference_points_cam, bev_mask, img_feats_flat,
            cross_spatial_shapes, num_cams=num_cams)
        bev_queries = self.norms[1](residual + bev_queries_out)

        # --- FFN ---
        residual = bev_queries
        bev_queries = self.ffns[0](bev_queries)
        bev_queries = self.norms[2](residual + bev_queries)

        return bev_queries


# ============================================================
# BEV Encoder
# ============================================================

class BEVEncoder(nn.Module):
    """BEV Encoder with 3 layers.

    State_dict keys (under encoder):
        layers.{0-2}.attentions.0.* : temporal self-attn
        layers.{0-2}.attentions.1.* : spatial cross-attn
        layers.{0-2}.ffns.0.* : FFN
        layers.{0-2}.norms.{0-2}.* : LayerNorms
    """

    def __init__(self, d_model=256, n_heads=8, d_ffn=512,
                 n_layers=3, dropout=0.1,
                 self_attn_n_levels=1, self_attn_n_points=8,
                 cross_attn_n_levels=1, cross_attn_n_points=8,
                 num_cams=6):
        super().__init__()
        self.layers = nn.ModuleList([
            BEVEncoderLayer(
                d_model=d_model, n_heads=n_heads, d_ffn=d_ffn,
                dropout=dropout,
                self_attn_n_levels=self_attn_n_levels,
                self_attn_n_points=self_attn_n_points,
                cross_attn_n_levels=cross_attn_n_levels,
                cross_attn_n_points=cross_attn_n_points,
                num_cams=num_cams)
            for _ in range(n_layers)
        ])

    def forward(self, bev_queries, img_feats_flat, bev_ref_points,
                reference_points_cam, bev_mask, bev_spatial_shapes,
                cross_spatial_shapes, bev_pos=None, num_cams=6):
        for layer in self.layers:
            bev_queries = layer(
                bev_queries, img_feats_flat, bev_ref_points,
                reference_points_cam, bev_mask, bev_spatial_shapes,
                cross_spatial_shapes, bev_pos=bev_pos, num_cams=num_cams)
        return bev_queries


# ============================================================
# Decoder Layer
# ============================================================

class DecoderLayer(nn.Module):
    """Single DETR decoder layer with iterative refinement.

    State_dict keys (under decoder.layers.{i}):
        attentions.0.attn.in_proj_weight/bias, attn.out_proj.*
        attentions.1.sampling_offsets/attention_weights/value_proj/output_proj.*
        ffns.0.layers.0.0.*, ffns.0.layers.1.*
        norms.0/1/2.*
    """

    def __init__(self, d_model=256, n_heads=8, d_ffn=512, dropout=0.1,
                 cross_attn_n_levels=1, cross_attn_n_points=4):
        super().__init__()
        # attentions: ModuleList with 2 elements
        self.attentions = nn.ModuleList([
            DecoderSelfAttention(d_model, n_heads, dropout=dropout),
            MSDeformAttn(d_model=d_model, n_levels=cross_attn_n_levels,
                         n_heads=n_heads, n_points=cross_attn_n_points),
        ])

        # ffns: ModuleList with 1 element
        self.ffns = nn.ModuleList([
            FFN(d_model, d_ffn, dropout),
        ])

        # norms: 3 LayerNorms
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model),
            nn.LayerNorm(d_model),
            nn.LayerNorm(d_model),
        ])

    def forward(self, query, memory, ref_points, spatial_shapes,
                query_pos=None):
        """
        Args:
            query: (B, num_queries, C)
            memory: (B, HW, C) - flattened BEV features
            ref_points: (B, num_queries, n_levels, 2)
            spatial_shapes: list of (H, W) for memory levels
            query_pos: (B, num_queries, C) positional encoding

        Returns:
            query: (B, num_queries, C)
        """
        # --- Self-attention ---
        residual = query
        q = query
        if query_pos is not None:
            q = q + query_pos
        query = self.attentions[0](q, q, query)
        query = self.norms[0](residual + query)

        # --- Cross-attention to BEV memory ---
        residual = query
        q = query
        if query_pos is not None:
            q = q + query_pos
        query_out = self.attentions[1](
            q, ref_points, memory, spatial_shapes)
        query = self.norms[1](residual + query_out)

        # --- FFN ---
        residual = query
        query = self.ffns[0](query)
        query = self.norms[2](residual + query)

        return query


# ============================================================
# Decoder
# ============================================================

class Decoder(nn.Module):
    """DETR decoder with 6 layers.

    State_dict keys (under decoder):
        layers.{0-5}.attentions.0/1.*, ffns.0.*, norms.0/1/2.*
    """

    def __init__(self, d_model=256, n_heads=8, d_ffn=512,
                 n_layers=6, dropout=0.1,
                 cross_attn_n_levels=1, cross_attn_n_points=4):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model, n_heads=n_heads, d_ffn=d_ffn,
                dropout=dropout,
                cross_attn_n_levels=cross_attn_n_levels,
                cross_attn_n_points=cross_attn_n_points)
            for _ in range(n_layers)
        ])

    def forward(self, query, memory, ref_points, spatial_shapes,
                query_pos=None, reg_branches=None):
        """
        Args:
            query: (B, num_queries, C)
            memory: (B, HW, C)
            ref_points: (B, num_queries, n_levels, 2) or (B, num_queries, 3)
            spatial_shapes: list of (H, W)
            query_pos: (B, num_queries, C)
            reg_branches: nn.ModuleList of regression heads for iterative
                         refinement (updates reference points each layer)

        Returns:
            intermediate: list of (B, num_queries, C) outputs from each layer
            ref_points_out: final reference points
        """
        intermediate = []
        B = query.shape[0]

        # Ensure ref_points has the right shape: (B, num_queries, n_levels, 2)
        if ref_points.dim() == 3:
            # (B, num_queries, 3) -> take xy, expand for n_levels
            ref_2d = ref_points[..., :2].unsqueeze(2)  # (B, nq, 1, 2)
        else:
            ref_2d = ref_points

        for lid, layer in enumerate(self.layers):
            query = layer(query, memory, ref_2d, spatial_shapes,
                          query_pos=query_pos)
            intermediate.append(query)

            # Iterative refinement: update reference points
            if reg_branches is not None:
                tmp = reg_branches[lid](query)
                # Update reference points using predicted offsets
                new_ref = torch.zeros_like(ref_points)
                new_ref[..., :2] = (tmp[..., :2] + inverse_sigmoid(
                    ref_points[..., :2])).sigmoid()
                if ref_points.shape[-1] == 3:
                    new_ref[..., 2] = (tmp[..., 4] + inverse_sigmoid(
                        ref_points[..., 2])).sigmoid()
                ref_points = new_ref.detach()
                ref_2d = ref_points[..., :2].unsqueeze(2)

        return intermediate, ref_points


# ============================================================
# Transformer (encoder + decoder + misc embeddings)
# ============================================================

class PerceptionTransformer(nn.Module):
    """BEVFormer Transformer: encoder + decoder + embeddings.

    State_dict keys (under transformer):
        encoder.layers.* : BEV encoder
        decoder.layers.* : DETR decoder
        level_embeds: [4, 256]
        cams_embeds: [6, 256]
        can_bus_mlp.* : CAN bus MLP
        reference_points.weight/bias: [3, 256]
    """

    def __init__(self, d_model=256, n_heads=8, d_ffn=512,
                 num_enc_layers=3, num_dec_layers=6,
                 num_cams=6, num_feature_levels=4,
                 bev_h=50, bev_w=50, dropout=0.1,
                 pc_range=None, num_points_in_pillar=4,
                 img_h=480, img_w=800):
        super().__init__()
        self.d_model = d_model
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_cams = num_cams
        self.num_feature_levels = num_feature_levels
        if pc_range is None:
            pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.img_h = img_h
        self.img_w = img_w

        # Encoder (cross-attention uses n_levels=1 per camera as in original config)
        self.encoder = BEVEncoder(
            d_model=d_model, n_heads=n_heads, d_ffn=d_ffn,
            n_layers=num_enc_layers, dropout=dropout,
            self_attn_n_levels=1, self_attn_n_points=8,
            cross_attn_n_levels=1, cross_attn_n_points=8,
            num_cams=num_cams)

        # Decoder
        self.decoder = Decoder(
            d_model=d_model, n_heads=n_heads, d_ffn=d_ffn,
            n_layers=num_dec_layers, dropout=dropout,
            cross_attn_n_levels=1, cross_attn_n_points=4)

        # Level embeddings (added to multi-scale image features)
        self.level_embeds = nn.Parameter(
            torch.Tensor(num_feature_levels, d_model))
        nn.init.normal_(self.level_embeds)

        # Camera embeddings (added per camera)
        self.cams_embeds = nn.Parameter(
            torch.Tensor(num_cams, d_model))
        nn.init.normal_(self.cams_embeds)

        # CAN bus MLP
        self.can_bus_mlp = CanBusMLP(
            in_dim=18, hidden_dim=128, out_dim=d_model)

        # Reference points linear layer for decoder queries
        self.reference_points = nn.Linear(d_model, 3)

    def get_reference_points_3d(self, device, dtype):
        """Generate 3D reference points for BEV grid with pillar sampling.

        Returns:
            ref_3d: (D, B=1, bev_h*bev_w, 3) normalized 3D reference points
        """
        D = self.num_points_in_pillar
        H, W = self.bev_h, self.bev_w
        Z = self.pc_range[5] - self.pc_range[2]  # 8.0

        zs = torch.linspace(
            0.5, Z - 0.5, D, dtype=dtype, device=device
        ).view(-1, 1, 1).expand(D, H, W) / Z
        xs = torch.linspace(
            0.5, W - 0.5, W, dtype=dtype, device=device
        ).view(1, 1, W).expand(D, H, W) / W
        ys = torch.linspace(
            0.5, H - 0.5, H, dtype=dtype, device=device
        ).view(1, H, 1).expand(D, H, W) / H

        ref_3d = torch.stack((xs, ys, zs), -1)  # (D, H, W, 3)
        ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
        # (D, H*W, 3) -> (D, 1, H*W, 3)
        ref_3d = ref_3d[:, None]
        return ref_3d

    def point_sampling(self, reference_points, lidar2img, img_h, img_w):
        """Project 3D reference points to camera image coordinates.

        Args:
            reference_points: (D, B, num_query, 3) normalized [0,1]
            lidar2img: (B, num_cams, 4, 4)
            img_h, img_w: image dimensions

        Returns:
            reference_points_cam: (num_cams, B, num_query, D, 2) in [0,1]
            bev_mask: (num_cams, B, num_query, D) boolean
        """
        pc = self.pc_range
        ref = reference_points.clone()
        ref[..., 0:1] = ref[..., 0:1] * (pc[3] - pc[0]) + pc[0]
        ref[..., 1:2] = ref[..., 1:2] * (pc[4] - pc[1]) + pc[1]
        ref[..., 2:3] = ref[..., 2:3] * (pc[5] - pc[2]) + pc[2]

        # Add homogeneous coordinate
        ref = torch.cat([ref, torch.ones_like(ref[..., :1])], -1)
        # ref: (D, B, num_query, 4)

        D, B, num_query = ref.shape[:3]
        num_cam = lidar2img.shape[1]

        # (D, B, 1, num_query, 4, 1)
        ref = ref.view(D, B, 1, num_query, 4).unsqueeze(-1)
        ref = ref.expand(-1, -1, num_cam, -1, -1, -1)

        # (1, B, num_cam, 1, 4, 4)
        l2i = lidar2img.view(1, B, num_cam, 1, 4, 4)
        l2i = l2i.expand(D, -1, -1, num_query, -1, -1)

        # Project: (D, B, num_cam, num_query, 4, 1) -> squeeze
        ref_cam = torch.matmul(l2i, ref).squeeze(-1)

        eps = 1e-5
        mask = ref_cam[..., 2:3] > eps
        ref_cam = ref_cam[..., 0:2] / torch.clamp(ref_cam[..., 2:3], min=eps)
        ref_cam[..., 0] /= img_w
        ref_cam[..., 1] /= img_h

        mask = (mask
                & (ref_cam[..., 0:1] > 0.0)
                & (ref_cam[..., 0:1] < 1.0)
                & (ref_cam[..., 1:2] > 0.0)
                & (ref_cam[..., 1:2] < 1.0))

        # ref_cam: (D, B, num_cam, num_query, 2)
        # -> (num_cam, B, num_query, D, 2)
        ref_cam = ref_cam.permute(2, 1, 3, 0, 4)
        mask = mask.squeeze(-1).permute(2, 1, 3, 0)

        return ref_cam, mask

    @staticmethod
    def get_default_lidar2img(num_cams=6, device='cpu'):
        """Return lidar2img matrices for nuScenes v1.0-mini sample
        ca9a282c9e77460f8360f564131a8af5 (first keyframe).

        Computed from the calibrated_sensor and ego_pose metadata:
            lidar2img = viewpad(intrinsic) @ inv(cam2global) @ lidar2global

        Camera order: FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT

        Returns:
            lidar2img: (1, num_cams, 4, 4) tensor
        """
        # fmt: off
        lidar2img_list = [
            # CAM_FRONT
            [[1263.4881310658986, 820.4207963981752, 24.73538332651016, -328.9915887805],
             [6.937362876127535, 516.2185427762092, -1256.527762123457, -627.647274771842],
             [-0.003542212411213258, 0.999802298499569, 0.019565700759667102, -0.4292221797941238],
             [0.0, 0.0, 0.0, 1.0]],
            # CAM_FRONT_LEFT
            [[51.735891794922004, 1516.134104097482, 38.203188136822604, -248.1182537451171],
             [-389.6552231440645, 306.77311600862845, -1266.3831357913477, -671.3391927000974],
             [-0.819554247797848, 0.5728736473548813, 0.012108636703708407, -0.5106567477259887],
             [0.0, 0.0, 0.0, 1.0]],
            # CAM_FRONT_RIGHT
            [[1369.3507087252467, -605.4496755462368, -29.29658020778624, -469.1855640048184],
             [400.13215148803295, 304.8248168234193, -1257.8030709918235, -727.3929916573209],
             [0.8340582831918065, 0.5516517112865864, 0.005212453714103082, -0.6078003270167756],
             [0.0, 0.0, 0.0, 1.0]],
            # CAM_BACK
            [[-813.043166065008, -825.3453104780422, -14.480529239691625, -837.8834242142998],
             [5.7940717606915175, -475.4852444033304, -812.914062090951, -710.9691029240383],
             [-0.004668203505499156, -0.9999586913425677, -0.007798941241917679, -1.007525480580398],
             [0.0, 0.0, 0.0, 1.0]],
            # CAM_BACK_LEFT
            [[-1149.5392303247443, 940.9229648721968, 8.063046726400245, -642.0285223586698],
             [-442.2411716483507, -114.56587151389417, -1270.2458400363512, -520.4483240071451],
             [-0.9481973029110318, -0.3163290533386733, -0.029288304254537875, -0.43581627449702864],
             [0.0, 0.0, 0.0, 1.0]],
            # CAM_BACK_RIGHT
            [[304.42313405171575, -1463.425610380557, -61.18949508469049, -322.7224958717359],
             [461.55255282524763, -127.43022641982672, -1268.1888147593554, -589.4029597434226],
             [0.9340952306605895, -0.35649516421388244, -0.019424158907449675, -0.4928893159585641],
             [0.0, 0.0, 0.0, 1.0]],
        ]
        # fmt: on

        lidar2img = torch.tensor(lidar2img_list, dtype=torch.float32,
                                 device=device)  # (6, 4, 4)
        # Scale from original 1600x900 pixel space to model input 800x480
        # so that point_sampling (which divides by img_w=800, img_h=480)
        # produces correct [0,1] normalized coordinates.
        lidar2img[:, 0, :] *= 800.0 / 1600.0   # x scale
        lidar2img[:, 1, :] *= 480.0 / 900.0     # y scale
        return lidar2img.unsqueeze(0)  # (1, num_cams, 4, 4)

    def forward(self, img_feats, bev_queries, bev_pos,
                query_embed, reg_branches=None):
        """
        Args:
            img_feats: list of (B*num_cams, C, H, W) multi-scale features
            bev_queries: (B, bev_h*bev_w, C)
            bev_pos: (B, bev_h*bev_w, C) positional encoding
            query_embed: (num_queries, 2*C) object query embeddings
            reg_branches: nn.ModuleList for iterative bbox refinement

        Returns:
            hs: list of (B, num_queries, C) intermediate decoder outputs
            init_ref_points: (B, num_queries, 3) initial reference points
            inter_ref_points: (B, num_queries, 3) refined reference points
        """
        B = bev_queries.shape[0]
        device = bev_queries.device

        # --- Flatten multi-scale features & add level/camera embeddings ---
        feat_flatten_list = []
        spatial_shapes = []
        for lvl, feat in enumerate(img_feats):
            # feat: (B*num_cams, C, H, W)
            BN, C, H, W = feat.shape
            spatial_shapes.append((H, W))
            feat_flat = feat.flatten(2).permute(0, 2, 1)  # (BN, H*W, C)
            # Add level embedding
            feat_flat = feat_flat + self.level_embeds[lvl][None, None, :]
            feat_flatten_list.append(feat_flat)

        # (B*num_cams, sum(Hi*Wi), C)
        img_feats_flat = torch.cat(feat_flatten_list, dim=1)

        # Add camera embeddings
        # img_feats_flat: (B*num_cams, L, C)
        L = img_feats_flat.shape[1]
        img_feats_flat = img_feats_flat.view(B, self.num_cams, L, C)
        img_feats_flat = img_feats_flat + self.cams_embeds[None, :, None, :]
        img_feats_flat = img_feats_flat.view(B * self.num_cams, L, C)

        # --- CAN bus embedding (zeros for inference without CAN data) ---
        can_bus = torch.zeros(B, 18, device=device)
        can_bus_emb = self.can_bus_mlp(can_bus)  # (B, C)
        bev_queries = bev_queries + can_bus_emb[:, None, :]

        # --- BEV reference points for self-attention ---
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, self.bev_h - 0.5, self.bev_h,
                           device=device) / self.bev_h,
            torch.linspace(0.5, self.bev_w - 0.5, self.bev_w,
                           device=device) / self.bev_w,
            indexing='ij')
        bev_ref = torch.stack([ref_x.flatten(), ref_y.flatten()], -1)
        # Self-attn ref points: (B, bev_h*bev_w, 1, 2)
        bev_ref_points = bev_ref[None, :, None, :].expand(B, -1, 1, -1)

        bev_spatial_shapes = [(self.bev_h, self.bev_w)]

        # --- Compute 3D reference points and project to cameras ---
        ref_3d = self.get_reference_points_3d(
            device, bev_queries.dtype)  # (D, 1, bev_h*bev_w, 3)
        ref_3d = ref_3d.expand(-1, B, -1, -1)

        lidar2img = self.get_default_lidar2img(
            self.num_cams, device)  # (1, num_cams, 4, 4)
        lidar2img = lidar2img.expand(B, -1, -1, -1)

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, lidar2img, self.img_h, self.img_w)
        # reference_points_cam: (num_cams, B, num_query, D, 2)
        # bev_mask: (num_cams, B, num_query, D)

        # --- Encode: BEV queries attend to image features ---
        bev_embed = self.encoder(
            bev_queries, img_feats_flat,
            bev_ref_points, reference_points_cam, bev_mask,
            bev_spatial_shapes, spatial_shapes,
            bev_pos=bev_pos, num_cams=self.num_cams)

        # --- Decode: object queries attend to BEV features ---
        # Split query_embedding into query_pos and query content
        query_pos, query = torch.split(
            query_embed, self.d_model, dim=1)  # each (num_queries, C)
        query_pos = query_pos.unsqueeze(0).expand(B, -1, -1)
        query = query.unsqueeze(0).expand(B, -1, -1)
        # Initialize query content as zeros (standard DETR)
        query = torch.zeros_like(query)

        # Reference points for decoder: project query_pos to 3D coords
        ref_points_init = self.reference_points(query_pos)  # (B, nq, 3)
        ref_points_init = ref_points_init.sigmoid()
        init_ref_points = ref_points_init

        # Memory: flatten BEV
        memory = bev_embed  # (B, bev_h*bev_w, C)
        memory_spatial = [(self.bev_h, self.bev_w)]

        intermediate, inter_ref_points = self.decoder(
            query, memory, ref_points_init, memory_spatial,
            query_pos=query_pos, reg_branches=reg_branches)

        return intermediate, init_ref_points, inter_ref_points


# ============================================================
# Detection Head (pts_bbox_head)
# ============================================================

class BEVFormerHead(nn.Module):
    """BEVFormer detection head.

    State_dict keys (under pts_bbox_head):
        bev_embedding.weight: [2500, 256]
        positional_encoding.row_embed.weight: [50, 128]
        positional_encoding.col_embed.weight: [50, 128]
        query_embedding.weight: [900, 512]
        code_weights: [10] (buffer)
        cls_branches.{0-5}.{0,1,3,4,6}.*
        reg_branches.{0-5}.{0,2,4}.*
        transformer.*
    """

    def __init__(self, d_model=256, num_classes=10, num_queries=900,
                 code_size=10, num_dec_layers=6, num_cams=6,
                 num_feature_levels=4, bev_h=50, bev_w=50,
                 pc_range=None):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.code_size = code_size
        self.num_dec_layers = num_dec_layers
        self.bev_h = bev_h
        self.bev_w = bev_w
        if pc_range is None:
            pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.pc_range = pc_range

        # BEV embedding: learnable queries for BEV grid
        self.bev_embedding = nn.Embedding(bev_h * bev_w, d_model)

        # Positional encoding for BEV
        self.positional_encoding = LearnedPositionalEncoding(
            num_feats=d_model // 2, row_num=bev_h, col_num=bev_w)

        # Object query embedding
        self.query_embedding = nn.Embedding(num_queries, d_model * 2)

        # Code weights (registered buffer, not a parameter)
        self.register_buffer(
            'code_weights',
            torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]))

        # Transformer
        self.transformer = PerceptionTransformer(
            d_model=d_model, n_heads=8, d_ffn=512,
            num_enc_layers=3, num_dec_layers=num_dec_layers,
            num_cams=num_cams, num_feature_levels=num_feature_levels,
            bev_h=bev_h, bev_w=bev_w)

        # Classification branches: 6 independent nn.Sequential
        # Each: Linear(256,256)->LN(256)->ReLU->Linear(256,256)->LN(256)->ReLU->Linear(256,10)
        self.cls_branches = nn.ModuleList()
        for _ in range(num_dec_layers):
            self.cls_branches.append(nn.Sequential(
                nn.Linear(d_model, d_model),    # .0
                nn.LayerNorm(d_model),           # .1
                nn.ReLU(inplace=True),           # .2
                nn.Linear(d_model, d_model),    # .3
                nn.LayerNorm(d_model),           # .4
                nn.ReLU(inplace=True),           # .5
                nn.Linear(d_model, num_classes), # .6
            ))

        # Regression branches: 6 independent nn.Sequential
        # Each: Linear(256,256)->ReLU->Linear(256,256)->ReLU->Linear(256,10)
        self.reg_branches = nn.ModuleList()
        for _ in range(num_dec_layers):
            self.reg_branches.append(nn.Sequential(
                nn.Linear(d_model, d_model),    # .0
                nn.ReLU(inplace=True),           # .1
                nn.Linear(d_model, d_model),    # .2
                nn.ReLU(inplace=True),           # .3
                nn.Linear(d_model, code_size),  # .4
            ))

        # Initialize bias for classification (prior probability)
        bias_init = -math.log((1 - 0.01) / 0.01)
        for branch in self.cls_branches:
            nn.init.constant_(branch[-1].bias, bias_init)

    def forward(self, img_feats):
        """
        Args:
            img_feats: list of (B*num_cams, C, H, W) multi-scale features

        Returns:
            cls_scores: (B, num_queries, num_classes) from last decoder layer
            bbox_preds: (B, num_queries, code_size) from last decoder layer
        """
        B_cams = img_feats[0].shape[0]
        B = B_cams // self.transformer.num_cams
        device = img_feats[0].device

        # BEV queries and positional encoding
        bev_queries = self.bev_embedding.weight.unsqueeze(0).expand(
            B, -1, -1)  # (B, bev_h*bev_w, C)
        bev_pos = self.positional_encoding(
            self.bev_h, self.bev_w, device).expand(
            B, -1, -1)  # (B, bev_h*bev_w, C)

        # Run transformer
        intermediate, init_ref, inter_ref = self.transformer(
            img_feats, bev_queries, bev_pos,
            self.query_embedding.weight,
            reg_branches=self.reg_branches)

        # Use the LAST decoder layer's output
        last_hs = intermediate[-1]  # (B, num_queries, C)
        last_idx = self.num_dec_layers - 1

        cls_scores = self.cls_branches[last_idx](last_hs)
        bbox_preds = self.reg_branches[last_idx](last_hs)

        # Decode predictions following the original BEVFormer convention:
        # Raw output: [cx, cy, w, l, cz, h, sin, cos, vx, vy]
        ref = inter_ref  # (B, num_queries, 3)

        # cx, cy: add reference point offset, sigmoid, scale to pc_range
        cx = (bbox_preds[..., 0:1] + inverse_sigmoid(ref[..., 0:1])).sigmoid()
        cy = (bbox_preds[..., 1:2] + inverse_sigmoid(ref[..., 1:2])).sigmoid()
        cx = cx * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        cy = cy * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]

        # cz: add reference point offset, sigmoid, scale to pc_range
        cz = (bbox_preds[..., 4:5] + inverse_sigmoid(ref[..., 2:3])).sigmoid()
        cz = cz * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

        # Output format: [cx, cy, cz, w, l, h, sin, cos, vx, vy]
        # where cx, cy, cz are in world coordinates (meters)
        bbox_preds_out = torch.cat([
            cx,                            # cx (meters)
            cy,                            # cy (meters)
            cz,                            # cz (meters)
            bbox_preds[..., 2:4],          # w, l (raw)
            bbox_preds[..., 5:6],          # h (raw)
            bbox_preds[..., 6:8],          # sin, cos (raw)
            bbox_preds[..., 8:10],         # vx, vy (raw)
        ], dim=-1)

        return cls_scores, bbox_preds_out


# ============================================================
# Full BEVFormer-tiny Model
# ============================================================

class BEVFormerTiny(nn.Module):
    """BEVFormer-tiny: complete model from images to 3D detections.

    Input:  (B, num_cams, 3, H, W) camera images
    Output: cls_scores (B, num_queries, num_classes),
            bbox_preds (B, num_queries, code_size)

    State_dict top-level keys:
        img_backbone.* : ResNet-50
        img_neck.* : FPN
        pts_bbox_head.* : detection head + transformer
    """

    def __init__(self, num_cams=6, embed_dims=256, num_classes=10,
                 num_queries=900, bev_h=50, bev_w=50,
                 num_dec_layers=6, num_feature_levels=4):
        super().__init__()
        self.num_cams = num_cams

        # Backbone: ResNet-50
        self.img_backbone = ResNet50Backbone()

        # Neck: FPN (single level, 2048 -> 256)
        self.img_neck = FPN(in_channels=2048, out_channels=embed_dims)

        # Detection head (contains transformer, embeddings, branches)
        self.pts_bbox_head = BEVFormerHead(
            d_model=embed_dims,
            num_classes=num_classes,
            num_queries=num_queries,
            code_size=10,
            num_dec_layers=num_dec_layers,
            num_cams=num_cams,
            num_feature_levels=num_feature_levels,
            bev_h=bev_h,
            bev_w=bev_w)

    def extract_feat(self, imgs):
        """Extract features from multi-camera images.

        Args:
            imgs: (B*num_cams, 3, H, W)

        Returns:
            list of (B*num_cams, C, H_i, W_i) feature maps.
            For BEVFormer-tiny with 1 FPN level, returns a single-element list.
            For the encoder which expects 4 levels, we generate 4 scales
            by downsampling.
        """
        c5 = self.img_backbone(imgs)  # (BN, 2048, H/32, W/32)
        fpn_feats = self.img_neck(c5)  # [(BN, 256, H/32, W/32)]
        return fpn_feats

    def forward(self, imgs):
        """
        Args:
            imgs: (B, num_cams, 3, H, W)

        Returns:
            cls_scores: (B, num_queries, num_classes)
            bbox_preds: (B, num_queries, code_size)
        """
        B, N, C, H, W = imgs.shape

        # Extract features for all cameras
        imgs_flat = imgs.reshape(B * N, C, H, W)
        fpn_feats = self.extract_feat(imgs_flat)

        # BEVFormer-tiny uses 1 FPN level (matching original config)
        # Run detection head
        cls_scores, bbox_preds = self.pts_bbox_head(fpn_feats)

        return cls_scores, bbox_preds


# ============================================================
# Utility functions
# ============================================================

def inverse_sigmoid(x, eps=1e-5):
    """Inverse of sigmoid function (logit)."""
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


def build_bevformer_tiny(num_cams=6, img_h=480, img_w=800):
    """Build a BEVFormer-tiny model with default configuration.

    Returns:
        model: BEVFormerTiny instance ready for weight loading
    """
    model = BEVFormerTiny(
        num_cams=num_cams,
        embed_dims=256,
        num_classes=10,
        num_queries=900,
        bev_h=50,
        bev_w=50,
        num_dec_layers=6,
        num_feature_levels=4,
    )
    return model


def load_pretrained(model, ckpt_path):
    """Load pretrained weights from a BEVFormer-tiny checkpoint.

    The checkpoint from mmdetection3d stores weights under the 'state_dict'
    key. This function handles loading with proper key matching and reports
    any missing or unexpected keys.

    Args:
        model: BEVFormerTiny instance
        ckpt_path: path to bevformer_tiny_epoch_24.pth

    Returns:
        model: the model with loaded weights
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    # Load with strict=False to handle minor mismatches
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"Missing keys ({len(missing)}):")
        for k in missing:
            print(f"  {k}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}):")
        for k in unexpected:
            print(f"  {k}")

    if not missing and not unexpected:
        print("All checkpoint weights loaded successfully!")
    else:
        loaded = len(state_dict) - len(unexpected)
        total_model = len(dict(model.named_parameters())) + len(
            dict(model.named_buffers()))
        print(f"Loaded {loaded}/{len(state_dict)} checkpoint keys, "
              f"model has {total_model} parameters+buffers")

    return model
