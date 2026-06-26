# Depth Anything V3 ONNX Export

## Usage

```bash
pip install depth-anything-3 safetensors huggingface_hub onnx
python export_onnx.py --encoder vits  # or vitb, vitl
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--encoder` | `vits` | Model type: `vits`, `vitb`, `vitl` |
| `--output` | `da3_{encoder}.onnx` | Output ONNX file path |
| `--opset` | `17` | ONNX opset version |
| `--input_height` | `336` | Input image height (must be multiple of 14) |
| `--input_width` | `504` | Input image width (must be multiple of 14) |

## Dynamic Shape

Exported ONNX models support dynamic shapes for batch, height, and width dimensions:

- Input `image`: `[batch, 3, height, width]`
- Output `depth`: `[batch, 1, height, width]`

Height and width must be multiples of 14 (patch size).

## Required patches to depth_anything_3 package

TorchScript tracing bakes shape-dependent Python operations (`int()`, `float()`, dict caching) as constants. The following 4 patches replace them with tensor operations so that shapes propagate dynamically through the ONNX graph. Apply all patches to the installed `depth_anything_3` package before running the export.

Base path: `<site-packages>/depth_anything_3/`

---

### Patch 1: PositionGetter (rope.py)

`torch.cartesian_prod` is unsupported by ONNX. Replace with `torch.meshgrid` + `torch.stack`. Also remove dict caching (incompatible with traced shape values).

File: `model/dinov2/layers/rope.py` class `PositionGetter.__call__`

```python
# Before
    def __call__(self, batch_size, height, width, device):
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions
        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()

# After
    def __call__(self, batch_size, height, width, device):
        y_coords = torch.arange(height, device=device)
        x_coords = torch.arange(width, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        positions = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)
        return positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()
```

---

### Patch 2: RoPE frequency components (rope.py)

`int(positions.max())` is a data-dependent conversion that becomes a constant. Use `positions.shape[1]` (= H_patches*W_patches, always >= max(H_patches, W_patches)) as a dynamic upper bound instead. Also remove dict caching from `_compute_frequency_components`, and replace `torch.cat((angles, angles), dim=-1)` with `angles.repeat(1, 2)` (ailia SDK does not support Concat with the same tensor as both inputs).

File: `model/dinov2/layers/rope.py` class `RotaryPositionEmbedding2D`

```python
# Before (_compute_frequency_components)
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)
            angles = angles.to(dtype)
            angles = angles.repeat(1, 2)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)
        return self.frequency_cache[cache_key]

# After (_compute_frequency_components)
        exponents = torch.arange(0, dim, 2, device=device).float() / dim
        inv_freq = 1.0 / (self.base_frequency**exponents)
        positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        angles = torch.einsum("i,j->ij", positions, inv_freq)
        angles = angles.to(dtype)
        angles = torch.cat((angles, angles), dim=-1)
        cos_components = angles.cos().to(dtype)
        sin_components = angles.sin().to(dtype)
        return (cos_components, sin_components)
```

```python
# Before (forward)
        max_position = int(positions.max()) + 1

# After (forward)
        max_position = positions.shape[1]
```

---

### Patch 3: Position embedding interpolation (vision_transformer.py)

`float()` on shape values bakes scale_factor as a constant. Use `size=(h0, w0)` instead.

File: `model/dinov2/vision_transformer.py` method `interpolate_pos_encoding`

```python
# Before
        kwargs = {}
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]

# After
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            size=(h0, w0),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )
```

---

### Patch 4: Output size computation (dualdpt.py, head_utils.py)

`int()` on shape arithmetic bakes output dimensions as constants. Use integer division instead.

File: `model/dualdpt.py` method `_forward_impl`

```python
# Before
        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)

# After
        h_out = ph * self.patch_size // self.down_ratio
        w_out = pw * self.patch_size // self.down_ratio
```

File: `model/utils/head_utils.py` function `custom_interpolate`

```python
# Before
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

# After
        size = (x.shape[-2] * scale_factor, x.shape[-1] * scale_factor)
```
