"""
numpy による画像リサイズユーティリティ。
PyTorch / torchvision の座標系・補間アルゴリズムに準拠。

interpolation 引数には cv2 定数（int）または文字列を渡せる:
    cv2.INTER_NEAREST = 0  -> "nearest"
    cv2.INTER_LINEAR  = 1  -> "bilinear"
    cv2.INTER_CUBIC   = 2  -> "bicubic"
    cv2.INTER_AREA    = 3  -> "area"
"""

import numpy as np

# cv2 互換定数
INTER_NEAREST = 0
INTER_LINEAR = 1
INTER_CUBIC = 2
INTER_AREA = 3

_INTERP_STR = {
    0: "nearest",
    1: "bilinear",
    2: "bicubic",
    3: "area",
}


# ---------------------------------------------------------------------------
# 内部ユーティリティ
# ---------------------------------------------------------------------------


def _compute_output_size(in_h, in_w, size, max_size=None):
    """
    torchvision.transforms.functional.resize と同様の size 解釈。
    size:
      - int       : 短辺を size に合わせる（アスペクト比維持）
      - (h, w)    : そのサイズにする
    """
    if isinstance(size, int):
        scale = size / min(in_h, in_w)
        out_h = int(round(in_h * scale))
        out_w = int(round(in_w * scale))
        if max_size is not None and max(out_h, out_w) > max_size:
            scale = max_size / max(out_h, out_w)
            out_h = int(round(out_h * scale))
            out_w = int(round(out_w * scale))
    else:
        out_h, out_w = size
    return out_h, out_w


def _cast_output(out, src_dtype):
    """float32 の計算結果を元の dtype に戻す。整数型は round→clip。"""
    if np.issubdtype(src_dtype, np.integer):
        out = np.rint(out)
        info = np.iinfo(src_dtype)
        out = np.clip(out, info.min, info.max).astype(src_dtype)
    else:
        out = out.astype(src_dtype, copy=False)
    return out


# ---------------------------------------------------------------------------
# NEAREST
# ---------------------------------------------------------------------------


def resize_nearest(arr, size, max_size=None, exact=False):
    """
    PyTorch mode='nearest'（exact=False）または mode='nearest-exact'（exact=True）相当。
    arr: HWC or HW
    """
    is_2d = arr.ndim == 2
    if is_2d:
        arr = arr[..., None]

    in_h, in_w, _ = arr.shape
    out_h, out_w = _compute_output_size(in_h, in_w, size, max_size)

    # PyTorch CUDA カーネルと同じ演算順序: スケール比を float32 で先に計算
    scale_h = np.float32(in_h) / np.float32(out_h)
    scale_w = np.float32(in_w) / np.float32(out_w)
    ih = np.arange(out_h, dtype=np.float32)
    iw = np.arange(out_w, dtype=np.float32)

    if exact:
        y_idx = np.clip(np.floor((ih + 0.5) * scale_h).astype(np.int64), 0, in_h - 1)
        x_idx = np.clip(np.floor((iw + 0.5) * scale_w).astype(np.int64), 0, in_w - 1)
    else:
        y_idx = np.clip(np.floor(ih * scale_h).astype(np.int64), 0, in_h - 1)
        x_idx = np.clip(np.floor(iw * scale_w).astype(np.int64), 0, in_w - 1)

    out = arr[y_idx[:, None], x_idx[None, :], :]
    if is_2d:
        out = out[..., 0]
    return out


# ---------------------------------------------------------------------------
# BILINEAR (antialias=False)
# ---------------------------------------------------------------------------


def resize_bilinear(arr, size, max_size=None):
    """
    torch.nn.functional.interpolate(mode='bilinear', align_corners=False, antialias=False) 相当。
    arr: HWC or HW
    """
    is_2d = arr.ndim == 2
    if is_2d:
        arr = arr[..., None]

    in_h, in_w, _ = arr.shape
    out_h, out_w = _compute_output_size(in_h, in_w, size, max_size)

    y = (np.arange(out_h) + 0.5) * in_h / out_h - 0.5
    x = (np.arange(out_w) + 0.5) * in_w / out_w - 0.5

    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)

    y0c = np.clip(y0, 0, in_h - 1)
    x0c = np.clip(x0, 0, in_w - 1)
    y1c = np.clip(y0 + 1, 0, in_h - 1)
    x1c = np.clip(x0 + 1, 0, in_w - 1)

    wy = (y - y0).astype(np.float32)
    wx = (x - x0).astype(np.float32)

    wa = (1.0 - wy)[:, None] * (1.0 - wx)[None, :]
    wb = (1.0 - wy)[:, None] * wx[None, :]
    wc = wy[:, None] * (1.0 - wx)[None, :]
    wd = wy[:, None] * wx[None, :]

    arrf = arr.astype(np.float32, copy=False)
    out = (
        arrf[y0c[:, None], x0c[None, :], :] * wa[..., None]
        + arrf[y0c[:, None], x1c[None, :], :] * wb[..., None]
        + arrf[y1c[:, None], x0c[None, :], :] * wc[..., None]
        + arrf[y1c[:, None], x1c[None, :], :] * wd[..., None]
    )

    out = _cast_output(out, arr.dtype)
    if is_2d:
        out = out[..., 0]
    return out


# ---------------------------------------------------------------------------
# BILINEAR (antialias=True)
# ---------------------------------------------------------------------------


def _bilinear_axis0(arr, out_size):
    """axis=0 方向を通常 bilinear（antialias なし）でリサイズ。拡大時に使用。"""
    in_size = arr.shape[0]
    y = (np.arange(out_size, dtype=np.float32) + 0.5) * (in_size / out_size) - 0.5
    y0 = np.floor(y).astype(np.int64)
    wy = (y - y0).astype(np.float32).reshape((-1,) + (1,) * (arr.ndim - 1))
    y0c = np.clip(y0, 0, in_size - 1)
    y1c = np.clip(y0 + 1, 0, in_size - 1)
    return arr[y0c] * (1.0 - wy) + arr[y1c] * wy


def _bilinear_aa_axis0(arr, out_size):
    """
    axis=0 方向を antialias bilinear で縮小。
    PyTorch antialias=True bilinear の triangle kernel に準拠。
    arr: float32 前提
    """
    in_size = arr.shape[0]
    scale = in_size / out_size  # 縮小時 > 1
    support = scale  # triangle kernel 半径（source 空間）
    rest_shape = arr.shape[1:]
    out = np.zeros((out_size,) + rest_shape, dtype=np.float32)

    for oi in range(out_size):
        center = (oi + 0.5) * scale - 0.5
        i0 = max(int(np.floor(center - support)), 0)
        i1 = min(int(np.ceil(center + support)), in_size - 1)

        idx = np.arange(i0, i1 + 1, dtype=np.float32)
        w = np.maximum(0.0, 1.0 - np.abs(idx - center) / support)
        wsum = w.sum()
        if wsum == 0.0:
            out[oi] = arr[max(0, min(int(round(center)), in_size - 1))]
            continue

        w /= wsum
        w_bc = w.reshape((-1,) + (1,) * len(rest_shape))
        out[oi] = (arr[i0 : i1 + 1] * w_bc).sum(axis=0)

    return out


def resize_bilinear_aa(arr, size, max_size=None):
    """
    torchvision Resize(interpolation=BILINEAR, antialias=True) 相当。
    分離フィルタ（H→W の2パス）で実装。
    arr: HWC or HW
    """
    is_2d = arr.ndim == 2
    if is_2d:
        arr = arr[..., None]

    in_h, in_w, _ = arr.shape
    out_h, out_w = _compute_output_size(in_h, in_w, size, max_size)

    arrf = arr.astype(np.float32, copy=False)
    # 縮小軸は antialias (triangle kernel)、拡大軸は通常 bilinear（PyTorch と同じ挙動）
    tmp = _bilinear_aa_axis0(arrf, out_h) if out_h < in_h else _bilinear_axis0(arrf, out_h)
    tmpt = tmp.transpose(1, 0, 2)
    tmp = (_bilinear_aa_axis0(tmpt, out_w) if out_w < in_w else _bilinear_axis0(tmpt, out_w)).transpose(1, 0, 2)

    tmp = _cast_output(tmp, arr.dtype)
    if is_2d:
        tmp = tmp[..., 0]
    return tmp


# ---------------------------------------------------------------------------
# BICUBIC (antialias=False)
# ---------------------------------------------------------------------------


def _bicubic_kernel(x, a=-0.75):
    """Keys cubic kernel（vectorized）。PyTorch / PIL と同じ a=-0.75。"""
    ax = np.abs(np.asarray(x, dtype=np.float32))
    return np.where(
        ax <= 1,
        (a + 2) * ax**3 - (a + 3) * ax**2 + 1,
        np.where(ax <= 2, a * ax**3 - 5 * a * ax**2 + 8 * a * ax - 4 * a, 0.0),
    ).astype(np.float32)


def resize_bicubic(arr, size, max_size=None):
    """
    torch.nn.functional.interpolate(mode='bicubic', align_corners=False) 相当。
    arr: HWC or HW
    """
    is_2d = arr.ndim == 2
    if is_2d:
        arr = arr[..., None]

    in_h, in_w, c = arr.shape
    out_h, out_w = _compute_output_size(in_h, in_w, size, max_size)

    y = (np.arange(out_h, dtype=np.float32) + 0.5) * in_h / out_h - 0.5
    x = (np.arange(out_w, dtype=np.float32) + 0.5) * in_w / out_w - 0.5
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)

    arrf = arr.astype(np.float32, copy=False)
    out = np.zeros((out_h, out_w, c), dtype=np.float32)

    for dy in range(-1, 3):
        wy = _bicubic_kernel(y - (y0 + dy))
        yi = np.clip(y0 + dy, 0, in_h - 1)
        for dx in range(-1, 3):
            wx = _bicubic_kernel(x - (x0 + dx))
            xi = np.clip(x0 + dx, 0, in_w - 1)
            out += (
                arrf[yi[:, None], xi[None, :], :]
                * (wy[:, None] * wx[None, :])[..., None]
            )

    out = _cast_output(out, arr.dtype)
    if is_2d:
        out = out[..., 0]
    return out


# ---------------------------------------------------------------------------
# AREA
# ---------------------------------------------------------------------------


def _area_reduce_axis0(arr, out_size):
    """
    axis=0 方向を area averaging で縮小。
    PyTorch mode='area' の座標系:
        src_start = i * (in_size / out_size)
        src_end   = (i+1) * (in_size / out_size)
    境界ピクセルは部分的な重みを持つ。
    arr: float32 前提
    """
    in_size = arr.shape[0]
    scale = in_size / out_size
    rest_shape = arr.shape[1:]
    out = np.empty((out_size,) + rest_shape, dtype=np.float32)

    for oi in range(out_size):
        src0f = oi * scale
        src1f = src0f + scale
        i0 = int(src0f)
        i1 = min(int(np.ceil(src1f)), in_size)

        n = i1 - i0
        if n == 1:
            out[oi] = arr[i0]
            continue

        w = np.ones(n, dtype=np.float32)
        frac0 = src0f - i0
        if frac0 > 0.0:
            w[0] = 1.0 - frac0
        frac1 = i1 - src1f
        if frac1 > 0.0:
            w[-1] = 1.0 - frac1

        w_bc = w.reshape((-1,) + (1,) * len(rest_shape))
        out[oi] = (arr[i0:i1] * w_bc).sum(axis=0) / w.sum()

    return out


def resize_area(arr, size, max_size=None):
    """
    torch.nn.functional.interpolate(mode='area') 相当。
    大幅縮小に適している。分離フィルタ（H→W の2パス）で実装。
    arr: HWC or HW
    """
    is_2d = arr.ndim == 2
    if is_2d:
        arr = arr[..., None]

    in_h, in_w, _ = arr.shape
    out_h, out_w = _compute_output_size(in_h, in_w, size, max_size)

    arrf = arr.astype(np.float32, copy=False)
    tmp = _area_reduce_axis0(arrf, out_h)
    tmp = _area_reduce_axis0(tmp.transpose(1, 0, 2), out_w).transpose(1, 0, 2)

    tmp = _cast_output(tmp, arr.dtype)
    if is_2d:
        tmp = tmp[..., 0]
    return tmp


# ---------------------------------------------------------------------------
# 統合インターフェイス
# ---------------------------------------------------------------------------


def tv_resize(
    arr, size, interpolation="bilinear", antialias=True, max_size=None, exact=False
):
    """
    torchvision.transforms.functional.resize に近いインターフェイス。

    arr           : HWC or HW numpy array
    size          : int or (out_h, out_w)
    interpolation : "bilinear" | "nearest" | "bicubic" | "area"
                    または cv2 定数 (0=NEAREST, 1=LINEAR, 2=CUBIC, 3=AREA)
    antialias     : bool  (nearest / area / bicubic では無視)
    max_size      : int or None
    exact         : bool  (nearest のみ: nearest-exact モード)
    """
    if isinstance(interpolation, int):
        if interpolation not in _INTERP_STR:
            raise ValueError(
                f"cv2 interpolation constant {interpolation} is not supported"
            )
        interpolation = _INTERP_STR[interpolation]

    if interpolation == "nearest":
        return resize_nearest(arr, size, max_size=max_size, exact=exact)
    elif interpolation == "bilinear":
        if antialias:
            return resize_bilinear_aa(arr, size, max_size=max_size)
        else:
            return resize_bilinear(arr, size, max_size=max_size)
    elif interpolation == "bicubic":
        return resize_bicubic(arr, size, max_size=max_size)
    elif interpolation == "area":
        return resize_area(arr, size, max_size=max_size)
    else:
        raise ValueError(
            f"interpolation must be 'bilinear', 'nearest', 'bicubic', or 'area', got: {interpolation!r}"
        )
