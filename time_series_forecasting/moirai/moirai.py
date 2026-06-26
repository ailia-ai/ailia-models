import math
import os
import sys
import warnings
from logging import getLogger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

DATA_PATH = "input.csv"
SAVE_IMAGE_PATH = "output.png"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/moirai/"

VERSIONS = ("1.0", "1.1")

VERSION_SIZE_TO_FILES = {
    ("1.0", "small"): ("moirai-1.0-R-small.onnx", "moirai-1.0-R-small.onnx.prototxt"),
    ("1.0", "base"):  ("moirai-1.0-R-base.onnx",  "moirai-1.0-R-base.onnx.prototxt"),
    ("1.0", "large"): ("moirai-1.0-R-large.onnx", "moirai-1.0-R-large.onnx.prototxt"),
    ("1.1", "small"): ("moirai-1.1-R-small.onnx", "moirai-1.1-R-small.onnx.prototxt"),
    ("1.1", "base"):  ("moirai-1.1-R-base.onnx",  "moirai-1.1-R-base.onnx.prototxt"),
    ("1.1", "large"): ("moirai-1.1-R-large.onnx", "moirai-1.1-R-large.onnx.prototxt"),
}

# Static configuration of Moirai-1.x-R (all sizes).
PATCH_SIZES = (8, 16, 32, 64, 128)
MAX_PATCH = max(PATCH_SIZES)
# NormalFixedScale mixture component uses a fixed scale of 1e-3 (uni2ts default).
NORMAL_FIXED_SCALE = 1e-3

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser("Moirai", DATA_PATH, SAVE_IMAGE_PATH, fp16_support=False)
parser.add_argument("-i", "--input", type=str, default=DATA_PATH)
parser.add_argument(
    "--version",
    type=str,
    default="1.0",
    choices=list(VERSIONS),
    help="Moirai version: 1.0 (Apache-2.0) or 1.1 (CC-BY-NC-4.0)",
)
parser.add_argument(
    "--size",
    type=str,
    default="large",
    choices=["small", "base", "large"],
    help="Model size",
)
parser.add_argument(
    "--target",
    type=str,
    default=None,
    help="Target column name (defaults to first non-date column).",
)
parser.add_argument(
    "--feat",
    type=str,
    default=None,
    help=(
        "Comma-separated covariate column names whose future values are known "
        "(feat_dynamic_real)."
    ),
)
parser.add_argument(
    "--context_len",
    type=int,
    default=200,
    help="Context length (history)",
)
parser.add_argument(
    "--prediction_len",
    type=int,
    default=20,
    help="Prediction horizon length",
)
parser.add_argument(
    "--patch_size",
    type=str,
    default="auto",
    help="Patch size: 'auto' or one of {8, 16, 32, 64, 128}",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=100,
    help="Number of samples drawn from the predictive distribution",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Random seed used for sampling",
)
parser.add_argument(
    "--onnx",
    action="store_true",
    help="Use ONNX Runtime instead of ailia SDK",
)
args = update_parser(parser)


# ======================
# Input Preparation
# ======================


def _choose_patch_size(context_len, prediction_len):
    """Choose the patch size that minimizes total padding waste."""
    best_ps = PATCH_SIZES[0]
    best_waste = float("inf")
    for ps in PATCH_SIZES:
        n_ctx = math.ceil(context_len / ps)
        n_pred = math.ceil(prediction_len / ps)
        waste = (n_ctx * ps - context_len) + (n_pred * ps - prediction_len)
        if waste < best_waste:
            best_waste = waste
            best_ps = ps
    return best_ps


def _pad_left(arr, total_len):
    """Left-pad a 1-D array with zeros to *total_len*."""
    pad = total_len - len(arr)
    if pad > 0:
        return np.concatenate([np.zeros(pad, dtype=arr.dtype), arr])
    return arr


def _pad_right(arr, total_len):
    """Right-pad a 1-D array with zeros to *total_len*."""
    pad = total_len - len(arr)
    if pad > 0:
        return np.concatenate([arr, np.zeros(pad, dtype=arr.dtype)])
    return arr


def _pad_patch(patches, patch_size):
    """Right-pad each patch from *patch_size* to MAX_PATCH."""
    if patch_size == MAX_PATCH:
        return patches
    return np.pad(patches, ((0, 0), (0, MAX_PATCH - patch_size)))


def _build_inputs(history, prediction_len, patch_size,
                  feat_past=None, feat_future=None):
    """Build ONNX model inputs from time series data.

    Replicates the patching / masking logic of
    ``uni2ts.model.moirai.MoiraiForecast._convert`` in pure NumPy.

    Args:
        history: 1-D float32 array of past target values [context_len].
        prediction_len: number of future steps to predict.
        patch_size: patch size (one of PATCH_SIZES).
        feat_past: optional [n_cov, context_len] past covariate values.
        feat_future: optional [n_cov, prediction_len] future covariate values.

    Returns:
        dict of NumPy arrays ready for the ONNX model.
    """
    context_len = len(history)
    n_ctx = math.ceil(context_len / patch_size)
    n_pred = math.ceil(prediction_len / patch_size)
    padded_ctx_len = n_ctx * patch_size
    padded_pred_len = n_pred * patch_size

    # --- helper: generate time_id from past observation mask ---
    def _make_time_ids(past_obs_per_patch):
        cummax = np.maximum.accumulate(past_obs_per_patch.astype(np.int64))
        past_tid = np.clip(np.cumsum(cummax) - 1, 0, None)
        max_past = int(past_tid[-1]) if len(past_tid) > 0 else -1
        future_tid = np.arange(n_pred, dtype=np.int64) + max_past + 1
        return past_tid, future_tid

    # --- helper: compute per-patch sample_id from is_pad ---
    def _sample_id_from_is_pad(is_pad_patches):
        # 1 where at least one element in the patch is NOT padding.
        return (is_pad_patches == 0).astype(np.int64).max(axis=1)

    # ------------------------------------------------------------------
    # Target variate
    # ------------------------------------------------------------------
    # Past target: left-pad to padded_ctx_len
    past_vals = _pad_left(history.astype(np.float32), padded_ctx_len)
    past_patches = past_vals.reshape(n_ctx, patch_size)

    past_obs = _pad_left(np.ones(context_len, dtype=np.float32), padded_ctx_len)
    past_obs_patches = past_obs.reshape(n_ctx, patch_size)

    # past_is_pad: 0 = real data, 1 = padding.  Left-pad with 1.
    past_is_pad = np.concatenate([
        np.ones(padded_ctx_len - context_len, dtype=np.int32),
        np.zeros(context_len, dtype=np.int32),
    ]).reshape(n_ctx, patch_size)
    past_sid = _sample_id_from_is_pad(past_is_pad)  # [n_ctx]

    # Future target: zeros (unknown)
    fut_vals = np.zeros(padded_pred_len, dtype=np.float32).reshape(n_pred, patch_size)
    # Observed mask for future target: True for valid positions, False for
    # right-padding beyond prediction_len.
    fut_obs = _pad_right(
        np.ones(prediction_len, dtype=np.float32), padded_pred_len
    ).reshape(n_pred, patch_size)
    # Future is_pad: 0 for real, right-padded with 1.
    fut_is_pad = _pad_right(
        np.zeros(prediction_len, dtype=np.int32), padded_pred_len
    )
    # For is_pad right-padding, use value=1
    fut_is_pad[prediction_len:] = 1
    fut_sid = _sample_id_from_is_pad(fut_is_pad.reshape(n_pred, patch_size))

    # Pad patches to MAX_PATCH and concatenate past + future.
    tgt_target = np.concatenate([
        _pad_patch(past_patches, patch_size),
        _pad_patch(fut_vals, patch_size),
    ], axis=0)  # [n_ctx + n_pred, MAX_PATCH]
    tgt_obs = np.concatenate([
        _pad_patch(past_obs_patches, patch_size),
        _pad_patch(fut_obs, patch_size),
    ], axis=0)

    tgt_sid = np.concatenate([past_sid, fut_sid])  # [n_ctx + n_pred]

    # Time IDs (same logic as _generate_time_id in MoiraiForecast).
    past_obs_max = past_obs_patches.max(axis=1)  # max over patch dim
    past_tid, fut_tid = _make_time_ids(past_obs_max)
    tgt_tid = np.concatenate([past_tid, fut_tid])

    n_tokens = n_ctx + n_pred
    tgt_vid = np.zeros(n_tokens, dtype=np.int64)
    tgt_pmask = np.concatenate([
        np.zeros(n_ctx, dtype=bool),
        np.ones(n_pred, dtype=bool),
    ])

    # Collect all variates.
    all_target = [tgt_target]
    all_obs = [tgt_obs]
    all_sid = [tgt_sid]
    all_tid = [tgt_tid]
    all_vid = [tgt_vid]
    all_pmask = [tgt_pmask]

    # ------------------------------------------------------------------
    # Covariate variates (feat_dynamic_real)
    # ------------------------------------------------------------------
    if feat_past is not None:
        n_cov = feat_past.shape[0]
        for ci in range(n_cov):
            # Past covariate.
            cp = _pad_left(feat_past[ci].astype(np.float32), padded_ctx_len)
            cp_patches = cp.reshape(n_ctx, patch_size)
            cp_obs = _pad_left(
                np.ones(context_len, dtype=np.float32), padded_ctx_len
            ).reshape(n_ctx, patch_size)

            # Future covariate (known values).
            cf_raw = np.zeros(prediction_len, dtype=np.float32)
            cf_obs_raw = np.zeros(prediction_len, dtype=np.float32)
            if feat_future is not None:
                alen = min(prediction_len, feat_future.shape[1])
                cf_raw[:alen] = feat_future[ci, :alen].astype(np.float32)
                cf_obs_raw[:alen] = 1.0
            cf = _pad_right(cf_raw, padded_pred_len).reshape(n_pred, patch_size)
            cf_obs = _pad_right(cf_obs_raw, padded_pred_len).reshape(n_pred, patch_size)

            all_target.append(np.concatenate([
                _pad_patch(cp_patches, patch_size),
                _pad_patch(cf, patch_size),
            ], axis=0))
            all_obs.append(np.concatenate([
                _pad_patch(cp_obs, patch_size),
                _pad_patch(cf_obs, patch_size),
            ], axis=0))

            # sample_id: past same as target, future = 1.
            all_sid.append(np.concatenate([
                past_sid,
                np.ones(n_pred, dtype=np.int64),
            ]))
            all_tid.append(tgt_tid.copy())
            all_vid.append(np.full(n_tokens, ci + 1, dtype=np.int64))
            # Covariates are fully known -> prediction_mask = False.
            all_pmask.append(np.zeros(n_tokens, dtype=bool))

    # Concatenate all variates and add batch dimension.
    target = np.concatenate(all_target, axis=0)[np.newaxis]
    observed_mask = np.concatenate(all_obs, axis=0)[np.newaxis]
    sample_id = np.concatenate(all_sid)[np.newaxis]
    time_id = np.concatenate(all_tid)[np.newaxis]
    variate_id = np.concatenate(all_vid)[np.newaxis]
    prediction_mask = np.concatenate(all_pmask)[np.newaxis]
    total_seq = target.shape[1]
    patch_size_arr = np.full((1, total_seq), patch_size, dtype=np.int64)

    return {
        "target": target.astype(np.float32),
        "observed_mask": observed_mask.astype(bool),
        "sample_id": sample_id.astype(np.int64),
        "time_id": time_id.astype(np.int64),
        "variate_id": variate_id.astype(np.int64),
        "prediction_mask": prediction_mask.astype(bool),
        "patch_size": patch_size_arr,
    }


# ======================
# Distribution Sampling
# ======================


def _softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def _sample_negbinomial(rng, total_count, logits):
    """Sample from NegativeBinomial matching PyTorch's parameterisation.

    PyTorch NegBin uses a Gamma-Poisson mixture:
        rate ~ Gamma(concentration=total_count, rate=exp(-logits))
        x ~ Poisson(rate)
    NumPy Gamma uses (shape, scale) with scale = 1/rate = exp(logits).
    """
    tc = np.clip(total_count, 1e-6, 1e6)
    gamma_samples = rng.gamma(shape=tc, scale=np.exp(np.clip(logits, -20, 20)))
    return rng.poisson(lam=np.clip(gamma_samples, 0, 1e9)).astype(np.float32)


def _sample_from_mixture(rng, outputs, n_ctx, n_pred, patch_size,
                         prediction_len, num_samples):
    """Sample from the mixture distribution output by the ONNX model.

    The ONNX model outputs raw parameters for a 4-component mixture:
        [StudentT, NormalFixedScale, NegativeBinomial, LogNormal]
    plus the scaler's (loc, scale) for an affine transform.

    Returns:
        samples: ndarray [num_samples, prediction_len]
    """
    (
        weights_logits, st_df, st_loc, st_scale,
        normal_loc, nb_total_count, nb_logits,
        ln_loc, ln_scale, loc, scale,
    ) = outputs

    # Extract prediction positions for the target variate.
    # Target prediction patches sit at indices [n_ctx, n_ctx + n_pred).
    ps, pe = n_ctx, n_ctx + n_pred

    # weights_logits: [1, S, MAX_PATCH, 4] -> [n_pred, patch_size, 4]
    wl = weights_logits[0, ps:pe, :patch_size, :]

    # Per-element component params: [1, S, MAX_PATCH] -> [n_pred, patch_size]
    st_df_p = st_df[0, ps:pe, :patch_size]
    st_loc_p = st_loc[0, ps:pe, :patch_size]
    st_scale_p = st_scale[0, ps:pe, :patch_size]
    normal_loc_p = normal_loc[0, ps:pe, :patch_size]
    nb_tc_p = nb_total_count[0, ps:pe, :patch_size]
    nb_lg_p = nb_logits[0, ps:pe, :patch_size]
    ln_loc_p = ln_loc[0, ps:pe, :patch_size]
    ln_scale_p = ln_scale[0, ps:pe, :patch_size]

    # Scaler: [1, S, 1] -> broadcast to [n_pred, patch_size]
    s_loc = np.broadcast_to(
        loc[0, ps:pe, :],   # [n_pred, 1]
        (n_pred, patch_size),
    )
    s_scale = np.broadcast_to(
        scale[0, ps:pe, :],
        (n_pred, patch_size),
    )

    # Flatten patches -> [n_pred * patch_size], then trim to prediction_len.
    flat = n_pred * patch_size
    L = min(prediction_len, flat)

    wl_f = wl.reshape(flat, 4)[:L]
    st_df_f = st_df_p.reshape(flat)[:L]
    st_loc_f = st_loc_p.reshape(flat)[:L]
    st_scale_f = st_scale_p.reshape(flat)[:L]
    nl_f = normal_loc_p.reshape(flat)[:L]
    nb_tc_f = nb_tc_p.reshape(flat)[:L]
    nb_lg_f = nb_lg_p.reshape(flat)[:L]
    ll_f = ln_loc_p.reshape(flat)[:L]
    ls_f = ln_scale_p.reshape(flat)[:L]
    sl_f = s_loc.reshape(flat)[:L]
    ss_f = s_scale.reshape(flat)[:L]

    weights = _softmax(wl_f, axis=-1)  # [L, 4]

    samples = np.empty((num_samples, L), dtype=np.float32)
    for i in range(num_samples):
        # Select mixture component per element via inverse-CDF.
        cum = np.cumsum(weights, axis=-1)
        u = rng.uniform(size=(L, 1))
        comp = np.clip((u >= cum).sum(axis=-1), 0, 3)  # [L]

        # Sample from each component.
        s0 = st_loc_f + np.abs(st_scale_f) * rng.standard_t(
            df=np.clip(st_df_f, 1e-6, 1e6)
        )
        s1 = nl_f + NORMAL_FIXED_SCALE * rng.standard_normal(L).astype(np.float32)
        s2 = _sample_negbinomial(rng, nb_tc_f, nb_lg_f)
        s3 = np.exp(
            ll_f + np.clip(ls_f, 1e-8, 20.0) * rng.standard_normal(L).astype(np.float32)
        )

        all_comp = np.stack([s0, s1, s2, s3], axis=-1)  # [L, 4]
        base = all_comp[np.arange(L), comp]

        # Affine inverse-scaling: prediction = base * scale + loc
        samples[i] = base * ss_f + sl_f

    return samples


# ======================
# ONNX Inference
# ======================


def _run_onnx(net, inputs, use_onnx_runtime):
    if use_onnx_runtime:
        return net.run(None, inputs)
    ailia_inputs = [
        inputs["target"],
        inputs["observed_mask"],
        inputs["sample_id"],
        inputs["time_id"],
        inputs["variate_id"],
        inputs["prediction_mask"],
        inputs["patch_size"],
    ]
    return net.run(ailia_inputs)


# ======================
# Plotting
# ======================


def draw_result(
    history, trues, preds_quantiles, save_path, target_name, covariates=None
):
    has_cov = covariates is not None and len(covariates) > 0
    holiday_band = None
    if has_cov:
        for name, values in covariates.items():
            uniq = np.unique(values)
            if set(uniq.tolist()).issubset({0, 1}):
                holiday_band = (name, np.asarray(values))
                break

    n_rows = 2 + (len(covariates) if has_cov else 0)
    fig = plt.figure(figsize=(12, 2.5 * n_rows))
    gs = fig.add_gridspec(n_rows, 1, hspace=0.35)

    n_hist = len(history)
    n_pred = preds_quantiles["median"].shape[-1]
    x_hist = np.arange(n_hist)
    x_pred = np.arange(n_hist, n_hist + n_pred)
    median = preds_quantiles["median"]
    q10 = preds_quantiles["q10"]
    q90 = preds_quantiles["q90"]

    # (1) Full overview
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x_hist, history, label=f"History ({n_hist} steps)", color="darkblue")
    if 0 < len(trues):
        ax.plot(
            x_pred[: len(trues)],
            trues,
            label=f"Ground Truth ({len(trues)} steps)",
            color="darkblue",
            linestyle="--",
            alpha=0.5,
        )
    ax.plot(x_pred, median, label="Forecast (median)", color="red", linestyle="--")
    ax.fill_between(x_pred, q10, q90, color="red", alpha=0.2, label="80% interval")
    ax.set_ylabel(target_name)
    ax.set_title("Overview")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # (2) Zoom
    ax = fig.add_subplot(gs[1, 0])
    zoom_hist = min(3 * n_pred, n_hist)
    xh = np.arange(-zoom_hist, 0)
    xp = np.arange(0, n_pred)
    ax.plot(xh, history[-zoom_hist:], color="darkblue", label="History")
    if 0 < len(trues):
        ax.plot(
            xp[: len(trues)],
            trues,
            "--",
            color="darkblue",
            alpha=0.5,
            label="Ground Truth",
        )
    ax.plot(xp, median, "--", color="red", label="Forecast (median)")
    ax.fill_between(xp, q10, q90, color="red", alpha=0.2, label="80% interval")
    if holiday_band is not None:
        name, vals = holiday_band
        future_vals = vals[-n_pred:]
        for i, h in enumerate(future_vals):
            if h:
                ax.axvspan(i - 0.4, i + 0.4, color="orange", alpha=0.18)
        ax.axvline(0, color="gray", linestyle=":")
        ax.set_title(f"Zoomed forecast (orange bands = {name}=1)")
    else:
        ax.axvline(0, color="gray", linestyle=":")
        ax.set_title("Zoomed forecast")
    ax.set_ylabel(target_name)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # (3+) Covariate panels
    if has_cov:
        for ax_idx, (name, values) in enumerate(covariates.items()):
            ax_i = fig.add_subplot(gs[2 + ax_idx, 0])
            ax_i.plot(np.arange(len(values)), values, color="green")
            ax_i.set_ylabel(name)
            ax_i.grid(alpha=0.3)

    fig.axes[-1].set_xlabel("Time")
    plt.savefig(save_path)


# ======================
# Forecasting
# ======================


def time_series_forecasting(net):
    data_path = args.input if isinstance(args.input, str) else args.input[0]
    df = pd.read_csv(data_path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    target = args.target
    if target is None:
        target = df.columns[0]
    logger.info(f"target column: {target}")

    feat_cols = []
    if args.feat:
        feat_cols = [c.strip() for c in args.feat.split(",") if c.strip()]
        logger.info(f"feat_dynamic_real: {feat_cols}")

    df_input = df[[target] + feat_cols].copy()

    context_len = args.context_len
    prediction_len = args.prediction_len

    if len(df_input) < context_len + prediction_len:
        logger.warning(
            "Input length %d is shorter than context_len + prediction_len=%d",
            len(df_input),
            context_len + prediction_len,
        )

    history_df = df_input.iloc[-(context_len + prediction_len) : -prediction_len]
    truth_df = df_input.iloc[-prediction_len:]

    history_vals = history_df[target].values.astype(np.float32)

    # Resolve patch_size.
    if args.patch_size == "auto":
        patch_size = _choose_patch_size(context_len, prediction_len)
        logger.info(f"Auto-selected patch_size: {patch_size}")
    else:
        patch_size = int(args.patch_size)

    n_ctx = math.ceil(context_len / patch_size)
    n_pred = math.ceil(prediction_len / patch_size)

    # Prepare covariates.
    feat_past = None
    feat_future = None
    if feat_cols:
        feat_past = np.stack(
            [history_df[c].values.astype(np.float32) for c in feat_cols], axis=0
        )
        feat_future = np.stack(
            [truth_df[c].values.astype(np.float32) for c in feat_cols], axis=0
        )

    # Build ONNX inputs.
    inputs = _build_inputs(
        history_vals, prediction_len, patch_size,
        feat_past=feat_past, feat_future=feat_future,
    )

    # Run model.
    outputs = _run_onnx(net, inputs, use_onnx_runtime=args.onnx)

    # Sample from the mixture distribution.
    rng = np.random.default_rng(args.seed)
    samples = _sample_from_mixture(
        rng, outputs, n_ctx, n_pred, patch_size, prediction_len, args.num_samples,
    )

    median = np.quantile(samples, 0.5, axis=0)
    q10 = np.quantile(samples, 0.1, axis=0)
    q90 = np.quantile(samples, 0.9, axis=0)

    truth_vals = truth_df[target].values

    covariates = None
    if feat_cols:
        covariates = {
            c: pd.concat([history_df[c], truth_df[c]]).values for c in feat_cols
        }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        savepath = get_savepath(args.savepath, data_path, ext=".png")
        logger.info(f"saved at : {savepath}")
        draw_result(
            history_vals,
            truth_vals,
            {"median": median, "q10": q10, "q90": q90},
            savepath,
            target_name=target,
            covariates=covariates,
        )

    if getattr(args, "write_json", False):
        import json

        out = {
            "median": median.tolist(),
            "q10": q10.tolist(),
            "q90": q90.tolist(),
        }
        json_path = os.path.splitext(savepath)[0] + ".json"
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"saved json at : {json_path}")

    logger.info("Script finished successfully.")


def main():
    weight_path, model_path = VERSION_SIZE_TO_FILES[(args.version, args.size)]
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    if not args.onnx:
        net = ailia.Net(model_path, weight_path, env_id=args.env_id)
    else:
        import onnxruntime

        net = onnxruntime.InferenceSession(weight_path)

    time_series_forecasting(net)


if __name__ == "__main__":
    main()
