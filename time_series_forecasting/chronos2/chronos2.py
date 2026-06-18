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

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/chronos2/"

WEIGHT_PATH = "chronos-2-p4.onnx"
MODEL_PATH = "chronos-2-p4.onnx.prototxt"

# Architectural constants of amazon/chronos-2.
PATCH_SIZE = 16
NUM_OUTPUT_PATCHES = 4   # baked into the exported ONNX (4 * 16 = 64 steps)
PRED_HORIZON = NUM_OUTPUT_PATCHES * PATCH_SIZE
QUANTILE_LEVELS = [round(0.05 * i, 2) for i in range(1, 20)] + [0.5]
# Chronos-2 emits 21 quantiles ordered as configured in the model:
# 0.01, 0.05, 0.10, 0.15, ..., 0.90, 0.95, 0.99
QUANTILES = [
    0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99,
]


# ======================
# Argument Parser Config
# ======================

parser = get_base_parser("Chronos-2", DATA_PATH, SAVE_IMAGE_PATH, fp16_support = False)
parser.add_argument("-i", "--input", type=str, default=DATA_PATH)
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
        "(passed as future covariates to Chronos-2)."
    ),
)
parser.add_argument(
    "--context_len",
    type=int,
    default=192,
    help=(
        "Context length (history). Must be a multiple of "
        f"patch_size={PATCH_SIZE}; if not, the head of the series is left-"
        "padded with NaN."
    ),
)
parser.add_argument(
    "--prediction_len",
    type=int,
    default=20,
    help=(
        "Prediction horizon length. The exported ONNX always produces "
        f"{PRED_HORIZON} steps ({NUM_OUTPUT_PATCHES}*{PATCH_SIZE}); the "
        "trailing steps are discarded."
    ),
)
parser.add_argument(
    "--onnx",
    action="store_true",
    help="Use ONNX Runtime instead of ailia SDK",
)
args = update_parser(parser)


# ======================
# Helpers
# ======================


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


def _build_inputs(history: np.ndarray, future_covariates: np.ndarray | None,
                  context_len: int, prediction_len: int):
    """Pack the konbini-style data into the (context, mask, group_ids,
    future_covariates, future_covariates_mask) tuple expected by the
    exported Chronos-2 ONNX model.

    The exported graph runs the encoder once on a batch of B time series in
    a single group:
      - row 0      : the target series (forecast target)
      - row 1..K-1 : known future covariates (one row per covariate)

    All rows share the same ``context_len`` (left-padded with NaN if the
    user-supplied series is shorter), and the same ``num_output_patches *
    patch_size`` future window. For target rows the future values are NaN
    and ``future_covariates_mask=False``; for covariate rows the future
    values are taken from the user-supplied data and the mask is True.
    """
    n_target_rows = 1
    n_cov_rows = future_covariates.shape[0] if future_covariates is not None else 0
    n_rows = n_target_rows + n_cov_rows

    # Pad context window to a multiple of patch_size.
    pad_ctx = _round_up(context_len, PATCH_SIZE)
    pad_history = np.full((n_rows, pad_ctx), np.nan, dtype=np.float32)
    pad_history[0, -context_len:] = history.astype(np.float32)
    if n_cov_rows > 0:
        pad_history[1:, -context_len:] = future_covariates[
            :, :context_len
        ].astype(np.float32)

    # Future covariate values (provided for covariate rows, NaN for target).
    fut_cov = np.full((n_rows, PRED_HORIZON), np.nan, dtype=np.float32)
    fut_mask = np.zeros((n_rows, PRED_HORIZON), dtype=bool)
    if n_cov_rows > 0:
        fut_cov[1:, :prediction_len] = future_covariates[
            :, context_len: context_len + prediction_len
        ].astype(np.float32)
        fut_mask[1:, :prediction_len] = True

    context_mask = ~np.isnan(pad_history)
    pad_history = np.nan_to_num(pad_history, nan=0.0).astype(np.float32)
    group_ids = np.zeros(n_rows, dtype=np.int64)

    return {
        "context": pad_history,
        "context_mask": context_mask,
        "group_ids": group_ids,
        "future_covariates": np.where(fut_mask, fut_cov, 0.0).astype(np.float32),
        "future_covariates_mask": fut_mask,
    }


def _run(net, inputs, use_onnx_runtime: bool):
    if use_onnx_runtime:
        return net.run(None, inputs)[0]
    ailia_inputs = [
        inputs["context"],
        inputs["context_mask"],
        inputs["group_ids"],
        inputs["future_covariates"],
        inputs["future_covariates_mask"],
    ]
    return net.run(ailia_inputs)[0]


# ======================
# Plotting
# ======================


def draw_result(history, trues, preds_quantiles, save_path, target_name,
                covariates=None):
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
    median = preds_quantiles["median"]
    q10 = preds_quantiles["q10"]
    q90 = preds_quantiles["q90"]

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(np.arange(n_hist), history, label=f"History ({n_hist} steps)", color="darkblue")
    if 0 < len(trues):
        ax.plot(np.arange(n_hist, n_hist + len(trues)), trues, "--",
                color="darkblue", alpha=0.5,
                label=f"Ground Truth ({len(trues)} steps)")
    ax.plot(np.arange(n_hist, n_hist + n_pred), median, "--", color="red",
            label="Forecast (median)")
    ax.fill_between(np.arange(n_hist, n_hist + n_pred), q10, q90,
                    color="red", alpha=0.2, label="80% interval")
    ax.set_ylabel(target_name); ax.set_title("Overview")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    zoom_hist = min(3 * n_pred, n_hist)
    xh = np.arange(-zoom_hist, 0)
    xp = np.arange(0, n_pred)
    ax.plot(xh, history[-zoom_hist:], color="darkblue", label="History")
    if 0 < len(trues):
        ax.plot(xp[: len(trues)], trues, "--", color="darkblue", alpha=0.5,
                label="Ground Truth")
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
    ax.set_ylabel(target_name); ax.legend(loc="upper left"); ax.grid(alpha=0.3)

    if has_cov:
        for ax_idx, (name, values) in enumerate(covariates.items()):
            ax_i = fig.add_subplot(gs[2 + ax_idx, 0])
            ax_i.plot(np.arange(len(values)), values, color="green")
            ax_i.set_ylabel(name); ax_i.grid(alpha=0.3)

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
        logger.info(f"future covariates: {feat_cols}")

    cols_needed = [target] + feat_cols
    df_input = df[cols_needed].iloc[
        -(args.context_len + args.prediction_len) :
    ].copy()

    history_df = df_input.iloc[: args.context_len]
    truth_df = df_input.iloc[args.context_len:]

    history_vals = history_df[target].values.astype(np.float32)
    truth_vals = truth_df[target].values.astype(np.float32)

    if feat_cols:
        # Stack covariates as rows: shape (n_cov, context_len + prediction_len)
        covariate_arr = np.stack(
            [df_input[c].values.astype(np.float32) for c in feat_cols], axis=0
        )
    else:
        covariate_arr = None

    if args.prediction_len > PRED_HORIZON:
        logger.warning(
            "prediction_len=%d exceeds the export's fixed horizon of %d steps; "
            "re-export the ONNX with a larger --num_output_patches.",
            args.prediction_len, PRED_HORIZON,
        )

    inputs = _build_inputs(
        history=history_vals,
        future_covariates=covariate_arr,
        context_len=args.context_len,
        prediction_len=min(args.prediction_len, PRED_HORIZON),
    )

    output = _run(net, inputs, use_onnx_runtime=args.onnx)
    # output shape: [n_rows, num_quantiles=21, PRED_HORIZON]
    target_quantiles = output[0, :, : args.prediction_len]

    median = target_quantiles[QUANTILES.index(0.5)]
    q10 = target_quantiles[QUANTILES.index(0.1)]
    q90 = target_quantiles[QUANTILES.index(0.9)]

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
            history_vals, truth_vals,
            {"median": median, "q10": q10, "q90": q90},
            savepath, target_name=target, covariates=covariates,
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
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    time_series_forecasting(net)


if __name__ == "__main__":
    main()
