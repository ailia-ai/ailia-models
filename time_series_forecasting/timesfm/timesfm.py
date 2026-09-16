import sys
from typing import Literal, Sequence
import warnings
from logging import getLogger

import ailia

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_PATH = "timesfm-1.0-200m.onnx"
MODEL_PATH = "timesfm-1.0-200m.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/timesfm/"

DATA_PATH = "ETTh1.csv"
SAVE_IMAGE_PATH = "output.png"

BATCH_SIZE = 32


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("TimesFM", DATA_PATH, SAVE_IMAGE_PATH)
parser.add_argument("-i", "--input", type=str, default=DATA_PATH)
parser.add_argument("--target", type=str, default=None)
parser.add_argument(
    "--context_len",
    type=int,
    default=512,
    help="context length, max context length of 512.",
)
parser.add_argument(
    "--horizon_len",
    type=int,
    default=128,
    help="time series length to forecast, it must not exceed context len.",
)
parser.add_argument(
    "--forecast_horizon",
    type=int,
    default=None,
    help="How many ends of the time series of data to forecast.",
)
parser.add_argument(
    "--forecast_mode",
    default="median",
    choices=["mean", "median"],
    help="forecast_mode: mean or median",
)
parser.add_argument(
    "--onnx",
    action="store_true",
    help="By default, the ailia SDK is used, but with this option, you can switch to using ONNX Runtime",
)
parser.add_argument(
    "-w",
    "--write_json",
    action="store_true",
    help="Flag to output results to json file.",
)
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================


def draw_result(history, trues, preds, save_path):
    plt.figure(figsize=(12, 4))

    # Plotting the first time series from history
    plt.plot(
        range(len(history)),
        history,
        label=f"History ({len(history)} timesteps)",
        c="darkblue",
    )

    offset = len(history)
    if 0 < len(trues):
        plt.plot(
            range(offset, offset + len(trues)),
            trues,
            label=f"Ground Truth ({len(trues)} timesteps)",
            color="darkblue",
            linestyle="--",
            alpha=0.5,
        )
    plt.plot(
        range(offset, offset + len(preds)),
        preds,
        label=f"Forecast ({len(preds)} timesteps)",
        color="red",
        linestyle="--",
    )

    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.legend(fontsize=14)

    plt.savefig(save_path)


# ======================
# Main functions
# ======================


def preprocess(
    inputs: Sequence[np.ndarray], freq: Sequence[int], context_len, horizon_len
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:

    input_ts, input_padding, inp_freq = [], [], []

    pmap_pad = ((len(inputs) - 1) // BATCH_SIZE + 1) * BATCH_SIZE - len(inputs)

    for i, ts in enumerate(inputs):
        input_len = ts.shape[0]
        padding = np.zeros(shape=(input_len + horizon_len,), dtype=float)
        if input_len < context_len:
            num_front_pad = context_len - input_len
            ts = np.concatenate(
                [np.zeros(shape=(num_front_pad,), dtype=float), ts], axis=0
            )
            padding = np.concatenate(
                [np.ones(shape=(num_front_pad,), dtype=float), padding], axis=0
            )
        elif input_len > context_len:
            ts = ts[-context_len:]
            padding = padding[-(context_len + horizon_len) :]

        input_ts.append(ts)
        input_padding.append(padding)
        inp_freq.append(freq[i])

    # Padding the remainder batch.
    for _ in range(pmap_pad):
        input_ts.append(input_ts[-1])
        input_padding.append(input_padding[-1])
        inp_freq.append(inp_freq[-1])

    return (
        np.stack(input_ts, axis=0),
        np.stack(input_padding, axis=0),
        np.array(inp_freq).astype(np.int32).reshape(-1, 1),
        pmap_pad,
    )


def decode(
    net,
    input_ts: np.ndarray,
    paddings: np.ndarray,
    freq: np.ndarray,
    horizon_len: int,
    max_len: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    final_out = input_ts
    full_outputs = []

    output_patch_len = 128
    num_decode_patches = (horizon_len + output_patch_len - 1) // output_patch_len
    for _ in range(num_decode_patches):
        current_padding = paddings[:, 0 : final_out.shape[1]]
        input_ts = final_out[:, -max_len:]
        input_padding = current_padding[:, -max_len:]

        # feedforward
        if not args.onnx:
            output = net.predict([input_ts, input_padding, freq])
        else:
            output = net.run(
                None,
                {"input_ts": input_ts, "input_padding": input_padding, "freq": freq},
            )
        fprop_outputs = output[0]

        # (full batch, last patch, output_patch_len, index of mean forecast = 0)
        new_ts = fprop_outputs[:, -1, :output_patch_len, 0]
        new_full_ts = fprop_outputs[:, -1, :output_patch_len, :]
        # (full batch, last patch, output_patch_len, all output indices)
        full_outputs.append(new_full_ts)
        final_out = np.concatenate([final_out, new_ts], axis=-1)

    # `full_outputs` indexing starts at the forecast horizon.
    full_outputs = np.concatenate(full_outputs, axis=1)[:, 0:horizon_len, :]

    return (full_outputs[:, :, 0], full_outputs)


def forecast(
    net,
    inputs,
    freq=None,
    context_len=512,
    horizon_len=128,
    forecast_mode: Literal["mean", "median"] = "median",
):
    """
    Returns:
        A tuple for np.array:
        - the mean forecast of size (# inputs, # forecast horizon),
        - the full forecast (mean + quantiles) of size
    """
    inputs = [np.array(ts)[-context_len:] for ts in inputs]
    if freq is None:
        freq = [0] * len(inputs)

    input_ts, input_padding, inp_freq, pmap_pad = preprocess(
        inputs, freq, context_len, horizon_len
    )

    mean_outputs = []
    full_outputs = []
    for i in range(input_ts.shape[0] // BATCH_SIZE):
        input_ts_in = np.array(
            input_ts[i * BATCH_SIZE : (i + 1) * BATCH_SIZE],
            dtype=np.float32,
        )
        input_padding_in = np.array(
            input_padding[i * BATCH_SIZE : (i + 1) * BATCH_SIZE],
            dtype=np.float32,
        )
        inp_freq_in = np.array(
            inp_freq[
                i * BATCH_SIZE : (i + 1) * BATCH_SIZE,
                :,
            ],
            dtype=int,
        )

        mean_output, full_output = decode(
            net,
            input_ts=input_ts_in,
            paddings=input_padding_in,
            freq=inp_freq_in,
            horizon_len=horizon_len,
        )
        mean_outputs.append(mean_output)
        full_outputs.append(full_output)

    mean_outputs = np.concatenate(mean_outputs, axis=0)
    full_outputs = np.concatenate(full_outputs, axis=0)

    if pmap_pad > 0:
        mean_outputs = mean_outputs[:-pmap_pad, ...]
        full_outputs = full_outputs[:-pmap_pad, ...]

    if forecast_mode == "mean":
        point_forecast, experimental_quantile_forecast = mean_outputs, full_outputs
    elif forecast_mode == "median":
        median_index = 4
        point_forecast, experimental_quantile_forecast = (
            full_outputs[:, :, 1 + median_index],
            full_outputs,
        )
    else:
        raise ValueError(
            "Unsupported point forecast mode:"
            f" {forecast_mode}. Use 'mean' or 'median'."
        )

    return point_forecast, experimental_quantile_forecast


def time_series_forecasting(net):
    data_path = args.input[0]
    context_length = args.context_len
    horizon_len = args.horizon_len
    forecast_horizon = args.forecast_horizon
    forecast_mode = args.forecast_mode
    target = args.target

    df = pd.read_csv(data_path)

    if target is None:
        target = df.columns[-1]

    target = int(target) if target.isdigit() else target
    if isinstance(target, str):
        logger.info("target column: %s" % target)
        df = df[[target]]
    else:
        logger.info("target column index: %s" % target)
        df = df.iloc[:, [target]]

    df_train = (
        df[-(context_length + forecast_horizon) : -forecast_horizon]
        if forecast_horizon
        else df[-context_length:]
    )
    df_true = df[-forecast_horizon:] if forecast_horizon else df[:0]

    history = df_train.values.T
    trues = df_true.values.T

    preds, _ = forecast(
        net,
        history,
        context_len=context_length,
        horizon_len=horizon_len,
        forecast_mode=forecast_mode,
    )

    history = history.reshape(-1)
    trues = trues.reshape(-1)
    preds = preds.reshape(-1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        # plot result
        savepath = get_savepath(args.savepath, data_path, ext=".png")
        logger.info(f"saved at : {savepath}")
        draw_result(history, trues, preds, savepath)

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # net initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime

        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    # forecasting
    time_series_forecasting(net)


if __name__ == "__main__":
    main()
