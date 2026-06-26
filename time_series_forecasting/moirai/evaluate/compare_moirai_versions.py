"""Compare zero-shot forecasts from Moirai-1.0-R / Moirai-1.1-R /
Moirai-2.0-R / Chronos-2 on the same konbini sample data.

This script does **not** use the exported ONNX weights — it loads each model
directly via uni2ts (Moirai) or chronos-forecasting (Chronos-2). The goal is
to compare model families, not to benchmark the ONNX export.

Notes on weights / licenses
---------------------------
- Moirai-1.0-R is loaded from the **Apache-2.0** revision (last commit
  before the 2024-03-28 relicense). Hugging Face still serves the legacy
  ``model.ckpt`` from these revisions.
- Moirai-1.1-R and Moirai-2.0-R weights are CC-BY-NC-4.0; they are pulled
  here only for non-commercial side-by-side analysis.
- Chronos-2 (``amazon/chronos-2``) is **Apache-2.0** and can be used
  alongside the Apache-2.0 Moirai-1.0-R weights without a license switch.

We use ``--size large`` for Moirai-1.x (the largest publicly released size).
Moirai-2.0-R is only published in ``small``; Chronos-2 ships as a single
~120M-parameter model (no size variants).
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "input.csv"


# Last commit before the 2024-03-28 license change for Moirai-1.0-R.
APACHE_REVISIONS = {
    "small": "4a950dea3b2c38b9675082959109e1b36d40ab16",
    "base": "03e0d0f88ea7dee295d398d102fb582494b549e1",
    "large": "bc5caba1947b76c9efd513ada3675b8d5006f09a",
}


def load_moirai_10(size: str):
    """Load Moirai-1.0-R from the Apache-2.0 era model.ckpt."""
    from uni2ts.model.moirai import MoiraiModule

    repo = f"Salesforce/moirai-1.0-R-{size}"
    revision = APACHE_REVISIONS[size]
    path = hf_hub_download(repo_id=repo, filename="model.ckpt", revision=revision)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    module = MoiraiModule(**ck["hyper_parameters"]["module_kwargs"])
    module.load_state_dict(
        {
            k.removeprefix("module."): v
            for k, v in ck["state_dict"].items()
            if k.startswith("module.")
        },
        strict=True,
    )
    module.eval()
    return module


def load_moirai_11(size: str):
    """Load Moirai-1.1-R via the standard from_pretrained path."""
    from uni2ts.model.moirai import MoiraiModule

    return MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size}").eval()


def load_moirai_2(size: str = "small"):
    """Load Moirai-2.0-R."""
    from uni2ts.model.moirai2 import Moirai2Module

    return Moirai2Module.from_pretrained(f"Salesforce/moirai-2.0-R-{size}").eval()


def forecast_moirai_v1(module, df_in, target, feat_cols, ctx, pred, patch_size, num_samples, seed):
    """Run a Moirai-1.x forecast and return the median + 80% interval."""
    from uni2ts.model.moirai import MoiraiForecast

    ds = PandasDataset(
        df_in, target=target, feat_dynamic_real=feat_cols if feat_cols else None
    )
    _, tt = split(ds, offset=-pred)
    td = tt.generate_instances(prediction_length=pred, windows=1, distance=pred)
    fm = MoiraiForecast(
        module=module,
        prediction_length=pred,
        context_length=ctx,
        patch_size=patch_size,
        num_samples=num_samples,
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )
    fm.eval()
    torch.manual_seed(seed)
    np.random.seed(seed)
    fc = list(fm.create_predictor(batch_size=1).predict(td.input))[0]
    s = np.asarray(fc.samples)
    return (
        np.quantile(s, 0.5, axis=0),
        np.quantile(s, 0.1, axis=0),
        np.quantile(s, 0.9, axis=0),
    )


def forecast_chronos2(df_in, df_index, target, feat_cols, pred, ctx, seed):
    """Run a Chronos-2 forecast.

    Chronos-2 directly produces quantile forecasts. We expose the 0.1 / 0.5
    / 0.9 quantiles and treat the median as the point estimate.
    """
    from chronos import Chronos2Pipeline

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Format data into the long DataFrame Chronos-2 expects.
    history = df_in.iloc[:-pred].copy()
    history = history.rename(columns={target: "target"})
    history["timestamp"] = df_index[: len(history)]
    history["item_id"] = "series"

    future = df_in.iloc[-pred:][feat_cols].copy() if feat_cols else pd.DataFrame()
    future["timestamp"] = df_index[-pred:]
    future["item_id"] = "series"

    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    result = pipe.predict_df(
        df=history,
        future_df=future if feat_cols else None,
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
        prediction_length=pred,
        context_length=ctx,
    )
    median = result["0.5"].to_numpy()
    q10 = result["0.1"].to_numpy()
    q90 = result["0.9"].to_numpy()
    return median, q10, q90


def forecast_moirai_v2(module, df_in, target, feat_cols, ctx, pred, num_samples, seed):
    """Run a Moirai-2.0-R forecast.

    Moirai-2.0-R outputs quantile predictions directly instead of samples.
    The ``quantile_levels`` configured at training time are
    ``(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)``; we read off 0.1/0.5/0.9.
    """
    from uni2ts.model.moirai2 import Moirai2Forecast

    ds = PandasDataset(
        df_in, target=target, feat_dynamic_real=feat_cols if feat_cols else None
    )
    _, tt = split(ds, offset=-pred)
    td = tt.generate_instances(prediction_length=pred, windows=1, distance=pred)
    fm = Moirai2Forecast(
        module=module,
        prediction_length=pred,
        context_length=ctx,
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )
    fm.eval()
    torch.manual_seed(seed)
    np.random.seed(seed)
    fc = list(fm.create_predictor(batch_size=1).predict(td.input))[0]

    # gluonts QuantileForecast: fc.forecast_array shape [num_quantiles, pred_len]
    q_levels = list(module.quantile_levels)
    q_arr = np.asarray(fc.forecast_array)
    median = q_arr[q_levels.index(0.5)]
    q10 = q_arr[q_levels.index(0.1)]
    q90 = q_arr[q_levels.index(0.9)]
    return median, q10, q90


def metrics(median, q10, q90, truth):
    truth = np.asarray(truth)
    mae = float(np.mean(np.abs(median - truth)))
    rmse = float(np.sqrt(np.mean((median - truth) ** 2)))
    coverage80 = float(np.mean((truth >= q10) & (truth <= q90)))
    return {"MAE": mae, "RMSE": rmse, "PI80_coverage": coverage80}


def main():
    parser = argparse.ArgumentParser(description="Compare Moirai versions")
    parser.add_argument("--data", type=str, default=str(DATA_PATH))
    parser.add_argument("--target", type=str, default="sales")
    parser.add_argument("--feat", type=str, default="temperature,is_holiday")
    parser.add_argument("--context_len", type=int, default=200)
    parser.add_argument("--prediction_len", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save",
        type=str,
        default=str(HERE / "output_compare.png"),
        help="Where to save the comparison plot",
    )
    parser.add_argument(
        "--size_v1",
        type=str,
        default="large",
        choices=["small", "base", "large"],
        help="Moirai-1.x size (only one set of weights is downloaded per call)",
    )
    parser.add_argument(
        "--patch_v10",
        type=str,
        default="32",
        help="patch_size for Moirai-1.0-R: 'auto' or one of 8/16/32/64/128",
    )
    parser.add_argument(
        "--patch_v11",
        type=str,
        default="16",
        help="patch_size for Moirai-1.1-R: 'auto' or one of 8/16/32/64/128",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    feat_cols = [c.strip() for c in args.feat.split(",") if c.strip()]
    df_in = df[[args.target] + feat_cols].iloc[-(args.context_len + args.prediction_len):]
    df_index = df_in.index
    truth = df[args.target].iloc[-args.prediction_len:].values
    history = df[args.target].iloc[-(args.context_len + args.prediction_len):-args.prediction_len].values
    future_holiday = (
        df["is_holiday"].iloc[-args.prediction_len:].values
        if "is_holiday" in feat_cols
        else None
    )

    def _ps(s):
        return s if s == "auto" else int(s)

    runs = []

    # Moirai 1.0-R (Apache-2.0 era)
    print(f"--- Moirai-1.0-R-{args.size_v1} (Apache-2.0) -- patch={args.patch_v10} ---", flush=True)
    m10 = load_moirai_10(args.size_v1)
    median, q10, q90 = forecast_moirai_v1(
        m10, df_in, args.target, feat_cols,
        args.context_len, args.prediction_len, _ps(args.patch_v10),
        args.num_samples, args.seed,
    )
    runs.append({
        "name": f"Moirai-1.0-R-{args.size_v1} (Apache-2.0, p={args.patch_v10})",
        "median": median, "q10": q10, "q90": q90,
        "metrics": metrics(median, q10, q90, truth),
    })
    del m10

    # Moirai 1.1-R
    print(f"--- Moirai-1.1-R-{args.size_v1} (CC-BY-NC-4.0) -- patch={args.patch_v11} ---", flush=True)
    m11 = load_moirai_11(args.size_v1)
    median, q10, q90 = forecast_moirai_v1(
        m11, df_in, args.target, feat_cols,
        args.context_len, args.prediction_len, _ps(args.patch_v11),
        args.num_samples, args.seed,
    )
    runs.append({
        "name": f"Moirai-1.1-R-{args.size_v1} (CC-BY-NC-4.0, p={args.patch_v11})",
        "median": median, "q10": q10, "q90": q90,
        "metrics": metrics(median, q10, q90, truth),
    })
    del m11

    # Moirai 2.0-R-small (only public size)
    print("--- Moirai-2.0-R-small (CC-BY-NC-4.0, fixed patch=16) ---", flush=True)
    m2 = load_moirai_2("small")
    median, q10, q90 = forecast_moirai_v2(
        m2, df_in, args.target, feat_cols,
        args.context_len, args.prediction_len, args.num_samples, args.seed,
    )
    runs.append({
        "name": "Moirai-2.0-R-small (CC-BY-NC-4.0, p=16 fixed)",
        "median": median, "q10": q10, "q90": q90,
        "metrics": metrics(median, q10, q90, truth),
    })
    del m2

    # Chronos-2 (Apache-2.0) — single published model, ~120M params.
    print("--- Chronos-2 (Apache-2.0, amazon/chronos-2) ---", flush=True)
    median, q10, q90 = forecast_chronos2(
        df_in, df_index, args.target, feat_cols,
        args.prediction_len, args.context_len, args.seed,
    )
    runs.append({
        "name": "Chronos-2 (Apache-2.0, amazon/chronos-2)",
        "median": median, "q10": q10, "q90": q90,
        "metrics": metrics(median, q10, q90, truth),
    })

    # Print metrics table.
    print()
    print(f"{'model':<50} | {'MAE':>7} | {'RMSE':>7} | {'PI80':>6} | {'gap':>6}")
    print("-" * 90)
    for r in runs:
        if future_holiday is not None and future_holiday.sum() > 0 and (future_holiday == 0).sum() > 0:
            gap = float(r["median"][future_holiday == 1].mean()) - float(r["median"][future_holiday == 0].mean())
            gap_str = f"{gap:+6.2f}"
        else:
            gap_str = "  n/a"
        print(
            f"{r['name']:<50} | {r['metrics']['MAE']:>7.2f} | {r['metrics']['RMSE']:>7.2f} | "
            f"{r['metrics']['PI80_coverage']:>5.0%} | {gap_str}"
        )

    # Plot.
    n_pred = len(truth)
    zoom_hist = min(3 * n_pred, len(history))
    xh = np.arange(-zoom_hist, 0)
    xp = np.arange(0, n_pred)

    fig, axes = plt.subplots(len(runs), 1, figsize=(12, 3 * len(runs)), sharex=True, sharey=True)
    if len(runs) == 1:
        axes = [axes]
    for ax, r in zip(axes, runs):
        ax.plot(xh, history[-zoom_hist:], color="darkblue", label="History")
        ax.plot(xp, truth, "--", color="darkblue", alpha=0.5, label="Ground Truth")
        ax.plot(xp, r["median"], "--", color="red", label="Forecast (median)")
        ax.fill_between(xp, r["q10"], r["q90"], color="red", alpha=0.2, label="80% interval")
        if future_holiday is not None:
            for i, h in enumerate(future_holiday):
                if h:
                    ax.axvspan(i - 0.4, i + 0.4, color="orange", alpha=0.18)
        ax.axvline(0, color="gray", linestyle=":")
        m = r["metrics"]
        ax.set_title(
            f"{r['name']}    "
            f"MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  PI80={m['PI80_coverage']:.0%}"
        )
        ax.set_ylabel(args.target)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel(
        "Days from forecast start"
        + (" (orange = is_holiday=1)" if future_holiday is not None else "")
    )
    plt.tight_layout()
    plt.savefig(args.save)
    print(f"\nsaved: {args.save}")


if __name__ == "__main__":
    main()
