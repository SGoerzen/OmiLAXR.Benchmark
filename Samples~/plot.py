"""
SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2025 Sergej Görzen <sergej.goerzen@gmail.com>
This file is part of OmiLAXR.

Pretty compare plots for Unity Benchmark CSVs (auto-detects per-frame vs already-binned),
with optional visual-only smoothing (does NOT affect summary.csv).

Supports two CSV shapes:
A) Per-frame: t_s, phase(optional), frame_ms, fps, used_mem_mb, ...
B) Binned:    t_s, phase, frame_ms, fps, used_mem_mb, frame_ms_p95, frame_ms_max, spike_ratio, frames_in_bin

Outputs time series plots and a summary CSV per run.
"""


import argparse

import os

import sys

from pathlib import Path

from typing import List, Tuple, Dict


import pandas as pd

import matplotlib.pyplot as plt


from typing import List, Tuple, Dict, Optional



def parse_args():

    """Parse CLI arguments for plotting and aggregation."""
    ap = argparse.ArgumentParser()


    ap.add_argument("--csv", action="append", required=True,

                    help="Path to a Benchmark CSV. Use multiple times for multiple runs.")

    ap.add_argument("--label", action="append", default=None,

                    help="Label for each --csv (use multiple times, same count as --csv).")

    ap.add_argument("--labels", default=None,

                    help="Comma-separated labels for the runs (alternative to multiple --label).")


    ap.add_argument("--outdir", default="plots_pretty", help="Output folder for PNGs.")

    ap.add_argument("--phase", choices=["all", "warmup", "measure"], default="measure",

                    help="Filter phase. Default: measure (t_s>=0).")

    ap.add_argument("--bin", type=float, default=1.0,

                    help="Aggregation bin size in seconds if CSV is per-frame (default: 1.0).")


    ap.add_argument("--p95", action="store_true",

                    help="Also plot p95 line if available (recommended).")

    ap.add_argument("--max", action="store_true",

                    help="Also plot max line if available (recommended for spikes).")


    ap.add_argument("--tmin", type=float, default=None,

                    help="Optional lower time bound in seconds (e.g., -30).")

    ap.add_argument("--tmax", type=float, default=None,

                    help="Optional upper time bound in seconds (e.g., 120).")


    ap.add_argument("--downsample", type=int, default=1,

                    help="Keep every Nth row before aggregating (only for per-frame CSVs). Default: 1.")

    ap.add_argument("--clip_pct", type=float, default=0.0,

                    help="Optional clip at percentile for plotting (e.g., 99.5). 0 disables.")

    ap.add_argument("--spike_ms", type=float, default=16.7,

                    help="Frame time threshold (ms) for spike ratio (only used for per-frame aggregation).")

    ap.add_argument("--only", default=None,

                    help="Comma-separated metrics to plot. Default plots frame_ms,fps,used_mem_mb (+spike_ratio if present).")


    ap.add_argument("--yscale_frame", choices=["linear", "log"], default="linear",

                    help="Y-scale for frame_ms plot (linear/log).")



    ap.add_argument("--smooth", type=int, default=0,

                    help="Smooth plotted series over this many *binned points*. 0 disables. "

                         "Example: --smooth 11 (≈11s if bin=1s).")

    ap.add_argument("--smooth_method", choices=["rolling", "ewm"], default="rolling",

                    help="Smoothing method: rolling (moving average) or ewm (exponential).")

    ap.add_argument("--smooth_center", action="store_true",

                    help="Center the rolling window (rolling only).")

    ap.add_argument("--show_raw", action="store_true",

                    help="If smoothing is enabled, also show the raw binned line (faint/dashed).")


    ap.add_argument("--overhead", action="store_true",

                    help="Also write overhead plots (OmiLAXR - Baseline). Requires exactly 2 CSVs.")

    ap.add_argument("--overhead_baseline_index", type=int, default=0,

                    help="Which run index is baseline for overhead (default: 0).")

    ap.add_argument("--overhead_other_index", type=int, default=1,

                    help="Which run index is compared against baseline (default: 1).")


    return ap.parse_args()



def read_csv(path: str) -> pd.DataFrame:

    """Load a Benchmark CSV and normalize required columns."""
    df = pd.read_csv(path)

    if "t_s" not in df.columns:

        raise ValueError(f"{path}: missing required column 't_s'")


    df = df.copy()

    df["t_s"] = pd.to_numeric(df["t_s"], errors="coerce")

    df = df.dropna(subset=["t_s"])

    df = df.sort_values("t_s")



    if "phase" in df.columns:

        df["phase"] = df["phase"].astype(str)

    else:

        df["phase"] = df["t_s"].apply(lambda x: "warmup" if x < 0 else "measure")



    for c in df.columns:

        if c in ("t_s", "phase"):

            continue

        df[c] = pd.to_numeric(df[c], errors="coerce")


    return df


def compute_overhead_df(base: pd.DataFrame, other: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:


    """Compute metric deltas between baseline and other run."""
    merged = pd.merge(base, other, on="t_s", how="inner", suffixes=("_base", "_other"))

    out = pd.DataFrame()

    out["t_s"] = merged["t_s"]


    for m in metrics:

        mean = f"{m}_mean"

        p95  = f"{m}_p95"

        mx   = f"{m}_max"


        if f"{mean}_base" in merged.columns and f"{mean}_other" in merged.columns:

            out[f"{mean}_delta"] = merged[f"{mean}_other"] - merged[f"{mean}_base"]


        if f"{p95}_base" in merged.columns and f"{p95}_other" in merged.columns:

            out[f"{p95}_delta"] = merged[f"{p95}_other"] - merged[f"{p95}_base"]


        if f"{mx}_base" in merged.columns and f"{mx}_other" in merged.columns:

            out[f"{mx}_delta"] = merged[f"{mx}_other"] - merged[f"{mx}_base"]


    return out.sort_values("t_s")



def save_overhead_plot(delta_df: pd.DataFrame, metric: str, outpath: Path,

    """Save an overhead plot for a single metric."""
                       show_p95: bool, show_max: bool, clip_pct: float):

    plt.figure()


    mean_col = f"{metric}_mean_delta"

    p95_col  = f"{metric}_p95_delta"

    max_col  = f"{metric}_max_delta"


    if mean_col in delta_df.columns:

        plt.plot(delta_df["t_s"], clip_for_plot(delta_df[mean_col], clip_pct), label="Δ mean")


    if show_p95 and p95_col in delta_df.columns:

        plt.plot(delta_df["t_s"], clip_for_plot(delta_df[p95_col], clip_pct), linestyle="--", label="Δ p95")


    if show_max and max_col in delta_df.columns:

        plt.plot(delta_df["t_s"], clip_for_plot(delta_df[max_col], clip_pct), linestyle=":", label="Δ max")


    plt.axhline(0.0, linewidth=1)

    plt.xlabel("t (s)")

    plt.ylabel(f"Δ {metric} (other - baseline)")

    plt.title(f"Overhead: Δ {metric} over time")

    plt.legend(loc="best", fontsize="small")

    plt.tight_layout()

    plt.savefig(outpath, dpi=200)

    plt.close()



def filter_phase(df: pd.DataFrame, phase: str) -> pd.DataFrame:

    """Filter rows by phase (warmup/measure/all)."""
    if phase == "all":

        return df

    if phase == "measure":

        return df[(df["phase"] == "measure") & (df["t_s"] >= 0)]

    if phase == "warmup":

        return df[(df["phase"] == "warmup") | (df["t_s"] < 0)]

    return df



def apply_downsample(df: pd.DataFrame, n: int) -> pd.DataFrame:

    """Downsample a DataFrame by keeping every Nth row."""
    if n <= 1:

        return df

    return df.iloc[::n, :]



def clip_for_plot(s: pd.Series, clip_pct: float) -> pd.Series:

    """Clip values at a percentile for plotting."""
    if clip_pct <= 0:

        return s

    upper = s.quantile(clip_pct / 100.0)

    return s.clip(upper=upper)



def is_already_binned(df: pd.DataFrame) -> bool:

    """Detect whether the CSV is already binned."""
    for c in ("frames_in_bin", "frame_ms_p95", "frame_ms_max", "spike_ratio"):

        if c in df.columns:

            return True

    return False



def aggregate_per_bin_per_frame(df: pd.DataFrame, bin_s: float, spike_ms: float, metrics: List[str]) -> pd.DataFrame:

    """Aggregate per-frame samples into time bins."""
    if bin_s <= 0:

        raise ValueError("--bin must be > 0")


    tmp = df.copy()

    tmp["t_bin"] = (tmp["t_s"] / bin_s).astype(int) * bin_s



    if "frame_ms" in tmp.columns and tmp["frame_ms"].notna().any():

        tmp["spike"] = (tmp["frame_ms"] > spike_ms).astype(int)

    else:

        tmp["spike"] = 0


    aggs: Dict[str, tuple] = {}

    aggs["n"] = ("t_s", "count")

    aggs["spike_count"] = ("spike", "sum")


    for m in metrics:

        if m not in tmp.columns:

            continue

        aggs[f"{m}_mean"] = (m, "mean")

        aggs[f"{m}_p95"] = (m, lambda s: s.quantile(0.95))

        aggs[f"{m}_max"] = (m, "max")


    out = tmp.groupby("t_bin").agg(**aggs).reset_index().rename(columns={"t_bin": "t_s"})

    out["spike_ratio"] = out["spike_count"] / out["n"].clip(lower=1)

    out["frames_in_bin"] = out["n"]

    return out.sort_values("t_s")



def normalize_binned(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:

    """Normalize already-binned CSV columns to expected schema."""
    out = pd.DataFrame()

    out["t_s"] = df["t_s"]

    out["phase"] = df["phase"]


    for m in metrics:

        if m in df.columns:

            out[f"{m}_mean"] = df[m]


        p95_col = f"{m}_p95"

        max_col = f"{m}_max"

        if p95_col in df.columns:

            out[p95_col] = df[p95_col]

        if max_col in df.columns:

            out[max_col] = df[max_col]


    if "spike_ratio" in df.columns:

        out["spike_ratio"] = df["spike_ratio"]

    if "frames_in_bin" in df.columns:

        out["frames_in_bin"] = df["frames_in_bin"]


    return out.sort_values("t_s")



def smooth_series(s: pd.Series, window: int, method: str, center: bool) -> pd.Series:

    """Smooth a series using rolling or exponential average."""
    if window <= 1:

        return s

    if method == "ewm":


        return s.ewm(span=window, adjust=False, min_periods=max(2, window // 3)).mean()


    return s.rolling(window=window, center=center, min_periods=max(2, window // 3)).mean()



def plot_line(ax, x, y, label: str, linestyle: str = "-", alpha: float = 1.0):

    """Plot a single line with label and styling."""
    ax.plot(x, y, label=label, linestyle=linestyle, alpha=alpha)



def save_frame_ms_plot(runs: List[Tuple[str, pd.DataFrame]], outpath: Path,

    """Save frame time plot with mean/p95/max options."""
                       show_p95: bool, show_max: bool, clip_pct: float, yscale: str,

                       smooth: int, smooth_method: str, smooth_center: bool, show_raw: bool):

    fig, ax = plt.subplots()


    for label, df in runs:

        if "frame_ms_mean" not in df.columns:

            continue


        y_raw_mean = clip_for_plot(df["frame_ms_mean"], clip_pct)

        y_mean = smooth_series(y_raw_mean, smooth, smooth_method, smooth_center) if smooth > 1 else y_raw_mean


        if smooth > 1 and show_raw:

            plot_line(ax, df["t_s"], y_raw_mean, label=f"{label} (mean, raw)", linestyle="--", alpha=0.35)


        suffix = f", smooth={smooth}" if smooth > 1 else ""

        plot_line(ax, df["t_s"], y_mean, label=f"{label} (mean{suffix})", linestyle="-", alpha=1.0)


        if show_p95 and "frame_ms_p95" in df.columns:

            y_raw_p95 = clip_for_plot(df["frame_ms_p95"], clip_pct)

            y_p95 = smooth_series(y_raw_p95, smooth, smooth_method, smooth_center) if smooth > 1 else y_raw_p95

            if smooth > 1 and show_raw:

                plot_line(ax, df["t_s"], y_raw_p95, label=f"{label} (p95, raw)", linestyle="--", alpha=0.25)

            plot_line(ax, df["t_s"], y_p95, label=f"{label} (p95{suffix})", linestyle="--", alpha=1.0)


        if show_max and "frame_ms_max" in df.columns:

            y_raw_max = clip_for_plot(df["frame_ms_max"], clip_pct)

            y_max = smooth_series(y_raw_max, smooth, smooth_method, smooth_center) if smooth > 1 else y_raw_max

            if smooth > 1 and show_raw:

                plot_line(ax, df["t_s"], y_raw_max, label=f"{label} (max, raw)", linestyle=":", alpha=0.25)

            plot_line(ax, df["t_s"], y_max, label=f"{label} (max{suffix})", linestyle=":", alpha=1.0)


    ax.set_xlabel("t (s)")

    ax.set_ylabel("frame_ms")

    ax.set_yscale(yscale)


    title = "frame_ms over time"

    if clip_pct > 0:

        title += f" (clipped at p{clip_pct})"

    if smooth > 1:

        title += f" | smoothed ({smooth_method}, window={smooth})"


    ax.set_title(title)

    ax.legend(loc="best", fontsize="small")

    fig.tight_layout()

    fig.savefig(outpath, dpi=200)

    plt.close(fig)



def save_simple_metric_plot(runs: List[Tuple[str, pd.DataFrame]], metric: str, outpath: Path,

    """Save a plot for a single metric."""
                            show_p95: bool, show_max: bool, clip_pct: float,

                            smooth: int, smooth_method: str, smooth_center: bool, show_raw: bool):

    fig, ax = plt.subplots()


    mean_col = f"{metric}_mean"

    p95_col = f"{metric}_p95"

    max_col = f"{metric}_max"


    for label, df in runs:

        if mean_col not in df.columns:

            continue


        y_raw_mean = clip_for_plot(df[mean_col], clip_pct)

        y_mean = smooth_series(y_raw_mean, smooth, smooth_method, smooth_center) if smooth > 1 else y_raw_mean


        if smooth > 1 and show_raw:

            plot_line(ax, df["t_s"], y_raw_mean, label=f"{label} (mean, raw)", linestyle="--", alpha=0.35)


        suffix = f", smooth={smooth}" if smooth > 1 else ""

        plot_line(ax, df["t_s"], y_mean, label=f"{label} (mean{suffix})", linestyle="-", alpha=1.0)


        if show_p95 and p95_col in df.columns:

            y_raw_p95 = clip_for_plot(df[p95_col], clip_pct)

            y_p95 = smooth_series(y_raw_p95, smooth, smooth_method, smooth_center) if smooth > 1 else y_raw_p95

            if smooth > 1 and show_raw:

                plot_line(ax, df["t_s"], y_raw_p95, label=f"{label} (p95, raw)", linestyle="--", alpha=0.25)

            plot_line(ax, df["t_s"], y_p95, label=f"{label} (p95{suffix})", linestyle="--", alpha=1.0)


        if show_max and max_col in df.columns:

            y_raw_max = clip_for_plot(df[max_col], clip_pct)

            y_max = smooth_series(y_raw_max, smooth, smooth_method, smooth_center) if smooth > 1 else y_raw_max

            if smooth > 1 and show_raw:

                plot_line(ax, df["t_s"], y_raw_max, label=f"{label} (max, raw)", linestyle=":", alpha=0.25)

            plot_line(ax, df["t_s"], y_max, label=f"{label} (max{suffix})", linestyle=":", alpha=1.0)


    ax.set_xlabel("t (s)")

    ax.set_ylabel(metric)


    title = f"{metric} over time"

    if clip_pct > 0:

        title += f" (clipped at p{clip_pct})"

    if smooth > 1:

        title += f" | smoothed ({smooth_method}, window={smooth})"


    ax.set_title(title)

    ax.legend(loc="best", fontsize="small")

    fig.tight_layout()

    fig.savefig(outpath, dpi=200)

    plt.close(fig)



def save_spike_plot(runs: List[Tuple[str, pd.DataFrame]], outpath: Path, spike_ms: float,

    """Save a spike ratio plot."""
                    smooth: int, smooth_method: str, smooth_center: bool, show_raw: bool):

    fig, ax = plt.subplots()


    for label, df in runs:

        if "spike_ratio" not in df.columns:

            continue


        y_raw = df["spike_ratio"]

        y = smooth_series(y_raw, smooth, smooth_method, smooth_center) if smooth > 1 else y_raw


        if smooth > 1 and show_raw:

            plot_line(ax, df["t_s"], y_raw, label=f"{label} (raw)", linestyle="--", alpha=0.35)


        suffix = f", smooth={smooth}" if smooth > 1 else ""

        plot_line(ax, df["t_s"], y, label=f"{label}{suffix}", linestyle="-", alpha=1.0)


    ax.set_xlabel("t (s)")

    ax.set_ylabel("spike_ratio")


    title = f"Spike ratio over time (frame_ms > {spike_ms} ms)"

    if smooth > 1:

        title += f" | smoothed ({smooth_method}, window={smooth})"


    ax.set_title(title)

    ax.legend(loc="best", fontsize="small")

    fig.tight_layout()

    fig.savefig(outpath, dpi=200)

    plt.close(fig)



def save_frames_in_bin_plot(runs: List[Tuple[str, pd.DataFrame]], outpath: Path,

    """Save frames-per-bin plot as a sanity check."""
                            smooth: int, smooth_method: str, smooth_center: bool, show_raw: bool):

    fig, ax = plt.subplots()


    for label, df in runs:

        if "frames_in_bin" not in df.columns:

            continue


        y_raw = df["frames_in_bin"]

        y = smooth_series(y_raw, smooth, smooth_method, smooth_center) if smooth > 1 else y_raw


        if smooth > 1 and show_raw:

            plot_line(ax, df["t_s"], y_raw, label=f"{label} (raw)", linestyle="--", alpha=0.35)


        suffix = f", smooth={smooth}" if smooth > 1 else ""

        plot_line(ax, df["t_s"], y, label=f"{label}{suffix}", linestyle="-", alpha=1.0)


    ax.set_xlabel("t (s)")

    ax.set_ylabel("frames_in_bin")


    title = "Frames per bin (sanity check for sampling / FPS)"

    if smooth > 1:

        title += f" | smoothed ({smooth_method}, window={smooth})"


    ax.set_title(title)

    ax.legend(loc="best", fontsize="small")

    fig.tight_layout()

    fig.savefig(outpath, dpi=200)

    plt.close(fig)



def summarize_run(label: str, raw_df: pd.DataFrame, agg_df: pd.DataFrame, spike_ms: float) -> Dict[str, object]:

    """Summarize a run into a dict of scalar metrics."""
    out: Dict[str, object] = {}

    out["label"] = label

    out["t_min_s"] = float(raw_df["t_s"].min()) if len(raw_df) else float("nan")

    out["t_max_s"] = float(raw_df["t_s"].max()) if len(raw_df) else float("nan")

    out["rows_raw"] = int(len(raw_df))

    out["rows_binned"] = int(len(agg_df))



    if "frame_ms" in raw_df.columns and raw_df["frame_ms"].notna().any():

        s = raw_df["frame_ms"].dropna()

        out["frame_ms_avg"] = float(s.mean())

        out["frame_ms_p95"] = float(s.quantile(0.95))

        out["frame_ms_max"] = float(s.max())

        out["spikes_pct"] = float((s > spike_ms).mean() * 100.0)

    elif "frame_ms_mean" in agg_df.columns and agg_df["frame_ms_mean"].notna().any():


        s = agg_df["frame_ms_mean"].dropna()

        out["frame_ms_avg"] = float(s.mean())


    if "fps" in raw_df.columns and raw_df["fps"].notna().any():

        s = raw_df["fps"].dropna()

        out["fps_avg"] = float(s.mean())

        out["fps_p05"] = float(s.quantile(0.05))

        out["fps_min"] = float(s.min())

        out["fps_max"] = float(s.max())

    elif "fps_mean" in agg_df.columns and agg_df["fps_mean"].notna().any():

        s = agg_df["fps_mean"].dropna()

        out["fps_avg"] = float(s.mean())


    if "used_mem_mb" in raw_df.columns and raw_df["used_mem_mb"].notna().any():

        s = raw_df["used_mem_mb"].dropna()

        out["mem_mb_avg"] = float(s.mean())

        out["mem_mb_p95"] = float(s.quantile(0.95))

        out["mem_mb_min"] = float(s.min())

        out["mem_mb_max"] = float(s.max())

    elif "used_mem_mb_mean" in agg_df.columns and agg_df["used_mem_mb_mean"].notna().any():

        s = agg_df["used_mem_mb_mean"].dropna()

        out["mem_mb_avg"] = float(s.mean())


    return out


def filter_time_window(df: pd.DataFrame, tmin: Optional[float], tmax: Optional[float]) -> pd.DataFrame:

    """Filter rows by an optional time window."""
    if tmin is not None:

        df = df[df["t_s"] >= tmin]

    if tmax is not None:

        df = df[df["t_s"] <= tmax]

    return df


def main():

    """CLI entry point for plotting and summary generation."""
    args = parse_args()


    csv_paths = args.csv

    labels: List[str] = []

    if args.labels:

        labels = [x.strip() for x in args.labels.split(",") if x.strip()]

    elif args.label:

        labels = args.label


    if not labels or len(labels) != len(csv_paths):

        labels = [Path(p).stem for p in csv_paths]


    default_metrics = ["frame_ms", "fps", "used_mem_mb"]

    metrics = [x.strip() for x in args.only.split(",") if x.strip()] if args.only else default_metrics


    outdir = Path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)


    runs_plot: List[Tuple[str, pd.DataFrame]] = []

    summaries: List[Dict[str, object]] = []


    for path, label in zip(csv_paths, labels):

        if not os.path.isfile(path):

            print(f"ERROR: file not found: {path}", file=sys.stderr)

            sys.exit(1)


        df = read_csv(path)

        df = filter_phase(df, args.phase)

        df = filter_time_window(df, args.tmin, args.tmax)


        if is_already_binned(df):

            raw_for_summary = df.copy()

            agg = normalize_binned(df, metrics)

        else:

            df = apply_downsample(df, args.downsample)

            raw_for_summary = df.copy()

            agg = aggregate_per_bin_per_frame(df, args.bin, args.spike_ms, metrics)


        runs_plot.append((label, agg))

        summaries.append(summarize_run(label, raw_for_summary, agg, args.spike_ms))



    pd.DataFrame(summaries).to_csv(outdir / "summary.csv", index=False)



    save_frame_ms_plot(

        runs_plot,

        outdir / "frame_ms.png",

        show_p95=args.p95,

        show_max=args.max,

        clip_pct=args.clip_pct,

        yscale=args.yscale_frame,

        smooth=args.smooth,

        smooth_method=args.smooth_method,

        smooth_center=args.smooth_center,

        show_raw=args.show_raw,

    )


    for m in metrics:

        if m == "frame_ms":

            continue

        save_simple_metric_plot(

            runs_plot,

            m,

            outdir / f"{m}.png",

            show_p95=args.p95,

            show_max=args.max,

            clip_pct=args.clip_pct,

            smooth=args.smooth,

            smooth_method=args.smooth_method,

            smooth_center=args.smooth_center,

            show_raw=args.show_raw,

        )


    save_spike_plot(

        runs_plot,

        outdir / "spike_ratio.png",

        args.spike_ms,

        smooth=args.smooth,

        smooth_method=args.smooth_method,

        smooth_center=args.smooth_center,

        show_raw=args.show_raw,

    )


    save_frames_in_bin_plot(

        runs_plot,

        outdir / "frames_in_bin.png",

        smooth=args.smooth,

        smooth_method=args.smooth_method,

        smooth_center=args.smooth_center,

        show_raw=args.show_raw,

    )


    print(f"Saved plots to: {outdir.resolve()}")

    print(f"Also wrote: {str((outdir / 'summary.csv').resolve())}")


    if args.overhead:

        if len(runs_plot) != 2:

            raise SystemExit("--overhead requires exactly 2 --csv inputs (Baseline + OmiLAXR).")


    base_label, base_df = runs_plot[args.overhead_baseline_index]

    other_label, other_df = runs_plot[args.overhead_other_index]


    delta = compute_overhead_df(base_df, other_df, metrics)


    for m in metrics:

        save_overhead_plot(

            delta,

            m,

            outdir / f"overhead_{m}.png",

            show_p95=args.p95,

            show_max=args.max,

            clip_pct=args.clip_pct

        )



if __name__ == "__main__":

    main()
