import pathlib

import matplotlib.pyplot as plt
import numpy as np

from openequivariance.benchmark.plotting.plotting_utils import (
    calculate_tp_per_sec,
    grouped_barchart,
    load_benchmarks,
    set_grid,
)


def _parse_layout_label(label: str):
    if label.endswith("[mul_ir]"):
        return label[: -len(" [mul_ir]")], "mul_ir"
    if label.endswith("[ir_mul]"):
        return label[: -len(" [ir_mul]")], "ir_mul"
    return label, None


def plot_layout(data_folder):
    data_folder = pathlib.Path(data_folder)
    benchmarks, _ = load_benchmarks(data_folder)

    grouped = {}
    dtype_order = []
    for benchmark in benchmarks:
        dtype = benchmark["benchmark results"]["rep_dtype"]
        if dtype not in dtype_order:
            dtype_order.append(dtype)

        direction = benchmark["direction"]
        base_label, layout = _parse_layout_label(benchmark["config_label"])
        if layout is None:
            continue

        grouped.setdefault(dtype, {}).setdefault(direction, {}).setdefault(
            base_label, {"mul_ir": 0.0, "ir_mul": 0.0}
        )
        grouped[dtype][direction][base_label][layout] = calculate_tp_per_sec(benchmark)

    def _dtype_sort_key(dtype_name: str) -> int:
        if "float32" in dtype_name:
            return 0
        if "float64" in dtype_name:
            return 1
        return 2

    dtype_order = sorted(dtype_order, key=_dtype_sort_key)

    directions = [
        d for d in ["forward", "backward"] if any(d in grouped[x] for x in grouped)
    ]
    if not directions:
        raise ValueError("No forward/backward layout benchmark entries found to plot.")

    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(len(directions), max(1, len(dtype_order)))
    axs = gs.subplots(sharex="col")

    if len(directions) == 1 and len(dtype_order) == 1:
        axs = np.array([[axs]])
    elif len(directions) == 1:
        axs = np.array([axs])
    elif len(dtype_order) == 1:
        axs = np.array([[ax] for ax in axs])

    colormap = {"mul_ir": "#1f77b4", "ir_mul": "#2ca02c"}

    for row, direction in enumerate(directions):
        for col, dtype in enumerate(dtype_order):
            axis = axs[row][col]
            source = grouped.get(dtype, {}).get(direction, {})
            data = {
                label: {
                    "mul_ir": vals["mul_ir"],
                    "ir_mul": vals["ir_mul"],
                }
                for label, vals in source.items()
            }
            grouped_barchart(
                data,
                axis,
                bar_height_fontsize=0,
                colormap=colormap,
                group_spacing=6.0,
                xticklabel=(row == len(directions) - 1),
            )
            set_grid(axis)

            if row == 0:
                axis.set_title(dtype.replace("<class 'numpy.", "").replace("'>", ""))
            if col == 0:
                axis.set_ylabel(direction.capitalize())
            if row < len(directions) - 1:
                axis.tick_params(axis="x", labelbottom=False)

    fig.supylabel("Throughput (# tensor products / s)", x=0.03, y=0.56)
    fig.supxlabel("Problem")

    handles, labels = axs[0][0].get_legend_handles_labels()
    unique = [
        (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
    ]
    if unique:
        axs[0][0].legend(*zip(*unique))

    fig.tight_layout(rect=(0.03, 0.03, 1.0, 1.0))
    fig.savefig(str(data_folder / "layout_throughput_comparison.pdf"))

    print("Layout speedups (ir_mul / mul_ir):")
    print("\t".join(["dtype", "direction", "min", "mean", "median", "max"]))
    for dtype in dtype_order:
        for direction in directions:
            ratios = []
            for _, values in grouped.get(dtype, {}).get(direction, {}).items():
                if values["mul_ir"] > 0:
                    ratios.append(values["ir_mul"] / values["mul_ir"])
            if ratios:
                stats = [
                    np.min(ratios),
                    np.mean(ratios),
                    np.median(ratios),
                    np.max(ratios),
                ]
                stats_fmt = [f"{val:.3f}" for val in stats]
                print("\t".join([dtype, direction] + stats_fmt))
