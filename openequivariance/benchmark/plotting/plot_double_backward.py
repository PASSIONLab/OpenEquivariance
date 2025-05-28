import numpy as np
import matplotlib.pyplot as plt
import pathlib
from openequivariance.benchmark.plotting.plotting_utils import (
    set_grid,
    colormap,
    labelmap,
    grouped_barchart,
    load_benchmarks,
)


def plot_double_backward(data_folder):
    data_folder = pathlib.Path(data_folder)
    benchmarks, metadata = load_benchmarks(data_folder)

    configs = metadata["config_labels"]
    implementations = ["E3NNTensorProduct", "CUETensorProduct", "LoopUnrollTP"]

    def calculate_tp_per_sec(exp):
        return exp["benchmark results"]["batch_size"] / (
            np.mean(exp["benchmark results"]["time_millis"]) * 0.001
        )

    dataf32 = {"double_backward": {}}
    for i, desc in enumerate(configs):
        for direction in ["double_backward"]:
            dataf32[direction][desc] = {}
            for impl in implementations:
                f32_benches = [
                    b
                    for b in benchmarks
                    if b["benchmark results"]["rep_dtype"] == "<class 'numpy.float32'>"
                ]
                exp = filter(
                    f32_benches,
                    {
                        "config_label": desc,
                        "direction": direction,
                        "implementation_name": impl,
                    },
                    match_one=True,
                )
                dataf32[direction][desc][labelmap[impl]] = calculate_tp_per_sec(exp)

    dataf64 = {"double_backward": {}}
    for i, desc in enumerate(configs):
        for direction in ["double_backward"]:
            dataf64[direction][desc] = {}
            for impl in implementations:
                f64_benches = [
                    b
                    for b in benchmarks
                    if "float64" in b["benchmark results"]["rep_dtype"]
                ]

                exp = filter(
                    f64_benches,
                    {
                        "config_label": desc,
                        "direction": direction,
                        "implementation_name": impl,
                    },
                    match_one=True,
                )

                if exp is None:
                    print(desc)
                    print(direction)
                    print(impl)

                dataf64[direction][desc][labelmap[impl]] = calculate_tp_per_sec(exp)

    fig = plt.figure(figsize=(7, 3))
    gs = fig.add_gridspec(1, 2, hspace=0, wspace=0.1)
    axs = gs.subplots(sharex="col", sharey="row")

    grouped_barchart(
        dataf32["double_backward"],
        axs[0],
        bar_height_fontsize=0,
        colormap=colormap,
        group_spacing=6.0,
    )
    grouped_barchart(
        dataf64["double_backward"],
        axs[1],
        bar_height_fontsize=0,
        colormap=colormap,
        group_spacing=6.0,
    )

    for i in range(2):
        set_grid(axs[i])
        set_grid(axs[i])

    axs[0].set_xlabel("float32")
    axs[1].set_xlabel("float64")

    handles, labels = axs[0].get_legend_handles_labels()
    unique = [
        (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
    ]
    axs[0].legend(*zip(*unique))

    for ax in fig.get_axes():
        ax.label_outer()

    fig.supylabel("2nd Deriv. Throughput\n(# tensor products / s)", y=0.5)

    speedup_table = []
    for direction in ["double_backward"]:
        for impl in ["e3nn", "cuE"]:
            for dtype_label, dtype_set in [("f32", dataf32), ("f64", dataf64)]:
                speedups = [
                    measurement["ours"] / measurement[impl]
                    for _, measurement in dtype_set[direction].items()
                    if impl in measurement
                ]
                stats = (
                    np.min(speedups),
                    np.mean(speedups),
                    np.median(speedups),
                    np.max(speedups),
                )
                stats = [f"{stat:.2f}" for stat in stats]

                dir_print = direction
                result = [dir_print, impl, dtype_label] + stats
                speedup_table.append(result)

    print("\t\t".join(["Direction", "Base", "dtype", "min", "mean", "med", "max"]))
    for row in speedup_table:
        print("\t\t".join(row))

    fig.show()
    fig.tight_layout()
    fig.savefig(
        str(data_folder / "double_backward_throughput.pdf"), bbox_inches="tight"
    )
