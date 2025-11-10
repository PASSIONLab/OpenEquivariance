import matplotlib.pyplot as plt
import pathlib
from openequivariance.benchmark.plotting.plotting_utils import (
    set_grid,
    colormap,
    labelmap,
    grouped_barchart,
    calculate_tp_per_sec,
    generate_speedup_table,
    print_speedup_table,
    load_benchmarks,
    filter_experiments,
)


def plot_uvw(data_folder):
    data_folder = pathlib.Path(data_folder)
    benchmarks, metadata = load_benchmarks(data_folder)

    configs = metadata["config_labels"]
    implementations = metadata["implementations"]
    metadata["directions"]

    dataf32 = {"forward": {}, "backward": {}}
    for i, desc in enumerate(configs):
        for direction in ["forward", "backward"]:
            dataf32[direction][desc] = {}
            for impl in implementations:
                if True:  # direction == "forward" or impl != "CUETensorProduct" or 'mace' in desc:
                    f32_benches = [
                        b
                        for b in benchmarks
                        if b["benchmark results"]["rep_dtype"]
                        == "<class 'numpy.float32'>"
                    ]
                    exp = filter_experiments(
                        f32_benches,
                        {
                            "config_label": desc,
                            "direction": direction,
                            "implementation_name": impl,
                        },
                        match_one=True,
                    )
                    dataf32[direction][desc][labelmap[impl]] = calculate_tp_per_sec(exp)

    dataf64 = {"forward": {}, "backward": {}}
    for i, desc in enumerate(configs):
        for direction in ["forward", "backward"]:
            dataf64[direction][desc] = {}
            for impl in implementations:
                if True:  # direction == "forward" or impl != "CUETensorProduct" or 'mace' in desc:
                    f64_benches = [
                        b
                        for b in benchmarks
                        if b["benchmark results"]["rep_dtype"]
                        == "<class 'numpy.float64'>"
                    ]
                    exp = filter_experiments(
                        f64_benches,
                        {
                            "config_label": desc,
                            "direction": direction,
                            "implementation_name": impl,
                        },
                        match_one=True,
                    )
                    dataf64[direction][desc][labelmap[impl]] = calculate_tp_per_sec(exp)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams.update({"font.size": 11})

    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(2, 2)
    axs = gs.subplots(sharex=True, sharey="row")

    grouped_barchart(
        dataf32["forward"],
        axs[0][0],
        bar_height_fontsize=0,
        xticklabel=False,
        colormap=colormap,
        group_spacing=6.0,
    )
    grouped_barchart(
        dataf32["backward"],
        axs[1][0],
        bar_height_fontsize=0,
        xticklabel=True,
        colormap=colormap,
        group_spacing=6.0,
    )

    grouped_barchart(
        dataf64["forward"],
        axs[0][1],
        bar_height_fontsize=0,
        xticklabel=False,
        colormap=colormap,
        group_spacing=6.0,
    )
    grouped_barchart(
        dataf64["backward"],
        axs[1][1],
        bar_height_fontsize=0,
        xticklabel=True,
        colormap=colormap,
        group_spacing=6.0,
    )

    for i in range(2):
        for j in range(2):
            set_grid(axs[i][j])

    fig.supylabel("Throughput (# tensor products / s)", x=0.03, y=0.56)

    axs[0][0].set_ylabel("Forward")
    axs[1][0].set_ylabel("Backward")

    axs[1][0].set_xlabel("float32")
    axs[1][1].set_xlabel("float64")

    handles, labels = axs[0][1].get_legend_handles_labels()
    unique = [
        (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
    ]
    axs[0][1].legend(*zip(*unique))

    fig.show()
    fig.tight_layout()
    fig.savefig(str(data_folder / "uvw_throughput_comparison.pdf"))

    speedup_table = generate_speedup_table(
        data_dict={
            "forward": dataf32,
            "backward": dataf32,
        },  # Not used, we iterate differently
        directions=["forward", "backward"],
        implementations=["e3nn", "cuE"],
        dtype_configs=[("f32", dataf32), ("f64", dataf64)],
        filter_func=lambda label, measurement: "DiffDock" in label,
    )
    print_speedup_table(speedup_table, title="DiffDock")
