import numpy as np
import matplotlib.pyplot as plt
import pathlib
from openequivariance.benchmark.plotting.plotting_utils import (
    set_grid,
    colormap,
    labelmap,
    hatchmap,
    dtypes,
    directions,
    dtype_labelmap,
    grouped_barchart,
    load_benchmarks,
)


def plot_convolution(data_folder):
    data_folder = pathlib.Path(data_folder)
    benchmarks, metadata = load_benchmarks(data_folder)

    implementations = [
        "CUEConvolution",
        "CUEConvolutionFused",
        "LoopUnrollConvScatterSum",
        "LoopUnrollConvAtomic",
        "LoopUnrollConvDeterministic",
    ]

    graphs = ["1drf_radius6.0", "covid_spike_radius3.0", "carbon_lattice_radius6.0"]
    graph_lmap = {
        "covid_spike_radius3.0": "COVID spike",
        "1drf_radius6.0": "DHFR",
        "carbon_lattice_radius6.0": "carbon-lattice",
    }

    data = {}

    for direction in directions:
        data[direction] = {}
        for dtype in dtypes:
            data[direction][dtype] = {}
            for graph in graphs:
                data[direction][dtype][graph_lmap[graph]] = {}
                for impl in implementations:
                    exp = filter(
                        benchmarks,
                        {
                            "graph": graph,
                            "direction": direction,
                            "name": impl,
                            "irrep_dtype": dtype,
                        },
                        match_one=True,
                    )

                    data[direction][dtype][graph_lmap[graph]][labelmap[impl]] = np.mean(
                        exp["benchmark"]["time_millis"]
                    )

    fig = plt.figure(figsize=(5, 5))
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    axes = gs.subplots(sharex="col", sharey="row")

    for i, direction in enumerate(directions):
        for j, dtype in enumerate(dtypes):
            for k, graph in enumerate(graphs):
                normalizing_value = data[direction][dtype][graph_lmap[graph]][
                    "cuE-scattersum"
                ]
                for impl in implementations:
                    data[direction][dtype][graph_lmap[graph]][labelmap[impl]] = (
                        normalizing_value
                        / data[direction][dtype][graph_lmap[graph]][labelmap[impl]]
                    )

            grouped_barchart(
                data[direction][dtype],
                axes[i][j],
                bar_height_fontsize=0,
                rotate_xlabels=True,
                colormap=colormap,
                hatchmap=hatchmap,
                group_spacing=6.0,
            )

            axes[i][j].set_xlabel(dtype_labelmap[dtype])
            axes[i][j].set_ylabel(direction)
            axes[i][j].axhline(1.0, ls="--", c=colormap["cuE"])
            set_grid(axes[i][j])

    axes[1][0].set_ylim(0, 3.8)
    for ax in fig.get_axes():
        ax.label_outer()

    fig.supylabel("Speedup over cuE-scattersum", x=0.025, y=0.6)

    handles, labels = axes[0][0].get_legend_handles_labels()
    for i, l in enumerate(labels):
        if "fast" in l:
            labels[i] += " (ours)"

    unique = [
        (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
    ]
    fig.legend(*zip(*unique), loc="upper center", bbox_to_anchor=(0.55, 0.01))

    fig.show()
    fig.tight_layout()
    fig.savefig(str(data_folder / "kernel_fusion_speedup.pdf"), bbox_inches="tight")
