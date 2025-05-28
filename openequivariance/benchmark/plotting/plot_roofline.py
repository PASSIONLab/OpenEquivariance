import numpy as np
import pathlib
from openequivariance.benchmark.plotting.plotting_utils import (
    colormap,
    labelmap,
    load_benchmarks,
    roofline_plot,
)


def plot_roofline(data_folder):
    data_folder = pathlib.Path(data_folder)
    benchmarks, metadata = load_benchmarks(data_folder)

    configs = metadata["config_labels"]
    implementations = ["LoopUnrollTP", "CUETensorProduct"]

    data = {"forward": {}, "backward": {}}
    for i, desc in enumerate(configs):
        for direction in ["forward", "backward"]:
            data[direction][desc] = {}
            for impl in implementations:
                exp = filter(
                    benchmarks,
                    {
                        "config_label": desc,
                        "direction": direction,
                        "implementation_name": impl,
                    },
                    match_one=True,
                )
                data[direction][desc][labelmap[impl]] = (
                    exp["benchmark results"]["arithmetic_intensity (FLOPs / byte)"],
                    np.mean(exp["benchmark results"]["throughputs_gflops"]),
                )

    roofline_data = []
    marker_map = {
        "forward-cuE": "+",
        "backward-cuE": "X",
        "forward-ours": "P",
        "backward-ours": "X",
    }
    for i, desc in enumerate(configs):
        for direction in ["forward", "backward"]:
            ai, throughput = data[direction][desc][labelmap["LoopUnrollTP"]]
            label = f"{direction}-ours"
            roofline_data.append(
                {
                    "AI": float(ai),
                    "throughput": throughput / 1000,
                    "label": label,
                    "marker": marker_map[label],
                    "color": colormap["ours"],
                    "markersize": 80,
                }
            )

            label = f"{direction}-cuE"
            ai, throughput = data[direction][desc][labelmap["CUETensorProduct"]]
            roofline_data.append(
                {
                    "AI": float(ai),
                    "throughput": throughput / 1000,
                    "label": label,
                    "marker": marker_map[label],
                    "color": colormap["cuE"],
                    "markersize": 80,
                }
            )

    cpu_roofs = {"A100-SXM-80GB FP32 Peak": 19.5}
    mem_bottlenecks = {"HBM2": 2.039}
    AI_v = {"": 9.56}

    draw_bounds = {"xmin": 0.4, "xmax": 15, "ymin": 0.15, "ymax": 25}
    fig, ax = roofline_plot(
        draw_bounds,
        cpu_roofs,
        mem_bottlenecks,
        AI_v,
        roofline_data,
        fig_ratio=1.8,
        fig_dimension=4,
    )

    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
    ]
    ax.legend(*zip(*unique))

    fig.show()
    fig.savefig(str(data_folder / "roofline.pdf"))

    # Table of throughputs and arithmetic intensities

    header = r"""
\begin{tabular}{cccccc}
\toprule
\multirow{2}{*}{ID} & \multirow{2}{*}{Description} & \multirow{2}{*}{Dir.} & \multirow{2}{*}{AI} & \multicolumn{2}{c}{TFLOP/s} \\
\cmidrule(r){5-6}
                    &                              &                       &                     & \multicolumn{1}{l}{cuE} & ours \\
\midrule
    """

    rows = []

    dir_map = {"forward": "F", "backward": "B"}
    for i, desc in enumerate(sorted(configs)):
        for direction in ["forward", "backward"]:
            for impl in implementations:
                short_id, long_desc = desc.split("#")
                long_desc = long_desc.replace("->", "$\\rightarrow$").replace(
                    " x ", "$\ \\times\ $"
                )
                ai_ours, throughput_ours = data[direction][desc][
                    labelmap["LoopUnrollTP"]
                ]
                throughput_ours = f"{float(throughput_ours / 1000):.2f}"
                _, throughput_cue = data[direction][desc][labelmap["CUETensorProduct"]]
                throughput_cue = f"{float(throughput_cue / 1000):.2f}"

            result = [
                "\multirow{2}{*}{" + short_id + "}",
                "\multirow{2}{*}{" + long_desc + "}",
                dir_map[direction],
                f"{ai_ours:.1f}",
                throughput_cue,
                throughput_ours,
            ]
            if direction == "backward":
                result[0] = ""
                result[1] = ""
            rows.append(result)

    print(header)
    result = ""
    for i, row in enumerate(rows):
        result += " & ".join(row) + r"\\" + "\n"
        if row[2] == "B" and i < len(rows) - 1:
            result += "\cmidrule(r){3-6}" + "\n"

    print(result.replace("[", "").replace("]", "").replace("uvu", "B"))
    print("\\bottomrule\n\\end{tabular}")
