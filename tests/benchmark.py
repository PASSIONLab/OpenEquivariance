import itertools
import logging
import argparse
import os
import copy
import json
import pathlib
from pathlib import Path
import urllib.request

import numpy as np

from openequivariance.benchmark.logging_utils import getLogger
from openequivariance.extlib import DeviceProp
from openequivariance.implementations.E3NNTensorProduct import (
    E3NNTensorProduct,
    E3NNTensorProductCompiledCUDAGraphs,
    E3NNTensorProductCompiledMaxAutotuneCUDAGraphs,
)
from openequivariance.implementations.TensorProduct import TensorProduct
from openequivariance.implementations.CUETensorProduct import CUETensorProduct
from openequivariance.benchmark.TestBenchmarkSuite import (
    TestBenchmarkSuite,
    TestDefinition,
    Direction,
)
from openequivariance.benchmark.tpp_creation_utils import (
    ChannelwiseTPP,
    FullyConnectedTPProblem,
    SingleInstruction,
)

from openequivariance.implementations.convolution.TensorProductConv import (
    TensorProductConvAtomic,
    TensorProductConvDeterministic,
    TensorProductConvKahan,
    TensorProductConvScatterSum,
)

from openequivariance.implementations.convolution.CUEConv import CUEConv, CUEConvFused
from openequivariance.benchmark.ConvBenchmarkSuite import ConvBenchmarkSuite, load_graph

from openequivariance.benchmark.problems import (
    e3nn_torch_tetris_poly_problems,
    diffdock_problems,
    mace_problems,
    nequip_problems,
)

from torch._functorch import config

logger = getLogger()

CTPP = ChannelwiseTPP
FCTPP = FullyConnectedTPProblem

implementation_map = {
    "e3nn": E3NNTensorProductCompiledMaxAutotuneCUDAGraphs,
    "e3nn_uncompiled": E3NNTensorProduct,
    "cue": CUETensorProduct,
    "oeq": TensorProduct,
}

datatype_map = {"float32": np.float32, "float64": np.float64}

roofline_configs = [
    SingleInstruction(L1, L2, L3, cm, f"[{i + 1}]#{L1} x {L2} -> {L3} ({cm})")
    for i, (L1, L2, L3, cm) in enumerate(
        [
            ("128x1e", "1x1e", "128x1e", "uvu"),
            ("128x2e", "1x1e", "128x2e", "uvu"),
            ("128x3e", "1x3e", "128x3e", "uvu"),
            ("128x5e", "1x5e", "128x3e", "uvu"),
            ("128x5e", "1x3e", "128x5e", "uvu"),
            ("128x6e", "1x3e", "128x6e", "uvu"),
            ("128x7e", "1x4e", "128x7e", "uvu"),
            ("128x7e", "1x7e", "128x7e", "uvu"),
        ]
    )
]


def benchmark_uvu(params):
    float64_problems = mace_problems() + nequip_problems()
    for problem in float64_problems:
        problem.irrep_dtype = np.float64
        problem.weight_dtype = np.float64
    problems = mace_problems() + nequip_problems() + float64_problems

    implementations = [implementation_map[impl] for impl in params.implementations]
    directions = params.directions

    tests = [
        TestDefinition(
            implementation, problem, direction, correctness=False, benchmark=True
        )
        for implementation, problem, direction in itertools.product(
            implementations, problems, directions
        )
    ]

    # Handle the float64 Benzene case, since we run out of memory with torch compile
    tests = [
        test
        for test in tests
        if "benzene" not in test.problem.label
        or test.implementation != E3NNTensorProductCompiledMaxAutotuneCUDAGraphs
        or test.problem.irrep_dtype != np.float64
    ]

    if "e3nn" in params.implementations and "float64" in params.datatypes:
        tests.extend(
            [
                TestDefinition(
                    E3NNTensorProduct,
                    CTPP(
                        "64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e + 64x3o + 64x3e",
                        "0e + 1o + 2e + 3o",
                        "64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e + 64x3o + 64x3e",
                        "nequip-revmd17-benzene",
                        irrep_dtype=np.float64,
                        weight_dtype=np.float64,
                    ),
                    direction,
                    correctness=False,
                    benchmark=True,
                )
                for direction in ["forward", "backward"]
            ]
        )

    # Remove some more configurations for GPUs with limited memory
    if params.limited_memory:
        tests = [
            test
            for test in tests
            if (
                test.implementation == TensorProduct
                and "benzene" not in test.problem.label
            )
            or (
                test.implementation == CUETensorProduct
                and "benzene" not in test.problem.label
            )
            or (
                "benzene" not in test.problem.label
                and test.problem.irrep_dtype != np.float64
            )
        ]

    bench_suite = TestBenchmarkSuite(
        num_warmup=100,
        num_iter=100,
        bench_batch_size=params.batch_size,
        prng_seed=11111,
        test_name="uvu",
    )

    data_folder = bench_suite.run(tests, params.output_folder)

    if params.plot:
        plot({"data_folder": data_folder})


@config.patch("donated_buffer", False)
def benchmark_uvw(params) -> pathlib.Path:
    problems = list(
        itertools.chain(e3nn_torch_tetris_poly_problems(), diffdock_problems())
    )

    float64_problems = copy.deepcopy(problems)
    for problem in float64_problems:
        problem.irrep_dtype = np.float64
        problem.weight_dtype = np.float64

    problems += float64_problems

    implementations = [
        E3NNTensorProductCompiledCUDAGraphs,
        CUETensorProduct,
        TensorProduct,
    ]

    tests = [
        TestDefinition(
            implementation, problem, direction, correctness=False, benchmark=True
        )
        for problem, direction, implementation in itertools.product(
            problems, params.directions, implementations
        )
    ]

    bench_suite = TestBenchmarkSuite(
        num_warmup=100,
        num_iter=100,
        bench_batch_size=params.batch_size,
        prng_seed=11111,
        torch_op=True,
        test_name="uvw",
    )

    logger.setLevel(logging.INFO)
    data_folder = bench_suite.run(tests, output_folder=params.output_folder)

    if params.plot:
        import openequivariance.benchmark.plotting as plotting

        plotting.plot_uvw(data_folder)

    return data_folder


def benchmark_roofline(params):
    implementations = [TensorProduct, CUETensorProduct]
    directions = ["forward", "backward"]

    tests = [
        TestDefinition(
            implementation, problem, direction, correctness=False, benchmark=True
        )
        for implementation, problem, direction in itertools.product(
            implementations, roofline_configs, directions
        )
    ]

    bench_suite = TestBenchmarkSuite(
        num_warmup=100,
        num_iter=100,
        bench_batch_size=200000,
        prng_seed=11111,
        torch_op=False,
        test_name="roofline",
    )

    data_folder = bench_suite.run(tests, params.output_folder)

    if params.plot:
        plot({"data_folder": data_folder})


def download_graphs(params, filenames):
    download_prefix = "https://portal.nersc.gov/project/m1982/equivariant_nn_graphs/"

    if not Path(params.data).exists():
        os.makedirs(params.data, exist_ok=True)

    graphs = []
    for filename in filenames:
        target_path = Path(params.data) / filename
        if not target_path.exists():
            if params.disable_download:
                logging.critical(f"Error, {target_path} does not exist.")
                exit(1)
            else:
                logging.info(f"Downloading {download_prefix + filename}...")
                urllib.request.urlretrieve(download_prefix + filename, target_path)

        graphs.append(load_graph(str(target_path)))

    return graphs


def benchmark_convolution(params):
    filenames = [
        "covid_spike_radius3.0.pickle",
        "1drf_radius6.0.pickle",
        "carbon_lattice_radius6.0.pickle",
    ]

    graphs = download_graphs(params, filenames)

    if not params.disable_bench:
        configs = [
            ChannelwiseTPP(
                "128x0e+128x1o+128x2e",
                "1x0e+1x1o+1x2e+1x3o",
                "128x0e+128x1o+128x2e+128x3o",
            ),
            ChannelwiseTPP(
                "128x0e+128x1o+128x2e",
                "1x0e+1x1o+1x2e+1x3o",
                "128x0e+128x1o+128x2e+128x3o",
            ),
        ]  # MACE-large

        configs[1].irrep_dtype = np.float64
        configs[1].weight_dtype = np.float64

        bench = ConvBenchmarkSuite(configs, test_name="convolution")

        implementations = [
            TensorProductConvScatterSum,
            CUEConv,
            CUEConvFused,
            TensorProductConvDeterministic,
            TensorProductConvAtomic,
        ]

        if params.limited_memory:
            implementations = [
                impl
                for impl in implementations
                if impl != TensorProductConvScatterSum and impl != CUEConv
            ]

        output_folder = None
        for graph in graphs:
            for direction in ["forward", "backward"]:
                output_folder = bench.run(
                    implementations=implementations,
                    graph=graph,
                    direction=direction,
                    correctness=False,
                    benchmark=True,
                    output_folder=params.output_folder,
                )

    if params.plot:
        if not params.limited_memory:
            plot({"data_folder": output_folder})
        else:
            logger.critical(
                "Cannot plot convolution speedups over cuE with --limited-memory flag enabled."
            )


def benchmark_double_backward(params):
    implementations = [E3NNTensorProduct, CUETensorProduct, TensorProduct]
    problems = diffdock_problems() + mace_problems() + nequip_problems()
    float64_problems = copy.deepcopy(problems)

    for problem in float64_problems:
        problem.irrep_dtype = np.float64
        problem.weight_dtype = np.float64

    directions: list[Direction] = ["double_backward"]
    tests = [
        TestDefinition(
            implementation, problem, direction, correctness=False, benchmark=True
        )
        for problem, direction, implementation in itertools.product(
            problems + float64_problems, directions, implementations
        )
    ]

    logger = getLogger()
    logger.setLevel(logging.INFO)
    test_suite = TestBenchmarkSuite(bench_batch_size=20000, test_name="double_backward")
    test_suite.run(tests)

    data_folder = test_suite.run(tests, params.output_folder)
    if params.plot:
        plot({"data_folder": data_folder})


def benchmark_kahan_accuracy(params):
    filenames = ["carbon_lattice_radius6.0.pickle"]
    graphs = download_graphs(params, filenames)
    implementations = [TensorProductConvAtomic, TensorProductConvKahan]
    problems = [mace_problems()[0]]

    bench = ConvBenchmarkSuite(
        problems, test_name="kahan_convolution_accuracy", correctness_threshold=1e-4
    )
    directions = ["forward", "backward"]
    if params.double_backward:
        directions.append("double_backward")

    for graph in graphs:
        for direction in directions:
            bench.run(
                implementations=implementations,
                graph=graph,
                direction=direction,
                correctness=True,
                benchmark=False,
                output_folder=params.output_folder,
                high_precision_ref=True,
            )


def plot(params):
    import openequivariance.benchmark.plotting as plotting

    data_folder, test_name = None, None
    if isinstance(params, dict):
        data_folder = params["data_folder"]
    else:
        data_folder = params.data_folder

    with open(pathlib.Path(data_folder) / "metadata.json", "r") as f:
        metadata = json.load(f)
        test_name = metadata["test_name"]

    if test_name == "uvu":
        plotting.plot_uvu(data_folder)
    elif test_name == "uvw":
        plotting.plot_uvw(data_folder)
    elif test_name == "roofline":
        plotting.plot_roofline(data_folder)
    elif test_name == "convolution":
        plotting.plot_convolution(data_folder)
    elif test_name == "double_backward":
        plotting.plot_double_backward(data_folder)
    else:
        raise ValueError(f"Unknown test name: {test_name}. Cannot plot results.")


if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    dp = DeviceProp(0)
    paper_benchmark_gpu = "NVIDIA A100-SXM4-80GB"
    if dp.name != paper_benchmark_gpu:
        logger.warning(
            msg=f"Current GPU ({dp.name}) is not the {paper_benchmark_gpu} used in the paper. Runtime benchmarks may differ from our reported results."
        )
    parser = argparse.ArgumentParser(description="Benchmark openequivariance kernels")
    parser.add_argument(
        "--output_folder",
        "-o",
        type=str,
        default=None,
        help="Output folder for benchmark results",
    )

    subparsers = parser.add_subparsers(help="subcommand help", required=True)
    parser_uvu = subparsers.add_parser(
        "uvu", help="Run the UVU kernel benchmark without fusion"
    )
    parser_uvu.add_argument(
        "--batch_size", "-b", type=int, default=50000, help="Batch size for benchmark"
    )
    parser_uvu.add_argument(
        "--implementations",
        "-i",
        type=str,
        nargs="+",
        default=["e3nn", "cue", "oeq"],
        help="Implementations to benchmark",
        choices=["e3nn", "e3nn_uncompiled", "cue", "oeq"],
    )
    parser_uvu.add_argument(
        "--directions",
        "-d",
        type=str,
        nargs="+",
        default=["forward", "backward"],
        help="Directions to benchmark",
        choices=["forward", "backward"],
    )
    parser_uvu.add_argument(
        "--datatypes",
        "-t",
        type=str,
        nargs="+",
        default=["float32", "float64"],
        help="Data types to benchmark",
        choices=["float32", "float64"],
    )
    parser_uvu.add_argument(
        "--limited-memory",
        action="store_true",
        help="Disable tests requiring large amounts of memory.",
    )
    parser_uvu.add_argument("--plot", action="store_true", help="Plot the results.")
    parser_uvu.set_defaults(func=benchmark_uvu)

    parser_roofline = subparsers.add_parser(
        "roofline", help="Run the roofline comparison"
    )
    parser_roofline.add_argument(
        "--plot", action="store_true", help="Plot the results."
    )
    parser_roofline.set_defaults(func=benchmark_roofline)

    parser_conv = subparsers.add_parser(
        "conv", help="Run the fused convolution kernel benchmark"
    )
    parser_conv.add_argument(
        "--data", type=str, help="Folder containing graph data", required=True
    )
    parser_conv.add_argument(
        "--disable_download",
        action="store_true",
        help="Disable downloading data files if they do not exist",
    )
    parser_conv.add_argument(
        "--disable_bench",
        action="store_true",
        help="Disable benchmark (downloads data if needed)",
    )
    parser_conv.add_argument(
        "--limited-memory",
        action="store_true",
        help="Disable tests requiring large amounts of memory.",
    )
    parser_conv.add_argument("--plot", action="store_true", help="Plot the results.")
    parser_conv.set_defaults(func=benchmark_convolution)

    parser_uvw = subparsers.add_parser(
        "uvw", help="Run the UVW kernel benchmark without fusion"
    )
    parser_uvw.add_argument(
        "--batch_size", "-b", type=int, default=50000, help="Batch size for benchmark"
    )
    parser_uvw.add_argument(
        "--directions",
        "-d",
        type=str,
        nargs="+",
        default=["forward", "backward"],
        help="Directions to benchmark",
        choices=["forward", "backward"],
    )
    parser_uvw.add_argument("--plot", action="store_true", help="Plot the results.")
    parser_uvw.set_defaults(func=benchmark_uvw)

    parser_double_bwd = subparsers.add_parser(
        "double_backward", help="Run the higher derivative kernel benchmark"
    )
    parser_double_bwd.add_argument(
        "--batch_size", "-b", type=int, default=50000, help="Batch size for benchmark"
    )
    parser_double_bwd.set_defaults(func=benchmark_double_backward)

    parser_kahan = subparsers.add_parser(
        "kahan_conv", help="Run the Kahan convolution accuracy benchmark"
    )
    parser_kahan.add_argument(
        "--data",
        type=str,
        help="Folder to download graph data to (or already containing graphs)",
        required=True,
    )
    parser_kahan.add_argument(
        "--disable_download",
        action="store_true",
        help="Disable downloading data files if they do not exist",
    )
    parser_kahan.add_argument(
        "--double_backward",
        action="store_true",
        help="Run double backward test (high memory usage)",
    )
    parser_kahan.set_defaults(func=benchmark_kahan_accuracy)

    parser_plot = subparsers.add_parser(
        "plot", help="Generate a plot for a folder of benchmarks."
    )
    parser_plot.add_argument("data_folder", type=str)
    parser_plot.set_defaults(func=plot)

    args = parser.parse_args()
    args.func(args)
