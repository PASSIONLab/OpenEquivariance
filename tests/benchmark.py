import numpy as np
import numpy.linalg as la

import itertools, logging, argparse, os, copy
from pathlib import Path
import urllib.request

from openequivariance.benchmark.logging_utils import getLogger
from openequivariance.extlib import DeviceProp
from openequivariance.implementations.E3NNTensorProduct import E3NNTensorProduct, E3NNTensorProductCompiledCUDAGraphs, E3NNTensorProductCompiledMaxAutotuneCUDAGraphs 
from openequivariance.implementations.LoopUnrollTP import LoopUnrollTP
from openequivariance.implementations.CUETensorProduct import CUETensorProduct
from openequivariance.benchmark.TestBenchmarkSuite import TestBenchmarkSuite, TestDefinition, Direction
from openequivariance.benchmark.tpp_creation_utils import ChannelwiseTPP, FullyConnectedTPProblem, SingleInstruction
from openequivariance.benchmark.benchmark_routines.paper_benchmark_uvw import run_paper_uvw_benchmark

from openequivariance.implementations.convolution.LoopUnrollConv import *
from openequivariance.implementations.convolution.CUEConv import *
from openequivariance.benchmark.ConvBenchmarkSuite import *

logger = getLogger()

CTPP = ChannelwiseTPP
FCTPP = FullyConnectedTPProblem

implementation_map = {
    'e3nn': E3NNTensorProductCompiledMaxAutotuneCUDAGraphs,
    'e3nn_uncompiled': E3NNTensorProduct,
    'cue': CUETensorProduct,
    'oeq': LoopUnrollTP
}

datatype_map = {
    'float32': np.float32,
    'float64': np.float64
}

roofline_configs = [
    SingleInstruction(L1, L2, L3, cm, f"[{i+1}]#{L1} x {L2} -> {L3} ({cm})")
    for i, (L1, L2, L3, cm) in enumerate([
        ("128x1e", "1x1e", "128x1e", "uvu"), 
        ("128x2e", "1x1e", "128x2e", "uvu"),
        ("128x3e", "1x3e", "128x3e", "uvu"),
        ("128x5e", "1x5e", "128x3e", "uvu"),
        ("128x5e", "1x3e", "128x5e", "uvu"),
        ("128x6e", "1x3e", "128x6e", "uvu"),
        ("128x7e", "1x4e", "128x7e", "uvu"),
        ("128x7e", "1x7e", "128x7e", "uvu"),
    ])
]

def benchmark_uvu(params):
    from openequivariance.benchmark.benchmark_configs \
            import mace_nequip_problems 

    float64_problems = copy.deepcopy(mace_nequip_problems)
    for problem in float64_problems: 
        problem.irrep_dtype = np.float64
        problem.weight_dtype = np.float64 
    problems = mace_nequip_problems + float64_problems

    implementations = [
        implementation_map[impl] for impl in params.implementations
    ] 
    directions = params.directions
    datatypes = [datatype_map[dt] for dt in params.datatypes]

    tests = [TestDefinition(implementation, problem, direction, correctness=False, benchmark=True) 
             for implementation, problem, direction
             in itertools.product(implementations, problems, directions)]

    # Handle the float64 Benzene case, since we run out of memory with torch compile
    tests = [test for test in tests
            if 'benzene' not in test.problem.label
            or test.implementation != E3NNTensorProductCompiledMaxAutotuneCUDAGraphs 
            or test.problem.irrep_dtype != np.float64]

    if 'e3nn' in params.implementations and 'float64' in params.datatypes:
        tests.extend([TestDefinition(E3NNTensorProduct, 
            CTPP('64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e + 64x3o + 64x3e',  '0e + 1o + 2e + 3o', '64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e + 64x3o + 64x3e', 
                    'nequip-revmd17-benzene', irrep_dtype=np.float64, weight_dtype=np.float64), direction, correctness=False, benchmark=True) 
                    for direction in ['forward', 'backward']])

    # Remove some more configurations for GPUs with limited memory 
    if params.limited_memory:
        tests = [test for test in tests if 
                (test.implementation == LoopUnrollTP and 'benzene' not in test.problem.label)
                or (test.implementation == CUETensorProduct and 'benzene' not in test.problem.label)
                or ('benzene' not in test.problem.label and test.problem.irrep_dtype != np.float64)]

    bench_suite = TestBenchmarkSuite(
        correctness_threshold = 5e-5,
        num_warmup=100,
        num_iter=100,
        bench_batch_size=params.batch_size,
        prng_seed=11111,
        test_name="uvu")

    data_folder = bench_suite.run(tests, params.output_folder)

    if params.plot:
        plot({"data_folder": data_folder})

def benchmark_roofline(params):
    implementations =   [LoopUnrollTP, CUETensorProduct]
    directions = ['forward', 'backward']

    tests = [TestDefinition(implementation, problem, direction, correctness=False, benchmark=True) 
             for implementation, problem, direction
             in itertools.product(implementations, roofline_configs, directions)]

    bench_suite = TestBenchmarkSuite(
        correctness_threshold = 5e-5,
        num_warmup=100,
        num_iter=100,
        bench_batch_size=200000,
        prng_seed=11111,
        torch_op=False,
        test_name="roofline")

    data_folder = bench_suite.run(tests, params.output_folder)

    if params.plot:
        plot({"data_folder": data_folder})


def benchmark_convolution(params):
    filenames = [   "covid_spike_radius3.0.pickle", 
                    "1drf_radius6.0.pickle", 
                    "carbon_lattice_radius6.0.pickle"]
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

    if not params.disable_bench:
        configs = [ ChannelwiseTPP("128x0e+128x1o+128x2e", 
                        "1x0e+1x1o+1x2e+1x3o",
                        "128x0e+128x1o+128x2e+128x3o"),
                    ChannelwiseTPP("128x0e+128x1o+128x2e", 
                        "1x0e+1x1o+1x2e+1x3o",
                        "128x0e+128x1o+128x2e+128x3o"),
                    ] # MACE-large 

        configs[1].irrep_dtype = np.float64
        configs[1].weight_dtype = np.float64

        bench = ConvBenchmarkSuite(configs, torch_op=True, test_name="convolution") 

        implementations = [ LoopUnrollConvScatterSum, 
                            CUEConv,
                            LoopUnrollConvDeterministic, 
                            LoopUnrollConvAtomic]

        if params.limited_memory:
            implementations = [impl for impl in implementations 
                    if impl != LoopUnrollConvScatterSum
                    and impl != CUEConv]

        output_folder = None
        for graph in graphs: 
            for direction in ["forward", "backward"]:
                output_folder = bench.run(
                        implementations = implementations,
                        graph = graph,
                        direction=direction, 
                        correctness=False,
                        double_backward_correctness=False,
                        benchmark=True,
                        output_folder=params.output_folder)

    if params.plot:
        if not params.limited_memory:
            plot({"data_folder": output_folder})
        else:
            logger.critical("Cannot plot convolution speedups over cuE with --limited-memory flag enabled.")
 
def plot(params):
    import openequivariance.benchmark.plotting as plotting
    data_folder, test_name = None, None
    if isinstance(params, dict):
        data_folder = params["data_folder"]
    else:
        data_folder = params.data_folder

    with open(pathlib.Path(data_folder) / "metadata.json", 'r') as f:
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

if __name__=='__main__':
    logger.setLevel(logging.INFO)

    dp = DeviceProp(0)
    paper_benchmark_gpu = "NVIDIA A100-SXM4-80GB"
    if dp.name != paper_benchmark_gpu:
        logger.warning(msg=f"Current GPU ({dp.name}) is not the {paper_benchmark_gpu} used in the paper. Runtime benchmarks may differ from our reported results.")
    parser = argparse.ArgumentParser(description='Benchmark openequivariance kernels')
    parser.add_argument("--output_folder", "-o", type=str, default=None, help="Output folder for benchmark results")

    subparsers = parser.add_subparsers(help='subcommand help', required=True)
    parser_uvu = subparsers.add_parser('uvu', help='Run the UVU kernel benchmark without fusion') 
    parser_uvu.add_argument("--batch_size", "-b", type=int, default=50000, help="Batch size for benchmark")
    parser_uvu.add_argument("--implementations", "-i", type=str, nargs='+', 
            default=['e3nn', 'cue', 'oeq'], help="Implementations to benchmark",
            choices=['e3nn', 'e3nn_uncompiled', 'cue', 'oeq'])
    parser_uvu.add_argument("--directions", "-d", type=str, nargs='+',
            default=['forward', 'backward'], help="Directions to benchmark",
            choices=['forward', 'backward'])
    parser_uvu.add_argument("--datatypes", "-t", type=str, nargs='+',
            default=['float32', 'float64'], help="Data types to benchmark",
            choices=['float32', 'float64'])
    parser_uvu.add_argument("--limited-memory", action="store_true", help="Disable tests requiring large amounts of memory.")
    parser_uvu.add_argument("--plot", action="store_true", help="Plot the results.")
    parser_uvu.set_defaults(func=benchmark_uvu)

    parser_roofline = subparsers.add_parser('roofline', help='Run the roofline comparison')
    parser_roofline.add_argument("--plot", action="store_true", help="Plot the results.")
    parser_roofline.set_defaults(func=benchmark_roofline)

    parser_correctness = subparsers.add_parser('correctness', help='Run correctness tests')
    parser_correctness.set_defaults(func=correctness)

    parser_conv = subparsers.add_parser('conv', help='Run the fused convolution kernel benchmark')
    parser_conv.add_argument("--data", type=str, help="Folder containing graph data", required=True)
    parser_conv.add_argument("--disable_download", action='store_true', help="Disable downloading data files if they do not exist")
    parser_conv.add_argument("--disable_bench", action='store_true', help="Disable benchmark (downloads data if needed)")
    parser_conv.add_argument("--limited-memory", action="store_true", help="Disable tests requiring large amounts of memory.")
    parser_conv.add_argument("--plot", action="store_true", help="Plot the results.")
    parser_conv.set_defaults(func=benchmark_convolution)

    parser_uvw = subparsers.add_parser('uvw', help='Run the UVW kernel benchmark without fusion') 
    parser_uvw.add_argument("--batch_size", "-b", type=int, default=50000, help="Batch size for benchmark")
    parser_uvw.add_argument("--directions", "-d", type=str, nargs='+',
            default=['forward', 'backward'], help="Directions to benchmark",
            choices=['forward', 'backward'])
    parser_uvw.add_argument("--plot", action="store_true", help="Plot the results.")
    parser_uvw.set_defaults(func=run_paper_uvw_benchmark)

    parser_plot = subparsers.add_parser('plot', help="Generate a plot for a folder of benchmarks.")
    parser_plot.add_argument("data_folder", type=str)
    parser_plot.set_defaults(func=plot)

    args = parser.parse_args()
    args.func(args)