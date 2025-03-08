import numpy as np
import numpy.linalg as la

import itertools, logging, argparse, os
from pathlib import Path
import urllib.request

from openequivariance.logging_utils import getLogger
from openequivariance.extlib import DeviceProp
from openequivariance.implementations.E3NNTensorProduct import E3NNTensorProduct, E3NNTensorProductCompiledCUDAGraphs, E3NNTensorProductCompiledMaxAutotuneCUDAGraphs 
from openequivariance.implementations.LoopUnrollTP import LoopUnrollTP
from openequivariance.implementations.CUETensorProduct import CUETensorProduct
from tests.suites.TensorProductSuite import TensorProductSuite, TestDefinition, Direction
from openequivariance.interface.tpp_creation_utils import ChannelwiseTPP, FullyConnectedTPProblem, SingleInstruction
from openequivariance.benchmark.benchmark_routines.paper_benchmark_uvw import run_paper_uvw_benchmark

from openequivariance.implementations.convolution.LoopUnrollConv import *
from openequivariance.implementations.convolution.CUEConv import *
from tests.suites.ConvolutionSuite import *

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