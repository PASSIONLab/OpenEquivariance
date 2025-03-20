import os
import logging
import urllib
from pathlib import Path

from openequivariance.logging_utils import getLogger
from openequivariance.implementations import 
from openequivariance.interface.tpp_creation_utils import ChannelwiseTPP
from ...suites import 

logger = getLogger()

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
                logger.critical(f"Error, {target_path} does not exist.")
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
