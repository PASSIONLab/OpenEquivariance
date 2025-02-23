import itertools, sys, os, logging, copy, pathlib
import numpy as np

from openequivariance.benchmark.logging_utils import getLogger
from openequivariance.implementations.E3NNTensorProduct import E3NNTensorProductCompiledCUDAGraphs
from openequivariance.implementations.CUETensorProduct import CUETensorProduct
from openequivariance.implementations.LoopUnrollTP import LoopUnrollTP
from openequivariance.benchmark.TestBenchmarkSuite import TestBenchmarkSuite, TestDefinition, Direction
from openequivariance.benchmark.tpp_creation_utils import FullyConnectedTPProblem
from openequivariance.benchmark.benchmark_configs import e3nn_torch_tetris_polynomial, diffdock_configs

logger = getLogger()

def run_paper_uvw_benchmark(params) -> pathlib.Path:
    FCTPP = FullyConnectedTPProblem

    problems =  list(itertools.chain(
        e3nn_torch_tetris_polynomial,
        diffdock_configs
    ))

    float64_problems = copy.deepcopy(problems)
    for problem in float64_problems: 
        problem.irrep_dtype = np.float64
        problem.weight_dtype = np.float64
    
    #problems += float64_problems
    #problems = float64_problems

    implementations = [
        #E3NNTensorProductCompiledCUDAGraphs,
        #CUETensorProduct,
        LoopUnrollTP]

    tests = [TestDefinition(implementation, problem, direction, correctness=True, benchmark=True) 
                for problem, direction, implementation
                in itertools.product(problems, params.directions, implementations)]

    bench_suite = TestBenchmarkSuite(
            correctness_threshold = 5e-5,
            num_warmup=100,
            num_iter=100,
            bench_batch_size=params.batch_size,
            prng_seed=11111,
            torch_op=True,
            test_name="uvw"
        )
    
    logger.setLevel(logging.INFO)
    return bench_suite.run(tests, output_folder=params.output_folder)

if __name__ == '__main__':
    run_paper_uvw_benchmark()