import itertools
import logging
import copy
import pathlib
from typing import List
import numpy as np

from torch._functorch import config

from openequivariance.benchmark.logging_utils import getLogger
from openequivariance.implementations.E3NNTensorProduct import (
    E3NNTensorProductCompiledCUDAGraphs,
)
from openequivariance.implementations.CUETensorProduct import CUETensorProduct
from openequivariance.implementations.TensorProduct import TensorProduct
from openequivariance.benchmark.TestBenchmarkSuite import (
    TestBenchmarkSuite,
    TestDefinition,
)
from openequivariance.benchmark.benchmark_problems import (
    e3nn_torch_tetris_poly_problems,
    diffdock_problems,
)

logger = getLogger()


@config.patch("donated_buffer", False)
def run_paper_uvw_benchmark(params) -> pathlib.Path:
    problems = list(itertools.chain(e3nn_torch_tetris_poly_problems(), diffdock_problems()))

    float64_problems = copy.deepcopy(problems)
    for problem in float64_problems:
        problem.irrep_dtype = np.float64
        problem.weight_dtype = np.float64

    problems += float64_problems

    implementations: List[TensorProduct] = [
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


if __name__ == "__main__":
    run_paper_uvw_benchmark()
