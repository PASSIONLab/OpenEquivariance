from itertools import product
import logging

import e3nn
from e3nn import o3

from openequivariance.benchmark.TestBenchmarkSuite import TestBenchmarkSuite, TestDefinition, Direction
from openequivariance.implementations.E3NNTensorProduct import E3NNTensorProduct
from openequivariance.implementations.CUETensorProduct import CUETensorProduct 
from openequivariance.implementations.LoopUnrollTP import LoopUnrollTP

from openequivariance.implementations.e3nn_lite import TPProblem
from openequivariance.benchmark.tpp_creation_utils import FullyConnectedTPProblem, ChannelwiseTPP
from openequivariance.benchmark.logging_utils import getLogger
from openequivariance.benchmark.benchmark_configs import mace_nequip_problems


implementations = [
    LoopUnrollTP, 
    E3NNTensorProduct,
    CUETensorProduct,
]

problems = mace_nequip_problems

# [
#     # FullyConnectedTPProblem("1x1e", "1x1e", "1x1e"),
#     ChannelwiseTPP("1x1e", "1x1e", "1x1e"),
#     ChannelwiseTPP('32x0o + 32x0e + 32x1o + 32x1e', '0e + 1o', '32x0o + 32x0e + 32x1o + 32x1e')


# ]



directions : list[Direction] = [
    # 'forward',
    # 'backward',
    'double_backward',
]

tests = [TestDefinition(implementation, problem, direction, correctness=False) for  problem, direction, implementation,  in product(problems, directions, implementations)]

if __name__ == "__main__":

    logger = getLogger() 

    logger.setLevel(logging.INFO)
    test_suite = TestBenchmarkSuite(
        bench_batch_size=50000
    )
    test_suite.run(tests)