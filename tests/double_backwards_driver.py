from itertools import product
import logging

import e3nn
from e3nn import o3

from openequivariance.benchmark.TestBenchmarkSuite import TestBenchmarkSuite, TestDefinition, Direction
from openequivariance.implementations.E3NNTensorProduct import E3NNTensorProduct
from openequivariance.implementations.CUETensorProduct import CUETensorProduct 
from openequivariance.implementations.TensorProduct import TensorProduct 

from openequivariance.implementations.e3nn_lite import TPProblem
from openequivariance.benchmark.tpp_creation_utils import FullyConnectedTPProblem, ChannelwiseTPP
from openequivariance.benchmark.logging_utils import getLogger
from openequivariance.benchmark.benchmark_configs import mace_nequip_problems


implementations = [
    # E3NNTensorProduct,
    #CUETensorProduct,
    TensorProduct, 
]

problems = mace_nequip_problems

directions : list[Direction] = [
    #'forward',
    #'backward',
    'double_backward',
]

tests = [TestDefinition(implementation, problem, direction, correctness=True, benchmark=True) for  problem, direction, implementation,  in product(problems, directions, implementations)]

if __name__ == "__main__":

    logger = getLogger() 

    logger.setLevel(logging.INFO)
    test_suite = TestBenchmarkSuite(
        bench_batch_size=50000
    )
    test_suite.run(tests)