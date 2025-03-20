import openequivariance as oeq

from openequivariance.benchmark.tpp_creation_utils import ChannelwiseTPP, FullyConnectedTPProblem 
from openequivariance.benchmark.TestBenchmarkSuite import TestBenchmarkSuite, TestDefinition 
from openequivariance.benchmark.ConvBenchmarkSuite import *

import itertools

CTPP = ChannelwiseTPP
FCTPP = FullyConnectedTPProblem

def test_single_instruction_uvu():
    directions = ['forward', 'backward']

    multiplicities = [
        (1, 1, 1), (2, 1, 2), (4, 1, 4), (8, 1, 8), (16, 1, 16), 
        (32, 1, 32), (5, 1, 5), (13, 1, 13), (19, 1, 19),
        (33, 1, 33), (49, 1, 49), (50, 1, 50), (123, 1, 123),
        (128, 1, 128), (256, 1, 256), (512, 1, 512),
        (1, 2, 1), (1, 4, 1), (1, 8, 1), (1, 16, 1), (1, 32, 1),
        (16, 3, 16), (16, 4, 16), (16, 8, 16), (24, 24, 24), (32, 32, 32) 
    ]

    ir_types = [
        (0, 0, 0), (1, 1, 1), (1, 0, 1), (1, 2, 1),
        (2, 0, 2), (2, 2, 4), (2, 2, 2), (5, 3, 5), (7, 2, 5) 
    ]

    configurations = [
        (f"{m1}x{t1}e", f"{m2}x{t2}e", f"{m3}x{t3}e")
        for (m1, m2, m3) in multiplicities
        for (t1, t2, t3) in ir_types
    ]

    instructions=[(0, 0, 0, "uvu", True)]
    problems = [
        oeq.TPProblem(c[0], c[1], c[2], 
                        instructions, shared_weights=False, internal_weights=False)
        for c in configurations
    ]
    tests = [TestDefinition(oeq.TensorProduct, problem, direction, 
                            correctness=True, benchmark=False) 
                for problem, direction
                in itertools.product(problems, directions)]

    bench_suite = TestBenchmarkSuite(
        correctness_threshold = 5e-5, prng_seed=11111, torch_op=True, correctness_batch_size=500)
    bench_suite.run(tests, progressbar=True)    


if __name__=='__main__':
    test_single_instruction_uvu()
