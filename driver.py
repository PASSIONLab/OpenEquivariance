#import e3nn
#from src.implementations.E3NNTensorProduct import *

import itertools
import typing

from src.benchmark.logging_utils import *
from src.benchmark.e3nn_lite_utils import *
from build.kernel_wrapper import *
from src.benchmark.random_buffer_utils import get_random_buffers_forward, get_random_buffers_backward
from src.benchmark.TestBenchmarkSuite import TestBenchmarkSuite, TestDefinition, Direction
from src.benchmark.tpp_creation_utils import *
from src.implementations.LoopUnrollTP import LoopUnrollTP
from src.implementations.NumpyTensorProduct import NumpyTensorProduct
from src.implementations.MultiplicityOuterProductTP import MultiplicityOuterProductTP


from src.implementations.e3nn_lite import *

import numpy as np
import numpy.linalg as la

logger = getLogger()

def debug(tp_impl : type[TensorProduct], config : TPProblem, direction : Direction) -> None:
    assert issubclass(tp_impl, TensorProduct)
    assert isinstance(config, TPProblem)
    assert direction in typing.get_args(Direction)

    batch_size = 1000000
    prng_seed = 12345
    
    tp = tp_impl(config)

    from src.implementations.E3NNTensorProduct import E3NNTensorProduct
    e3nn_tp = E3NNTensorProduct(config)

    logger.debug(repr(config))
    if direction == "forward":
        in1, in2, weights, out = get_random_buffers_forward(tpp=config, batch_size=batch_size, prng_seed=prng_seed)

        print(f"{out =}")
        test_out = out.copy()
        tp.forward_cpu(
            L1_in=in1, 
            L2_in=in2, 
            L3_out=test_out, 
            weights=weights
            )   
        
        print(f"{test_out = }")

        ground_truth_out = out.copy()
        e3nn_tp.forward_cpu(
            L1_in=in1, 
            L2_in=in2, 
            L3_out=ground_truth_out,
            weights=weights
            )
        
        print(f"{ground_truth_out = }")

        print("LA.Norm:")
        print(la.norm((test_out - ground_truth_out).flatten(), ord=np.inf))

        print("test_output / ground_truth_output")
        print( test_out / ground_truth_out)

    elif direction == "backward":
        in1, in2, out_grad, weights, weights_grad, in1_grad, in2_grad = get_random_buffers_backward(tpp=config, batch_size=batch_size, prng_seed=prng_seed)

        test_in1_grad = in1_grad.copy()
        test_in2_grad = in2_grad.copy()
        test_weights_grad = weights_grad.copy()
        tp.backward_cpu(
            L1_in=in1,
            L1_grad=test_in1_grad,
            L2_in=in2,
            L2_grad=test_in2_grad,
            L3_grad=out_grad,
            weights=weights,
            weights_grad=test_weights_grad            
        )

        print(test_in1_grad)
        print(test_in2_grad)
        print(test_weights_grad)
    else:
        assert(False)

if __name__=='__main__':
   
    tests = [
        # single_inst_conf("32x5e", "1x3e", "32x5e", "uvu", True),
        # single_inst_conf("32x5e", "1x5e", "32x3e", "uvu", True),
        # mace_conf("32x3e + 32x2e", "1x0e + 1x1e", 3), # Last value is Lmax
        # ("32x3e + 32x2e + 32x1e + 32x0e", "1x0e + 1x1e + 1x2e", 3), 
        # ("32x2e + 32x1e + 32x0e", "1x0e + 1x1e", 3)s
    ]  
    
    FCTPP = FullyConnectedTPProblem
    basic_fully_connected_problems = [
        FCTPP("1x1e", "1x1e", "1x1e"),
        FCTPP("1x1e", "2x1e", "1x1e"), 
        FCTPP("2x1e", "1x1e", "1x1e"),
        FCTPP("2x1e", "2x1e", "1x1e"),
        FCTPP("2x1e", "2x1e", "2x1e"),
        
    ]

    increasing_multiplicty_fully_connected_problems = [
        FCTPP("2x1e", "2x1e", "4x1e"),
        FCTPP("4x1e", "4x1e", "4x1e"),
        FCTPP("8x1e", "8x1e", "8x1e"),
        FCTPP("16x1e", "16x1e", "16x1e"),
        FCTPP("32x1e", "32x1e", "32x1e"),
    ]

    basic_multi_interaction_problems = [
        FCTPP("2x1e + 1x0e", "2x1e", "4x1e"),
        FCTPP("2x1e", "2x1e + 1x0e", "4x1e"),
        FCTPP("2x1e + 1x0e", "2x1e + 1x0e", "4x1e"),
        FCTPP("2x1e + 1x0e", "2x1e + 1x0e", "4x1e + 1x0e"),
    ]

    problems = itertools.chain.from_iterable([
        # basic_fully_connected_problems,
        increasing_multiplicty_fully_connected_problems,
        # basic_multi_interaction_problems,
    ])

    implementations = [MultiplicityOuterProductTP]

    directions = ['forward']

    tests = [TestDefinition(implementation, problem, direction, benchmark=True) 
             for implementation, problem, direction 
             in itertools.product(implementations, problems, directions)]

    logger.setLevel(logging.INFO)
    bench_suite = TestBenchmarkSuite(
        correctness_threshold = 5e-6,
        bench_batch_size=100000,
    )
    bench_suite.run(tests)

    # debug(MultiplicityOuterProductTP, increasing_multiplicty_fully_connected_problems[3], direction="forward")