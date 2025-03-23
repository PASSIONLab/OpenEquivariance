import itertools, typing, os 

import numpy as np
import numpy.linalg as la

from openequivariance.benchmark.logging_utils import *
from openequivariance.implementations.e3nn_lite import *
from openequivariance.benchmark.e3nn_lite_utils import *
from openequivariance.extlib import *
from openequivariance.benchmark.random_buffer_utils import get_random_buffers_forward, get_random_buffers_backward
from openequivariance.benchmark.TestBenchmarkSuite import TestBenchmarkSuite, TestDefinition, Direction
from openequivariance.benchmark.tpp_creation_utils import *
from openequivariance.implementations.LoopUnrollTP import LoopUnrollTP
from openequivariance.implementations.MultiplicityOuterProductTP import MultiplicityOuterProductTP
from openequivariance.implementations.E3NNTensorProduct import (
    E3NNTensorProduct, 
    E3NNTensorProductCompiledCUDAGraphs, 
    E3NNTensorProductCompiledMaxAutotuneCUDAGraphs,
    )
from openequivariance.implementations.CUETensorProduct import CUETensorProduct

logger = getLogger()

def debug(tp_impl : type[TensorProductBase], config : TPProblem, direction : Direction) -> None:
    assert issubclass(tp_impl, TensorProductBase)
    assert isinstance(config, TPProblem)
    assert direction in typing.get_args(Direction)

    batch_size = 10_000
    prng_seed = 12345
    
    tp = tp_impl(config)

    from openequivariance.implementations.E3NNTensorProduct import E3NNTensorProduct
    ref_tp = E3NNTensorProduct(config)

    logger.debug(repr(config))

    np.set_printoptions(formatter={'float': '{:0.3f}'.format})
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
        ref_tp.forward_cpu(
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
            L1_in=in1.copy(),
            L1_grad=test_in1_grad,
            L2_in=in2.copy(),
            L2_grad=test_in2_grad,
            L3_grad=out_grad.copy(),
            weights=weights.copy(),
            weights_grad=test_weights_grad            
        )

        ref_in1_grad = in1_grad.copy()
        ref_in2_grad = in2_grad.copy()
        ref_weights_grad = weights_grad.copy()

        ref_tp.backward_cpu(
            L1_in=in1.copy(),
            L1_grad=ref_in1_grad, 
            L2_in=in2.copy(), 
            L2_grad=ref_in2_grad,
            L3_grad=out_grad.copy(), 
            weights=weights.copy(),
            weights_grad=ref_weights_grad, 
        )


        for name, ground_truth, test_result in [
            ("L1_grad", ref_in1_grad , test_in1_grad),
            ("L2_grad", ref_in2_grad , test_in2_grad),
            ("weight_grad", ref_weights_grad, test_weights_grad),
            ]:
            print(name)
            print(ground_truth, "ground truth")
            print(test_result , "test_result" )
            print(test_result / ground_truth, "ratio")
            print("LA.Norm:")
            print(la.norm((test_result - ground_truth).flatten(), ord=np.inf))
    else:
        assert(False)
    np.set_printoptions()

if __name__=='__main__':
    #warp_matmul.test_simple_kernel()
    #exit(1)

    

    FCTPP = FullyConnectedTPProblem
    ChannelTPP = ChannelwiseTPP 
    basic_fully_connected_problems = [
        FCTPP("1x1e", "1x1e", "1x1e"),
        FCTPP("1x1e", "1x1e", "2x1e"),
        FCTPP("1x1e", "2x1e", "1x1e"), 
        FCTPP("2x1e", "1x1e", "1x1e"),
        FCTPP("2x1e", "2x1e", "1x1e"),
        FCTPP("2x1e", "2x1e", "2x1e"),
        FCTPP("2x1e", "2x1e", "4x1e") 
    ]

    increasing_multiplicity_fully_connected_problems = [
        FCTPP("2x1e", "2x1e", "2x1e"),
        FCTPP("4x1e", "4x1e", "4x1e"),
        FCTPP("8x1e", "8x1e", "8x1e"),
        FCTPP("16x1e", "16x1e", "16x1e"),
        FCTPP("32x1e", "32x1e", "32x1e"),
    ]

    full_size_uvw_case = [
        FCTPP("32x1e", "32x1e", "32x1e"),
        FCTPP("32x2e", "32x2e", "32x2e"),
        FCTPP("32x3e", "32x3e", "32x3e"),
        FCTPP("32x4e", "32x4e", "32x4e"),
        FCTPP("32x5e", "32x5e", "32x5e"),
    ]

    basic_multi_interaction_problems = [
        FCTPP("2x1e + 1x0e", "2x1e", "4x1e"),
        FCTPP("2x1e", "2x1e + 1x0e", "4x1e"),
        FCTPP("2x1e + 1x0e", "2x1e + 1x0e", "4x1e"),
        FCTPP("32x1e + 32x0e", "32x1e + 32x0e", "32x1e + 32x0e"),
    ]

    conv_problems = [  
        #FCTPP("32x0e + 32x0e + 24x1e + 24x1o + 16x2e + 16x2o", "1x0e+1x1o+1x2e+1x3o", "0o + 6x0e")
        #FCTPP("17x5e", "3x3e", "16x5e", shared_weights=False, internal_weights=False),

        #FCTPP(  "10x1o + 10x1e + 32x0e + 16x0e + 32x0o + 16x0o", 
        #        "1x0e + 1x1o", 
        #        "10x1o + 10x1e + 32x0e + 16x0e + 32x0o + 16x0o",
        #        shared_weights=False, label='DiffDock L = 1'),

        #FCTPP(  "10x1o", 
        #        "1x0e", 
        #        "10x1o",
        #        shared_weights=False, label='DiffDock L = 1')

        #FCTPP("10x1o + 10x1e + 48x0e + 48x0o", "1x0e + 1x1o + 1x2e", "10x1o + 10x1e + 48x0e + 48x0o", shared_weights=False, label='DiffDock L = 2'),

        SingleInstruction("48x5e", "1x7e", "48x5e", "uvw", True),
        #ChannelwiseTPP("128x0e+128x1o+128x2e", 
        #        "1x0e+1x1o+1x2e+1x3o",
        #        "128x0e+128x1o+128x2e+128x3o")
        #ChannelwiseTPP("48x0e", "2x0e", "48x0e")
        #FCTPP("48x0e", "2x0e", "1x0e", shared_weights=False),
        #FCTPP("10x1o + 10x1e + 48x0e", "1x0e + 1x1o + 1x2e", "10x1o + 10x1e + 48x0e", shared_weights=False, label='DiffDock-L=2'),
    ]

    for problem in conv_problems:
        problem.irrep_dtype = np.float64
        problem.weight_dtype = np.float64

    problems = list(itertools.chain(
        # basic_fully_connected_problems,
        # increasing_multiplicity_fully_connected_problems,
        # full_size_uvw_case,
        #basic_multi_interaction_problems,
        conv_problems,
    ))
 
    implementations = [
        LoopUnrollTP,
        #E3NNTensorProduct,
        #MultiplicityOuterProductTP,
        #CUETensorProduct, 
        #ManyOneUVWTP
        ]

    directions = ['forward', 'backward'] 

    tests = [TestDefinition(implementation, problem, direction, 
                correctness=True, benchmark=False) 
             for problem, direction, implementation
             in itertools.product(problems, directions, implementations)]
 
    bench_suite = TestBenchmarkSuite(
        correctness_threshold = 5e-5,
        num_warmup=100,
        num_iter=30,
        correctness_batch_size=10,
        bench_batch_size=50_000,
        prng_seed=11111,
        torch_op=False
    )

    logger.setLevel(logging.INFO)
    bench_suite.run([tests[0]])
    #  debug(MultiplicityOuterProductTP, basic_fully_connected_problems[0], direction="forward")

    #from openequivariance.benchmark.correctness_utils import correctness_double_backward
    #result = correctness_double_backward(
    #    problem = conv_problems[0],
    #    test_implementation = LoopUnrollTP,
    #    reference_implementation = E3NNTensorProduct,
    #    batch_size = 100, 
    #    correctness_threshold = 1e-5, 
    #    prng_seed = 12345  
    #)
    #exit(1)