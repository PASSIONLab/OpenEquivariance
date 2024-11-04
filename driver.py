#import e3nn
#from src.implementations.E3NNTensorProduct import *

from src.benchmark.logging_utils import *
from build.kernel_wrapper import *
from src.benchmark.TestBenchmarkSuite import *
from src.implementations.LoopUnrollTP import *
from src.implementations.NumpyTensorProduct import *

from src.implementations.e3nn_lite import *

import numpy as np
import numpy.linalg as la

logger = getLogger()

def debug(tp_impl, config, direction="forward"): 
    L1, L2, L3 = config.irreps_in1, config.irreps_in2, config.irreps_out 
    batch_size = 1

    tp = tp_impl(config)

    rng = np.random.default_rng(12345)
    L1_in  = np.array(rng.uniform(size=(batch_size, L1.dim)), dtype=np.float32)
    L2_in  = np.array(rng.uniform(size=(batch_size, L2.dim)), dtype=np.float32)
    weights = np.array(rng.uniform(size=(batch_size, config.weight_numel)), dtype=np.float32) 

    L3_out = np.zeros((batch_size, L3.dim), dtype=np.float32)

    if direction == "forward":
        tp.exec_tensor_product_cpu(L1_in, L2_in, L3_out, weights)
        _, ground_truth = tp.test_correctness(L1_in, L2_in, weights, L3_out)
        print(la.norm((L3_out-ground_truth).flatten(), ord=np.inf))
        print(L3_out / ground_truth)
    elif direction == "backward":
        L3_grad = L3_out
        L3_grad[:] = rng.uniform(size=(batch_size, L3.dim)) 
        weights = np.array(rng.uniform(size=(batch_size, config.weight_numel)), dtype=np.float32) # Assumes no shared weights
        L1_grad, L2_grad, weights_grad = tp.backward_cpu(L1_in, L2_in, L3_grad, weights)
        print(L1_grad)
        print(L2_grad)
        print(weights_grad)
    else:
        assert(False)

if __name__=='__main__':
    tests = [
        #single_inst_conf("32x5e", "1x5e", "32x3e", "uvu", True),
        mace_conf("32x3e + 32x2e", "1x0e + 1x1e", 3), # Last value is Lmax
        #("32x3e + 32x2e + 32x1e + 32x0e", "1x0e + 1x1e + 1x2e", 3), 
        #("32x2e + 32x1e + 32x0e", "1x0e + 1x1e", 3)
    ]

    bench_suite = TestBenchmarkSuite(tests, bench_batch_size=1000000)
    bench_suite.run([LoopUnrollTP], direction="forward", reference_impl=NumpyTensorProduct)

    #debug(LoopUnrollTP, tests[0], direction="forward")