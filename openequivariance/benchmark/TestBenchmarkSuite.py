import json
import os
import time
import pathlib

from typing import NamedTuple, Optional, Literal, Any, get_args
from dataclasses import dataclass

from openequivariance.extlib.kernel_wrapper import DeviceProp
from openequivariance.implementations.TensorProductBase import TensorProductBase

from openequivariance.benchmark.logging_utils import getLogger
from openequivariance.extlib.kernel_wrapper import *
from openequivariance.implementations.e3nn_lite import TPProblem
from openequivariance.benchmark.correctness_utils import correctness_forward, correctness_backward
from openequivariance.benchmark.benchmark_utils import benchmark_forward, benchmark_backward
from openequivariance import package_root

logger = getLogger()

Direction = Literal['forward', 'backward']

class TestDefinition(NamedTuple):
    implementation : type[TensorProductBase]
    problem : TPProblem
    direction : Direction
    correctness : bool = True
    benchmark : bool = True

@dataclass(init=True, repr=False, eq=False)
class TestBenchmarkSuite:
    num_warmup : int = 10
    num_iter : int = 30
    correctness_batch_size : int = 10_000
    bench_batch_size : int = 10_000_000
    prng_seed : int = 12345
    reference_implementation : Optional[type[TensorProductBase]] = None
    correctness_threshold : float = 5e-7
    torch_op : bool = True

    @staticmethod
    def validate_inputs(test_list : list[TestDefinition]) -> None:
        """
        Just does empty list and type checking to catch bad input 
        """
        assert isinstance(test_list, list) 
        assert len(test_list) != 0
        for test in test_list:
            assert isinstance(test, TestDefinition)
            assert issubclass(test.implementation, TensorProductBase)
            assert isinstance(test.problem, TPProblem)
            assert test.direction in get_args(Direction)
            assert isinstance(test.correctness, bool)
            assert isinstance(test.benchmark, bool)

    @staticmethod
    def generate_metadata(test_list : list[TestDefinition]) -> dict[str, Any]: 
        impls, tpps, directions, corectnesses, benchmarks = zip(*test_list)
        config_strs = list(dict.fromkeys([str(tpp) for tpp in tpps]))
        config_reprs = list(dict.fromkeys([repr(tpp) for tpp in tpps]))
        config_labels = list(dict.fromkeys([tpp.label for tpp in tpps]))
        implementation_names = list(dict.fromkeys([impl.name() for impl in impls])) 
        directions = list(dict.fromkeys(directions))
        did_correctness = any(corectnesses)
        did_benchmark = any(benchmarks)
        
        dp = DeviceProp(0)

        metadata = {
                'config_strs' : config_strs,
                'config_reprs': config_reprs, 
                'config_labels' : config_labels,
                'implementations' : implementation_names,
                'directions' : directions,
                'did_correctness' :  did_correctness, 
                'did_benchmark' : did_benchmark,
                'gpu_name' : dp.name,
            }
        
        test_details = {}
        for test_ID, test in enumerate(test_list):
            test_details[test_ID] = {
                'implementation' : test.implementation.name(),
                'problem' : repr(test.problem),
                'direction' : test.direction, 
                'correctness' : test.correctness,
                'benchmark' : test.benchmark,
                }
        
        metadata['test details'] = test_details

        return metadata

    def run(self, test_list : list[TestDefinition]) -> pathlib.Path:        
        
        TestBenchmarkSuite.validate_inputs(test_list)

        millis_since_epoch = round(time.time() * 1000)
        output_folder = pathlib.Path(f'{package_root}/outputs/{millis_since_epoch}')
        output_folder.mkdir(parents=True)

        metadata = TestBenchmarkSuite.generate_metadata(test_list)

        with open(os.path.join(output_folder,'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2) 

        for test_ID, test in enumerate(test_list): 
            impl = test.implementation
            tpp = test.problem

            logger.info(f'Starting Test ID: {test_ID}')
            logger.info(f'Config: {str(tpp)}')
            logger.info(f'Irrep dtype: {tpp.irrep_dtype.__name__}')
            logger.info(f'Weight dype: {tpp.weight_dtype.__name__}')
            if(tpp.label):
                logger.info(f'Label: {tpp.label}')
            logger.info(f'Implementation Name: {impl.name()}')
            logger.info(f'Test Direction: {test.direction}')
            logger.info(f"Torch Overhead Included: {self.torch_op}")


            result = {
                "config_str" : str(tpp),
                "config_repr" : repr(tpp),
                "config_label" : tpp.label, 
                "direction" :  test.direction, 
                "implementation_name": impl.name(),
                "correctness": str(test.correctness),
                "benchmark": str(test.benchmark),
                "torch_overhead_included": self.torch_op
            }

            if test.direction == 'forward':
                if test.correctness:
                    logger.info("Starting correctness check...")
                    result['correctness results'] = correctness_forward(
                        problem=tpp,
                        test_implementation=impl,
                        reference_implementation=self.reference_implementation,
                        batch_size=self.correctness_batch_size,
                        correctness_threshold=self.correctness_threshold,
                        prng_seed=self.prng_seed,
                    )
                    logger.info("Finished correctness check...")
                if test.benchmark:
                    result['benchmark results'] = benchmark_forward(
                        problem=tpp,
                        implementation=impl,
                        batch_size=self.bench_batch_size,
                        num_warmup=self.num_warmup,
                        num_iter=self.num_iter,
                        prng_seed=self.prng_seed,
                        torch_op=self.torch_op
                    )


            if test.direction == 'backward':
                pass 
                if test.correctness: 
                    logger.info("Starting correctness check...")
                    result ['correctness results'] = correctness_backward(
                        problem=tpp,
                        test_implementation=impl,
                        reference_implementation=self.reference_implementation,
                        batch_size=self.correctness_batch_size,
                        correctness_threshold=self.correctness_threshold,
                        prng_seed=self.prng_seed
                    )
                    logger.info("Finished correctness check...")
                if test.benchmark: 
                    result ['benchmark results'] = benchmark_backward(
                        problem=tpp,
                        implementation=impl,
                        batch_size=self.bench_batch_size,
                        num_warmup=self.num_warmup,
                        num_iter=self.num_iter,
                        prng_seed=self.prng_seed,
                        torch_op=self.torch_op
                    )
    
            fname = pathlib.Path(f"{output_folder}/{test_ID}_{impl.name()}.json")

            pretty_result = json.dumps(obj=result, indent=2).replace('\\n', '\n')
            logger.debug(pretty_result)
            with open(fname, 'w') as f:
                json.dump(result, f, indent=2)

            logger.info(f'Finished Test ID: {test_ID}')
        
        return output_folder