import json
import os
import time
import pickle
import pathlib
import numpy as np

import openequivariance as oeq
from openequivariance.benchmark.logging_utils import getLogger
from openequivariance.implementations.convolution.ConvolutionBase import CoordGraph

logger = getLogger()


def load_graph(filename):
    coords, rows, cols = [None] * 3
    name = pathlib.Path(filename).stem
    with open(filename, "rb") as f:
        logger.info(f"Loading {name} from pickle...")
        result = pickle.load(f)
        coords, rows, cols, name = result["coords"], result["row"], result["col"], name
        logger.info(
            f"Graph {name} loaded with {len(coords)} nodes and {len(rows)} edges."
        )

    return CoordGraph(coords, rows.astype(np.int64), cols.astype(np.int64), name)


class ConvBenchmarkSuite:
    def __init__(
        self,
        configs,
        num_warmup=10,
        num_iter=30,
        reference_impl=None,
        test_name=None,
        prng_seed=12345,
        correctness_threshold=1e-5,
    ):
        self.configs = configs
        self.num_warmup = num_warmup
        self.num_iter = num_iter
        self.reference_impl = reference_impl
        self.prng_seed = 12345
        self.correctness_threshold = correctness_threshold
        self.exp_count = 0
        self.test_name = test_name

        self.millis_since_epoch = round(time.time() * 1000)

    def run(
        self,
        graph,
        implementations,
        direction,
        output_folder=None,
        correctness=True,
        benchmark=True,
        high_precision_ref=False,
    ):
        if output_folder is None:
            if oeq._check_package_editable():
                output_folder = (
                    oeq._editable_install_output_path / f"{self.millis_since_epoch}"
                )
            else:
                raise ValueError(
                    "output folder must be specified for non-editable installs."
                )
        else:
            output_folder = pathlib.Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        metadata = {
            "test_name": self.test_name,
            "configs": [str(config) for config in self.configs],
            "implementations": [impl.name() for impl in implementations],
            "graph": graph.name,
        }
        if self.exp_count == 0:
            with open(os.path.join(output_folder, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

        for config in self.configs:
            for impl in implementations:
                tc_name = f"{config}, {impl.name()}"
                logger.info(f"Starting {tc_name}, graph {graph.name}, {direction}")
                conv = impl(config)

                if direction == "forward":
                    if correctness:
                        correctness = conv.test_correctness_forward(
                            graph,
                            thresh=self.correctness_threshold,
                            prng_seed=self.prng_seed,
                            reference_implementation=self.reference_impl,
                            high_precision_ref=high_precision_ref,
                        )

                    if benchmark:
                        benchmark = conv.benchmark_forward(
                            self.num_warmup, self.num_iter, graph, prng_seed=12345
                        )

                if direction == "backward":
                    if correctness:
                        correctness = conv.test_correctness_backward(
                            graph,
                            thresh=self.correctness_threshold,
                            prng_seed=self.prng_seed,
                            reference_implementation=self.reference_impl,
                            high_precision_ref=high_precision_ref,
                        )

                    if benchmark:
                        benchmark = conv.benchmark_backward(
                            self.num_warmup, self.num_iter, graph, prng_seed=12345
                        )

                if direction == "double_backward":
                    if correctness:
                        correctness = conv.test_correctness_double_backward(
                            self.graph,
                            thresh=self.correctness_threshold,
                            prng_seed=self.prng_seed,
                            reference_implementation=self.reference_impl,
                            high_precision_ref=high_precision_ref,
                        )

                    assert not benchmark

                result = {
                    "config": str(config),
                    "irrep_dtype": str(config.irrep_dtype),
                    "weight_dtype": str(config.weight_dtype),
                    "torch_overhead_included": conv.torch_op,
                    "direction": direction,
                    "graph": graph.name,
                    "name": impl.name(),
                    "correctness": correctness,
                    "benchmark": benchmark,
                }

                fname = pathlib.Path(
                    f"{output_folder}/{self.exp_count}_{impl.name()}_{graph.name}.json"
                )
                with open(fname, "w") as f:
                    json.dump(result, f, indent=2)
                self.exp_count += 1

                logger.info(f"Finished {tc_name}, graph {graph.name}")

        return output_folder
