import numpy as np

from openequivariance.benchmark.random_buffer_utils import (
    get_random_buffers_forward,
    get_random_buffers_backward,
    get_random_buffers_double_backward,
)
from openequivariance.benchmark.perf_metrics_utils import (
    calculate_minimum_flops_forward,
    calculate_minimum_memory_streamed_forward,
    calculate_minimum_memory_streamed_backward,
)
from openequivariance.implementations.utils import calculate_total_nnz
from openequivariance.implementations.TensorProductBase import TensorProductBase
from openequivariance.implementations.e3nn_lite import TPProblem
from openequivariance.implementations.CUETensorProduct import CUETensorProduct
from openequivariance.benchmark.logging_utils import getLogger, bcolors

logger = getLogger()


def calculate_performance_statistics(
    problem: TPProblem,
    batch_size: int,
    total_flops: int,
    total_memory_streamed: int,
    time_millis: np.ndarray,
) -> dict:
    result = {}

    throughputs_gflops = [float(x) for x in total_flops / (time_millis * 1e6)]
    bandwidth_gbps = [float(x) for x in total_memory_streamed / (time_millis * 1e6)]
    nnz = calculate_total_nnz(problem)
    time_millis = [float(x) for x in time_millis]

    result |= {
        "total_cg_nnz": nnz,
        "flops_per_tp": total_flops / batch_size,
        "L1": str(problem.irreps_in1),
        "L2": str(problem.irreps_in2),
        "L3": str(problem.irreps_out),
        "L1_rep_len": problem.irreps_in1.dim,
        "L2_rep_len": problem.irreps_in2.dim,
        "L3_rep_len": problem.irreps_out.dim,
        "rep_dtype": str(problem.irrep_dtype),
        "weight_dtype": str(problem.weight_dtype),
        "arithmetic_intensity (FLOPs / byte)": total_flops / total_memory_streamed,
        "batch_size": batch_size,
        "time_millis": time_millis,
        "throughputs_gflops": throughputs_gflops,
        "bandwidth_gbps": bandwidth_gbps,
    }

    logger.info(
        f"{bcolors.OKCYAN}Avg. Throughput: {bcolors.ENDC} {bcolors.WARNING}{np.mean(throughputs_gflops):.2f} ± {np.std(throughputs_gflops):.2f} GFLOPs{bcolors.ENDC}"
    )
    logger.info(
        f"{bcolors.OKCYAN}Avg. Bandwidth : {bcolors.ENDC} {bcolors.WARNING}{np.mean(bandwidth_gbps):.2f} ± {np.std(bandwidth_gbps):.2f} GBPs{bcolors.ENDC}"
    )
    logger.info(
        f"{bcolors.OKCYAN}Avg. Walltime  : {bcolors.ENDC} {bcolors.WARNING}{np.mean(time_millis):.2f} ± {np.std(time_millis):.2f} ms{bcolors.ENDC}"
    )
    return result


def benchmark_forward(
    problem: TPProblem,
    implementation: type[TensorProductBase],
    batch_size: int,
    num_warmup: int,
    num_iter: int,
    prng_seed: int,
    torch_op: bool,
) -> dict:
    """
    This function sets up the necessary materials and calls the internal benchmarker
    """
    result = {
        "tp_direction": "forward",
        "num_warmup": num_warmup,
        "num_iter": num_iter,
        "prng_seed": prng_seed,
    }

    L1_in, L2_in, weights, L3_buffer = get_random_buffers_forward(
        problem, batch_size, prng_seed
    )
    if problem.shared_weights and implementation == CUETensorProduct:
        weights = weights[np.newaxis, :]

    logger.info("Initialized input / output data.")
    tp = implementation(problem, torch_op=torch_op)

    # BENCHMARK
    try:
        time_millis = tp.benchmark_forward(
            num_warmup=num_warmup,
            num_iter=num_iter,
            L1_in=L1_in,
            L2_in=L2_in,
            weights=weights,
            L3_buffer=L3_buffer,
        )
    except NotImplementedError:
        logger.warning(
            "Benchmarking is not implemented, time millis replaced with -1's"
        )
        time_millis = np.full(shape=num_iter, fill_value=-1)

    # FLOPS
    try:
        flops = tp.calculate_flops_forward(batch_size=batch_size)
    except NotImplementedError:
        logger.warning(
            "Actual flop count not calculated, so minimum values are being used"
        )
        flops = calculate_minimum_flops_forward(problem, batch_size=batch_size)

    # DATA
    try:
        memory_streamed = tp.calculate_memory_streamed_backward(batch_size=batch_size)
    except NotImplementedError:
        logger.warning(
            "Actual memory streamed not calculated, so minimum values are being used"
        )
        memory_streamed = calculate_minimum_memory_streamed_forward(
            problem, batch_size=batch_size
        )

    result |= calculate_performance_statistics(
        problem=problem,
        batch_size=batch_size,
        total_flops=flops["total"],
        total_memory_streamed=memory_streamed["total"],
        time_millis=time_millis,
    )

    return result


def benchmark_backward(
    problem: TPProblem,
    implementation: type[TensorProductBase],
    batch_size: int,
    num_warmup: int,
    num_iter: int,
    prng_seed: int,
    torch_op: bool,
) -> dict:
    result = {
        "tp_direction": "backward",
        "num_warmup": num_warmup,
        "num_iter": num_iter,
        "prng_seed": prng_seed,
    }

    in1, in2, out_grad, weights, weights_grad, in1_grad, in2_grad = (
        get_random_buffers_backward(problem, batch_size, prng_seed)
    )
    if problem.shared_weights and implementation == CUETensorProduct:
        weights = weights[np.newaxis, :]

    logger.info("Initialized input / output data.")
    tp = implementation(problem, torch_op=torch_op)

    try:
        time_millis = tp.benchmark_backward(
            num_warmup=num_warmup,
            num_iter=num_iter,
            L1_in=in1,
            L2_in=in2,
            L3_buffer=out_grad,
            weights=weights,
            L1_grad=in1_grad,
            L2_grad=in2_grad,
            weights_grad=weights_grad,
        )
    except NotImplementedError:
        logger.warning(
            "Benchmarking is not implemented, time millis replaced with -1's"
        )
        time_millis = np.full(shape=num_iter, fill_value=-1)

    try:
        flops = tp.calculate_flops_backward(batch_size=batch_size)
    except NotImplementedError:
        try:
            flops = calculate_minimum_flops_forward(tpp=problem, batch_size=batch_size)
            logger.warning(
                "Actual flops was not calculated, so minimum values are being used"
            )
        except NotImplementedError:
            logger.warning(
                "Minimum Backwards flops calculations are not implemented, -1 is a placeholder"
            )
            flops = {"total": -1}

    try:
        memory_streamed = tp.calculate_memory_streamed_backward(batch_size=batch_size)
    except NotImplementedError:
        logger.warning(
            "Actual memory streamed was not calculated, so minimum values are being"
        )
        memory_streamed = calculate_minimum_memory_streamed_backward(
            tpp=problem, batch_size=batch_size
        )

    result |= calculate_performance_statistics(
        problem=problem,
        batch_size=batch_size,
        total_flops=flops["total"],
        total_memory_streamed=memory_streamed["total"],
        time_millis=time_millis,
    )

    return result


def benchmark_double_backward(
    problem: TPProblem,
    implementation: type[TensorProductBase],
    batch_size: int,
    num_warmup: int,
    num_iter: int,
    prng_seed: int,
    torch_op: bool,
) -> dict:
    result = {
        "tp_direction": "double_backward",
        "num_warmup": num_warmup,
        "num_iter": num_iter,
        "prng_seed": prng_seed,
    }

    in1, in2, out_grad, weights, weights_grad, in1_grad, in2_grad, out_double_grad = (
        get_random_buffers_double_backward(problem, batch_size, prng_seed)
    )

    if problem.shared_weights and implementation == CUETensorProduct:
        weights = weights[np.newaxis, :]

    logger.info("Initialized input / output data.")
    tp = implementation(problem, torch_op=torch_op)

    try:
        time_millis = tp.benchmark_double_backward(
            num_warmup=num_warmup,
            num_iter=num_iter,
            L1_in=in1,
            L2_in=in2,
            L3_buffer=out_grad,
            weights=weights,
            L1_grad=in1_grad,
            L2_grad=in2_grad,
            weights_grad=weights_grad,
            L3_double_grad=out_double_grad,
        )
    except NotImplementedError:
        logger.warning(
            "Benchmarking is not implemented, time millis replaced with -1's"
        )
        time_millis = np.full(shape=num_iter, fill_value=-1)

    try:
        flops = tp.calculate_flops_backward(batch_size=batch_size)
    except NotImplementedError:
        try:
            flops = calculate_minimum_flops_forward(tpp=problem, batch_size=batch_size)
            logger.warning(
                "Actual flops was not calculated, so minimum values are being used"
            )
        except NotImplementedError:
            logger.warning(
                "Minimum Backwards flops calculations are not implemented, -1 is a placeholder"
            )
            flops = {"total": -1}

    try:
        memory_streamed = tp.calculate_memory_streamed_backward(batch_size=batch_size)
    except NotImplementedError:
        logger.warning(
            "Actual memory streamed was not calculated, so minimum values are being"
        )
        memory_streamed = calculate_minimum_memory_streamed_backward(
            tpp=problem, batch_size=batch_size
        )

    result |= calculate_performance_statistics(
        problem=problem,
        batch_size=batch_size,
        total_flops=flops["total"],
        total_memory_streamed=memory_streamed["total"],
        time_millis=time_millis,
    )

    return result
