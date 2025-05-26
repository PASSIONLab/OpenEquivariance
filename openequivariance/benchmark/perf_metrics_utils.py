import math

from openequivariance.implementations.utils import (
    count_cg_non_zero,
    sparse_outer_product_work,
)
from openequivariance.implementations.TensorProductBase import TensorProductBase
from openequivariance.implementations.e3nn_lite import TPProblem
from openequivariance.benchmark.logging_utils import getLogger
import numpy as np

logger = getLogger()


def calculate_minimum_memory_streamed_forward(
    tpp: TPProblem, batch_size: int
) -> dict[str, int]:
    """
    This represents an absolute minimum amount of memory that could be streamed on an ideal machine
    It returns the number of bytes streamed total and from each source
    """
    data_size = {}
    irrep_word_size = np.dtype(tpp.irrep_dtype).itemsize
    weight_word_size = np.dtype(tpp.weight_dtype).itemsize

    data_size["input 1"] = tpp.irreps_in1.dim * batch_size * irrep_word_size
    data_size["input 2"] = tpp.irreps_in2.dim * batch_size * irrep_word_size
    data_size["output"] = tpp.irreps_out.dim * batch_size * irrep_word_size
    data_size["weights"] = tpp.weight_numel * batch_size * weight_word_size
    data_size["total"] = sum(data_size.values())
    return data_size


def calculate_minimum_memory_streamed_backward(tpp: TPProblem, batch_size: int) -> dict:
    """
    This represents an absolute minimum amount of memory that could be streamed on an ideal machine
    It returns the number of bytes streamed total and from each source
    """
    data_size = {}
    irrep_word_size = np.dtype(tpp.irrep_dtype).itemsize
    weight_word_size = np.dtype(tpp.weight_dtype).itemsize

    data_size["input 1"] = tpp.irreps_in1.dim * batch_size * irrep_word_size
    data_size["input 1 grad"] = tpp.irreps_in1.dim * batch_size * irrep_word_size
    data_size["input 2"] = tpp.irreps_in2.dim * batch_size * irrep_word_size
    data_size["input 2 grad"] = tpp.irreps_in2.dim * batch_size * irrep_word_size
    data_size["output grad"] = tpp.irreps_out.dim * batch_size * irrep_word_size
    data_size["weights"] = tpp.weight_numel * batch_size * weight_word_size
    data_size["weights grad"] = tpp.weight_numel * batch_size * weight_word_size
    data_size["total"] = sum(data_size.values())
    return data_size


def calculate_minimum_flops_forward(tpp: TPProblem, batch_size: int) -> dict:
    """
    This is not actually calcuating the minimum value.
    Ideally you might share the outer product values between two inputs across multiple inputs.
    This is assuming that you form those values and reuse them once per CG decomp.
    """
    logger.warning("Minimum flops Calculation is not the true minimum")
    flops_count = {}
    flops_count["outer_products"] = 0
    flops_count["CG_decomposition"] = 0
    flops_count["linear_combination"] = 0
    for ins in tpp.instructions:  # type : Instruction
        l1, l2, l3 = (
            tpp.irreps_in1[ins.i_in1].ir.l,
            tpp.irreps_in2[ins.i_in2].ir.l,
            tpp.irreps_out[ins.i_out].ir.l,
        )

        flops_count["outer_products"] += sparse_outer_product_work(
            TensorProductBase.load_cg_tensor(l1, l2, l3)
        )
        flops_count["CG_decomposition"] += count_cg_non_zero(l1, l2, l3) * (
            ins.path_shape[0] * ins.path_shape[1]
        )
        flops_count["linear_combination"] += (
            (2 * l3 + 1) * math.prod(ins.path_shape) if ins.has_weight else 0
        )

    flops_count["outer_products"] *= batch_size
    flops_count["CG_decomposition"] *= 2 * batch_size
    flops_count["linear_combination"] *= 2 * batch_size

    flops_count["total"] = sum(flops_count.values())
    return flops_count


def calculate_minimum_flops_backward(tpp: TPProblem, batch_size: int) -> dict:
    """
    This is not actually calcuating the minumum value.
    Ideally you might share the outer product values between two inputs across multiple inputs.
    This is assuming that you form those values and reuse them once per CG decomp.
    """
    raise NotImplementedError("this needs to be implemented properly")
