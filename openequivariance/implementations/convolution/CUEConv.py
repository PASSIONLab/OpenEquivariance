import numpy as np
import itertools
from typing import Iterator

from openequivariance.implementations.CUETensorProduct import CUETensorProduct
from openequivariance.implementations.convolution.ConvolutionBase import ConvolutionBase


class CUEConv(ConvolutionBase):
    def __init__(self, config, idx_dtype=np.int64, torch_op=True):
        super().__init__(config, idx_dtype, torch_op)

        global torch
        import torch

        self.reference_tp = CUETensorProduct(config, torch_op)
        self.cue_tp = self.reference_tp.cue_tp

        from openequivariance.implementations.convolution.scatter import scatter_sum

        self.scatter_sum = scatter_sum

    def forward(self, L1_in, L2_in, weights, rows, cols):
        tp_outputs = self.cue_tp(L1_in[cols], L2_in, weights)
        return self.scatter_sum(
            src=tp_outputs, index=rows, dim=0, dim_size=L1_in.shape[0]
        )

    @staticmethod
    def name():
        return "CUEConvolution"


class CUEConvFused(ConvolutionBase):
    def __init__(self, config, idx_dtype=np.int64, torch_op=True):
        super().__init__(config, idx_dtype, torch_op)

        global torch
        import torch
        import e3nn.o3 as o3

        np_to_torch_dtype = {np.float32: torch.float32, np.float64: torch.float64}

        import cuequivariance as cue
        from cuequivariance_torch.primitives.tensor_product import (
            TensorProductUniform4x1dIndexed,
        )

        class O3_e3nn(cue.O3):
            def __mul__(  # pylint: disable=no-self-argument
                rep1: "O3_e3nn", rep2: "O3_e3nn"
            ) -> Iterator["O3_e3nn"]:
                return [O3_e3nn(l=ir.l, p=ir.p) for ir in cue.O3.__mul__(rep1, rep2)]

            @classmethod
            def clebsch_gordan(
                cls, rep1: "O3_e3nn", rep2: "O3_e3nn", rep3: "O3_e3nn"
            ) -> np.ndarray:
                rep1, rep2, rep3 = cls._from(rep1), cls._from(rep2), cls._from(rep3)

                if rep1.p * rep2.p == rep3.p:
                    return o3.wigner_3j(rep1.l, rep2.l, rep3.l).numpy()[None] * np.sqrt(
                        rep3.dim
                    )
                return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

            def __lt__(  # pylint: disable=no-self-argument
                rep1: "O3_e3nn", rep2: "O3_e3nn"
            ) -> bool:
                rep2 = rep1._from(rep2)
                return (rep1.l, rep1.p) < (rep2.l, rep2.p)

            @classmethod
            def iterator(cls) -> Iterator["O3_e3nn"]:
                for l in itertools.count(0):
                    yield O3_e3nn(l=l, p=1 * (-1) ** l)
                    yield O3_e3nn(l=l, p=-1 * (-1) ** l)

        descriptor = (
            cue.descriptors.channelwise_tensor_product(
                cue.Irreps(O3_e3nn, str(config.irreps_in1)),
                cue.Irreps(O3_e3nn, str(config.irreps_in2)),
                cue.Irreps(O3_e3nn, str(config.irreps_out)),
            )
            .squeeze_modes()
            .flatten_coefficient_modes()
        )

        self.tp = TensorProductUniform4x1dIndexed(
            descriptor.polynomial.operations[0][1],
            "cuda",
            math_dtype=np_to_torch_dtype[config.irrep_dtype],
        )

    def forward(self, L1_in, L2_in, weights, rows, cols):
        return self.tp(weights, L1_in, L2_in, None, rows, None, cols, L1_in.shape[0])

    @staticmethod
    def name():
        return "CUEConvolutionFused"
