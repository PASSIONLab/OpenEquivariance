import numpy as np

from typing import Iterator, Optional
from openequivariance.implementations.e3nn_lite import Irrep, Irreps, TPProblem

"""
This was taken from 

https://github.com/e3nn/e3nn/blob/0.5.4/e3nn/o3/_tensor_product/_sub.py

And adopted to create TPP's to avoid torch dependence
"""


class FullyConnectedTPProblem(TPProblem):
    def __init__(self, irreps_in1, irreps_in2, irreps_out, **kwargs) -> None:
        irreps_in1 = Irreps(irreps_in1)
        irreps_in2 = Irreps(irreps_in2)
        irreps_out = Irreps(irreps_out)

        instr = [
            (i_1, i_2, i_out, "uvw", True, 1.0)
            for i_1, (_, ir_1) in enumerate(irreps_in1)
            for i_2, (_, ir_2) in enumerate(irreps_in2)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2
        ]
        super().__init__(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instr,
            **kwargs,
        )


class ElementwiseTPProblem(TPProblem):
    def __init__(self, irreps_in1, irreps_in2, filter_ir_out=None, **kwargs) -> None:
        irreps_in1 = Irreps(irreps_in1).simplify()
        irreps_in2 = Irreps(irreps_in2).simplify()
        if filter_ir_out is not None:
            try:
                filter_ir_out = [Irrep(ir) for ir in filter_ir_out]
            except ValueError:
                raise ValueError(
                    f"filter_ir_out (={filter_ir_out}) must be an iterable of e3nn.o3.Irrep"
                )

        assert irreps_in1.num_irreps == irreps_in2.num_irreps

        irreps_in1 = list(irreps_in1)
        irreps_in2 = list(irreps_in2)

        i = 0
        while i < len(irreps_in1):
            mul_1, ir_1 = irreps_in1[i]
            mul_2, ir_2 = irreps_in2[i]

            if mul_1 < mul_2:
                irreps_in2[i] = (mul_1, ir_2)
                irreps_in2.insert(i + 1, (mul_2 - mul_1, ir_2))

            if mul_2 < mul_1:
                irreps_in1[i] = (mul_2, ir_1)
                irreps_in1.insert(i + 1, (mul_1 - mul_2, ir_1))
            i += 1

        out = []
        instr = []
        for i, ((mul, ir_1), (mul_2, ir_2)) in enumerate(zip(irreps_in1, irreps_in2)):
            assert mul == mul_2
            for ir in ir_1 * ir_2:
                if filter_ir_out is not None and ir not in filter_ir_out:
                    continue

                i_out = len(out)
                out.append((mul, ir))
                instr += [(i, i, i_out, "uuu", False)]

        super().__init__(irreps_in1, irreps_in2, out, instr, **kwargs)


class FullTPProblem(TPProblem):
    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        filter_ir_out: Iterator[Irrep] = None,
        **kwargs,
    ) -> None:
        irreps_in1 = Irreps(irreps_in1).simplify()
        irreps_in2 = Irreps(irreps_in2).simplify()
        if filter_ir_out is not None:
            try:
                filter_ir_out = [Irrep(ir) for ir in filter_ir_out]
            except ValueError:
                raise ValueError(
                    f"filter_ir_out (={filter_ir_out}) must be an iterable of e3nn.o3.Irrep"
                )

        out = []
        instr = []
        for i_1, (mul_1, ir_1) in enumerate(irreps_in1):
            for i_2, (mul_2, ir_2) in enumerate(irreps_in2):
                for ir_out in ir_1 * ir_2:
                    if filter_ir_out is not None and ir_out not in filter_ir_out:
                        continue

                    i_out = len(out)
                    out.append((mul_1 * mul_2, ir_out))
                    instr += [(i_1, i_2, i_out, "uvuv", False)]

        out = Irreps(out)
        out, p, _ = out.sort()

        instr = [
            (i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instr
        ]

        super().__init__(irreps_in1, irreps_in2, out, instr, **kwargs)


class ChannelwiseTPP(TPProblem):
    """
    Modified from mace/mace/modules/irreps_tools.py.
    """

    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        label: Optional[str] = None,
        irrep_dtype=np.float32,
        weight_dtype=np.float32,
    ):
        trainable = True
        irreps1 = Irreps(irreps_in1)
        irreps2 = Irreps(irreps_in2)
        irreps_out = Irreps(irreps_out)

        # Collect possible irreps and their instructions
        irreps_out_list = []
        instructions = []
        for i, (mul, ir_in) in enumerate(irreps1):
            for j, (_, ir_edge) in enumerate(irreps2):
                for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                    if ir_out in irreps_out:
                        k = len(irreps_out_list)  # instruction index
                        irreps_out_list.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", trainable))

        irreps_out = Irreps(irreps_out_list)
        irreps_out, permut, _ = irreps_out.sort()

        instructions = [
            (i_in1, i_in2, permut[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        instructions = sorted(instructions, key=lambda x: x[2])
        super().__init__(
            irreps1,
            irreps2,
            irreps_out,
            instructions,
            internal_weights=False,
            shared_weights=False,
            label=label,
            irrep_dtype=irrep_dtype,
            weight_dtype=weight_dtype,
        )


class SingleInstruction(TPProblem):
    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_in3: Irreps,
        mode: str,
        label: Optional[str] = None,
    ):
        trainable = True
        irreps1 = Irreps(irreps_in1)
        irreps2 = Irreps(irreps_in2)
        irreps3 = Irreps(irreps_in3)
        instructions = [(0, 0, 0, mode, trainable)]

        super().__init__(
            irreps1,
            irreps2,
            irreps3,
            instructions,
            internal_weights=False,
            shared_weights=False,
            label=label,
        )
