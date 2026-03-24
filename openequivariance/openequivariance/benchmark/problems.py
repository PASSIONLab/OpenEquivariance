from typing import Iterator, Optional

import numpy as np

from openequivariance.core.e3nn_lite import Irrep, Irreps, TPProblem

"""
This was taken from
https://github.com/e3nn/e3nn/blob/0.5.4/e3nn/o3/_tensor_product/_sub.py
Adapted to create TPPs to avoid torch dependence.
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
            except ValueError as exc:
                raise ValueError(
                    f"filter_ir_out (={filter_ir_out}) must be an iterable of e3nn.o3.Irrep"
                ) from exc

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
            except ValueError as exc:
                raise ValueError(
                    f"filter_ir_out (={filter_ir_out}) must be an iterable of e3nn.o3.Irrep"
                ) from exc

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

        irreps_out_list = []
        instructions = []
        for i, (mul, ir_in) in enumerate(irreps1):
            for j, (_, ir_edge) in enumerate(irreps2):
                for ir_out in ir_in * ir_edge:
                    if ir_out in irreps_out:
                        k = len(irreps_out_list)
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


FCTPP = FullyConnectedTPProblem
CTPP = ChannelwiseTPP

# source: https://github.com/e3nn/e3nn/blob/main/examples/tetris.py
# running tetris will output the layers. I've only extracted the fully connected layers here.
_e3nn_torch_tetris = [
    # 0th Layer
    FCTPP("1x0e", "1x0e", "150x0e + 50x1o + 50x2e"),  # sc
    FCTPP("1x0e", "1x0e", "1x0e"),  # lin1
    FCTPP("1x0e + 1x1o + 1x2e", "1x0e", "150x0e + 50x1o + 50x2e"),  # lin2
    FCTPP("1x0e + 1x1o + 1x2e", "1x0e", "1x0e"),  # alpha
    # 1st Layer
    FCTPP(
        "50x0e + 50x1o + 50x2e", "1x0e", "250x0e + 50x1o + 50x1e + 50x2o + 50x2e"
    ),  # sc
    FCTPP("50x0e + 50x1o + 50x2e", "1x0e", "50x0e + 50x1o + 50x2e"),  # lin1
    # FCTPP("50x0e + 50x1o + 50x2e", "1x0e + 1x1o + 1x2e",  "150x0e + 200x1o + 100x1e + 100x2o + 200x2e"), #tp
    FCTPP(
        "150x0e + 200x1o + 100x1e + 100x2o + 200x2e",
        "1x0e",
        "250x0e + 50x1o + 50x1e + 50x2o + 50x2e",
    ),  # lin2
    FCTPP("150x0e + 200x1o + 100x1e + 100x2o + 200x2e", "1x0e", "1x0e"),  # alpha
    # 2nd Layer
    FCTPP(
        "50x0e + 50x1o + 50x1e + 50x2o + 50x2e",
        "1x0e",
        "50x0o + 250x0e + 50x1o + 50x1e + 50x2o + 50x2e",
    ),  # sc
    FCTPP(
        "50x0e + 50x1o + 50x1e + 50x2o + 50x2e",
        "1x0e",
        "50x0e + 50x1o + 50x1e + 50x2o + 50x2e",
    ),  # lin1
    FCTPP(
        "100x0o + 150x0e + 300x1o + 250x1e + 250x2o + 300x2e",
        "1x0e",
        "50x0o + 250x0e + 50x1o + 50x1e + 50x2o + 50x2e",
    ),  # lin2
    FCTPP(
        "100x0o + 150x0e + 300x1o + 250x1e + 250x2o + 300x2e", "1x0e", "1x0e"
    ),  # alpha
    # 3rd Layer
    FCTPP("50x0o + 50x0e + 50x1o + 50x1e + 50x2o + 50x2e", "1x0e", "1x0o + 6x0e"),  # sc
    FCTPP(
        "50x0o + 50x0e + 50x1o + 50x1e + 50x2o + 50x2e",
        "1x0e",
        "50x0o + 50x0e + 50x1o + 50x1e + 50x2o + 50x2e",
    ),  # lin1
    FCTPP("150x0o + 150x0e", "1x0e", "1x0o + 6x0e"),  # lin2
    FCTPP("150x0o + 150x0e", "1x0e", "1x0e"),  # alpha
]


def e3nn_torch_tetris_poly_problems():
    # source: https://github.com/e3nn/e3nn/blob/f95297952303347a8a3cfe971efe449c710c43b2/examples/tetris_polynomial.py#L66-L68
    return [
        FCTPP(
            "1x0e + 1x1o + 1x2e + 1x3o",
            "1x0e + 1x1o + 1x2e + 1x3o",
            "64x0e + 24x1e + 24x1o + 16x2e + 16x2o",
            label="tetris-poly-1",
        ),  # tp1
        FCTPP(
            "64x0e + 24x1e + 24x1o + 16x2e + 16x2o",
            "1x0e + 1x1o + 1x2e",
            "0o + 6x0e",
            label="tetris-poly-2",
        ),  # tp2
    ]


# https://github.com/gcorso/DiffDock/blob/b4704d94de74d8cb2acbe7ec84ad234c09e78009/models/tensor_layers.py#L299
# specific irreps come from Vivek's communication with DiffDock team
def diffdock_problems():
    return [
        FCTPP(
            "10x1o + 10x1e + 48x0e + 48x0o",
            "1x0e + 1x1o",
            "10x1o + 10x1e + 48x0e + 48x0o",
            shared_weights=False,
            label="DiffDock-L=1",
        ),
        FCTPP(
            "10x1o + 10x1e + 48x0e + 48x0o",
            "1x0e + 1x1o + 1x2e",
            "10x1o + 10x1e + 48x0e + 48x0o",
            shared_weights=False,
            label="DiffDock-L=2",
        ),
    ]


def mace_problems():
    return [
        CTPP(*config)
        for config in [
            (
                "128x0e+128x1o+128x2e",
                "1x0e+1x1o+1x2e+1x3o",
                "128x0e+128x1o+128x2e+128x3o",
                "mace-large",
            ),
            (
                "128x0e+128x1o",
                "1x0e+1x1o+1x2e+1x3o",
                "128x0e+128x1o+128x2e",
                "mace-medium",
            ),
        ]
    ]


def nequip_problems():
    return [
        CTPP(*config)
        for config in [
            (
                "32x0o + 32x0e + 32x1o + 32x1e + 32x2o + 32x2e",
                "0e + 1o + 2e",
                "32x0o + 32x0e + 32x1o + 32x1e + 32x2o + 32x2e",
                "nequip-lips",
            ),
            (
                "64x0o + 64x0e + 64x1o + 64x1e",
                "0e + 1o",
                "64x0o + 64x0e + 64x1o + 64x1e",
                "nequip-revmd17-aspirin",
            ),
            (
                "64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e",
                "0e + 1o + 2e",
                "64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e",
                "nequip-revmd17-toluene",
            ),
            (
                "64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e + 64x3o + 64x3e",
                "0e + 1o + 2e + 3o",
                "64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e + 64x3o + 64x3e",
                "nequip-revmd17-benzene",
            ),
            (
                "32x0o + 32x0e + 32x1o + 32x1e",
                "0e + 1o",
                "32x0o + 32x0e + 32x1o + 32x1e",
                "nequip-water",
            ),
        ]
    ]


# https://github.com/atomicarchitects/nequix/blob/main/configs/nequix-mp-1.yml
def nequix_problems():
    return [
        CTPP(
            "89x0e",
            "1x0e+1x1o+1x2e+1x3o",
            "89x0e+89x1o+89x2e+89x3o",
            "nequix-mp-1-first_layer",
        ),
        CTPP(
            "128x0e+64x1o+32x2e+32x3o",
            "1x0e+1x1o+1x2e+1x3o",
            "128x0e+128x1o+128x2e+128x3o+64x1o+64x0e+64x2e+64x1o+64x3o+64x2e+32x2e+32x1o+32x3o+32x0e+32x2e+32x1o+32x3o+32x3o+32x2e+32x1o+32x3o+32x0e+32x2e",
            "nequix-mp-1-main_layers",
        ),
        CTPP(
            "128x0e+64x1o+32x2e+32x3o",
            "1x0e+1x1o+1x2e+1x3o",
            "128x0e+64x0e+32x0e+32x0e",
            "nequix-mp-1-last_layer",
        ),
    ]


# https://github.com/MDIL-SNU/SevenNet/tree/main/sevenn/pretrained_potentials/SevenNet_l3i5
def seven_net_problems():
    return [
        CTPP(
            "128x0e",
            "1x0e+1x1e+1x2e+1x3e",
            "128x0e+128x1e+128x2e+128x3e",
            "SevenNet_l3i5-first-layer",
        ),
        CTPP(
            "128x0e+64x1e+32x2e+32x3e",
            "1x0e+1x1e+1x2e+1x3e",
            "128x0e+64x0e+32x0e+32x0e+128x1e+64x1e+64x1e+64x1e+32x1e+32x1e+32x1e+32x1e+32x1e+128x2e+64x2e+64x2e+64x2e+32x2e+32x2e+32x2e+32x2e+32x2e+32x2e+32x2e+128x3e+64x3e+64x3e+32x3e+32x3e+32x3e+32x3e+32x3e+32x3e+32x3e",
            "SevenNet_l3i5-main-layers",
        ),
        CTPP(
            "128x0e+64x1e+32x2e+32x3e",
            "1x0e+1x1e+1x2e+1x3e",
            "128x0e+64x0e+32x0e+32x0e",
            "SevenNet_l3i5-last-layer",
        ),
    ]


def e3tools_problems():
    return [
        FCTPP(in1, in2, out, label=label, shared_weights=sw, internal_weights=iw)
        for (in1, in2, out, label, sw, iw) in [
            (
                "64x0e+16x1o",
                "1x0e+1x1o",
                "80x0e+16x1o",
                "e3tools_conv",
                False,
                False,
            ),
            (
                "64x0e+16x1o",
                "1x0e+1x1o",
                "64x0e+16x1o",
                "e3tools_transformer",
                True,
                False,  # Should be true, we don't support currently
            ),
        ]
    ]
