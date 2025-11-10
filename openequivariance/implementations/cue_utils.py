"""Utilities for cuEquivariance integration."""

import itertools
from typing import Iterator

import numpy as np


def create_O3_e3nn_class():
    """
    Create and return the O3_e3nn class that bridges cuEquivariance and e3nn.

    This class must be created dynamically because it requires imports of
    cuequivariance and e3nn.o3, which may not always be available.

    Returns:
        Type[O3_e3nn]: A class that extends cuequivariance.O3 with e3nn-compatible methods
    """
    import cuequivariance as cue
    import e3nn.o3 as o3

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

    return O3_e3nn
