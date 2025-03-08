'''
This file contains lightly modified code from E3NN. The code has been modified to remove
all dependency on Pytorch.

The TensorProductProblem class does not maintain any internal weights.

https://github.com/e3nn/e3nn/blob/0.5.3/e3nn/o3/_tensor_product/_tensor_product.py
https://github.com/e3nn/e3nn/blob/0.5.3/e3nn/o3/_tensor_product/_instruction.py
https://github.com/e3nn/e3nn/blob/0.5.3/e3nn/o3/_irreps.py.
https://github.com/e3nn/e3nn/blob/main/e3nn/o3/_wigner.py

MIT License for e3nn:
Euclidean neural networks (e3nn) Copyright (c) 2020, The Regents of the
University of California, through Lawrence Berkeley National Laboratory
(subject to receipt of any required approvals from the U.S. Dept. of Energy), 
Ecole Polytechnique Federale de Lausanne (EPFL), Free University of Berlin 
and Kostiantyn Lapchevskyi. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
'''

__all__ = [
    "Irrep",
    "_MulIr",
    "Irreps",
    "Instruction",
]

import itertools
from typing import Tuple, NamedTuple, Union, List, Any, Optional
from math import sqrt, prod
import collections
import sys 
import numpy as np
import numpy.linalg as la
import functools, math

def perm_inverse(p):
    r"""
    compute the inverse permutation
    """
    return tuple(p.index(i) for i in range(len(p)))

class Irrep(tuple):
    def __new__(cls, l: Union[int, "Irrep", str, tuple], p=None):
        if p is None:
            if isinstance(l, Irrep):
                return l

            if isinstance(l, str):
                try:
                    name = l.strip()
                    l = int(name[:-1])
                    assert l >= 0
                    p = {
                        "e": 1,
                        "o": -1,
                        "y": (-1) ** l,
                    }[name[-1]]
                except Exception:
                    raise ValueError(f'unable to convert string "{name}" into an Irrep')
            elif isinstance(l, tuple):
                l, p = l

        if not isinstance(l, int) or l < 0:
            raise ValueError(f"l must be positive integer, got {l}")
        if p not in (-1, 1):
            raise ValueError(f"parity must be on of (-1, 1), got {p}")
        return super().__new__(cls, (l, p))

    @property
    def l(self) -> int:
        r"""The degree of the representation, :math:`l = 0, 1, \dots`."""
        return self[0]

    @property
    def p(self) -> int:
        r"""The parity of the representation, :math:`p = \pm 1`."""
        return self[1]

    def __repr__(self) -> str:
        p = {+1: "e", -1: "o"}[self.p]
        return f"{self.l}{p}"

    @classmethod
    def iterator(cls, lmax=None):
        r"""Iterator through all the irreps of :math:`O(3)`

        Examples
        --------
        >>> it = Irrep.iterator()
        >>> next(it), next(it), next(it), next(it)
        (0e, 0o, 1o, 1e)
        """
        for l in itertools.count():
            yield Irrep(l, (-1) ** l)
            yield Irrep(l, -((-1) ** l))

            if l == lmax:
                break

    @property
    def dim(self) -> int:
        """The dimension of the representation, :math:`2 l + 1`."""
        return 2 * self.l + 1

    def is_scalar(self) -> bool:
        """Equivalent to ``l == 0 and p == 1``"""
        return self.l == 0 and self.p == 1

    def __mul__(self, other):
        r"""Generate the irreps from the product of two irreps.

        Returns
        -------
        generator of `e3nn.o3.Irrep`
        """
        other = Irrep(other)
        p = self.p * other.p
        lmin = abs(self.l - other.l)
        lmax = self.l + other.l
        for l in range(lmin, lmax + 1):
            yield Irrep(l, p)

    def count(self, _value):
        raise NotImplementedError

    def index(self, _value):
        raise NotImplementedError

    def __rmul__(self, other):
        r"""
        >>> 3 * Irrep('1e')
        3x1e
        """
        assert isinstance(other, int)
        return Irreps([(other, self)])

    def __add__(self, other):
        return Irreps(self) + Irreps(other)

    def __contains__(self, _object):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class _MulIr(tuple):
    def __new__(cls, mul, ir=None):
        if ir is None:
            mul, ir = mul

        assert isinstance(mul, int)
        assert isinstance(ir, Irrep)
        return super().__new__(cls, (mul, ir))

    @property
    def mul(self) -> int:
        return self[0]

    @property
    def ir(self) -> Irrep:
        return self[1]

    @property
    def dim(self) -> int:
        return self.mul * self.ir.dim

    def __repr__(self) -> str:
        return f"{self.mul}x{self.ir}"

    def __getitem__(self, item) -> Union[int, Irrep]:  # pylint: disable=useless-super-delegation
        return super().__getitem__(item)

    def count(self, _value):
        raise NotImplementedError

    def index(self, _value):
        raise NotImplementedError


class Irreps(tuple):
    def __new__(cls, irreps=None) -> Union[_MulIr, "Irreps"]:
        if isinstance(irreps, Irreps):
            return super().__new__(cls, irreps)

        out = []
        if isinstance(irreps, Irrep):
            out.append(_MulIr(1, Irrep(irreps)))
        elif isinstance(irreps, str):
            try:
                if irreps.strip() != "":
                    for mul_ir in irreps.split("+"):
                        if "x" in mul_ir:
                            mul, ir = mul_ir.split("x")
                            mul = int(mul)
                            ir = Irrep(ir)
                        else:
                            mul = 1
                            ir = Irrep(mul_ir)

                        assert isinstance(mul, int) and mul >= 0
                        out.append(_MulIr(mul, ir))
            except Exception:
                raise ValueError(f'Unable to convert string "{irreps}" into an Irreps')
        elif irreps is None:
            pass
        else:
            for mul_ir in irreps:
                mul = None
                ir = None

                if isinstance(mul_ir, str):
                    mul = 1
                    ir = Irrep(mul_ir)
                elif isinstance(mul_ir, Irrep):
                    mul = 1
                    ir = mul_ir
                elif isinstance(mul_ir, _MulIr):
                    mul, ir = mul_ir
                elif len(mul_ir) == 2:
                    mul, ir = mul_ir
                    ir = Irrep(ir)

                if not (isinstance(mul, int) and mul >= 0 and ir is not None):
                    raise ValueError(f'Unable to interpret "{mul_ir}" as an irrep.')

                out.append(_MulIr(mul, ir))
        return super().__new__(cls, out)

    @staticmethod
    def spherical_harmonics(lmax: int, p: int = -1) -> "Irreps":
        return Irreps([(1, (l, p**l)) for l in range(lmax + 1)])

    def slices(self):
        r"""List of slices corresponding to indices for each irrep.

        Examples
        --------

        >>> Irreps('2x0e + 1e').slices()
        [slice(0, 2, None), slice(2, 5, None)]
        """
        s = []
        i = 0
        for mul_ir in self:
            s.append(slice(i, i + mul_ir.dim))
            i += mul_ir.dim
        return s

    def __getitem__(self, i) -> Union[_MulIr, "Irreps"]:
        x = super().__getitem__(i)
        if isinstance(i, slice):
            return Irreps(x)
        return x

    def __contains__(self, ir) -> bool:
        ir = Irrep(ir)
        return ir in (irrep for _, irrep in self)

    def count(self, ir) -> int:
        r"""Multiplicity of ``ir``.

        Parameters
        ----------
        ir : `e3nn.o3.Irrep`

        Returns
        -------
        `int`
            total multiplicity of ``ir``
        """
        ir = Irrep(ir)
        return sum(mul for mul, irrep in self if ir == irrep)

    def index(self, _object):
        raise NotImplementedError

    def __add__(self, irreps) -> "Irreps":
        irreps = Irreps(irreps)
        return Irreps(super().__add__(irreps))

    def __mul__(self, other) -> "Irreps":
        r"""
        >>> (Irreps('2x1e') * 3).simplify()
        6x1e
        """
        if isinstance(other, Irreps):
            raise NotImplementedError("Use o3.TensorProduct for this, see the documentation")
        return Irreps(super().__mul__(other))

    def __rmul__(self, other) -> "Irreps":
        r"""
        >>> 2 * Irreps('0e + 1e')
        1x0e+1x1e+1x0e+1x1e
        """
        return Irreps(super().__rmul__(other))

    def simplify(self) -> "Irreps":
        out = []
        for mul, ir in self:
            if out and out[-1][1] == ir:
                out[-1] = (out[-1][0] + mul, ir)
            elif mul > 0:
                out.append((mul, ir))
        return Irreps(out)

    def remove_zero_multiplicities(self) -> "Irreps":
        out = [(mul, ir) for mul, ir in self if mul > 0]
        return Irreps(out)

    def sort(self):
        Ret = collections.namedtuple("sort", ["irreps", "p", "inv"])
        out = [(ir, i, mul) for i, (mul, ir) in enumerate(self)]
        out = sorted(out)
        inv = tuple(i for _, i, _ in out)
        p = perm_inverse(inv)
        irreps = Irreps([(mul, ir) for ir, _, mul in out])
        return Ret(irreps, p, inv)

    def regroup(self) -> "Irreps":
        return self.sort().irreps.simplify()

    @property
    def dim(self) -> int:
        return sum(mul * ir.dim for mul, ir in self)

    @property
    def num_irreps(self) -> int:
        return sum(mul for mul, _ in self)

    @property
    def ls(self) -> List[int]:
        return [l for mul, (l, p) in self for _ in range(mul)]

    @property
    def lmax(self) -> int:
        if len(self) == 0:
            raise ValueError("Cannot get lmax of empty Irreps")
        return max(self.ls)

    def __repr__(self) -> str:
        return "+".join(f"{mul_ir}" for mul_ir in self)


class Instruction(NamedTuple):
    i_in1: int
    i_in2: int
    i_out: int
    connection_mode: str
    has_weight: bool
    path_weight: float
    path_shape: tuple

