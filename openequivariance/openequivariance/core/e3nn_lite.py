# ruff: noqa: E743
"""
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
"""

import itertools
from typing import Tuple, NamedTuple, Union, List, Any, Optional
from math import sqrt, prod
import collections
import numpy as np
import numpy.linalg as la
import functools
import copy


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

        Examples:

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

    def count(self, ir: Irrep) -> int:
        r"""
        :returns: total multiplicity of ``ir``.
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
            raise NotImplementedError(
                "Use o3.TensorProduct for this, see the documentation"
            )
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


class TPProblem:
    """
    Specification for a CG tensor product. All parameters from
    e3nn's ``o3.TensorProduct`` are available, along with additional
    parameters for the types of weights and irreps.

    :param irreps_in1: Irreps for the first CG argument
    :param irreps_in2: Irreps for the second CG argument
    :param irreps_out: Irreps for the output
    :param instructions: A list of 5-tuples, each of
           the form ``(i_in1, i_in2, i_out, has_weight, path_weight)``.
           ``i_in1``, ``i_in2``, and ``i_out`` each index
           an Irrep from ``irreps_in1``, ``irreps_in2``, and
           ``irreps_in3``, respectively. ``has_weight`` (True / False)
           controls whether trainable weights are included for the
           instruction, and ``path_weight`` controls output normalization.
    :param irrep_dtype: Datatype of irrep inputs; one of ``np.float32`` or ``np.float64``.
           *Default*: ``np.float32``.
    :param weight_dtype: Datatype of weights; one of ``np.float32`` or ``np.float64``.
           *Default*: ``np.float32``.
    :param label: A name for this problem specification (useful for testing / benchmarking).
    :param shared_weights: If True, all elements in a batch of inputs share a common set of
           weights. If False, each batch element has a unique set of weights. *Default*: True.
    :param internal_weights: Must be False; OpenEquivariance does not support internal weights. *Default*: False.
    :param irrep_normalization: One of ``["component", "norm", "none"]``. *Default*: "component".
    :param path_normalization: One of ``["element", "path", "none"]``. *Default*: "element".
    """

    instructions: List[Any]
    shared_weights: bool
    internal_weights: bool
    weight_numel: int
    label: str
    _profiling_str: str
    _in1_dim: int
    _in2_dim: int

    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        instructions: List[tuple],
        in1_var: Optional[List[float]] = None,
        in2_var: Optional[List[float]] = None,
        out_var: Optional[List[float]] = None,
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights: Optional[bool] = False,
        shared_weights: Optional[bool] = None,
        label: Optional[str] = None,
        irrep_dtype: type[np.generic] = np.float32,
        weight_dtype: type[np.generic] = np.float32,
    ) -> None:
        # === Setup ===
        super().__init__()

        assert irrep_normalization in ["component", "norm", "none"]
        assert path_normalization in ["element", "path", "none"]
        assert issubclass(irrep_dtype, np.generic)
        assert issubclass(weight_dtype, np.generic)

        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)
        self.irreps_out = Irreps(irreps_out)

        self.instructions_raw = instructions
        self.in1_var = in1_var
        self.in2_var = in2_var
        self.out_var = out_var
        self.irrep_normalization = irrep_normalization
        self.path_normalization = path_normalization
        self.label = label if label is not None else ""
        del irreps_in1, irreps_in2, irreps_out

        instructions = [x if len(x) == 6 else x + (1.0,) for x in instructions]
        instructions = [
            Instruction(
                i_in1=i_in1,
                i_in2=i_in2,
                i_out=i_out,
                connection_mode=connection_mode,
                has_weight=has_weight,
                path_weight=path_weight,
                path_shape={
                    "uvw": (
                        self.irreps_in1[i_in1].mul,
                        self.irreps_in2[i_in2].mul,
                        self.irreps_out[i_out].mul,
                    ),
                    "uvu": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uvv": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uuw": (self.irreps_in1[i_in1].mul, self.irreps_out[i_out].mul),
                    "uuu": (self.irreps_in1[i_in1].mul,),
                    "uvuv": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uvu<v": (
                        self.irreps_in1[i_in1].mul
                        * (self.irreps_in2[i_in2].mul - 1)
                        // 2,
                    ),
                    "u<vw": (
                        self.irreps_in1[i_in1].mul
                        * (self.irreps_in2[i_in2].mul - 1)
                        // 2,
                        self.irreps_out[i_out].mul,
                    ),
                }[connection_mode],
            )
            for i_in1, i_in2, i_out, connection_mode, has_weight, path_weight in instructions
        ]

        if in1_var is None:
            in1_var = [1.0 for _ in range(len(self.irreps_in1))]
        else:
            in1_var = [float(var) for var in in1_var]
            assert len(in1_var) == len(self.irreps_in1), (
                "Len of ir1_var must be equal to len(irreps_in1)"
            )

        if in2_var is None:
            in2_var = [1.0 for _ in range(len(self.irreps_in2))]
        else:
            in2_var = [float(var) for var in in2_var]
            assert len(in2_var) == len(self.irreps_in2), (
                "Len of ir2_var must be equal to len(irreps_in2)"
            )

        if out_var is None:
            out_var = [1.0 for _ in range(len(self.irreps_out))]
        else:
            out_var = [float(var) for var in out_var]
            assert len(out_var) == len(self.irreps_out), (
                "Len of out_var must be equal to len(irreps_out)"
            )

        def num_elements(ins):
            return {
                "uvw": (
                    self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul
                ),
                "uvu": self.irreps_in2[ins.i_in2].mul,
                "uvv": self.irreps_in1[ins.i_in1].mul,
                "uuw": self.irreps_in1[ins.i_in1].mul,
                "uuu": 1,
                "uvuv": 1,
                "uvu<v": 1,
                "u<vw": self.irreps_in1[ins.i_in1].mul
                * (self.irreps_in2[ins.i_in2].mul - 1)
                // 2,
            }[ins.connection_mode]

        normalization_coefficients = []
        for ins in instructions:
            mul_ir_in1 = self.irreps_in1[ins.i_in1]
            mul_ir_in2 = self.irreps_in2[ins.i_in2]
            mul_ir_out = self.irreps_out[ins.i_out]
            assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
            assert (
                abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l)
                <= mul_ir_out.ir.l
                <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
            )
            assert ins.connection_mode in [
                "uvw",
                "uvu",
                "uvv",
                "uuw",
                "uuu",
                "uvuv",
                "uvu<v",
                "u<vw",
            ]

            if irrep_normalization == "component":
                alpha = mul_ir_out.ir.dim
            if irrep_normalization == "norm":
                alpha = mul_ir_in1.ir.dim * mul_ir_in2.ir.dim
            if irrep_normalization == "none":
                alpha = 1

            if path_normalization == "element":
                x = sum(
                    in1_var[i.i_in1] * in2_var[i.i_in2] * num_elements(i)
                    for i in instructions
                    if i.i_out == ins.i_out
                )
            if path_normalization == "path":
                x = in1_var[ins.i_in1] * in2_var[ins.i_in2] * num_elements(ins)
                x *= len([i for i in instructions if i.i_out == ins.i_out])
            if path_normalization == "none":
                x = 1

            if x > 0.0:
                alpha /= x

            alpha *= out_var[ins.i_out]
            alpha *= ins.path_weight

            normalization_coefficients += [sqrt(alpha)]

        self.instructions = [
            Instruction(
                ins.i_in1,
                ins.i_in2,
                ins.i_out,
                ins.connection_mode,
                ins.has_weight,
                alpha,
                ins.path_shape,
            )
            for ins, alpha in zip(instructions, normalization_coefficients)
        ]

        self._in1_dim = self.irreps_in1.dim
        self._in2_dim = self.irreps_in2.dim

        if shared_weights is False and internal_weights is None:
            internal_weights = False

        if shared_weights is None:
            shared_weights = True

        assert shared_weights or not internal_weights
        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        # === Determine weights ===
        self.weight_numel = sum(
            prod(ins.path_shape) for ins in self.instructions if ins.has_weight
        )
        self.output_mask = None

        self.irrep_dtype = irrep_dtype
        self.weight_dtype = weight_dtype

    def __str__(self) -> str:
        """Simple representation, definitely incomplete"""
        result = ""
        result += f"{self.__class__.__name__}"
        result += f"({self.irreps_in1.simplify()} x {self.irreps_in2.simplify()}) -> {self.irreps_out.simplify()}"
        return result

    def __repr__(self) -> str:
        """More complete, yet maybe incomplete representation"""
        result = ""
        result += f"{self.__class__.__name__}"
        result += f"({self.irreps_in1.simplify()} x {self.irreps_in2.simplify()}) -> {self.irreps_out.simplify()}\n"
        result += f"{self.irrep_normalization = }\n"
        result += f"{self.path_normalization = }\n"
        result += f"{self.internal_weights = }\n"
        result += f"{self.shared_weights = }\n"
        result += f"{self.in1_var = }\n"
        result += f"{self.in2_var = }\n"
        result += f"{self.out_var = }\n"
        result += f"num weights {self.weight_numel} \n"
        result += "|      index      |       l         |        m        | mode  |    weights   | \n"
        result += "| in1 | in2 | out | in1 | in2 | out | in1 | in2 | out |       | exist | path | \n"
        for ins in self.instructions:  # type : Instruction
            mul_irrep_in1 = self.irreps_in1[ins.i_in1]
            mul_irrep_in2 = self.irreps_in2[ins.i_in2]
            mul_irrep_out = self.irreps_out[ins.i_out]

            assert isinstance(mul_irrep_in1, _MulIr)
            assert isinstance(mul_irrep_in2, _MulIr)
            assert isinstance(mul_irrep_out, _MulIr)

            result += f"| {ins.i_in1:3} | {ins.i_in2:3} | {ins.i_out:3} |"
            result += f" {mul_irrep_in1.ir.l:3} | {mul_irrep_in2.ir.l:3} | {mul_irrep_out.ir.l:3} |"
            result += f" {mul_irrep_in1.mul:3} | {mul_irrep_in2.mul:3} | {mul_irrep_out.mul:3} |"
            result += f" {ins.connection_mode:<5} |"
            result += f" {str(ins.has_weight):<5} |"
            result += f" {ins.path_weight:<4.2f} | "
            result += "\n"
        result = result.replace("self.", "")
        return result

    def weight_range_and_shape_for_instruction(
        self, instruction: int
    ) -> Tuple[int, int, tuple]:
        if not self.instructions[instruction].has_weight:
            raise ValueError(f"Instruction {instruction} has no weights.")
        offset = sum(prod(ins.path_shape) for ins in self.instructions[:instruction])
        ins = self.instructions[instruction]
        return offset, offset + prod(ins.path_shape), ins.path_shape

    def clone(self):
        return copy.deepcopy(self)


def change_basis_real_to_complex(l: int, dtype=None) -> np.ndarray:
    # https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    q = np.zeros((2 * l + 1, 2 * l + 1), dtype=np.complex128)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = 1 / 2**0.5
        q[l + m, l - abs(m)] = -1j / 2**0.5
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1) ** m / 2**0.5
        q[l + m, l - abs(m)] = 1j * (-1) ** m / 2**0.5
    q = (
        -1j
    ) ** l * q  # Added factor of 1j**l to make the Clebsch-Gordan coefficients real

    dtype = {np.float32: np.complex64, np.float64: np.complex128, None: np.complex128}[
        dtype
    ]

    return q.astype(dtype)


def wigner_3j(l1: int, l2: int, l3: int, dtype=np.float64) -> np.ndarray:
    r"""Wigner 3j symbols :math:`C_{lmn}`.

    It satisfies the following two properties:

        .. math::

            C_{lmn} = C_{ijk} D_{il}(g) D_{jm}(g) D_{kn}(g) \qquad \forall g \in SO(3)

        where :math:`D` are given by `wigner_D`.

        .. math::

            C_{ijk} C_{ijk} = 1

    Parameters
    ----------
    l1 : int
        :math:`l_1`

    l2 : int
        :math:`l_2`

    l3 : int
        :math:`l_3`

    dtype : np.dtype or None
        ``dtype`` of the returned tensor. Default is np.float64

    Returns
    -------
    `np.ndarray`
        tensor :math:`C` of shape :math:`(2l_1+1, 2l_2+1, 2l_3+1)`
    """
    assert abs(l2 - l3) <= l1 <= l2 + l3
    assert isinstance(l1, int) and isinstance(l2, int) and isinstance(l3, int)
    C = _so3_clebsch_gordan(l1, l2, l3)

    # make sure we always get a copy so mutation doesn't ruin the stored tensors
    return C.copy().astype(dtype)


@functools.lru_cache(maxsize=None)
def _so3_clebsch_gordan(l1: int, l2: int, l3: int) -> np.ndarray:
    Q1 = change_basis_real_to_complex(l1, dtype=np.float64)
    Q2 = change_basis_real_to_complex(l2, dtype=np.float64)
    Q3 = change_basis_real_to_complex(l3, dtype=np.float64)
    C = _su2_clebsch_gordan(l1, l2, l3).astype(np.complex128)
    C = np.einsum("ij,kl,mn,ikn->jlm", Q1, Q2, np.conj(Q3.T), C)

    # make it real
    assert np.all(np.abs(np.imag(C)) < 1e-5)
    C = np.real(C)

    # normalization
    C = C / la.norm(C.flatten())
    return C


# Taken from http://qutip.org/docs/3.1.0/modules/qutip/utilities.html

# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################


@functools.lru_cache(maxsize=None)
def _su2_clebsch_gordan(
    j1: Union[int, float], j2: Union[int, float], j3: Union[int, float]
) -> np.ndarray:
    """Calculates the Clebsch-Gordon matrix
    for SU(2) coupling j1 and j2 to give j3.
    Parameters
    ----------
    j1 : float
        Total angular momentum 1.
    j2 : float
        Total angular momentum 2.
    j3 : float
        Total angular momentum 3.
    Returns
    -------
    cg_matrix : numpy.array
        Requested Clebsch-Gordan matrix.
    """
    assert isinstance(j1, (int, float))
    assert isinstance(j2, (int, float))
    assert isinstance(j3, (int, float))
    mat = np.zeros(
        (int(2 * j1 + 1), int(2 * j2 + 1), int(2 * j3 + 1)), dtype=np.float64
    )
    if int(2 * j3) in range(int(2 * abs(j1 - j2)), int(2 * (j1 + j2)) + 1, 2):
        for m1 in (x / 2 for x in range(-int(2 * j1), int(2 * j1) + 1, 2)):
            for m2 in (x / 2 for x in range(-int(2 * j2), int(2 * j2) + 1, 2)):
                if abs(m1 + m2) <= j3:
                    mat[int(j1 + m1), int(j2 + m2), int(j3 + m1 + m2)] = (
                        _su2_clebsch_gordan_coeff((j1, m1), (j2, m2), (j3, m1 + m2))
                    )
    return mat


def _su2_clebsch_gordan_coeff(idx1, idx2, idx3):
    """Calculates the Clebsch-Gordon coefficient
    for SU(2) coupling (j1,m1) and (j2,m2) to give (j3,m3).
    Parameters
    ----------
    j1 : float
        Total angular momentum 1.
    j2 : float
        Total angular momentum 2.
    j3 : float
        Total angular momentum 3.
    m1 : float
        z-component of angular momentum 1.
    m2 : float
        z-component of angular momentum 2.
    m3 : float
        z-component of angular momentum 3.
    Returns
    -------
    cg_coeff : float
        Requested Clebsch-Gordan coefficient.
    """
    from fractions import Fraction
    from math import factorial

    j1, m1 = idx1
    j2, m2 = idx2
    j3, m3 = idx3

    if m3 != m1 + m2:
        return 0
    vmin = int(max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

    def f(n: int) -> int:
        assert n == round(n)
        return factorial(round(n))

    C = (
        (2.0 * j3 + 1.0)
        * Fraction(
            f(j3 + j1 - j2)
            * f(j3 - j1 + j2)
            * f(j1 + j2 - j3)
            * f(j3 + m3)
            * f(j3 - m3),
            f(j1 + j2 + j3 + 1) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2),
        )
    ) ** 0.5

    S = 0
    for v in range(vmin, vmax + 1):
        S += (-1) ** int(v + j2 + m2) * Fraction(
            f(j2 + j3 + m1 - v) * f(j1 - m1 + v),
            f(v) * f(j3 - j1 + j2 - v) * f(j3 + m3 - v) * f(v + j1 - j2 - m3),
        )
    C = C * S
    return C
