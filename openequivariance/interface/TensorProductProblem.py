from typing import List, Optional
from .e3nn_lite import *

import numpy as np

class TPProblem: 
    instructions: List[Any]
    shared_weights: bool
    internal_weights: bool
    weight_numel: int
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
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        label: Optional[str] = None, 
        irrep_dtype : type[np.generic] = np.float32,
        weight_dtype : type[np.generic] = np.float32) -> None:

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
        self.label = label
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
                    "uvw": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul, self.irreps_out[i_out].mul),
                    "uvu": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uvv": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uuw": (self.irreps_in1[i_in1].mul, self.irreps_out[i_out].mul),
                    "uuu": (self.irreps_in1[i_in1].mul,),
                    "uvuv": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uvu<v": (self.irreps_in1[i_in1].mul * (self.irreps_in2[i_in2].mul - 1) // 2,),
                    "u<vw": (self.irreps_in1[i_in1].mul * (self.irreps_in2[i_in2].mul - 1) // 2, self.irreps_out[i_out].mul),
                }[connection_mode],
            )
            for i_in1, i_in2, i_out, connection_mode, has_weight, path_weight in instructions
        ]

        if in1_var is None:
            in1_var = [1.0 for _ in range(len(self.irreps_in1))]
        else:
            in1_var = [float(var) for var in in1_var]
            assert len(in1_var) == len(self.irreps_in1), "Len of ir1_var must be equal to len(irreps_in1)"

        if in2_var is None:
            in2_var = [1.0 for _ in range(len(self.irreps_in2))]
        else:
            in2_var = [float(var) for var in in2_var]
            assert len(in2_var) == len(self.irreps_in2), "Len of ir2_var must be equal to len(irreps_in2)"

        if out_var is None:
            out_var = [1.0 for _ in range(len(self.irreps_out))]
        else:
            out_var = [float(var) for var in out_var]
            assert len(out_var) == len(self.irreps_out), "Len of out_var must be equal to len(irreps_out)"

        def num_elements(ins):
            return {
                "uvw": (self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul),
                "uvu": self.irreps_in2[ins.i_in2].mul,
                "uvv": self.irreps_in1[ins.i_in1].mul,
                "uuw": self.irreps_in1[ins.i_in1].mul,
                "uuu": 1,
                "uvuv": 1,
                "uvu<v": 1,
                "u<vw": self.irreps_in1[ins.i_in1].mul * (self.irreps_in2[ins.i_in2].mul - 1) // 2,
            }[ins.connection_mode]

        normalization_coefficients = []
        for ins in instructions:
            mul_ir_in1 = self.irreps_in1[ins.i_in1]
            mul_ir_in2 = self.irreps_in2[ins.i_in2]
            mul_ir_out = self.irreps_out[ins.i_out]
            assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
            assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
            assert ins.connection_mode in ["uvw", "uvu", "uvv", "uuw", "uuu", "uvuv", "uvu<v", "u<vw"]

            if irrep_normalization == "component":
                alpha = mul_ir_out.ir.dim
            if irrep_normalization == "norm":
                alpha = mul_ir_in1.ir.dim * mul_ir_in2.ir.dim
            if irrep_normalization == "none":
                alpha = 1

            if path_normalization == "element":
                x = sum(in1_var[i.i_in1] * in2_var[i.i_in2] * num_elements(i) for i in instructions if i.i_out == ins.i_out)
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
            Instruction(ins.i_in1, ins.i_in2, ins.i_out, ins.connection_mode, ins.has_weight, alpha, ins.path_shape)
            for ins, alpha in zip(instructions, normalization_coefficients)
        ]

        self._in1_dim = self.irreps_in1.dim
        self._in2_dim = self.irreps_in2.dim

        if shared_weights is False and internal_weights is None:
            internal_weights = False

        if shared_weights is None:
            shared_weights = True

        if internal_weights is None:
            internal_weights = shared_weights and any(i.has_weight for i in self.instructions)

        assert shared_weights or not internal_weights
        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        # === Determine weights ===
        self.weight_numel = sum(prod(ins.path_shape) for ins in self.instructions if ins.has_weight)
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
        result += f"|      index      |       l         |        m        | mode  |    weights   | \n"
        result += f"| in1 | in2 | out | in1 | in2 | out | in1 | in2 | out |       | exist | path | \n"
        for ins in self.instructions: # type : Instruction
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
        result = result.replace("self.","")
        return result 
    
    def weight_range_and_shape_for_instruction(self, instruction: int) -> Tuple[int, int, tuple]: 
        if not self.instructions[instruction].has_weight:
            raise ValueError(f"Instruction {instruction} has no weights.")
        offset = sum(prod(ins.path_shape) for ins in self.instructions[:instruction])
        ins = self.instructions[instruction]
        return offset, offset + prod(ins.path_shape), ins.path_shape