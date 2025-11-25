import numpy as np

from openequivariance.templates.jinja_utils import get_jinja_environment
from openequivariance.core.ComputationSchedule import ComputationSchedule
from openequivariance.core.TensorProductBase import TensorProductBase
from openequivariance.core.dtype_enum import dtype_to_enum

from openequivariance.core.utils import (
    filter_and_analyze_problem,
    count_cg_non_zero,
)

class LoopUnrollTP(TensorProductBase):
    def __init__(self, config, dp, postprocess_kernel, torch_op):
        super().__init__(config, torch_op=torch_op)

        env = get_jinja_environment()
        template = env.get_template("loop_unroll_batch.cuh")

        analysis = filter_and_analyze_problem(config)
        self.is_uvw = analysis["is_uvw"]

        def generate_forward_schedule(warps_per_block):
            self.forward_schedule = ComputationSchedule(
                self.config,
                smem_limit=dp.maxSharedMemPerBlock,
                warps_per_block=warps_per_block,
                warp_size=dp.warpsize,
                block_count=dp.multiprocessorCount * 4,
                direction="forward",
                irrep_dtype=config.irrep_dtype,
                weight_dtype=config.weight_dtype,
                include_scratch=self.is_uvw,
                stream_weights=self.is_uvw,
            )

        def generate_backward_schedule(warps_per_block):
            self.backward_schedule = ComputationSchedule(
                self.config,
                smem_limit=dp.maxSharedMemPerBlock,
                warps_per_block=warps_per_block,
                warp_size=dp.warpsize,
                block_count=dp.multiprocessorCount * 4,
                direction="backward",
                irrep_dtype=config.irrep_dtype,
                weight_dtype=config.weight_dtype,
                include_scratch=self.is_uvw,
                stream_weights=self.is_uvw,
            )

        def generate_double_backward_schedule(warps_per_block):
            self.double_backward_schedule = ComputationSchedule(
                self.config,
                smem_limit=dp.maxSharedMemPerBlock,
                warps_per_block=warps_per_block,
                warp_size=dp.warpsize,
                block_count=dp.multiprocessorCount,
                direction="double_backward",
                irrep_dtype=config.irrep_dtype,
                weight_dtype=config.weight_dtype,
                include_scratch=self.is_uvw,
                stream_weights=self.is_uvw,
                schedule_type=3,
            )

        scheduler_generators = [
            generate_forward_schedule,
            generate_backward_schedule,
            generate_double_backward_schedule,
        ]

        for generate_schedule in scheduler_generators:
            warp_count = 8
            while warp_count > 0:
                try:
                    generate_schedule(warp_count)
                    break
                except Exception:
                    warp_count -= 2
                    if warp_count == 0:
                        raise RuntimeError(
                            "Tensor product schedule generation failed, shared memory inadequate!"
                        )

        self.jit_kernel = postprocess_kernel(
            template.render(
                forward_schedule=self.forward_schedule,
                backward_schedule=self.backward_schedule,
                double_backward_schedule=self.double_backward_schedule,
            )
        )

        self.kernelProp = {
            "L1_dim": self.L1.dim,
            "L2_dim": self.L2.dim,
            "L3_dim": self.L3.dim,
            "weight_numel": self.config.weight_numel,
            "shared_weights": int(self.config.shared_weights),
            "opt_level": 3,
            "irrep_dtype": dtype_to_enum[self.config.irrep_dtype],
            "weight_dtype": dtype_to_enum[self.config.weight_dtype],
        }


    def calculate_flops_forward(self, batch_size: int) -> dict:
        if self.is_uvw:
            return super().calculate_flops_forward(batch_size)
        else:
            tpp = self.config
            flop_count = {
                "CG_decomposition": 0,
                "linear_combination": 0,
                "outer_products": 0,
            }
            for ins in tpp.instructions:
                l1, l2, l3 = (
                    tpp.irreps_in1[ins.i_in1].ir.l,
                    tpp.irreps_in2[ins.i_in2].ir.l,
                    tpp.irreps_out[ins.i_out].ir.l,
                )
                flop_count["CG_decomposition"] += count_cg_non_zero(l1, l2, l3) * (
                    ins.path_shape[0] * ins.path_shape[1]
                )
                flop_count["linear_combination"] += (
                    (2 * l3 + 1) * np.prod(ins.path_shape) if ins.has_weight else 0
                )

            flop_count["CG_decomposition"] *= 3 * batch_size
            flop_count["linear_combination"] *= (
                batch_size  # Weights do not require FMA here
            )
            flop_count["total"] = sum(flop_count.values())
            return flop_count

    def calculate_flops_backward(self, batch_size: int) -> dict:
        if self.is_uvw:
            return super().calculate_flops_backward(batch_size)
        else:
            tpp = self.config
            flop_count = {"backward": 0}
            for ins in tpp.instructions:
                l1, l2, l3 = (
                    tpp.irreps_in1[ins.i_in1].ir.l,
                    tpp.irreps_in2[ins.i_in2].ir.l,
                    tpp.irreps_out[ins.i_out].ir.l,
                )
                flop_count["backward"] += count_cg_non_zero(l1, l2, l3) * (
                    ins.path_shape[0] * ins.path_shape[1]
                )

            flop_count["backward"] *= 9 * batch_size
            flop_count["total"] = sum(flop_count.values())
            return flop_count
