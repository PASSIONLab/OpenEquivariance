import numpy as np

from openequivariance.core.ConvolutionBase import ConvolutionBase
from openequivariance.core.ComputationSchedule import (
    ComputationSchedule,
    SMEMCapacityException,
)

from openequivariance.core.utils import dtype_to_enum
from openequivariance.templates.jinja_utils import get_jinja_environment
from openequivariance.core.utils import filter_and_analyze_problem


class LoopUnrollConv(ConvolutionBase):
    def __init__(
        self,
        config,
        dp,
        postprocess_kernel,
        *,
        idx_dtype: type[np.generic] = np.int64,
        torch_op: bool = False,
        deterministic: bool = False,
        kahan: bool = False,
    ):
        super().__init__(
            config, idx_dtype=idx_dtype, torch_op=torch_op, deterministic=deterministic
        )

        if kahan:
            assert deterministic

        env = get_jinja_environment()
        template = env.get_template("loop_unroll_conv_atomic.cuh")

        analysis = filter_and_analyze_problem(config)
        self.is_uvw = analysis["is_uvw"]

        if config.shared_weights:
            assert not deterministic, (
                "Deterministic convolution does not support shared weights"
            )

        forward_schedule_type = 3
        backward_schedule_type = 2
        if deterministic:
            backward_schedule_type = 3
            template = env.get_template("loop_unroll_conv_det.cuh")

        def generate_forward_schedule(warps_per_block):
            self.forward_schedule = ComputationSchedule(
                self.config,
                smem_limit=dp.maxSharedMemPerBlock // 4 * 3,
                warps_per_block=warps_per_block,
                block_count=dp.multiprocessorCount,
                direction="forward",
                irrep_dtype=config.irrep_dtype,
                weight_dtype=config.weight_dtype,
                schedule_type=forward_schedule_type,
                warp_size=dp.warpsize,
                include_scratch=self.is_uvw,
                stream_weights=self.is_uvw,
                kahan=kahan,
            )

        def generate_backward_schedule(warps_per_block):
            self.backward_schedule = ComputationSchedule(
                self.config,
                smem_limit=dp.maxSharedMemPerBlock,
                warps_per_block=warps_per_block,
                block_count=dp.multiprocessorCount * 2,
                direction="backward",
                irrep_dtype=config.irrep_dtype,
                weight_dtype=config.weight_dtype,
                schedule_type=backward_schedule_type,
                warp_size=dp.warpsize,
                include_scratch=self.is_uvw,
                stream_weights=self.is_uvw,
                kahan=kahan,
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
                kahan=kahan,
            )

        scheduler_generators = [
            generate_forward_schedule,
            generate_backward_schedule,
            generate_double_backward_schedule,
        ]

        for generate_schedule in scheduler_generators:
            warp_count = 6
            while warp_count > 0:
                try:
                    generate_schedule(warp_count)
                    break
                except SMEMCapacityException:
                    warp_count -= 1
                    if warp_count == 0:
                        raise SMEMCapacityException(
                            "Tensor product schedule generation failed, shared memory inadequate!"
                        )

        if not deterministic:
            for segment in self.forward_schedule.segments:
                for key in segment.L3Map.storeback_procedure:
                    segment.L3Map.storeback_procedure[key] = "atomic_accumulate"

            for segment in self.backward_schedule.segments:
                for key in segment.L1Map.storeback_procedure:
                    segment.L1Map.storeback_procedure[key] = "atomic_accumulate"

            for segment in self.double_backward_schedule.segments:
                for key in segment.L1Map.storeback_procedure:
                    segment.L1Map.storeback_procedure[key] = "atomic_accumulate"

        idx_type_map = {np.int32: "int", np.int64: "long"}

        self.forward_workspace_offset = None
        self.backward_workspace_offset = None
        self.double_backwardB_offset = None

        self.workspace_size = 1
        if deterministic:
            destination_index_bytes = 32  # Add extra to account for padding
            self.workspace_size = max(
                (
                    self.forward_schedule.L3.dim * np.dtype(config.irrep_dtype).itemsize
                    + destination_index_bytes
                )
                * self.forward_schedule.total_warps,
                (
                    self.backward_schedule.L1.dim
                    * np.dtype(config.irrep_dtype).itemsize
                    + destination_index_bytes
                )
                * self.backward_schedule.total_warps,
                (
                    self.double_backward_schedule.L1.dim
                    * np.dtype(config.irrep_dtype).itemsize
                    + destination_index_bytes
                )
                * self.double_backward_schedule.total_warps,
            )

            self.forward_workspace_offset = (
                self.forward_schedule.L3.dim
                * np.dtype(config.irrep_dtype).itemsize
                * self.forward_schedule.total_warps
            )
            self.backward_workspace_offset = (
                self.backward_schedule.L1.dim
                * np.dtype(config.irrep_dtype).itemsize
                * self.backward_schedule.total_warps
            )
            self.double_backwardB_offset = (
                self.double_backward_schedule.L1.dim
                * np.dtype(config.irrep_dtype).itemsize
                * self.double_backward_schedule.total_warps
            )

            self.forward_workspace_offset = (self.forward_workspace_offset + 7) // 8 * 8
            self.backward_workspace_offset = (
                (self.backward_workspace_offset + 7) // 8 * 8
            )
            self.double_backwardB_offset = (self.double_backwardB_offset + 7) // 8 * 8

        self.kernel_prop = {
            "L1_dim": self.L1.dim,
            "L2_dim": self.L2.dim,
            "L3_dim": self.L3.dim,
            "weight_numel": self.config.weight_numel,
            "workspace_size": self.workspace_size,
            "opt_level": 3,
            "shared_weights": int(config.shared_weights),
            "deterministic": int(self.deterministic),
            "irrep_dtype": dtype_to_enum[self.config.irrep_dtype],
            "weight_dtype": dtype_to_enum[self.config.weight_dtype],
            "idx_dtype": dtype_to_enum[self.idx_dtype],
        }

        self.jit_kernel = template.render(
            forward_schedule=self.forward_schedule,
            backward_schedule=self.backward_schedule,
            double_backward_schedule=self.double_backward_schedule,
            idx_type=idx_type_map[idx_dtype],
            forward_workspace_offset=self.forward_workspace_offset,
            backward_workspace_offset=self.backward_workspace_offset,
            double_backwardB_offset=self.double_backwardB_offset,
        )
        self.jit_kernel = postprocess_kernel(self.jit_kernel)

        # with open("scratch.txt", "w") as f:
        #    f.write(self.jit_kernel)
