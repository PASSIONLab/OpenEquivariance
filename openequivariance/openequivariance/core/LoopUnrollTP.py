import json

import numpy as np

from openequivariance.core.logging import getLogger
from openequivariance.core.ComputationSchedule import (
    ComputationSchedule,
    SMEMCapacityException,
)
from openequivariance.core.TensorProductBase import TensorProductBase
from openequivariance.core.utils import (
    dtype_to_enum,
    filter_and_analyze_problem,
    hash_str_64,
)
from openequivariance.templates.jinja_utils import get_jinja_environment

logger = getLogger()


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
                except SMEMCapacityException:
                    warp_count -= 2
                    if warp_count == 0:
                        raise RuntimeError(
                            "Tensor product schedule generation failed, shared memory inadequate!"
                        )
                except Exception:
                    raise

        self.jit_kernel = postprocess_kernel(
            template.render(
                forward_schedule=self.forward_schedule,
                backward_schedule=self.backward_schedule,
                double_backward_schedule=self.double_backward_schedule,
            )
        )

        self.kernel_prop = {
            "L1_dim": self.L1.dim,
            "L2_dim": self.L2.dim,
            "L3_dim": self.L3.dim,
            "weight_numel": self.config.weight_numel,
            "shared_weights": int(self.config.shared_weights),
            "opt_level": 3,
            "irrep_dtype": dtype_to_enum[self.config.irrep_dtype],
            "weight_dtype": dtype_to_enum[self.config.weight_dtype],
            # Not relevant, included for compatibility with convolution
            "workspace_size": 0,
            "deterministic": 1,
            "idx_dtype": 0,
        }

        self.kernel_string = json.dumps(
            {
                "kernel": self.jit_kernel,
                "forward_config": vars(self.forward_schedule.launch_config),
                "backward_config": vars(self.backward_schedule.launch_config),
                "double_backward_config": vars(
                    self.double_backward_schedule.launch_config
                ),
                "kernel_prop": self.kernel_prop,
            }
        )
        self.hash = hash_str_64(self.kernel_string)
        logger.info(f"Kernel File Size: {len(self.jit_kernel) // 1024} KB")
