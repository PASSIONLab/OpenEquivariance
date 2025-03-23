import numpy as np

from openequivariance.extlib import *
from openequivariance.templates.jinja_utils import *
from openequivariance.implementations.ComputationSchedule import ComputationSchedule 

from openequivariance.implementations.TensorProductBase import TensorProductBase 
from openequivariance.benchmark.logging_utils import getLogger, bcolors
from openequivariance.benchmark.e3nn_lite_utils import count_cg_non_zero
logger = getLogger()

class LoopUnrollTP(TensorProductBase):
    def __init__(self, config, torch_op=True):
        super().__init__(config, torch_op=torch_op)
        L1, L2, L3 = self.L1, self.L2, self.L3

        env = get_jinja_environment()
        template = env.get_template("loop_unroll_batch.cuh")
        env.globals['enumerate'] = enumerate 

        dp = DeviceProp(0)

        if len(config.instructions) == 0:
            raise ValueError("Tensor product problem has no valid intructions!")

        for inst in config.instructions:
            assert(inst.connection_mode == config.instructions[0].connection_mode)         
        assert(config.instructions[0].connection_mode in ["uvu", "uvw"]) 
        assert(config.irrep_dtype == config.weight_dtype)
        self.is_uvw = (config.instructions[0].connection_mode == "uvw")

        def generate_forward_schedule(warps_per_block):
            self.forward_schedule = ComputationSchedule(self.config, 
                    smem_limit=dp.maxSharedMemPerBlock, warps_per_block=warps_per_block,
                    block_count=dp.multiprocessorCount * 4,
                    direction = "forward",
                    irrep_dtype = config.irrep_dtype,
                    weight_dtype = config.weight_dtype,
                    include_scratch=self.is_uvw,
                    stream_weights=self.is_uvw)

        def generate_backward_schedule(warps_per_block):
            self.backward_schedule = ComputationSchedule(self.config, 
                    smem_limit=dp.maxSharedMemPerBlock, warps_per_block=warps_per_block,
                    block_count=dp.multiprocessorCount * 3,
                    direction = "backward",
                    irrep_dtype = config.irrep_dtype,
                    weight_dtype = config.weight_dtype,
                    include_scratch=self.is_uvw,
                    stream_weights=self.is_uvw)

        # Latent error: warps per block must be a multiple of 4 or we run into
        # problems for uvw, float64 backward pass. Need to eventually fix.

        try:
            generate_forward_schedule(8)
        except Exception as e:
            generate_forward_schedule(4)

        try:
            generate_backward_schedule(8)
        except Exception as e:
            generate_backward_schedule(4)

        self.jit_kernel = template.render(
            forward_schedule=self.forward_schedule,
            backward_schedule=self.backward_schedule)

        with open("scratch.txt", "w") as f:
            f.write(self.jit_kernel)

        logger.info("Starting NVRTC")
        self.internal = JITTPImpl(self.jit_kernel,
                self.forward_schedule.launch_config,
                self.backward_schedule.launch_config)
        logger.info("Kernel compiled!")

        logger.info(f"CUDA Kernel File Size: {len(self.jit_kernel) // 1000} KB")

        if self.torch_op:
            self.setup_torch_custom_op()

        self.reorder_weights_e3nn_to_oeq = lambda input, output, has_batch_dim: \
                self.forward_schedule.reorder_weights(input, output, "forward", has_batch_dim) 
        self.reorder_weights_oeq_to_e3nn = lambda input, output, has_batch_dim: \
                self.forward_schedule.reorder_weights(input, output, "backward", has_batch_dim) 


    @staticmethod
    def name():
        return "LoopUnrollTP"
 
    def calculate_flops_forward(self, batch_size : int) -> dict:
        if self.is_uvw:
            return super().calculate_flops_forward(batch_size)
        else:
            tpp = self.config
            flop_count = {'CG_decomposition': 0, 'linear_combination': 0, 'outer_products': 0}
            for ins in tpp.instructions: 
                l1, l2, l3 = tpp.irreps_in1[ins.i_in1].ir.l, tpp.irreps_in2[ins.i_in2].ir.l, tpp.irreps_out[ins.i_out].ir.l
                flop_count["CG_decomposition"] += count_cg_non_zero(l1, l2, l3) * (ins.path_shape[0] * ins.path_shape[1])
                flop_count["linear_combination"] += (2 * l3 + 1) * np.prod(ins.path_shape) if ins.has_weight else 0

            flop_count["CG_decomposition"] *= 3 * batch_size
            flop_count["linear_combination"] *= batch_size    # Weights do not require FMA here
            flop_count["total"] = sum(flop_count.values())
            return flop_count

    def calculate_flops_backward(self, batch_size : int) -> dict:
        if self.is_uvw:
            return super().calculate_flops_backward(batch_size)
        else:
            tpp = self.config
            flop_count = {'backward': 0} 
            for ins in tpp.instructions: 
                l1, l2, l3 = tpp.irreps_in1[ins.i_in1].ir.l, tpp.irreps_in2[ins.i_in2].ir.l, tpp.irreps_out[ins.i_out].ir.l
                flop_count["backward"] += count_cg_non_zero(l1, l2, l3) * (ins.path_shape[0] * ins.path_shape[1])

            flop_count["backward"] *= 9 * batch_size
            flop_count["total"] = sum(flop_count.values())
            return flop_count