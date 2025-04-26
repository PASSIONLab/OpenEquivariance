from openequivariance.implementations.convolution.ConvolutionBase import *
from openequivariance.implementations.ComputationSchedule import ComputationSchedule
from openequivariance.implementations.TensorProduct import *
from openequivariance.templates.jinja_utils import *
from openequivariance.extlib import *

from openequivariance.implementations.utils import filter_and_analyze_problem 

class LoopUnrollConv(ConvolutionBase):
    def __init__(self, config, idx_dtype=np.int64, 
            torch_op=False, deterministic=False):
        super().__init__(config, idx_dtype, torch_op, deterministic)
        L1, L2, L3 = self.L1, self.L2, self.L3 

        env = get_jinja_environment()
        template = env.get_template("loop_unroll_conv_atomic.cuh")
        dp = DeviceProp(0)

        analysis = filter_and_analyze_problem(config)
        self.is_uvw = analysis["is_uvw"]

        forward_schedule_type = 3
        backward_schedule_type = 2
        if deterministic:
            backward_schedule_type = 3
            template = env.get_template("loop_unroll_conv_det.cuh")

        self.forward_schedule = ComputationSchedule(self.config, 
                smem_limit=dp.maxSharedMemPerBlock // 4 * 3, warps_per_block=6,
                block_count=dp.multiprocessorCount,
                direction = "forward",
                irrep_dtype = config.irrep_dtype,
                weight_dtype = config.weight_dtype,
                schedule_type=forward_schedule_type,
                warp_size=dp.warpsize,
                include_scratch=self.is_uvw,
                stream_weights=self.is_uvw)

        self.backward_schedule = ComputationSchedule(self.config, 
                smem_limit=dp.maxSharedMemPerBlock, warps_per_block=6,
                block_count=dp.multiprocessorCount * 2,
                direction = "backward",
                irrep_dtype = config.irrep_dtype,
                weight_dtype = config.weight_dtype,
                schedule_type=backward_schedule_type,
                warp_size=dp.warpsize)

        if not deterministic:
            for segment in self.forward_schedule.segments:
                for key in segment.L3Map.storeback_procedure:
                    segment.L3Map.storeback_procedure[key] = "atomic_accumulate"

            for segment in self.backward_schedule.segments:
                for key in segment.L1Map.storeback_procedure:
                    segment.L1Map.storeback_procedure[key] = "atomic_accumulate"

        idx_type_map = {np.int32: "int", np.int64: "long"}

        if self.torch_op:
            self.setup_torch_module()

        self.forward_workspace_offset = None
        self.backward_workspace_offset = None

        workspace_size = 1
        if deterministic:
            destination_index_bytes = 32 # Add extra to account for padding
            workspace_size = max(
                (self.forward_schedule.L3.dim * np.dtype(config.irrep_dtype).itemsize + destination_index_bytes) * self.forward_schedule.total_warps,
                (self.backward_schedule.L1.dim * np.dtype(config.irrep_dtype).itemsize + destination_index_bytes) * self.backward_schedule.total_warps)

            self.forward_workspace_offset = self.forward_schedule.L3.dim * np.dtype(config.irrep_dtype).itemsize * self.forward_schedule.total_warps
            self.backward_workspace_offset = self.backward_schedule.L1.dim * np.dtype(config.irrep_dtype).itemsize * self.backward_schedule.total_warps

            self.forward_workspace_offset = (self.forward_workspace_offset + 7) // 8 * 8
            self.backward_workspace_offset = (self.backward_workspace_offset + 7) // 8 * 8

        self.allocate_workspace(workspace_size)

        self.jit_kernel = template.render(
            forward_schedule=self.forward_schedule,
            backward_schedule=self.backward_schedule,
            idx_type=idx_type_map[idx_dtype],
            forward_workspace_offset=self.forward_workspace_offset,
            backward_workspace_offset=self.backward_workspace_offset)
        self.jit_kernel = postprocess_kernel(self.jit_kernel)

        if self.torch_op and extlib.TORCH_COMPILE:
            global torch
            import torch

            internal_cls = torch.classes.torch_tp_jit.TorchJITConv
        else:
            internal_cls = JITConvImpl 

        logger.info("Starting kernel compiler...")
        self.internal = internal_cls(self.jit_kernel,
                vars(self.forward_schedule.launch_config), 
                vars(self.backward_schedule.launch_config),
                {"L3_dim": self.L3.dim})
        logger.info("Kernel compiled!")

        self.reorder_weights_e3nn_to_oeq = lambda input, output, has_batch_dim: \
                self.forward_schedule.reorder_weights(input, output, "forward", has_batch_dim) 
        self.reorder_weights_oeq_to_e3nn = lambda input, output, has_batch_dim: \
                self.forward_schedule.reorder_weights(input, output, "backward", has_batch_dim) 

    @staticmethod
    def name():
        return "LoopUnrollConv"

    @classmethod
    def register_torch_fakes(cls):
        global torch
        import torch

        @torch._library.register_fake_class("torch_tp_jit::TorchJITConv")
        class TorchJITConv:
            def __init__(self, kernel_plaintext: str, 
                        fwd_config: dict[str, int], 
                        bwd_config: dict[str, int], 
                        kernel_dims: dict[str, int]) -> None:
                self.kernel_plaintext, self.fwd_config, self.bwd_config, self.kernel_dims = kernel_plaintext, fwd_config, bwd_config, kernel_dims

            @classmethod
            def __obj_unflatten__(cls, flattened_product):
                return cls(**dict(flattened_product))

            def __len__(self):
                return 0
            
            def __setstate__(self, state):
                self.kernel_plaintext, self.fwd_config, self.bwd_config, self.kernel_dims = state 
            
        @torch.library.register_fake("torch_tp_jit::jit_conv_forward")
        def fake_forward(jit, L1_in, L2_in, W, rows, cols, workspace_buffer, sender_perm):
            return L1_in.new_empty(L1_in.shape[0], jit.wrapped_obj.kernel_dims["L3_dim"]) 

        @torch.library.register_fake("torch_tp_jit::jit_conv_backward")
        def fake_backward(jit, L1_in, L2_in, W, L3_grad, rows, cols, workspace_buffer, sender_perm):
            return torch.empty_like(L1_in), torch.empty_like(L2_in), torch.empty_like(W) 

    @classmethod
    def register_autograd(cls):
        forward_op = torch.ops.torch_tp_jit.jit_conv_forward
        backward_op = torch.ops.torch_tp_jit.jit_conv_backward

        def setup_context(ctx, inputs, output):
            ctx.jit, ctx.L1_in, ctx.L2_in, ctx.W, ctx.rows, ctx.cols, ctx.workspace_buffer, ctx.sender_perm = inputs
        
        def backward(ctx, grad_output):
            L1_grad, L2_grad, W_grad= backward_op(ctx.jit, ctx.L1_in, ctx.L2_in, ctx.W, grad_output, ctx.rows, ctx.cols, ctx.workspace_buffer, ctx.sender_perm)
            return None, L1_grad, L2_grad, W_grad, None, None, None, None

        torch.library.register_autograd("torch_tp_jit::jit_conv_forward", backward, setup_context=setup_context)

        def setup_context_double_backward(ctx, inputs, output):
            ctx.jit, ctx.L1_in, ctx.L2_in, ctx.W, ctx.grad_output, ctx.rows, ctx.cols, ctx.workspace_buffer, ctx.sender_perm = inputs
            ctx.inputs = inputs

        def double_backward(ctx, E, F, G):
            jit, A, B, C, D, rows, cols, wspace, sender_perm = ctx.jit, ctx.L1_in, ctx.L2_in, ctx.grad_output, ctx.W, ctx.rows, ctx.cols, ctx.workspace_buffer, ctx.sender_perm

            op1 = backward_op(jit, E, F, D, C, rows, cols, wspace, sender_perm)
            op2 = backward_op(jit, A, B, G, C, rows, cols, wspace, sender_perm)
            op3 = forward_op(jit, E, B, D, rows, cols, wspace, sender_perm)
            op4 = backward_op(jit, E, B, D, C, rows, cols, wspace, sender_perm) # op4 and op5 could be combined with op3 and op6
            op5 = backward_op(jit, A, F, D, C, rows, cols, wspace, sender_perm)
            op6 = forward_op(jit, A, F, D, rows, cols, wspace, sender_perm)
            op7 = forward_op(jit, A, B, G, rows, cols, wspace, sender_perm)

            return None, op1[0] + op2[0], op1[1] + op2[1], op4[2] + op5[2], (op3 + op6 + op7), None, None, None, None

        torch.library.register_autograd("torch_tp_jit::jit_conv_backward", double_backward, setup_context=setup_context_double_backward)

if extlib.TORCH_COMPILE:
    LoopUnrollConv.register_torch_fakes()
    LoopUnrollConv.register_autograd()