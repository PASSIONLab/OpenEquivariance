import numpy as np

from openequivariance.implementations.convolution.ConvolutionBase import ConvolutionBase
from openequivariance.implementations.ComputationSchedule import (
    ComputationSchedule,
    SMEMCapacityException,
)


from openequivariance.templates.jinja_utils import get_jinja_environment
from openequivariance import extlib
from openequivariance.extlib import JITConvImpl, postprocess_kernel, DeviceProp

from openequivariance.implementations.utils import filter_and_analyze_problem

from openequivariance.benchmark.logging_utils import getLogger

logger = getLogger()


class LoopUnrollConv(ConvolutionBase):
    def __init__(
        self,
        config,
        idx_dtype=np.int64,
        torch_op=False,
        deterministic=False,
        kahan=False,
    ):
        super().__init__(config, idx_dtype, torch_op, deterministic)

        if kahan:
            assert deterministic

        env = get_jinja_environment()
        template = env.get_template("loop_unroll_conv_atomic.cuh")
        dp = DeviceProp(0)

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

        workspace_size = 1
        if deterministic:
            destination_index_bytes = 32  # Add extra to account for padding
            workspace_size = max(
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

        self.allocate_workspace(workspace_size)

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

        if self.torch_op and extlib.TORCH_COMPILE:
            global torch
            import torch

            internal_cls = torch.classes.libtorch_tp_jit.TorchJITConv
        else:
            internal_cls = JITConvImpl

        logger.info("Starting kernel compiler...")
        self.internal = internal_cls(
            self.jit_kernel,
            vars(self.forward_schedule.launch_config),
            vars(self.backward_schedule.launch_config),
            vars(self.double_backward_schedule.launch_config),
            {
                "L3_dim": self.L3.dim,
                "opt_level": 3,
                "shared_weights": int(config.shared_weights),
            },
        )
        logger.info("Kernel compiled!")

        # with open("scratch.txt", "w") as f:
        #    f.write(self.jit_kernel)

    def reorder_weights_e3nn_to_oeq(self, input, output, has_batch_dim):
        return self.forward_schedule.reorder_weights(
            input, output, "forward", has_batch_dim
        )

    def reorder_weights_oeq_to_e3nn(self, input, output, has_batch_dim):
        return self.forward_schedule.reorder_weights(
            input, output, "backward", has_batch_dim
        )

    @staticmethod
    def name():
        return "LoopUnrollConv"

    @classmethod
    def register_torch_fakes(cls):
        global torch
        import torch

        @torch._library.register_fake_class("libtorch_tp_jit::TorchJITConv")
        class TorchJITConv:
            def __init__(
                self,
                kernel_plaintext: str,
                fwd_config: dict[str, int],
                bwd_config: dict[str, int],
                dbl_bwd_config: dict[str, int],
                kernel_dims: dict[str, int],
            ) -> None:
                (
                    self.kernel_plaintext,
                    self.fwd_config,
                    self.bwd_config,
                    self.dbl_bwd_config,
                    self.kernel_dims,
                ) = (
                    kernel_plaintext,
                    fwd_config,
                    bwd_config,
                    dbl_bwd_config,
                    kernel_dims,
                )

            @classmethod
            def __obj_unflatten__(cls, flattened_product):
                return cls(**dict(flattened_product))

            def __len__(self):
                return 0

            def __setstate__(self, state):
                (
                    self.kernel_plaintext,
                    self.fwd_config,
                    self.bwd_config,
                    self.dbl_bwd_config,
                    self.kernel_dims,
                ) = state

        @torch.library.register_fake("libtorch_tp_jit::jit_conv_forward")
        def fake_forward(
            jit, L1_in, L2_in, W, rows, cols, workspace_buffer, sender_perm
        ):
            L3_dim = None
            if hasattr(jit, "wrapped_obj"):
                L3_dim = jit.wrapped_obj.kernel_dims["L3_dim"]
            else:
                L3_dim = jit.get_L3_dim()

            return L1_in.new_empty(L1_in.shape[0], L3_dim)

        @torch.library.register_fake("libtorch_tp_jit::jit_conv_backward")
        def fake_backward(
            jit, L1_in, L2_in, W, L3_grad, rows, cols, workspace_buffer, sender_perm
        ):
            return torch.empty_like(L1_in), torch.empty_like(L2_in), torch.empty_like(W)

    @classmethod
    def register_autograd(cls):
        backward_op = torch.ops.libtorch_tp_jit.jit_conv_backward
        double_backward_op = torch.ops.libtorch_tp_jit.jit_conv_double_backward

        def setup_context(ctx, inputs, output):
            (
                ctx.jit,
                ctx.L1_in,
                ctx.L2_in,
                ctx.W,
                ctx.rows,
                ctx.cols,
                ctx.workspace_buffer,
                ctx.sender_perm,
            ) = inputs

        def backward(ctx, grad_output):
            L1_grad, L2_grad, W_grad = backward_op(
                ctx.jit,
                ctx.L1_in,
                ctx.L2_in,
                ctx.W,
                grad_output,
                ctx.rows,
                ctx.cols,
                ctx.workspace_buffer,
                ctx.sender_perm,
            )
            return None, L1_grad, L2_grad, W_grad, None, None, None, None

        torch.library.register_autograd(
            "libtorch_tp_jit::jit_conv_forward", backward, setup_context=setup_context
        )

        def setup_context_double_backward(ctx, inputs, output):
            (
                ctx.jit,
                ctx.L1_in,
                ctx.L2_in,
                ctx.W,
                ctx.grad_output,
                ctx.rows,
                ctx.cols,
                ctx.workspace_buffer,
                ctx.sender_perm,
            ) = inputs
            ctx.inputs = inputs

        def double_backward(ctx, E, F, G):
            result = double_backward_op(
                ctx.jit,
                ctx.L1_in,
                ctx.L2_in,
                ctx.W,
                ctx.grad_output,
                E,
                F,
                G,
                ctx.rows,
                ctx.cols,
                ctx.workspace_buffer,
                ctx.sender_perm,
            )
            return (
                None,
                result[0],
                result[1],
                result[2],
                result[3],
                None,
                None,
                None,
                None,
            )

        torch.library.register_autograd(
            "libtorch_tp_jit::jit_conv_backward",
            double_backward,
            setup_context=setup_context_double_backward,
        )


if extlib.TORCH_COMPILE:
    LoopUnrollConv.register_torch_fakes()
    LoopUnrollConv.register_autograd()
