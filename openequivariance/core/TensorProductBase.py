import numpy as np

from openequivariance.core.e3nn_lite import TPProblem
from openequivariance.benchmark.logging_utils import getLogger
from openequivariance.core.utils import benchmark
from openequivariance.extlib import DeviceBuffer

logger = getLogger()


class TensorProductBase:
    next_tp_id = 0  # Assign unique IDs to each TP instance

    """
    Each class implementation of a TensorProduct uses
    a different internal representation, which it can
    initialize uniquely.
    """

    def __init__(self, config: TPProblem, torch_op: bool = False):
        assert isinstance(config, TPProblem)
        assert isinstance(torch_op, bool)
        config = config.clone()
        self.config, self.torch_op = config, torch_op
        self.L1, self.L2, self.L3 = (
            config.irreps_in1,
            config.irreps_in2,
            config.irreps_out,
        )
        self.irrep_dtype, self.weight_dtype = config.irrep_dtype, config.weight_dtype

        self.tp_id = TensorProductBase.next_tp_id
        TensorProductBase.next_tp_id += 1

        if torch_op:
            global torch
            import torch

    def __call__(self, L1_in, L2_in, weights):
        return self.forward(L1_in, L2_in, weights)

    def reorder_weights_from_e3nn(self, weights, has_batch_dim: bool = True):
        r"""
        Reorders weights from ``e3nn`` canonical order to the order used by ``oeq``.

        :param weights: Weights in ``e3nn`` canonical order, either an
                        np.ndarray or a torch.Tensor. Tensor of dimensions ``[B, problem.weight_numel]``
                        when ``has_batch_dim=True``, otherwise of dimensions ``[problem.weight_numel]``.

        :param has_batch_dim: If ``True``, treats the first dimension of weights as a batch dimension. Default: ``True``.

        :return: Weights in ``oeq`` order. Output type is identical to input.
        """
        return weights

    def reorder_weights_to_e3nn(self, weights, has_batch_dim: bool = True):
        r"""
        Reorders weights from ``oeq`` canonical order to the order used by ``e3nn``.

        :param weights: Weights in ``oeq`` canonical order, either an
                        np.ndarray or a torch.Tensor. Tensor of dimensions ``[B, problem.weight_numel]``
                        when ``has_batch_dim=True``, otherwise of dimensions ``[problem.weight_numel]``.

        :param has_batch_dim: If ``True``, treats the first dimension of wieghts as a batch dimension. Default: ``True``.

        :return: Weights in ``e3nn`` order. Output type is identical to input.
        """
        return weights

    def forward_raw(
        self,
        batch: np.uint64,
        L1_in: np.uint64,
        L2_in: np.uint64,
        L3_out: np.uint64,
        weights: np.uint64,
    ) -> None:
        self.internal.exec_tensor_product_rawptr(batch, L1_in, L2_in, L3_out, weights)

    def backward_raw(
        self,
        batch_size: np.uint64,
        L1_in: np.uint64,
        L1_grad: np.uint64,
        L2_in: np.uint64,
        L2_grad: np.uint64,
        weights: np.uint64,
        weights_grad: np.uint64,
        L3_grad: np.uint64,
    ):
        self.internal.backward_rawptr(
            batch_size, L1_in, L1_grad, L2_in, L2_grad, weights, weights_grad, L3_grad
        )

    def forward_cpu(
        self,
        L1_in: np.ndarray,
        L2_in: np.ndarray,
        L3_out: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        weights_chunked = self.reorder_weights_from_e3nn(
            weights, not self.config.shared_weights
        )

        batch = L1_in.shape[0]
        L1_d = DeviceBuffer(L1_in)
        L2_d = DeviceBuffer(L2_in)
        L3_d = DeviceBuffer(L3_out)
        weights_d = DeviceBuffer(weights_chunked)
        self.internal.exec_tensor_product_rawptr(
            batch,
            L1_d.data_ptr(),
            L2_d.data_ptr(),
            L3_d.data_ptr(),
            weights_d.data_ptr(),
        )
        L3_d.copy_to_host()

    def backward_cpu(
        self, L1_in, L1_grad, L2_in, L2_grad, L3_grad, weights, weights_grad
    ) -> None:
        weights_chunked = self.reorder_weights_from_e3nn(
            weights, not self.config.shared_weights
        )

        batch = L1_in.shape[0]
        L1_d, L2_d, L3_d = (
            DeviceBuffer(L1_in),
            DeviceBuffer(L2_in),
            DeviceBuffer(L3_grad),
        )
        L1_grad_d, L2_grad_d = DeviceBuffer(L1_grad), DeviceBuffer(L2_grad)
        weights_d, weights_grad_d = (
            DeviceBuffer(weights_chunked),
            DeviceBuffer(weights_grad),
        )

        self.internal.backward_rawptr(
            batch,
            L1_d.data_ptr(),
            L1_grad_d.data_ptr(),
            L2_d.data_ptr(),
            L2_grad_d.data_ptr(),
            weights_d.data_ptr(),
            weights_grad_d.data_ptr(),
            L3_d.data_ptr(),
        )

        L1_grad_d.copy_to_host()
        L2_grad_d.copy_to_host()
        weights_grad_d.copy_to_host()

        weights_grad[:] = self.reorder_weights_to_e3nn(
            weights_grad, not self.config.shared_weights
        )

    def benchmark_forward(
        self,
        num_warmup: int,
        num_iter: int,
        L1_in: np.ndarray,
        L2_in: np.ndarray,
        L3_buffer: np.ndarray,
        weights: np.ndarray,
        with_torch_overhead: bool = True,
        kernel_names=["forward"],
    ) -> np.ndarray:
        torch_L1_in = torch.tensor(L1_in).to(device="cuda").detach()
        torch_L2_in = torch.tensor(L2_in).to(device="cuda").detach()
        torch_weights = torch.tensor(weights).to(device="cuda").detach()

        mode = "gpu_time" if with_torch_overhead else "torch_kernel_time"
        return benchmark(
            (lambda: self.forward(torch_L1_in, torch_L2_in, torch_weights)),
            num_warmup,
            num_iter,
            mode=mode,
            kernel_names=kernel_names,
        )

    def benchmark_backward(
        self,
        num_warmup: int,
        num_iter: int,
        L1_in: np.ndarray,
        L2_in: np.ndarray,
        L3_buffer: np.ndarray,
        weights: np.ndarray,
        with_torch_overhead: bool = True,
        kernel_names=["backward"],
    ) -> np.ndarray:
        torch_L1_in = torch.tensor(L1_in, requires_grad=True, device="cuda")
        torch_L2_in = torch.tensor(L2_in, requires_grad=True, device="cuda")
        torch_weights = torch.tensor(weights, requires_grad=True, device="cuda")
        torch_out = self.forward(torch_L1_in, torch_L2_in, torch_weights)
        torch_L3_grad_in = torch.tensor(L3_buffer, device="cuda")

        mode = "gpu_time" if with_torch_overhead else "torch_kernel_time"

        return benchmark(
            (
                lambda: torch_out.backward(
                    gradient=torch_L3_grad_in,
                    retain_graph=True,
                    inputs=[torch_L1_in, torch_L2_in, torch_weights],
                )
            ),
            num_warmup,
            num_iter,
            mode=mode,
            kernel_names=kernel_names,
        )

    def benchmark_double_backward(
        self,
        num_warmup: int,
        num_iter: int,
        L1_in: np.ndarray,
        L2_in: np.ndarray,
        weights: np.ndarray,
        weights_grad: np.ndarray,
        with_torch_overhead: bool = True,
        kernel_names=["double_backward_A", "double_backward_B"],
    ) -> np.ndarray:
        torch_L1_in = torch.tensor(L1_in, requires_grad=True, device="cuda")
        torch_L2_in = torch.tensor(L2_in, requires_grad=True, device="cuda")
        torch_weights = torch.tensor(weights, requires_grad=True, device="cuda")

        torch_out = self(torch_L1_in, torch_L2_in, torch_weights)
        torch_out_grad = (
            torch_out.clone().detach().to(device="cuda").requires_grad_(True)
        )

        (torch_L1_grad, torch_L2_grad, torch_weights_grad) = torch.autograd.grad(
            outputs=torch_out,
            inputs=[torch_L1_in, torch_L2_in, torch_weights],
            grad_outputs=torch_out_grad,
            create_graph=True,
            retain_graph=True,
        )

        dummy = (
            torch.norm(torch_L1_grad)
            + torch.norm(torch_L2_grad)
            + torch.norm(torch_weights_grad)
        )
        dummy_grad = torch.tensor(float(dummy), device="cuda", requires_grad=True)

        torch_L1_grad = torch.tensor(L1_in, requires_grad=True, device="cuda")
        torch_L2_grad = torch.tensor(L2_in, requires_grad=True, device="cuda")
        torch_weights_grad = torch.tensor(
            weights_grad, requires_grad=True, device="cuda"
        )

        mode = "gpu_time" if with_torch_overhead else "torch_kernel_time"

        return benchmark(
            (
                lambda: torch.autograd.grad(
                    outputs=dummy,
                    inputs=[torch_L1_in, torch_L2_in, torch_weights, torch_out_grad],
                    grad_outputs=dummy_grad,
                    retain_graph=True,
                )
            ),
            num_warmup,
            num_iter,
            mode=mode,
            kernel_names=kernel_names,
        )

    def calculate_memory_streamed_forward(self, batch_size: int) -> dict:
        raise NotImplementedError("This needs to be implemented in your class")

    def calculate_memory_streamed_backward(self, batch_size: int) -> dict:
        raise NotImplementedError("This needs to be implemented in your class")

    def calculate_memory_streamed_double_backward(self, batch_size: int) -> dict:
        raise NotImplementedError("This needs to be implemented in your class")

    def calculate_flops_forward(self, batch_size: int) -> dict:
        raise NotImplementedError("This needs to be implemented in your class")

    def calculate_flops_backward(self, batch_size: int) -> dict:
        raise NotImplementedError("This needs to be implemented in your class")

    def calculate_flops_double_backward(self, batch_size: int) -> dict:
        raise NotImplementedError("This needs to be implemented in your class")
