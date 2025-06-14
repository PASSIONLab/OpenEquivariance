import numpy as np

from openequivariance.extlib import DeviceBuffer, GPUTimer

from openequivariance.implementations.e3nn_lite import TPProblem, wigner_3j
from openequivariance.benchmark.logging_utils import getLogger

logger = getLogger()


class TensorProductBase:
    next_tp_id = 0  # Assign unique IDs to each TP instance

    @staticmethod
    def load_cg_tensor(l1, l2, l3):
        return wigner_3j(l1, l2, l3)

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
        self.reorder_weights_e3nn_to_oeq, self.reorder_weights_oeq_to_e3nn = None, None

        self.tp_id = TensorProductBase.next_tp_id
        TensorProductBase.next_tp_id += 1

        if torch_op:
            global torch
            import torch

    def __call__(self, L1_in, L2_in, weights):
        return self.forward(L1_in, L2_in, weights)

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
        weights_chunked = np.zeros_like(weights)
        if self.reorder_weights_e3nn_to_oeq is not None:
            self.reorder_weights_e3nn_to_oeq(
                weights, weights_chunked, not self.config.shared_weights
            )
        else:
            weights_chunked = weights

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
        weights_chunked = np.zeros_like(weights)
        if self.reorder_weights_e3nn_to_oeq is not None:
            self.reorder_weights_e3nn_to_oeq(
                weights, weights_chunked, not self.config.shared_weights
            )
        else:
            weights_chunked = weights

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

        if self.reorder_weights_oeq_to_e3nn is not None:
            weights_grad_copy = weights_grad.copy()
            self.reorder_weights_oeq_to_e3nn(
                weights_grad_copy, weights_grad, not self.config.shared_weights
            )

    def benchmark_forward(
        self,
        num_warmup: int,
        num_iter: int,
        L1_in: np.ndarray,
        L2_in: np.ndarray,
        L3_buffer: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        time_millis = np.zeros(num_iter, dtype=np.float32)

        # GPUTimer introduces significantly less overhead when kernel runtime < 1ms
        timer = GPUTimer()

        if self.torch_op:
            torch_L1_in = torch.tensor(L1_in).to(device="cuda").detach()
            torch_L2_in = torch.tensor(L2_in).to(device="cuda").detach()
            torch_weights = torch.tensor(weights).to(device="cuda").detach()

            for i in range(num_warmup):
                self.forward(torch_L1_in, torch_L2_in, torch_weights)

            for i in range(num_iter):
                timer.clear_L2_cache()
                timer.start()
                self.forward(torch_L1_in, torch_L2_in, torch_weights)
                time_millis[i] = timer.stop_clock_get_elapsed()
        else:
            batch = L1_in.shape[0]
            L1_d, L2_d, L3_d = (
                DeviceBuffer(L1_in),
                DeviceBuffer(L2_in),
                DeviceBuffer(L3_buffer),
            )
            weights_d = DeviceBuffer(weights)

            for i in range(num_warmup):
                self.internal.exec_tensor_product_rawptr(
                    batch,
                    L1_d.data_ptr(),
                    L2_d.data_ptr(),
                    L3_d.data_ptr(),
                    weights_d.data_ptr(),
                )

            for i in range(num_iter):
                timer.clear_L2_cache()
                timer.start()
                self.internal.exec_tensor_product_rawptr(
                    batch,
                    L1_d.data_ptr(),
                    L2_d.data_ptr(),
                    L3_d.data_ptr(),
                    weights_d.data_ptr(),
                )
                time_millis[i] = timer.stop_clock_get_elapsed()

        return time_millis

    def benchmark_backward(
        self,
        num_warmup: int,
        num_iter: int,
        L1_in: np.ndarray,
        L2_in: np.ndarray,
        L3_buffer: np.ndarray,
        weights: np.ndarray,
        L1_grad: np.ndarray,
        L2_grad: np.ndarray,
        weights_grad: np.ndarray,
    ) -> np.ndarray:
        time_millis = np.zeros(num_iter, dtype=np.float32)
        timer = GPUTimer()

        if self.torch_op:
            torch_L1_in = torch.tensor(L1_in, requires_grad=True, device="cuda")
            torch_L2_in = torch.tensor(L2_in, requires_grad=True, device="cuda")
            torch_weights = torch.tensor(weights, requires_grad=True, device="cuda")
            torch_out = self.forward(torch_L1_in, torch_L2_in, torch_weights)
            torch_L3_grad_in = torch.tensor(L3_buffer, device="cuda")

            for i in range(num_warmup):
                torch_out.backward(
                    gradient=torch_L3_grad_in,
                    retain_graph=True,
                    inputs=[torch_L1_in, torch_L2_in, torch_weights],
                )

            for i in range(num_iter):
                torch_L1_in.grad.zero_()
                torch_L2_in.grad.zero_()
                torch_weights.grad.zero_()

                timer.clear_L2_cache()
                timer.start()
                torch_out.backward(
                    gradient=torch_L3_grad_in,
                    retain_graph=True,
                    inputs=[torch_L1_in, torch_L2_in, torch_weights],
                )
                time_millis[i] = timer.stop_clock_get_elapsed()

            L1_grad[:] = torch_L1_in.grad.numpy(force=True)
            L2_grad[:] = torch_L2_in.grad.numpy(force=True)
            weights_grad[:] = torch_weights.grad.numpy(force=True)
        else:
            batch = L1_in.shape[0]
            L1_d, L2_d, L3_d = (
                DeviceBuffer(L1_in),
                DeviceBuffer(L2_in),
                DeviceBuffer(L3_buffer),
            )
            L1_grad_d, L2_grad_d = DeviceBuffer(L1_grad), DeviceBuffer(L2_grad)
            weights_d, weights_grad_d = (
                DeviceBuffer(weights),
                DeviceBuffer(weights_grad),
            )

            for i in range(num_warmup):
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

            for i in range(num_iter):
                timer.clear_L2_cache()
                timer.start()
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
                time_millis[i] = timer.stop_clock_get_elapsed()

        return time_millis

    def benchmark_double_backward(
        self,
        num_warmup: int,
        num_iter: int,
        L1_in: np.ndarray,
        L2_in: np.ndarray,
        L3_buffer: np.ndarray,
        weights: np.ndarray,
        L1_grad: np.ndarray,
        L2_grad: np.ndarray,
        weights_grad: np.ndarray,
        L3_double_grad: np.ndarray,
    ) -> np.ndarray:
        time_millis = np.zeros(num_iter, dtype=np.float32)
        timer = GPUTimer()

        if self.torch_op:
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
            torch_L3_double_grad = torch.tensor(
                L3_double_grad, device="cuda", requires_grad=True
            )

            (torch_L1_grad, torch_L2_grad, torch_weights_grad, torch_L3_double_grad) = (
                torch.autograd.grad(
                    outputs=dummy,
                    inputs=[torch_L1_in, torch_L2_in, torch_weights, torch_out_grad],
                    grad_outputs=dummy_grad,
                    retain_graph=True,
                )
            )

            for i in range(num_warmup):
                (
                    torch_L1_grad,
                    torch_L2_grad,
                    torch_weights_grad,
                    torch_L3_double_grad,
                ) = torch.autograd.grad(
                    outputs=dummy,
                    inputs=[torch_L1_in, torch_L2_in, torch_weights, torch_out_grad],
                    grad_outputs=dummy_grad,
                    retain_graph=True,
                )

            for i in range(num_iter):
                timer.clear_L2_cache()
                timer.start()
                (
                    torch_L1_grad,
                    torch_L2_grad,
                    torch_weights_grad,
                    torch_L3_double_grad,
                ) = torch.autograd.grad(
                    outputs=dummy,
                    inputs=[torch_L1_in, torch_L2_in, torch_weights, torch_out_grad],
                    grad_outputs=dummy_grad,
                    retain_graph=True,
                )
                time_millis[i] = timer.stop_clock_get_elapsed()

            L1_grad[:] = torch_L1_grad.numpy(force=True)
            L2_grad[:] = torch_L2_grad.numpy(force=True)
            weights_grad[:] = torch_weights_grad.numpy(force=True)
            L3_double_grad[:] = torch_L3_double_grad.numpy(force=True)
        else:
            batch = L1_in.shape[0]
            L1_d, L2_d, L3_d = (
                DeviceBuffer(L1_in),
                DeviceBuffer(L2_in),
                DeviceBuffer(L3_buffer),
            )
            L1_grad_d, L2_grad_d = DeviceBuffer(L1_grad), DeviceBuffer(L2_grad)
            weights_d, weights_grad_d = (
                DeviceBuffer(weights),
                DeviceBuffer(weights_grad),
            )

            for i in range(num_warmup):
                self.internal.double_backward(
                    batch,
                    L1_d.data_ptr(),
                    L1_grad_d.data_ptr(),
                    L2_d.data_ptr(),
                    L2_grad_d.data_ptr(),
                    weights_d.data_ptr(),
                    weights_grad_d.data_ptr(),
                    L3_d.data_ptr(),
                )

            for i in range(num_iter):
                timer.clear_L2_cache()
                timer.start()
                self.internal.double_backward(
                    batch,
                    L1_d.data_ptr(),
                    L1_grad_d.data_ptr(),
                    L2_d.data_ptr(),
                    L2_grad_d.data_ptr(),
                    weights_d.data_ptr(),
                    weights_grad_d.data_ptr(),
                    L3_d.data_ptr(),
                )
                time_millis[i] = timer.stop_clock_get_elapsed()

        return time_millis

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
