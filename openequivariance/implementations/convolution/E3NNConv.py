import numpy as np

from openequivariance.implementations.convolution.ConvolutionBase import ConvolutionBase
from openequivariance.implementations.E3NNTensorProduct import E3NNTensorProduct


class E3NNConv(ConvolutionBase):
    def __init__(self, config, idx_dtype=np.int64, torch_op=True):
        assert torch_op
        super().__init__(config, idx_dtype, torch_op)

        from e3nn import o3
        import torch

        if config.irrep_dtype == np.float64:
            torch.set_default_dtype(torch.float64)

        self.e3nn_tp = o3.TensorProduct(
            config.irreps_in1,
            config.irreps_in2,
            config.irreps_out,
            config.instructions_raw,
            in1_var=config.in1_var,
            in2_var=config.in2_var,
            out_var=config.out_var,
            irrep_normalization=config.irrep_normalization,
            path_normalization=config.path_normalization,
            internal_weights=config.internal_weights,
            shared_weights=config.shared_weights,
        ).to(device="cuda")

        self.reference_tp = E3NNTensorProduct(config)

        if config.irrep_dtype == np.float64:
            torch.set_default_dtype(torch.float32)  # Reset to default

        from openequivariance.implementations.convolution.scatter import scatter_sum

        self.scatter_sum = scatter_sum

    def forward(self, L1_in, L2_in, weights, rows, cols):
        tp_outputs = self.reference_tp(L1_in[cols], L2_in, weights)
        return self.scatter_sum(
            src=tp_outputs, index=rows, dim=0, dim_size=L1_in.shape[0]
        )

    @staticmethod
    def name():
        return "E3NNConvolution"

    def forward_cpu(self, L1_in, L2_in, weights, L3_out, graph):
        tp_outputs = np.zeros((graph.nnz, self.L3.dim), dtype=L3_out.dtype)
        self.reference_tp.forward_cpu(L1_in[graph.cols], L2_in, tp_outputs, weights)
        np.add.at(L3_out, graph.rows, tp_outputs)

    def backward_cpu(
        self,
        L1_in: np.ndarray,
        L1_grad: np.ndarray,
        L2_in: np.ndarray,
        L2_grad: np.ndarray,
        L3_grad: np.ndarray,
        weights: np.ndarray,
        weights_grad: np.ndarray,
        graph,
    ):
        L1_grad_bcast = np.zeros((graph.nnz, self.L1.dim), dtype=L1_grad.dtype)
        self.reference_tp.backward_cpu(
            L1_in[graph.cols],
            L1_grad_bcast,
            L2_in,
            L2_grad,
            L3_grad[graph.rows],
            weights,
            weights_grad,
        )
        np.add.at(L1_grad, graph.cols, L1_grad_bcast)
