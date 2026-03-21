import torch

from openequivariance.core.utils import IrrepLayoutUtils


class NumpyDoubleBackwardMixin:
    """
    Adds a Numpy double backward method to any TensorProduct
    with the forward pass defined in PyTorch and the relevant
    derivatives registered.
    """

    def double_backward_cpu(
        self, in1, in2, out_grad, weights, weights_dgrad, in1_dgrad, in2_dgrad
    ):
        assert self.torch_op

        layout = self.config.layout

        in1_kernel = IrrepLayoutUtils.transpose_irrep_layout(
            in1, self.config.irreps_in1, layout, "mul_ir"
        )
        in2_kernel = IrrepLayoutUtils.transpose_irrep_layout(
            in2, self.config.irreps_in2, layout, "mul_ir"
        )
        out_grad_kernel = IrrepLayoutUtils.transpose_irrep_layout(
            out_grad, self.config.irreps_out, layout, "mul_ir"
        )
        in1_dgrad_kernel = IrrepLayoutUtils.transpose_irrep_layout(
            in1_dgrad, self.config.irreps_in1, layout, "mul_ir"
        )
        in2_dgrad_kernel = IrrepLayoutUtils.transpose_irrep_layout(
            in2_dgrad, self.config.irreps_in2, layout, "mul_ir"
        )

        in1_torch = torch.tensor(in1_kernel).to("cuda").requires_grad_(True)
        in2_torch = torch.tensor(in2_kernel).to("cuda").requires_grad_(True)
        weights_torch = torch.tensor(weights).to("cuda").requires_grad_(True)
        out_grad_torch = torch.tensor(out_grad_kernel).to("cuda").requires_grad_(True)
        in1_dgrad_torch = torch.tensor(in1_dgrad_kernel).to("cuda")
        in2_dgrad_torch = torch.tensor(in2_dgrad_kernel).to("cuda")
        weights_dgrad_torch = torch.tensor(weights_dgrad).to("cuda")
        out_torch = self.forward(in1_torch, in2_torch, weights_torch)

        in1_grad, in2_grad, weights_grad = torch.autograd.grad(
            outputs=out_torch,
            inputs=[in1_torch, in2_torch, weights_torch],
            grad_outputs=out_grad_torch,
            create_graph=True,
            retain_graph=True,
        )

        a, b, c, d = torch.autograd.grad(
            outputs=[in1_grad, in2_grad, weights_grad],
            inputs=[in1_torch, in2_torch, weights_torch, out_grad_torch],
            grad_outputs=[in1_dgrad_torch, in2_dgrad_torch, weights_dgrad_torch],
        )

        a_np = a.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()
        c_np = c.detach().cpu().numpy()
        d_np = d.detach().cpu().numpy()

        a_np = IrrepLayoutUtils.transpose_irrep_layout(
            a_np, self.config.irreps_in1, "mul_ir", layout
        )
        b_np = IrrepLayoutUtils.transpose_irrep_layout(
            b_np, self.config.irreps_in2, "mul_ir", layout
        )
        d_np = IrrepLayoutUtils.transpose_irrep_layout(
            d_np, self.config.irreps_out, "mul_ir", layout
        )

        return (a_np, b_np, c_np, d_np)


class NumpyDoubleBackwardMixinConv:
    """
    Similar, but for fused graph convolution.
    """

    def double_backward_cpu(
        self, in1, in2, out_grad, weights, weights_dgrad, in1_dgrad, in2_dgrad, graph
    ):
        assert self.torch_op

        layout = self.config.layout

        in1_kernel = IrrepLayoutUtils.transpose_irrep_layout(
            in1, self.config.irreps_in1, layout, "mul_ir"
        )
        in2_kernel = IrrepLayoutUtils.transpose_irrep_layout(
            in2, self.config.irreps_in2, layout, "mul_ir"
        )
        out_grad_kernel = IrrepLayoutUtils.transpose_irrep_layout(
            out_grad, self.config.irreps_out, layout, "mul_ir"
        )
        in1_dgrad_kernel = IrrepLayoutUtils.transpose_irrep_layout(
            in1_dgrad, self.config.irreps_in1, layout, "mul_ir"
        )
        in2_dgrad_kernel = IrrepLayoutUtils.transpose_irrep_layout(
            in2_dgrad, self.config.irreps_in2, layout, "mul_ir"
        )

        in1_torch = torch.tensor(in1_kernel).to("cuda").requires_grad_(True)
        in2_torch = torch.tensor(in2_kernel).to("cuda").requires_grad_(True)
        weights_torch = torch.tensor(weights).to("cuda").requires_grad_(True)
        out_grad_torch = torch.tensor(out_grad_kernel).to("cuda").requires_grad_(True)
        in1_dgrad_torch = torch.tensor(in1_dgrad_kernel).to("cuda")
        in2_dgrad_torch = torch.tensor(in2_dgrad_kernel).to("cuda")
        weights_dgrad_torch = torch.tensor(weights_dgrad).to("cuda")

        torch_rows = torch.tensor(graph.rows, device="cuda")
        torch_cols = torch.tensor(graph.cols, device="cuda")
        torch_transpose_perm = torch.tensor(graph.transpose_perm, device="cuda")

        out_torch = self.forward(
            in1_torch,
            in2_torch,
            weights_torch,
            torch_rows,
            torch_cols,
            torch_transpose_perm,
        )

        in1_grad, in2_grad, weights_grad = torch.autograd.grad(
            outputs=out_torch,
            inputs=[in1_torch, in2_torch, weights_torch],
            grad_outputs=out_grad_torch,
            create_graph=True,
            retain_graph=True,
        )

        a, b, c, d = torch.autograd.grad(
            outputs=[in1_grad, in2_grad, weights_grad],
            inputs=[in1_torch, in2_torch, weights_torch, out_grad_torch],
            grad_outputs=[in1_dgrad_torch, in2_dgrad_torch, weights_dgrad_torch],
        )

        a_np = a.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()
        c_np = c.detach().cpu().numpy()
        d_np = d.detach().cpu().numpy()

        a_np = IrrepLayoutUtils.transpose_irrep_layout(
            a_np, self.config.irreps_in1, "mul_ir", layout
        )
        b_np = IrrepLayoutUtils.transpose_irrep_layout(
            b_np, self.config.irreps_in2, "mul_ir", layout
        )
        d_np = IrrepLayoutUtils.transpose_irrep_layout(
            d_np, self.config.irreps_out, "mul_ir", layout
        )

        return (a_np, b_np, c_np, d_np)
