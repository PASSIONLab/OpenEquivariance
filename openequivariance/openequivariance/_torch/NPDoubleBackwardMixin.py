import torch


def _none_to_zeros(values, refs):
    return tuple(
        ref * 0 if value is None else value for value, ref in zip(values, refs)
    )


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

        in1_torch = torch.tensor(in1, device="cuda", requires_grad=True)
        in2_torch = torch.tensor(in2, device="cuda", requires_grad=True)
        weights_torch = torch.tensor(weights, device="cuda", requires_grad=True)
        out_grad_torch = torch.tensor(out_grad, device="cuda", requires_grad=True)
        in1_dgrad_torch = torch.tensor(in1_dgrad, device="cuda", requires_grad=False)
        in2_dgrad_torch = torch.tensor(in2_dgrad, device="cuda", requires_grad=False)
        weights_dgrad_torch = torch.tensor(
            weights_dgrad, device="cuda", requires_grad=False
        )
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

        return (
            a.detach().cpu().numpy(),
            b.detach().cpu().numpy(),
            c.detach().cpu().numpy(),
            d.detach().cpu().numpy(),
        )

    def triple_backward_cpu(
        self,
        in1,
        in2,
        out_grad,
        weights,
        weights_dgrad,
        in1_dgrad,
        in2_dgrad,
        out_tgrad,
        weights_tgrad,
        in1_tgrad,
        in2_tgrad,
    ):
        assert self.torch_op

        in1_torch = torch.tensor(in1, device="cuda", requires_grad=True)
        in2_torch = torch.tensor(in2, device="cuda", requires_grad=True)
        weights_torch = torch.tensor(weights, device="cuda", requires_grad=True)
        out_grad_torch = torch.tensor(out_grad, device="cuda", requires_grad=True)
        in1_dgrad_torch = torch.tensor(in1_dgrad, device="cuda", requires_grad=True)
        in2_dgrad_torch = torch.tensor(in2_dgrad, device="cuda", requires_grad=True)
        weights_dgrad_torch = torch.tensor(
            weights_dgrad, device="cuda", requires_grad=True
        )
        out_tgrad_torch = torch.tensor(out_tgrad, device="cuda", requires_grad=False)
        in1_tgrad_torch = torch.tensor(in1_tgrad, device="cuda", requires_grad=False)
        in2_tgrad_torch = torch.tensor(in2_tgrad, device="cuda", requires_grad=False)
        weights_tgrad_torch = torch.tensor(
            weights_tgrad, device="cuda", requires_grad=False
        )

        out_torch = self.forward(in1_torch, in2_torch, weights_torch)
        in1_grad, in2_grad, weights_grad = torch.autograd.grad(
            outputs=out_torch,
            inputs=[in1_torch, in2_torch, weights_torch],
            grad_outputs=out_grad_torch,
            create_graph=True,
            retain_graph=True,
        )
        double_grads = torch.autograd.grad(
            outputs=[in1_grad, in2_grad, weights_grad],
            inputs=[in1_torch, in2_torch, weights_torch, out_grad_torch],
            grad_outputs=[in1_dgrad_torch, in2_dgrad_torch, weights_dgrad_torch],
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        double_grads = _none_to_zeros(
            double_grads, (in1_torch, in2_torch, weights_torch, out_grad_torch)
        )
        triple_grads = torch.autograd.grad(
            outputs=double_grads,
            inputs=[
                in1_torch,
                in2_torch,
                weights_torch,
                out_grad_torch,
                in1_dgrad_torch,
                in2_dgrad_torch,
                weights_dgrad_torch,
            ],
            grad_outputs=[
                in1_tgrad_torch,
                in2_tgrad_torch,
                weights_tgrad_torch,
                out_tgrad_torch,
            ],
            allow_unused=True,
        )
        triple_grads = _none_to_zeros(
            triple_grads,
            (
                in1_torch,
                in2_torch,
                weights_torch,
                out_grad_torch,
                in1_dgrad_torch,
                in2_dgrad_torch,
                weights_dgrad_torch,
            ),
        )

        return tuple(grad.detach().cpu().numpy() for grad in triple_grads)


class NumpyDoubleBackwardMixinConv:
    """
    Similar, but for fused graph convolution.
    """

    def double_backward_cpu(
        self, in1, in2, out_grad, weights, weights_dgrad, in1_dgrad, in2_dgrad, graph
    ):
        assert self.torch_op

        in1_torch = torch.tensor(in1, device="cuda", requires_grad=True)
        in2_torch = torch.tensor(in2, device="cuda", requires_grad=True)
        weights_torch = torch.tensor(weights, device="cuda", requires_grad=True)
        out_grad_torch = torch.tensor(out_grad, device="cuda", requires_grad=True)
        in1_dgrad_torch = torch.tensor(in1_dgrad, device="cuda", requires_grad=False)
        in2_dgrad_torch = torch.tensor(in2_dgrad, device="cuda", requires_grad=False)
        weights_dgrad_torch = torch.tensor(
            weights_dgrad, device="cuda", requires_grad=False
        )

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

        return (
            a.detach().cpu().numpy(),
            b.detach().cpu().numpy(),
            c.detach().cpu().numpy(),
            d.detach().cpu().numpy(),
        )

    def triple_backward_cpu(
        self,
        in1,
        in2,
        out_grad,
        weights,
        weights_dgrad,
        in1_dgrad,
        in2_dgrad,
        out_tgrad,
        weights_tgrad,
        in1_tgrad,
        in2_tgrad,
        graph,
    ):
        assert self.torch_op

        in1_torch = torch.tensor(in1, device="cuda", requires_grad=True)
        in2_torch = torch.tensor(in2, device="cuda", requires_grad=True)
        weights_torch = torch.tensor(weights, device="cuda", requires_grad=True)
        out_grad_torch = torch.tensor(out_grad, device="cuda", requires_grad=True)
        in1_dgrad_torch = torch.tensor(in1_dgrad, device="cuda", requires_grad=True)
        in2_dgrad_torch = torch.tensor(in2_dgrad, device="cuda", requires_grad=True)
        weights_dgrad_torch = torch.tensor(
            weights_dgrad, device="cuda", requires_grad=True
        )
        out_tgrad_torch = torch.tensor(out_tgrad, device="cuda", requires_grad=False)
        in1_tgrad_torch = torch.tensor(in1_tgrad, device="cuda", requires_grad=False)
        in2_tgrad_torch = torch.tensor(in2_tgrad, device="cuda", requires_grad=False)
        weights_tgrad_torch = torch.tensor(
            weights_tgrad, device="cuda", requires_grad=False
        )

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
        double_grads = torch.autograd.grad(
            outputs=[in1_grad, in2_grad, weights_grad],
            inputs=[in1_torch, in2_torch, weights_torch, out_grad_torch],
            grad_outputs=[in1_dgrad_torch, in2_dgrad_torch, weights_dgrad_torch],
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        double_grads = _none_to_zeros(
            double_grads, (in1_torch, in2_torch, weights_torch, out_grad_torch)
        )
        triple_grads = torch.autograd.grad(
            outputs=double_grads,
            inputs=[
                in1_torch,
                in2_torch,
                weights_torch,
                out_grad_torch,
                in1_dgrad_torch,
                in2_dgrad_torch,
                weights_dgrad_torch,
            ],
            grad_outputs=[
                in1_tgrad_torch,
                in2_tgrad_torch,
                weights_tgrad_torch,
                out_tgrad_torch,
            ],
            allow_unused=True,
        )
        triple_grads = _none_to_zeros(
            triple_grads,
            (
                in1_torch,
                in2_torch,
                weights_torch,
                out_grad_torch,
                in1_dgrad_torch,
                in2_dgrad_torch,
                weights_dgrad_torch,
            ),
        )

        return tuple(grad.detach().cpu().numpy() for grad in triple_grads)
