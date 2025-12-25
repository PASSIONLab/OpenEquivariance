import torch

class NumpyDoubleBackwardMixin:
    '''
    Adds a Numpy double backward method to any TensorProduct 
    with the forward pass defined in PyTorch and the relevant
    derivatives registered. 
    '''
    def double_backward_cpu(self, in1, in2, out_grad, weights, weights_dgrad, in1_dgrad, in2_dgrad):
        assert self.torch_op

        in1_torch = torch.tensor(in1).to('cuda').requires_grad_(True)
        in2_torch = torch.tensor(in2).to('cuda').requires_grad_(True)
        weights_torch = torch.tensor(weights).to('cuda').requires_grad_(True)
        out_grad_torch = torch.tensor(out_grad).to('cuda').requires_grad_(True)
        in1_dgrad_torch = torch.tensor(in1_dgrad).to('cuda')
        in2_dgrad_torch = torch.tensor(in2_dgrad).to('cuda')
        weights_dgrad_torch = torch.tensor(weights_dgrad).to('cuda')
        out_torch = self.forward(in1_torch, in2_torch, weights_torch)

        in1_grad, in2_grad, weights_grad = torch.autograd.grad(
            outputs=out_torch,
            inputs=[in1_torch, in2_torch, weights_torch],
            grad_outputs=out_grad_torch,
            create_graph=True,
            retain_graph=True
        )

        a, b, c, d = torch.autograd.grad(
            outputs=[in1_grad, in2_grad, weights_grad],
            inputs=[in1_torch, in2_torch, weights_torch, out_grad_torch],
            grad_outputs=[in1_dgrad_torch, in2_dgrad_torch, weights_dgrad_torch]
        )

        return a.detach().cpu().numpy(), b.detach().cpu().numpy(), c.detach().cpu().numpy(), d.detach().cpu().numpy()


class NumpyDoubleBackwardMixinConv:
    '''
    Similar, but for fused graph convolution.
    '''
    def double_backward_cpu(self, in1, in2, out_grad, weights, weights_dgrad, in1_dgrad, in2_dgrad, graph):
        assert self.torch_op

        in1_torch = torch.tensor(in1).to('cuda').requires_grad_(True)
        in2_torch = torch.tensor(in2).to('cuda').requires_grad_(True)
        weights_torch = torch.tensor(weights).to('cuda').requires_grad_(True)
        out_grad_torch = torch.tensor(out_grad).to('cuda').requires_grad_(True)
        in1_dgrad_torch = torch.tensor(in1_dgrad).to('cuda')
        in2_dgrad_torch = torch.tensor(in2_dgrad).to('cuda')
        weights_dgrad_torch = torch.tensor(weights_dgrad).to('cuda')

        torch_rows = torch.tensor(graph.rows, device="cuda")
        torch_cols = torch.tensor(graph.cols, device="cuda")
        torch_transpose_perm = torch.tensor(graph.transpose_perm, device="cuda")

        out_torch = self.forward(in1_torch, in2_torch, weights_torch, torch_rows, torch_cols, torch_transpose_perm)

        in1_grad, in2_grad, weights_grad = torch.autograd.grad(
            outputs=out_torch,
            inputs=[in1_torch, in2_torch, weights_torch],
            grad_outputs=out_grad_torch,
            create_graph=True,
            retain_graph=True
        )

        a, b, c, d = torch.autograd.grad(
            outputs=[in1_grad, in2_grad, weights_grad],
            inputs=[in1_torch, in2_torch, weights_torch, out_grad_torch],
            grad_outputs=[in1_dgrad_torch, in2_dgrad_torch, weights_dgrad_torch]
        )

        return a.detach().cpu().numpy(), b.detach().cpu().numpy(), c.detach().cpu().numpy(), d.detach().cpu().numpy()


