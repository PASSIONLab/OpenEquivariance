import pytest
import torch

from tests.conftest import Executable
from openequivariance import TensorProduct, TensorProductConv


@pytest.fixture
def simple_oeq_tp_fwd_executable(tpp, tp_buffers):
    tp_oeq = TensorProduct(tpp)
    return Executable(tp_oeq, tp_buffers)


@pytest.fixture
def simple_oeq_tp_bwd_executable(tpp, tp_buffers):
    tp_oeq = TensorProduct(tpp)

    # Set up backward-executing callable
    def backward_fn(X, Y, W):
        X.requires_grad_(True)
        Y.requires_grad_(True)
        W.requires_grad_(True)
        output = tp_oeq(X, Y, W).sum()
        output.backward()
        return output

    return Executable(backward_fn, tp_buffers)


@pytest.fixture
def simple_oeq_tp_double_bwd_executable(tpp, tp_buffers):
    tp_oeq = TensorProduct(tpp)

    def double_backward_fn(X, Y, W):
        # Forward pass
        X.requires_grad_(True)
        Y.requires_grad_(True)
        W.requires_grad_(True)

        # First forward
        out = tp_oeq(X, Y, W)
        out_grad = out.clone().detach().requires_grad_(True)

        # First backward (compute gradients w.r.t inputs)
        in1_grad, in2_grad, w_grad = torch.autograd.grad(
            outputs=out,
            inputs=(X, Y, W),
            grad_outputs=out_grad,
            create_graph=True,
        )

        # Dummy loss to propagate second backward
        dummy = torch.norm(in1_grad) + torch.norm(in2_grad) + torch.norm(w_grad)

        # Second backward
        dummy_grad = torch.tensor(1.0, device="cuda")
        dummy.backward(
            dummy_grad,
            retain_graph=True,
            inputs=(out_grad, X, Y, W),
        )

        return dummy

    return Executable(double_backward_fn, tp_buffers)


@pytest.fixture
def simple_oeq_conv_atomic_fwd_executable(tpp, conv_buffers):
    tp_conv = TensorProductConv(tpp, torch_op=True, deterministic=False)

    return Executable(tp_conv, conv_buffers)


@pytest.fixture
def simple_oeq_conv_atomic_bwd_executable(tpp, conv_buffers):
    tp_conv = TensorProductConv(tpp, torch_op=True, deterministic=False)

    # Set up backward-executing callable
    def backward_fn(X, Y, W, receivers, senders):
        X.requires_grad_(True)
        Y.requires_grad_(True)
        W.requires_grad_(True)
        output = tp_conv(
            X, Y, W, receivers, senders
        ).sum()  # Scalar output for backward
        output.backward()
        return output

    return Executable(backward_fn, conv_buffers)


@pytest.fixture
def simple_oeq_conv_atomic_double_bwd_executable(tpp, conv_buffers):
    tp_conv = TensorProductConv(tpp, torch_op=True, deterministic=False)

    def double_backward_fn(X, Y, W, receivers, senders):
        # First forward pass
        X.requires_grad_(True)
        Y.requires_grad_(True)
        W.requires_grad_(True)

        out = tp_conv(X, Y, W, receivers, senders)
        out_grad = out.clone().detach().requires_grad_(True)

        # First backward (gradients w.r.t inputs)
        in1_grad, in2_grad, w_grad = torch.autograd.grad(
            outputs=out,
            inputs=(X, Y, W),
            grad_outputs=out_grad,
            create_graph=True,
        )

        # Dummy loss for second backward
        dummy = torch.norm(in1_grad) + torch.norm(in2_grad) + torch.norm(w_grad)

        # Second backward
        dummy_grad = torch.tensor(1.0, device="cuda")
        dummy.backward(
            dummy_grad,
            retain_graph=True,
            inputs=(out_grad, X, Y, W),
        )

        return dummy

    return Executable(double_backward_fn, conv_buffers)


@pytest.fixture
def simple_oeq_conv_det_fwd_executable(tpp, conv_buffers):
    tp_conv = TensorProductConv(tpp, torch_op=True, deterministic=False)

    return Executable(tp_conv, conv_buffers)


@pytest.fixture
def simple_oeq_conv_det_bwd_executable(tpp, conv_buffers):
    tp_conv = TensorProductConv(tpp, torch_op=True, deterministic=False)

    # Set up backward-executing callable
    def backward_fn(X, Y, W, receivers, senders):
        X.requires_grad_(True)
        Y.requires_grad_(True)
        W.requires_grad_(True)
        output = tp_conv(
            X, Y, W, receivers, senders
        ).sum()  # Scalar output for backward
        output.backward()
        return output

    return Executable(backward_fn, conv_buffers)


@pytest.fixture
def simple_oeq_conv_det_double_bwd_executable(tpp, conv_buffers):
    tp_conv = TensorProductConv(tpp, torch_op=True, deterministic=False)

    def double_backward_fn(X, Y, W, receivers, senders):
        # First forward pass
        X.requires_grad_(True)
        Y.requires_grad_(True)
        W.requires_grad_(True)

        out = tp_conv(X, Y, W, receivers, senders)
        out_grad = out.clone().detach().requires_grad_(True)

        # First backward (gradients w.r.t inputs)
        in1_grad, in2_grad, w_grad = torch.autograd.grad(
            outputs=out,
            inputs=(X, Y, W),
            grad_outputs=out_grad,
            create_graph=True,
        )

        # Dummy loss for second backward
        dummy = torch.norm(in1_grad) + torch.norm(in2_grad) + torch.norm(w_grad)

        # Second backward
        dummy_grad = torch.tensor(1.0, device="cuda")
        dummy.backward(
            dummy_grad,
            retain_graph=True,
            inputs=(out_grad, X, Y, W),
        )

        return dummy

    return Executable(double_backward_fn, conv_buffers)
