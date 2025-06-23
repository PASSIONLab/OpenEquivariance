import torch

import pytest
from pytest_lazy_fixtures import lf


from openequivariance import TPProblem, TensorProduct, TensorProductConv
from openequivariance.benchmark.TestBenchmarkSuite import Direction

from tests.conftest import Executable, BufferName, sig_tp, tp_shapes


def tp_buffers(
    num_batch: int, tpp: TPProblem, direction: Direction, cuda, gen
) -> dict[BufferName, torch.Tensor]:
    signature = sig_tp(direction)
    shapes = tp_shapes(num_batch, tpp)

    buffers = {}

    for input in signature.inputs:
        buffers[input] = torch.rand(shapes[input], device=cuda, generator=gen)

    for output in signature.outputs:
        buffers[output] = torch.zeros(shapes[output], device=cuda)

    return buffers


@pytest.fixture
def simple_oeq_tp_fwd_executable(tpp):
    tp_oeq = TensorProduct(tpp)

    def func(x, y, w, **kwargs):
        return torch.ops.libtorch_tp_jit.jit_tp_backward(tp_oeq.internal, x, y, w)

    return Executable(func, tp_buffers)


@pytest.fixture
def simple_oeq_tp_bwd_executable(tpp, tp_buffers):
    tp_oeq = TensorProduct(tpp)

    def func(x, y, w, z_grad, **kwargs):
        return torch.ops.libtorch_tp_jit.jit_tp_backward(
            tp_oeq.internal,
            x,
            y,
            w,
            z_grad,
        )

    return Executable(func, tp_buffers)


@pytest.fixture
def simple_oeq_tp_double_bwd_executable(tpp, tp_buffers):
    tp_oeq = TensorProduct(tpp)

    def func(x, y, w, z_grad, x_dgrad, y_dgrad, w_dgrad, **kwargs):
        return torch.ops.libtorch_tp_jit.jit_tp_double_backward(
            tp_oeq.internal,
            x,
            y,
            w,
            z_grad,
            x_dgrad,
            y_dgrad,
            w_dgrad,
        )

    return Executable(func, tp_buffers)


@pytest.fixture
def simple_oeq_conv_atomic_fwd_executable(tpp, conv_buffers):
    tp_conv = TensorProductConv(tpp, torch_op=True, deterministic=False)

    def func(x, y, w, rows, cols, transpose_perm, **kwargs):
        return torch.ops.libtorch_tp_jit.jit_conv_forward(
            tp_conv.internal,
            x,
            y,
            w,
            rows,
            cols,
            tp_conv.workspace_buffer,
            transpose_perm,
        )

    return Executable(func, conv_buffers)


@pytest.fixture
def simple_oeq_conv_atomic_bwd_executable(tpp, conv_buffers):
    tp_conv = TensorProductConv(tpp, torch_op=True, deterministic=False)

    def func(x, y, w, z_grad, rows, cols, transpose_perm, **kwargs):
        return torch.ops.libtorch_tp_jit.jit_conv_backward(
            tp_conv.internal,
            x,
            y,
            w,
            z_grad,
            rows,
            cols,
            tp_conv.workspace_buffer,
            transpose_perm,
        )

    return Executable(func, conv_buffers)


@pytest.fixture
def simple_oeq_conv_atomic_double_bwd_executable(tpp, conv_buffers):
    tp_conv = TensorProductConv(tpp, torch_op=True, deterministic=False)

    def func(
        x, y, w, z_grad, x_dgrad, y_dgrad, w_dgrad, rows, cols, transpose_perm, **kwargs
    ):
        return torch.ops.libtorch_tp_jit.jit_conv_double_backward(
            tp_conv.internal,
            x,
            y,
            w,
            z_grad,
            x_dgrad,
            y_dgrad,
            w_dgrad,
            rows,
            cols,
            tp_conv.workspace_buffer,
            transpose_perm,
        )

    return Executable(func, conv_buffers)


@pytest.fixture
def simple_oeq_conv_det_fwd_executable(tpp, conv_buffers):
    tp_conv = TensorProductConv(tpp, torch_op=True, deterministic=False)

    def func(x, y, w, rows, cols, transpose_perm, **kwargs):
        return torch.ops.libtorch_tp_jit.jit_conv_forward(
            tp_conv.internal,
            x,
            y,
            w,
            rows,
            cols,
            tp_conv.workspace_buffer,
            transpose_perm,
        )

    return Executable(func, conv_buffers)


@pytest.fixture
def simple_oeq_conv_det_bwd_executable(tpp, conv_buffers):
    tp_conv = TensorProductConv(tpp, torch_op=True, deterministic=False)

    def func(x, y, w, z_grad, rows, cols, transpose_perm, **kwargs):
        return torch.ops.libtorch_tp_jit.jit_conv_backward(
            tp_conv.internal,
            x,
            y,
            w,
            z_grad,
            rows,
            cols,
            tp_conv.workspace_buffer,
            transpose_perm,
        )

    return Executable(func, conv_buffers)


@pytest.fixture
def simple_oeq_conv_det_double_bwd_executable(tpp, conv_buffers):
    tp_conv = TensorProductConv(tpp, torch_op=True, deterministic=False)

    def func(
        x, y, w, z_grad, x_dgrad, y_dgrad, w_dgrad, rows, cols, transpose_perm, **kwargs
    ):
        return torch.ops.libtorch_tp_jit.jit_conv_double_backward(
            tp_conv.internal,
            x,
            y,
            w,
            z_grad,
            x_dgrad,
            y_dgrad,
            w_dgrad,
            rows,
            cols,
            tp_conv.workspace_buffer,
            transpose_perm,
        )

    return Executable(func, conv_buffers)


@pytest.fixture(
    params=[
        lf("simple_oeq_tp_fwd_executable"),
        lf("simple_oeq_tp_bwd_executable"),
        lf("simple_oeq_tp_double_bwd_executable"),
        lf("simple_oeq_conv_atomic_fwd_executable"),
        lf("simple_oeq_conv_atomic_bwd_executable"),
        lf("simple_oeq_conv_atomic_double_bwd_executable"),
        lf("simple_oeq_conv_det_fwd_executable"),
        lf("simple_oeq_conv_det_bwd_executable"),
        lf("simple_oeq_conv_det_double_bwd_executable"),
    ]
)
def simple_executable(request):
    return request.param
