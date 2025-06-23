import torch


import pytest
from pytest_check import check

from tests.conftest import Executable


def test_torch_dim_check(simple_executable):
    func, buffers_tuple = simple_executable.func, simple_executable.buffers

    for i in range(len(buffers_tuple)):
        buffers = [buf for buf in buffers_tuple]
        buffers[i] = torch.unsqueeze(buffers[i], 0)
        with check:
            with pytest.raises(RuntimeError) as exception_info:
                func(*buffers)

        with check:
            assert "Shape mismatch" in str(exception_info.value), (
                f"{buffers_tuple._fields[i]} did not trigger the expected error"
            )


def test_torch_size_check(simple_executable):
    assert isinstance(simple_executable, Executable)
    func, buffers_tuple = simple_executable.func, simple_executable.buffers

    assert len(buffers_tuple)
    for i in range(len(buffers_tuple)):
        assert buffers_tuple[i].dim()
        for j in range(buffers_tuple[i].dim()):
            sizes = [size for size in buffers_tuple[i].shape]
            sizes[j] += 1
            buffers = [buf for buf in buffers_tuple]
            buffers[i] = buffers[i].resize_(sizes)

            # with pytest.raises(RuntimeError) as exception_info:
            func(*buffers)

            # with check:
            #     assert "Shape mismatch" in str(exception_info.value), f"{buffers_tuple._fields[i]} did not trigger the expected error"
