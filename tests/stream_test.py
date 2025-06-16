# ruff: noqa: E731, F401
import json

from typing import NamedTuple
import logging

import pytest
from pytest_check import check
from pytest_lazy_fixtures import lf

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from tests.conftest import Executable
from tests.executable_utils import (
    simple_oeq_tp_fwd_executable,
    simple_oeq_tp_bwd_executable,
    simple_oeq_tp_double_bwd_executable,
    #
    simple_oeq_conv_atomic_fwd_executable,
    simple_oeq_conv_atomic_bwd_executable,
    simple_oeq_conv_atomic_double_bwd_executable,
    #
    simple_oeq_conv_det_fwd_executable,
    simple_oeq_conv_det_bwd_executable,
    simple_oeq_conv_det_double_bwd_executable,
)


class KernelExpectation(NamedTuple):
    kernel_name: str
    expected_appearances: int


KE = KernelExpectation

cuda = torch.device("cuda")


@pytest.fixture
def oeq_tp_fwd_kernel_expectations():
    return [KE("forward", 1)]


@pytest.fixture
def oeq_tp_bwd_kernel_expectations():
    return [KE("forward", 1), KE("backward", 1)]


@pytest.fixture
def oeq_tp_double_bwd_kernel_expectations():
    return [
        KE("forward", 1),
        KE("backward", 1),
        KE("double_backward_A", 1),
        KE("double_backward_B", 1),
    ]


@pytest.fixture
def oeq_conv_atomic_fwd_kernel_expectations():
    return [KE("forward", 1)]


@pytest.fixture
def oeq_conv_atomic_bwd_kernel_expectations():
    return [
        KE("forward", 1),
        KE("backward", 1),
    ]


@pytest.fixture
def oeq_conv_atomic_double_bwd_expectations():
    return [
        KE("forward", 1),
        KE("backward", 1),
        KE("double_backward_A", 1),
        KE("double_backward_B", 1),
    ]


@pytest.fixture
def oeq_conv_det_fwd_kernel_expectations():
    return [KE("forward", 1), KE("fixup_forward", 1)]


@pytest.fixture()
def oeq_conv_det_bwd_kernel_expectations():
    return [
        KE("forward", 1),
        KE("fixup_forward", 1),
        KE("backward", 1),
        KE("fixup_backward", 1),
    ]


@pytest.fixture()
def oeq_conv_det_double_bwd_kernel_expectations():
    return [
        KE("forward", 1),
        KE("fixup_forward", 2),
        KE("backward", 1),
        KE("fixup_backward", 1),
        KE("double_backward_A", 1),
        KE("double_backward_B", 1),
        KE("fixup_double_backwardB", 1),
    ]


@pytest.fixture(
    params=[
        (lf("simple_oeq_tp_fwd_executable"), lf("oeq_tp_fwd_kernel_expectations")),
        (lf("simple_oeq_tp_bwd_executable"), lf("oeq_tp_bwd_kernel_expectations")),
        (
            lf("simple_oeq_tp_double_bwd_executable"),
            lf("oeq_tp_double_bwd_kernel_expectations"),
        ),
        (
            lf("simple_oeq_conv_atomic_fwd_executable"),
            lf("oeq_conv_atomic_fwd_kernel_expectations"),
        ),
        (
            lf("simple_oeq_conv_atomic_bwd_executable"),
            lf("oeq_conv_atomic_bwd_kernel_expectations"),
        ),
        (
            lf("simple_oeq_conv_atomic_double_bwd_executable"),
            lf("oeq_conv_atomic_double_bwd_expectations"),
        ),
        (
            lf("simple_oeq_conv_det_fwd_executable"),
            lf("oeq_conv_det_fwd_kernel_expectations"),
        ),
        (
            lf("simple_oeq_conv_det_bwd_executable"),
            lf("oeq_conv_det_bwd_kernel_expectations"),
        ),
        (
            lf("simple_oeq_conv_det_double_bwd_executable"),
            lf("oeq_conv_det_double_bwd_kernel_expectations"),
        ),
    ]
)
def executable_and_expectations(request):
    return request.param


def test_separate_streams(
    request,
    tmp_path,
    executable_and_expectations: tuple[Executable, list[KernelExpectation]],
):
    executable, expectations = executable_and_expectations
    COUNT = 5
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        streams = [1, 2]
        for priority in streams:
            s = torch.cuda.Stream(device=cuda, priority=priority)
            with torch.cuda.stream(s):
                with record_function(f"executable_{priority}"):
                    for _ in range(COUNT):
                        executable()

    prof.export_chrome_trace(str(tmp_path / "trace"))

    trace = None
    with open(tmp_path / "trace", "r") as f:
        trace = json.load(f)

    gpu_annotations = []
    for event in trace["traceEvents"]:
        if "gpu_user_annotation" == event.get("cat") and "executable_" in event.get(
            "name", ""
        ):
            gpu_annotations.append(event)

    names = [x["name"] for x in gpu_annotations]
    tids = [x["tid"] for x in gpu_annotations]

    logger = logging.getLogger()
    logger.debug(msg=names)
    logger.debug(msg=tids)

    with check:
        assert len(names) == len(streams)
        assert len(tids) == len(streams)
        assert len(set(tids)) == len(set(names)), "The CUDA streams are not unique"

    for tid in set(tids):
        for kernel_expectation in expectations:
            criteria = (
                lambda event: (event.get("cat") == "kernel")
                and (event.get("name", "").startswith(kernel_expectation.kernel_name))
                and (event.get("tid") == tid)
            )
            matching = list(filter(criteria, trace["traceEvents"]))
            num_matching = len(matching)
            with check:
                assert (
                    num_matching == COUNT * kernel_expectation.expected_appearances
                ), f"{tid}_{kernel_expectation.kernel_name}"
