"""Tests for static receiver-streaming computation schedules."""

from dataclasses import replace

import numpy as np
import pytest

from openequivariance.core.FactorizedComputationSchedule import (
    FactorizedLaunchConfig,
    Input,
    factorized_schedule_from_problem,
)
from openequivariance.core.e3nn_lite import Irreps, TPProblem, wigner_3j


def _problem(mode="uvu", edge_mul=1, channels=4):
    return TPProblem(
        Irreps(f"{channels}x1e"),
        Irreps(f"{edge_mul}x1e"),
        Irreps(f"{channels}x1e"),
        [(0, 0, 0, mode, True)],
        shared_weights=False,
        internal_weights=False,
        irrep_dtype=np.float64,
        weight_dtype=np.float64,
    )


def test_factorized_schedule_preserves_layout_couplings_and_geometry():
    """Lowering resolves the same normalized CG entries and launch geometry."""
    problem = _problem()
    schedule = factorized_schedule_from_problem(problem)
    path = schedule.paths[0]
    assert (schedule.input_dim, schedule.edge_dim, schedule.output_dim) == (
        problem.irreps_in1.dim,
        problem.irreps_in2.dim,
        problem.irreps_out.dim,
    )
    assert schedule.weight_numel == problem.weight_numel
    assert schedule.launch_config.num_threads == 128
    assert schedule.launch_config.logical_cohort_width == 32
    assert schedule.launch_config.shared_memory_bytes == 0
    assert (Input.X.value, Input.SH.value, Input.W.value) == (0, 1, 2)

    output_irrep_dim = problem.irreps_out[0].ir.dim
    actual = np.zeros((path.input_irrep_dim, path.edge_irrep_dim, output_irrep_dim))
    for output in path.outputs:
        for term in output.terms:
            actual[
                term.input_component, term.edge_component, output.output_component
            ] = term.coefficient
    np.testing.assert_array_equal(
        actual, wigner_3j(1, 1, 1) * problem.instructions[0].path_weight
    )
    assert len(schedule.output_slots) == output_irrep_dim
    assert len(schedule.input_gradient_slots) == path.input_irrep_dim
    assert sum(len(output.terms) for output in path.outputs) == np.count_nonzero(actual)


def test_factorized_schedule_caches_complete_ffi_specializations():
    """Create source and FFI metadata together once per specialization."""
    schedule = factorized_schedule_from_problem(_problem())
    kernel = schedule.kernel(np.float32, is_hip=False)
    assert kernel is schedule.kernel(np.float32, is_hip=False)
    assert kernel.ffi_attributes["source"] == kernel.jit_kernel
    assert kernel.ffi_attributes["hash"] == kernel.hash
    assert kernel.ffi_attributes["input_dim"] == schedule.input_dim
    assert kernel.ffi_attributes["edge_dim"] == schedule.edge_dim
    assert kernel.ffi_attributes["weight_dim"] == schedule.weight_numel
    assert kernel.ffi_attributes["output_dim"] == schedule.output_dim
    assert kernel.ffi_attributes["num_threads"] == schedule.launch_config.num_threads
    assert (
        kernel.ffi_attributes["logical_cohort_width"]
        == schedule.launch_config.logical_cohort_width
    )
    assert (
        kernel.ffi_attributes["shared_memory_bytes"]
        == schedule.launch_config.shared_memory_bytes
    )

    tuned_schedule = replace(
        schedule, launch_config=FactorizedLaunchConfig(num_threads=256)
    )
    tuned_kernel = tuned_schedule.kernel(np.float32, is_hip=False)
    assert tuned_kernel.jit_kernel == kernel.jit_kernel
    assert tuned_kernel.hash != kernel.hash


def test_factorized_schedule_reduces_only_exact_scalar_uvw():
    """Only UVW paths with one input and output multiplicity reduce to UVU."""
    reducible = TPProblem(
        Irreps("1x1e"),
        Irreps("2x0e"),
        Irreps("1x1e"),
        [(0, 0, 0, "uvw", True)],
        shared_weights=False,
        internal_weights=False,
    )
    assert (
        factorized_schedule_from_problem(reducible).weight_numel
        == reducible.weight_numel
    )
    with pytest.raises(ValueError, match="one input and one output"):
        factorized_schedule_from_problem(_problem(mode="uvw"))


def test_factorized_schedule_rejects_unreferenced_output_block():
    """Reject an output irrep block with no producing instruction."""
    problem = TPProblem(
        Irreps("4x1e"),
        Irreps("1x0e"),
        Irreps("4x1e + 4x0e"),
        [(0, 0, 0, "uvu", True)],
        shared_weights=False,
        internal_weights=False,
        irrep_dtype=np.float64,
        weight_dtype=np.float64,
    )
    with pytest.raises(ValueError, match="output irrep blocks \\[1\\]"):
        factorized_schedule_from_problem(problem)
