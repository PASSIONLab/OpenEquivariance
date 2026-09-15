"""Build schedules for receiver-streaming convolutions.

This implementation adapts the streaming concept and computational schedule
proposed by Chorošajev and Bény [CB2026]_.

.. [CB2026] Chorošajev and Bény, *Sobek: Streaming Equivariant Tensor Product
    Convolutions*, arXiv (2026).
    https://doi.org/10.48550/arXiv.2607.18074
"""

from dataclasses import dataclass
from enum import IntEnum
from functools import cache

import numpy as np

from openequivariance.core.e3nn_lite import TPProblem, wigner_3j
from openequivariance.core.utils import calc_weight_offsets, hash_str_64
from openequivariance.templates.jinja_utils import (
    cpp_scalar_type,
    get_jinja_environment,
)


class Input(IntEnum):
    """Floating operands whose derivative buffers may be inactive."""

    X = 0
    SH = 1
    W = 2


@dataclass(frozen=True, slots=True)
class FactorizedLaunchConfig:
    """Runtime-grid launch geometry shared by generated streaming kernels."""

    num_threads: int = 128
    logical_cohort_width: int = 32
    shared_memory_bytes: int = 0

    def __post_init__(self):
        """Validate launch geometry required by the generated kernels."""
        if self.num_threads <= 0:
            raise ValueError("the launch must contain at least one thread")
        if self.logical_cohort_width != 32:
            raise ValueError("the generated reductions require 32-thread cohorts")
        if self.num_threads % self.logical_cohort_width:
            raise ValueError("the thread count must contain complete logical cohorts")
        if self.shared_memory_bytes < 0:
            raise ValueError("the shared-memory allocation cannot be negative")


@dataclass(frozen=True, slots=True)
class FactorizedAccumulatorSlot:
    """One register accumulator and its packed-array address."""

    name: str
    array_index: int
    irrep_dim: int


@dataclass(frozen=True, slots=True)
class FactorizedCoupling:
    """One nonzero component coupling."""

    input_component: int
    edge_component: int
    output_component: int
    coefficient: float


@dataclass(frozen=True, slots=True)
class FactorizedCouplingPath:
    """One weighted sparse coupling path."""

    input_start: int
    edge_start: int
    output_start: int
    weight_start: int
    input_irrep_dim: int
    edge_irrep_dim: int
    output_irrep_dim: int
    edge_mul: int
    couplings: tuple[FactorizedCoupling, ...]


@dataclass(frozen=True, slots=True)
class FactorizedScheduledCoupling:
    """One component coupling with its input-gradient accumulator."""

    input_component: int
    edge_component: int
    coefficient: float
    input_accumulator: FactorizedAccumulatorSlot


@dataclass(frozen=True, slots=True)
class FactorizedPathOutput:
    """Sparse terms and accumulator for one output component."""

    output_component: int
    accumulator: FactorizedAccumulatorSlot
    terms: tuple[FactorizedScheduledCoupling, ...]


@dataclass(frozen=True, slots=True)
class FactorizedComputationPath:
    """Static offsets and sparse couplings for one weighted instruction."""

    input_start: int
    edge_start: int
    weight_start: int
    input_irrep_dim: int
    edge_irrep_dim: int
    edge_mul: int
    outputs: tuple[FactorizedPathOutput, ...]


@dataclass(frozen=True, slots=True)
class FactorizedKernel:
    """Rendered source and FFI attributes for one kernel specialization."""

    jit_kernel: str
    hash: int
    ffi_attributes: dict[str, object]


@dataclass(frozen=True, slots=True, eq=False)
class FactorizedComputationSchedule:
    """Static sparse schedule for a receiver-owned generated convolution.

    Paths retain the public flattened input, edge, output, and per-edge weight
    layouts of the source :class:`~openequivariance.core.e3nn_lite.TPProblem`.
    Output and sender-gradient slots are fully resolved before rendering, so
    the CUDA or HIP template only executes this schedule. Graph size and edge
    count remain runtime values because the forward pass follows CSR rows.
    """

    paths: tuple[FactorizedComputationPath, ...]
    output_slots: tuple[FactorizedAccumulatorSlot, ...]
    input_gradient_slots: tuple[FactorizedAccumulatorSlot, ...]
    input_dim: int
    edge_dim: int
    output_dim: int
    weight_numel: int
    channels: int
    launch_config: FactorizedLaunchConfig = FactorizedLaunchConfig()
    layout: str = "mul_ir"

    @cache
    def kernel(
        self,
        dtype: object,
        *,
        is_hip: bool,
        forward_jvp_active: tuple[bool, bool, bool] = (True, True, True),
        backward_jvp_active: tuple[bool, bool, bool, bool] = (True, True, True, True),
    ) -> FactorizedKernel:
        """Render and cache one CUDA or HIP derivative specialization."""
        if len(forward_jvp_active) != len(Input):
            raise ValueError("forward JVP activity must contain X, SH, and W")
        if len(backward_jvp_active) != len(Input) + 1:
            raise ValueError("backward JVP activity must also contain dout")
        source = (
            get_jinja_environment(is_hip=is_hip)
            .get_template("factorized_projected.cuh")
            .render(
                scalar=cpp_scalar_type(dtype),
                schedule=self,
                Input=Input,
                forward_jvp_active=forward_jvp_active,
                backward_jvp_active=backward_jvp_active,
                backward_jvp_dout_active=backward_jvp_active[len(Input)],
            )
        )
        launch = self.launch_config
        cache_source = (
            f"{source}\0{launch.num_threads}\0{launch.logical_cohort_width}"
            f"\0{launch.shared_memory_bytes}"
        )
        kernel_hash = hash_str_64(cache_source)
        attributes = {
            "source": source,
            "hash": kernel_hash,
            "input_dim": self.input_dim,
            "edge_dim": self.edge_dim,
            "weight_dim": self.weight_numel,
            "output_dim": self.output_dim,
            "channels": self.channels,
            "num_threads": launch.num_threads,
            "logical_cohort_width": launch.logical_cohort_width,
            "shared_memory_bytes": launch.shared_memory_bytes,
        }
        return FactorizedKernel(source, kernel_hash, attributes)

    def weight_reordering_info(self, weights_in, has_batch_dim: bool):
        """Describe canonical-to-native weight transposes for OEQ utilities."""
        batch_dim = weights_in.shape[0]
        specs = []
        for path in self.paths:
            start = path.weight_start
            stop = start + self.channels * path.edge_mul
            parent_shape = [self.channels, path.edge_mul]
            child_shape = list(parent_shape)
            parent_range = [slice(start, stop)]
            child_range = [slice(start, stop)]
            weights_subrange = [slice(None), slice(None)]
            transpose_perm = [1, 0]
            reshape_size = [-1]
            if has_batch_dim:
                parent_shape = [batch_dim] + parent_shape
                child_shape = [batch_dim] + child_shape
                parent_range.insert(0, slice(0, batch_dim))
                child_range.insert(0, slice(0, batch_dim))
                weights_subrange.insert(0, slice(0, batch_dim))
                transpose_perm = [0, 2, 1]
                reshape_size = [batch_dim, -1]
            specs.append(
                {
                    "parent_range": tuple(parent_range),
                    "parent_shape": parent_shape,
                    "weights_subrange": tuple(weights_subrange),
                    "child_range": tuple(child_range),
                    "child_shape": child_shape,
                    "transpose_perm": transpose_perm,
                    "reshape_size": reshape_size,
                    "transpose_child_shape": [
                        child_shape[index] for index in transpose_perm
                    ],
                }
            )
        return specs


def factorized_schedule_from_problem(
    problem: TPProblem,
) -> FactorizedComputationSchedule:
    """Lower a supported tensor-product problem to a sparse streaming schedule.

    Each instruction in a homogeneous weighted ``uvu`` problem becomes one
    execution path.
    Channel-wise UVW paths are treated as UVU.

    :param problem: External-weight, unshared ``TPProblem`` in ``"mul_ir"``
        layout with homogeneous weighted ``"uvu"`` instructions, or an exactly
        reducible scalar-multiplicity ``"uvw"`` instruction set. Instructions
        using the two connection modes cannot be mixed in one problem.
    :return: A static sparse execution schedule preserving external layouts.
    :raises ValueError: If the layout, instructions, channel structure, or
        flat weight coverage is unsupported.
    """
    if problem.layout != "mul_ir":
        raise ValueError("generated factorized convolution requires mul_ir layout")
    if problem.shared_weights or problem.internal_weights:
        raise ValueError(
            "generated factorized convolution requires external unshared weights"
        )

    declared_modes = {
        instruction.connection_mode for instruction in problem.instructions
    }
    if not declared_modes.issubset({"uvu", "uvw"}):
        raise ValueError(
            "generated convolution supports homogeneous weighted uvu or uvw paths"
        )
    if len(declared_modes) != 1:
        raise ValueError(
            "generated convolution requires all paths to use one connection mode"
        )
    declared_connection_mode = next(iter(declared_modes), None)
    reduce_scalar_uvw = declared_connection_mode == "uvw" and all(
        problem.irreps_in1[instruction.i_in1].mul == 1
        and problem.irreps_out[instruction.i_out].mul == 1
        for instruction in problem.instructions
    )
    if declared_connection_mode == "uvw" and not reduce_scalar_uvw:
        raise ValueError(
            "receiver-streaming generated convolution supports UVW only when "
            "every path has one input and one output multiplicity"
        )

    # Extract the sparse component couplings in instruction order.
    input_slices = problem.irreps_in1.slices()
    edge_slices = problem.irreps_in2.slices()
    output_slices = problem.irreps_out.slices()
    weight_offsets = calc_weight_offsets(problem)
    coupling_paths = []
    referenced_outputs = set()
    channel_count = None

    for path_index, instruction in enumerate(problem.instructions):
        if not instruction.has_weight:
            raise ValueError("generated factorized convolution requires weighted paths")

        input_mul_ir = problem.irreps_in1[instruction.i_in1]
        edge_mul_ir = problem.irreps_in2[instruction.i_in2]
        output_mul_ir = problem.irreps_out[instruction.i_out]
        referenced_outputs.add(instruction.i_out)
        if input_mul_ir.mul != output_mul_ir.mul:
            raise ValueError("uvu input and output channel multiplicities must match")
        if channel_count is None:
            channel_count = input_mul_ir.mul
        elif input_mul_ir.mul != channel_count:
            raise ValueError("generated uvu convolution requires uniform channels")

        cg = np.asarray(
            wigner_3j(input_mul_ir.ir.l, edge_mul_ir.ir.l, output_mul_ir.ir.l),
            dtype=np.float64,
        ) * float(instruction.path_weight)
        nonzero = np.nonzero(cg)
        if len(nonzero[0]) == 0:
            raise ValueError(f"instruction {path_index} has an empty coupling tensor")

        couplings = []
        for input_component, edge_component, output_component in zip(
            *nonzero, strict=True
        ):
            couplings.append(
                FactorizedCoupling(
                    input_component=int(input_component),
                    edge_component=int(edge_component),
                    output_component=int(output_component),
                    coefficient=float(
                        cg[input_component, edge_component, output_component]
                    ),
                )
            )

        coupling_paths.append(
            FactorizedCouplingPath(
                input_start=input_slices[instruction.i_in1].start,
                edge_start=edge_slices[instruction.i_in2].start,
                output_start=output_slices[instruction.i_out].start,
                weight_start=weight_offsets[path_index],
                input_irrep_dim=input_mul_ir.ir.dim,
                edge_irrep_dim=edge_mul_ir.ir.dim,
                output_irrep_dim=output_mul_ir.ir.dim,
                edge_mul=edge_mul_ir.mul,
                couplings=tuple(couplings),
            )
        )

    if channel_count is None:
        raise ValueError("generated factorized convolution requires at least one path")
    missing_outputs = [
        output
        for output, mul_ir in enumerate(problem.irreps_out)
        if mul_ir.mul * mul_ir.ir.dim > 0 and output not in referenced_outputs
    ]
    if missing_outputs:
        raise ValueError(
            "factorized schedule does not reference positive-dimensional output "
            f"irrep blocks {missing_outputs}"
        )
    covered_weight_count = sum(
        np.prod(instruction.path_shape)
        for instruction in problem.instructions
        if instruction.has_weight
    )
    if covered_weight_count != problem.weight_numel:
        raise ValueError(
            "factorized schedule does not cover the complete weight vector"
        )

    # Group couplings by output component and assign their accumulator slots.
    output_slots = []
    input_gradient_slots = []
    output_slot_indices = {}
    input_gradient_slot_indices = {}

    def intern_slot(slots, slot_indices, prefix, start, component, irrep_dim):
        key = (start, component, irrep_dim)
        existing = slot_indices.get(key)
        if existing is not None:
            return slots[existing]
        slot_indices[key] = len(slots)
        slot = FactorizedAccumulatorSlot(
            name=f"{prefix}_{start}_{component}",
            array_index=start + component,
            irrep_dim=irrep_dim,
        )
        slots.append(slot)
        return slot

    paths = []
    for coupling_path in coupling_paths:
        outputs = []
        for output_component in range(coupling_path.output_irrep_dim):
            scheduled_terms = []
            for coupling in coupling_path.couplings:
                if coupling.output_component != output_component:
                    continue
                scheduled_terms.append(
                    FactorizedScheduledCoupling(
                        input_component=coupling.input_component,
                        edge_component=coupling.edge_component,
                        coefficient=coupling.coefficient,
                        input_accumulator=intern_slot(
                            input_gradient_slots,
                            input_gradient_slot_indices,
                            "input_gradient",
                            coupling_path.input_start,
                            coupling.input_component,
                            coupling_path.input_irrep_dim,
                        ),
                    )
                )

            outputs.append(
                FactorizedPathOutput(
                    output_component=output_component,
                    accumulator=intern_slot(
                        output_slots,
                        output_slot_indices,
                        "output",
                        coupling_path.output_start,
                        output_component,
                        coupling_path.output_irrep_dim,
                    ),
                    terms=tuple(scheduled_terms),
                )
            )

        paths.append(
            FactorizedComputationPath(
                input_start=coupling_path.input_start,
                edge_start=coupling_path.edge_start,
                weight_start=coupling_path.weight_start,
                input_irrep_dim=coupling_path.input_irrep_dim,
                edge_irrep_dim=coupling_path.edge_irrep_dim,
                edge_mul=coupling_path.edge_mul,
                outputs=tuple(outputs),
            )
        )

    return FactorizedComputationSchedule(
        paths=tuple(paths),
        output_slots=tuple(output_slots),
        input_gradient_slots=tuple(input_gradient_slots),
        input_dim=problem.irreps_in1.dim,
        edge_dim=problem.irreps_in2.dim,
        output_dim=problem.irreps_out.dim,
        weight_numel=problem.weight_numel,
        channels=channel_count,
    )


__all__ = [
    "FactorizedAccumulatorSlot",
    "FactorizedCoupling",
    "FactorizedCouplingPath",
    "FactorizedComputationPath",
    "FactorizedComputationSchedule",
    "FactorizedKernel",
    "FactorizedLaunchConfig",
    "FactorizedPathOutput",
    "FactorizedScheduledCoupling",
    "Input",
    "factorized_schedule_from_problem",
]
