"""
OpenEquivariance must consume weights in e3nn's canonical order.

e3nn's order is defined by a single sentence: each instruction owns a contiguous
block of the flat weight vector that reshapes to ``instruction.path_shape``, and
the blocks are concatenated in instruction order. That block contracts through a
plain einsum -- ``uv,ijk,bui,bvj->buk`` for uvu, ``uvw,ijk,bui,bvj->bwk`` for uvw.

OEQ stores weights in a different order today, produced by two mechanisms in
``ComputationSchedule``:

1. The kernel's inner loop walks ``v`` serially and ``u`` across lanes, so each
   instruction's tile is stored ``[v][u]`` rather than e3nn's ``[u][v]``. This is
   invisible whenever ``mul_v == 1``.
2. ``ProblemSplitter`` tiles every multiplicity into blocks of ``warp_size`` (32,
   capped at 32 for uvw) and concatenates the tiles, so a 64x64 weight matrix is
   stored as four 32x32 blocks instead of one row-major matrix.

Both are compile-time-constant address maps, so both can be folded into the
generated kernel. These tests assert the end state: the same flat weight vector
goes into ``o3.TensorProduct`` and into OEQ and produces the same answer, and
``reorder_weights_from_e3nn`` / ``reorder_weights_to_e3nn`` are the identity.

e3nn itself is the reference throughout, via the repo's existing
``E3NNTensorProduct`` adapter -- the same one ``benchmark/correctness.py`` uses.
"""

from itertools import product

import numpy as np
import pytest
import torch
from pytest_check import check

import openequivariance as oeq
from openequivariance._torch.E3NNTensorProduct import E3NNTensorProduct
from openequivariance.benchmark.problems import mace_problems, nequip_problems


@pytest.fixture(params=[np.float32, np.float64], ids=["F32", "F64"], scope="module")
def dtype(request):
    return request.param


# Tolerances for torch.testing.assert_close. Comparisons are against e3nn running
# in the same dtype, so the gap is summation order, which widens as derivatives
# stack up.
TOLERANCES = {
    np.float32: {
        "fwd": {"rtol": 1e-4, "atol": 1e-5},
        "bwd": {"rtol": 1e-3, "atol": 1e-4},
        "double_bwd": {"rtol": 2e-3, "atol": 1e-3},
    },
    np.float64: {
        "fwd": {"rtol": 1e-9, "atol": 1e-11},
        "bwd": {"rtol": 1e-8, "atol": 1e-10},
        "double_bwd": {"rtol": 1e-7, "atol": 1e-9},
    },
}


def torch_dtype_of(problem):
    return torch.float32 if problem.irrep_dtype == np.float32 else torch.float64


def random_inputs(problem, batch, seed, requires_grad=False):
    """
    e3nn-ordered weights, straight from the RNG. Nothing here permutes anything.
    """
    rng = np.random.default_rng(seed)
    td = torch_dtype_of(problem)

    def rand(*shape):
        arr = np.asarray(rng.uniform(size=shape), dtype=problem.irrep_dtype)
        return torch.tensor(arr, dtype=td, device="cuda", requires_grad=requires_grad)

    in1 = rand(batch, problem.irreps_in1.dim)
    in2 = rand(batch, problem.irreps_in2.dim)
    weight_shape = (
        (problem.weight_numel,)
        if problem.shared_weights
        else (batch, problem.weight_numel)
    )
    weights = rand(*weight_shape)
    return in1, in2, weights


def check_all_close(named_triples, tol):
    """
    Soft-assert each (name, actual, expected) so one run reports every mismatched
    tensor rather than stopping at the first.
    """
    for name, actual, expected in named_triples:
        with check:
            torch.testing.assert_close(
                actual, expected, msg=lambda m, name=name: f"{name}: {m}", **tol
            )


class WeightOrderCorrectness:
    """
    Every subclass supplies a ``problem`` fixture; the tests below are shared.
    """

    batch_size = 512

    def tol(self, problem, direction):
        return TOLERANCES[problem.irrep_dtype][direction]

    @pytest.fixture(scope="class")
    def tp(self, problem, with_jax):
        if with_jax:
            import openequivariance.jax.TensorProduct as jax_tp

            return jax_tp(problem)
        return oeq.TensorProduct(problem)

    @pytest.fixture(scope="class")
    def reference_tp(self, problem):
        """
        e3nn's ``o3.TensorProduct``, built by the repo's own adapter. Callable and
        differentiable, so autograd supplies the reference gradients.
        """
        return E3NNTensorProduct(problem)

    # -- The invariant, stated directly --------------------------------------

    def test_reorder_from_e3nn_is_identity(self, tp, problem):
        """
        With e3nn as the native order there is nothing to permute, so the public
        reordering hooks must be the identity. They stay in the API for
        compatibility, but they may no longer move data.
        """
        _, _, weights = random_inputs(problem, self.batch_size, seed=12345)
        weights_np = weights.detach().cpu().numpy()

        reordered = tp.reorder_weights_from_e3nn(
            weights_np, has_batch_dim=not problem.shared_weights
        )

        np.testing.assert_array_equal(np.asarray(reordered), weights_np)

    def test_reorder_to_e3nn_is_identity(self, tp, problem):
        _, _, weights = random_inputs(problem, self.batch_size, seed=12345)
        weights_np = weights.detach().cpu().numpy()

        reordered = tp.reorder_weights_to_e3nn(
            weights_np, has_batch_dim=not problem.shared_weights
        )

        np.testing.assert_array_equal(np.asarray(reordered), weights_np)

    # -- The drop-in claim: same weight vector, same answer ------------------

    def test_forward_matches_e3nn(self, tp, reference_tp, problem, with_jax):
        in1, in2, weights = random_inputs(problem, self.batch_size, seed=12345)
        expected = reference_tp(in1, in2, weights)

        if with_jax:
            import jax.numpy as jnp

            raw = tp.forward(
                *[jnp.asarray(t.cpu().numpy()) for t in (in1, in2, weights)]
            )
            actual = torch.as_tensor(np.asarray(raw), device="cuda")
        else:
            actual = tp(in1, in2, weights)

        torch.testing.assert_close(actual, expected, **self.tol(problem, "fwd"))

    def test_backward_matches_e3nn(self, tp, reference_tp, problem, with_jax):
        if with_jax:
            pytest.skip("Covered by test_jax_vjp_matches_e3nn")

        in1, in2, weights = random_inputs(
            problem, self.batch_size, seed=12345, requires_grad=True
        )
        inputs = [in1, in2, weights]

        expected_out = reference_tp(in1, in2, weights)
        out_grad = torch.rand_like(expected_out)

        ref_grads = torch.autograd.grad(expected_out, inputs, grad_outputs=out_grad)
        oeq_grads = torch.autograd.grad(
            tp(in1, in2, weights), inputs, grad_outputs=out_grad
        )

        check_all_close(
            zip(["in1_grad", "in2_grad", "weight_grad"], oeq_grads, ref_grads),
            self.tol(problem, "bwd"),
        )

    def test_double_backward_matches_e3nn(self, tp, reference_tp, problem, with_jax):
        """
        The backward kernel writes weight gradients back out, so it encodes the
        weight order in the reverse direction too. Differentiating through it with
        a random cotangent puts both directions in one graph.
        """
        if with_jax:
            pytest.skip("N/A for JAX")

        in1, in2, weights = random_inputs(problem, 64, seed=23456, requires_grad=True)
        inputs = [in1, in2, weights]

        out_grad = torch.rand_like(reference_tp(in1, in2, weights))

        def first_order(forward_fn):
            return torch.autograd.grad(
                forward_fn(in1, in2, weights),
                inputs,
                grad_outputs=out_grad,
                create_graph=True,
            )

        ref_grads = first_order(reference_tp)
        # A random cotangent on the weight-gradient slot is what makes this
        # sensitive to the order the backward kernel writes weight grads in.
        cotangents = [torch.rand_like(g) for g in ref_grads]

        ref = torch.autograd.grad(ref_grads, inputs, grad_outputs=cotangents)
        actual = torch.autograd.grad(first_order(tp), inputs, grad_outputs=cotangents)

        check_all_close(
            zip(
                ["in1_double_grad", "in2_double_grad", "weight_double_grad"],
                actual,
                ref,
            ),
            self.tol(problem, "double_bwd"),
        )

    def test_jax_vjp_matches_e3nn(self, tp, reference_tp, problem, with_jax):
        if not with_jax:
            pytest.skip("N/A for PyTorch")

        import jax
        import jax.numpy as jnp

        in1, in2, weights = random_inputs(
            problem, self.batch_size, seed=12345, requires_grad=True
        )
        jax_inputs = [
            jnp.asarray(t.detach().cpu().numpy()) for t in (in1, in2, weights)
        ]

        out, vjp_fn = jax.vjp(tp.forward, *jax_inputs)
        jax_grads = vjp_fn(jnp.ones_like(out))

        ref_out = reference_tp(in1, in2, weights)
        ref_grads = torch.autograd.grad(
            ref_out, [in1, in2, weights], grad_outputs=torch.ones_like(ref_out)
        )

        check_all_close(
            [
                (name, torch.as_tensor(np.asarray(actual), device="cuda"), expected)
                for name, actual, expected in zip(
                    ["in1_grad", "in2_grad", "weight_grad"], jax_grads, ref_grads
                )
            ],
            self.tol(problem, "bwd"),
        )


def single_instruction_problem(mode, m, ir, dtype):
    return oeq.TPProblem(
        f"{m[0]}x{ir[0]}e",
        f"{m[1]}x{ir[1]}e",
        f"{m[2]}x{ir[2]}e",
        [(0, 0, 0, mode, True)],
        shared_weights=False,
        internal_weights=False,
        irrep_dtype=dtype,
        weight_dtype=dtype,
        label=f"{mode}_{m[0]}x{m[1]}x{m[2]}",
    )


def mul_id(param):
    m, ir = param
    return f"{m[0]}x{ir[0]}e__x__{m[1]}x{ir[1]}e---{m[2]}x{ir[2]}e"


class TestUVU(WeightOrderCorrectness):
    """
    ``mul_v == 1`` rows already agree with e3nn at every multiplicity, so they
    are regression guards. Rows with ``mul_v > 1`` are transposed today.
    """

    muls = [
        (1, 1, 1),
        (32, 1, 32),
        (64, 1, 64),
        (128, 1, 128),
        (4, 3, 4),
        (16, 4, 16),
        (32, 2, 32),
        (64, 2, 64),
        (24, 24, 24),
        (33, 3, 33),
    ]
    # Weight order is a function of the multiplicities and connection mode only,
    # so a few representative irrep combinations are enough here.
    irs = [(0, 0, 0), (1, 2, 1), (5, 3, 5)]

    @pytest.fixture(params=list(product(muls, irs)), ids=mul_id, scope="class")
    def problem(self, request, dtype):
        m, ir = request.param
        return single_instruction_problem("uvu", m, ir, dtype)


class TestUVW(WeightOrderCorrectness):
    """
    ``mul <= 32`` with ``mul_v == 1`` already agrees with e3nn. Anything past the
    32-wide tiling threshold, or with ``mul_v > 1``, is reordered today.
    """

    muls = [
        (1, 1, 1),
        (16, 1, 16),
        (32, 1, 32),
        (33, 1, 33),
        (50, 1, 50),
        (64, 1, 64),
        (4, 3, 2),
        (8, 8, 8),
        (24, 24, 24),
    ]
    irs = [(0, 0, 0), (1, 2, 1), (5, 3, 5)]

    @pytest.fixture(params=list(product(muls, irs)), ids=mul_id, scope="class")
    def problem(self, request, dtype):
        m, ir = request.param
        return single_instruction_problem("uvw", m, ir, dtype)


class TestMultiInstruction(WeightOrderCorrectness):
    """
    Several instructions spread across shared-memory segments, so the per-segment
    weight offsets have to line up with e3nn's instruction-order concatenation.
    """

    problems = [
        (
            "mace_like_uvu",
            "64x0e+64x1o+64x2e",
            "1x0e+1x1o",
            "64x0e+64x1o+64x2e",
            [
                (0, 0, 0, "uvu", True),
                (0, 1, 1, "uvu", True),
                (1, 0, 1, "uvu", True),
                (1, 1, 2, "uvu", True),
                (2, 0, 2, "uvu", True),
            ],
        ),
        (
            "multi_v_uvu",
            "48x0e+48x1o",
            "4x0e",
            "48x0e+48x1o",
            [(0, 0, 0, "uvu", True), (1, 0, 1, "uvu", True)],
        ),
        (
            "multi_uvw",
            "40x0e+40x1o",
            "2x0e",
            "40x0e+40x1o",
            [(0, 0, 0, "uvw", True), (1, 0, 1, "uvw", True)],
        ),
    ]

    @pytest.fixture(params=problems, ids=lambda p: p[0], scope="class")
    def problem(self, request, dtype):
        label, irreps_in1, irreps_in2, irreps_out, instructions = request.param
        return oeq.TPProblem(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions,
            shared_weights=False,
            internal_weights=False,
            irrep_dtype=dtype,
            weight_dtype=dtype,
            label=label,
        )


class TestSharedWeights(WeightOrderCorrectness):
    """
    Shared weights take a different store-back path in the backward kernel:
    atomics into one buffer rather than per-batch writes.
    """

    problems = [
        ("shared_uvu_multi_v", "64x1o", "4x0e", "64x1o", "uvu"),
        ("shared_uvw_large", "50x1o", "1x0e", "50x1o", "uvw"),
    ]

    @pytest.fixture(params=problems, ids=lambda p: p[0], scope="class")
    def problem(self, request, dtype):
        label, irreps_in1, irreps_in2, irreps_out, mode = request.param
        return oeq.TPProblem(
            irreps_in1,
            irreps_in2,
            irreps_out,
            [(0, 0, 0, mode, True)],
            shared_weights=True,
            internal_weights=False,
            irrep_dtype=dtype,
            weight_dtype=dtype,
            label=label,
        )


class TestProductionModels(WeightOrderCorrectness):
    """
    Real configurations, all uvu with ``mul_v == 1``. These already match e3nn
    today; they are here so the fix does not regress the common path.
    """

    problems = mace_problems() + nequip_problems()

    @pytest.fixture(params=problems, ids=lambda p: p.label, scope="class")
    def problem(self, request, dtype):
        problem = request.param.clone()
        problem.irrep_dtype, problem.weight_dtype = dtype, dtype
        return problem


class TestConvWeightOrder:
    """
    The fused convolution kernels stage weights through their own templates
    (``loop_unroll_conv_atomic.cuh`` / ``loop_unroll_conv_det.cuh``), so they
    need coverage separate from the batched tensor product.

    e3nn has no fused convolution, so the reference is e3nn's tensor product on
    gathered inputs followed by a scatter-add -- the definition of the fused op.
    """

    node_count = 128
    edge_count = 512

    problems = [
        ("conv_uvu_multi_v", "32x1o", "4x0e", "32x1o", "uvu"),
        ("conv_uvw_large", "50x1o", "1x0e", "50x1o", "uvw"),
        ("conv_uvu_channelwise", "64x0e+64x1o", "1x0e", "64x0e+64x1o", "uvu"),
    ]

    @pytest.fixture(params=problems, ids=lambda p: p[0], scope="class")
    def problem(self, request, dtype):
        label, irreps_in1, irreps_in2, irreps_out, mode = request.param
        instructions = [(0, 0, 0, mode, True)]
        if "+" in irreps_in1:
            instructions.append((1, 0, 1, mode, True))
        return oeq.TPProblem(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions,
            shared_weights=False,
            internal_weights=False,
            irrep_dtype=dtype,
            weight_dtype=dtype,
            label=label,
        )

    @pytest.fixture(scope="class")
    def edge_index(self):
        from torch_geometric import EdgeIndex

        rng = np.random.default_rng(12345)
        rows = np.sort(rng.integers(0, self.node_count, size=self.edge_count))
        cols = rng.integers(0, self.node_count, size=self.edge_count)

        ei = EdgeIndex(
            data=np.stack([rows, cols]).tolist(),
            sort_order="row",
            sparse_size=(self.node_count, self.node_count),
            device="cuda",
            dtype=torch.long,
        )
        ei.fill_cache_()
        return ei

    @pytest.fixture(
        params=[False, True], ids=["atomic", "deterministic"], scope="class"
    )
    def conv_and_determinism(self, request, problem, with_jax):
        if with_jax:
            from openequivariance.jax import TensorProductConv as cls
        else:
            cls = oeq.TensorProductConv
        return cls(problem, deterministic=request.param), request.param

    @pytest.fixture(scope="class")
    def reference_tp(self, problem):
        return E3NNTensorProduct(problem)

    def test_conv_forward_matches_e3nn(
        self, conv_and_determinism, reference_tp, problem, edge_index, with_jax
    ):
        if with_jax:
            pytest.skip("N/A for JAX")

        conv, deterministic = conv_and_determinism
        rows, cols = edge_index[0], edge_index[1]

        rng = np.random.default_rng(12345)
        td = torch_dtype_of(problem)

        def rand(*shape):
            arr = np.asarray(rng.uniform(size=shape), dtype=problem.irrep_dtype)
            return torch.tensor(arr, dtype=td, device="cuda")

        in1 = rand(self.node_count, problem.irreps_in1.dim)
        in2 = rand(self.edge_count, problem.irreps_in2.dim)
        weights = rand(self.edge_count, problem.weight_numel)

        messages = reference_tp(in1[cols], in2, weights)
        expected = torch.zeros(
            self.node_count, problem.irreps_out.dim, device="cuda", dtype=td
        )
        expected.index_add_(0, rows, messages)

        sender_perm = edge_index.get_csc()[1] if deterministic else None
        actual = conv(in1, in2, weights, rows, cols, sender_perm)

        torch.testing.assert_close(
            actual, expected, **TOLERANCES[problem.irrep_dtype]["fwd"]
        )

    def test_conv_reorder_is_identity(self, conv_and_determinism, problem):
        conv, _ = conv_and_determinism
        rng = np.random.default_rng(12345)
        weights = np.asarray(
            rng.uniform(size=(self.edge_count, problem.weight_numel)),
            dtype=problem.weight_dtype,
        )

        np.testing.assert_array_equal(
            np.asarray(conv.reorder_weights_from_e3nn(weights, has_batch_dim=True)),
            weights,
        )
        np.testing.assert_array_equal(
            np.asarray(conv.reorder_weights_to_e3nn(weights, has_batch_dim=True)),
            weights,
        )
