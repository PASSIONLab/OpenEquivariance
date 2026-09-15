"""Numerical coverage for public receiver-streaming convolution selection."""

import numpy as np
import pytest


def test_streaming_mode_requires_an_eligible_problem():
    """Reject irreducible UVW and aggregation options in streaming mode."""
    from openequivariance.core.e3nn_lite import TPProblem
    from openequivariance.jax import (
        StreamingUnavailableError,
        TensorProductConv,
        streaming_support,
    )

    irreducible = TPProblem(
        "2x0e",
        "1x0e",
        "2x0e",
        [(0, 0, 0, "uvw", True)],
        shared_weights=False,
        internal_weights=False,
    )
    supported, reason = streaming_support(irreducible)
    assert not supported
    assert "UVW" in reason
    with pytest.raises(StreamingUnavailableError, match="mode='streaming'"):
        TensorProductConv(irreducible, mode="streaming")

    eligible = TPProblem(
        "1x0e",
        "2x0e",
        "1x0e",
        [(0, 0, 0, "uvw", True)],
        shared_weights=False,
        internal_weights=False,
    )
    with pytest.raises(StreamingUnavailableError, match="deterministic"):
        TensorProductConv(eligible, deterministic=True, mode="streaming")

    mixed_dtype = TPProblem(
        "1x0e",
        "1x0e",
        "1x0e",
        [(0, 0, 0, "uvu", True)],
        shared_weights=False,
        internal_weights=False,
        irrep_dtype=np.float32,
        weight_dtype=np.float64,
    )
    supported, reason = streaming_support(mixed_dtype)
    assert not supported
    assert "matching irrep and weight dtypes" in reason
    with pytest.raises(StreamingUnavailableError, match="matching irrep"):
        TensorProductConv(mixed_dtype, mode="streaming")


def test_standard_sorted_indices_keep_endpoints_and_sender_permutation():
    """Keep valid sorted endpoints and the deterministic sender permutation."""
    import jax.numpy as jnp

    from openequivariance.jax import (
        LoopUnrollTensorProductConv,
        TensorProductConv,
    )
    from openequivariance.jax.StreamingTensorProductConv import (
        StreamingTensorProductConv,
    )

    class FakeLoopUnroll(LoopUnrollTensorProductConv):
        """Provide loop-unroll aggregation without requiring CUDA in this unit test."""

        def __init__(self):
            """Avoid constructing a CUDA kernel for this selector unit test."""
            self.last_sender_perm = None

        def forward(self, X, Y, W, rows, cols, sender_perm=None, **kwargs):
            del kwargs
            self.last_sender_perm = sender_perm
            values = X[cols, :1] * Y[:, :1] * W[:, :1]
            return jnp.zeros((X.shape[0], 1), dtype=X.dtype).at[rows].add(values)

    selector = object.__new__(TensorProductConv)
    loop = FakeLoopUnroll()
    selector._impl = loop
    selector.deterministic = False
    selector.kahan = False
    X = jnp.asarray([[2.0], [3.0], [5.0]])
    Y = jnp.asarray([[7.0], [11.0], [13.0]])
    W = jnp.asarray([[17.0], [19.0], [23.0]])
    rows = jnp.asarray([1, 2, 0], dtype=jnp.int32)
    cols = jnp.asarray([1, 0, 2], dtype=jnp.int32)
    topology = StreamingTensorProductConv._prepare_topology(rows, cols, X.shape[0])

    ordinary = jnp.asarray(
        [[5.0 * 13.0 * 23.0], [3.0 * 7.0 * 17.0], [2.0 * 11.0 * 19.0]]
    )
    sorted_output = selector.forward(
        X,
        Y[topology.order],
        W[topology.order],
        topology.receivers,
        topology.senders,
        indices_are_sorted=True,
    )
    np.testing.assert_allclose(sorted_output, ordinary)
    np.testing.assert_allclose(selector.forward(X, Y, W, rows, cols), ordinary)

    selector.deterministic = True
    np.testing.assert_allclose(
        selector.forward(
            X,
            Y[topology.order],
            W[topology.order],
            topology.receivers,
            topology.senders,
            sender_perm=jnp.arange(Y.shape[0], dtype=jnp.int32),
            indices_are_sorted=True,
        ),
        ordinary,
    )
    np.testing.assert_array_equal(
        loop.last_sender_perm, jnp.arange(Y.shape[0], dtype=jnp.int32)
    )

    np.testing.assert_allclose(
        selector.forward(
            X,
            Y[topology.order],
            W[topology.order],
            topology.receivers,
            topology.senders,
            indices_are_sorted=True,
            row_ptr=topology.row_ptr,
        ),
        ordinary,
    )


def test_streaming_row_ptr_requires_expected_shape_and_dtype():
    """Reject cached receiver offsets with incompatible metadata."""
    import jax.numpy as jnp

    from openequivariance.jax.StreamingTensorProductConv import (
        StreamingTensorProductConv,
    )

    rows = jnp.asarray([0, 0, 2], dtype=jnp.int32)
    cols = jnp.asarray([1, 2, 0], dtype=jnp.int32)
    correct = jnp.asarray([0, 2, 2, 3], dtype=jnp.int32)
    StreamingTensorProductConv._validate_row_ptr(correct, rows, cols, 3)
    with pytest.raises(ValueError, match=r"shape \[N \+ 1\]"):
        StreamingTensorProductConv._validate_row_ptr(correct[:-1], rows, cols, 3)
    with pytest.raises(ValueError, match="int32"):
        StreamingTensorProductConv._validate_row_ptr(
            correct.astype(jnp.float32), rows, cols, 3
        )


@pytest.fixture(scope="module")
def gpu_context(with_jax):
    """Provide JAX only when the explicitly requested GPU backend is ready."""
    if not with_jax:
        pytest.skip("requires --jax")
    import jax
    import jax.numpy as jnp

    if jax.default_backend() != "gpu":
        pytest.skip("requires a JAX GPU backend")
    return jax, jnp


def _scalar_case(gpu_context, mode):
    """Create a V=2 scalar-output case and its standard OEQ reference."""
    jax, jnp = gpu_context
    from openequivariance.core.e3nn_lite import TPProblem
    from openequivariance.jax.TensorProductConv import TensorProductConv

    problem = TPProblem(
        "1x1e",
        "2x1e",
        "1x1e",
        [(0, 0, 0, mode, True)],
        shared_weights=False,
        internal_weights=False,
    )
    generated = TensorProductConv(problem, mode="auto")
    assert generated.uses_streaming_kernel
    reference = TensorProductConv(problem, requires_jvp=True, mode="standard")
    keys = jax.random.split(jax.random.key(728), 7)
    dtype = problem.irrep_dtype
    x = jax.random.normal(keys[0], (5, problem.irreps_in1.dim), dtype=dtype)
    y = jax.random.normal(keys[1], (8, problem.irreps_in2.dim), dtype=dtype)
    w = jax.random.normal(keys[2], (8, problem.weight_numel), dtype=dtype)
    tangents = tuple(
        jax.random.normal(key, operand.shape, dtype=dtype)
        for key, operand in zip(keys[3:6], (x, y, w), strict=True)
    )
    return (
        generated,
        reference,
        (x, y, w),
        tangents,
        jax.random.normal(keys[6], (5, problem.irreps_out.dim), dtype=dtype),
        jnp.array([3, 0, 4, 1, 0, 2, 3, 1], jnp.int32),
        jnp.array([0, 2, 1, 3, 4, 0, 2, 4], jnp.int32),
    )


def _assert_tree_close(actual, expected, *, atol=2e-3, rtol=2e-3):
    """Compare matching JAX array tuples using float32-safe tolerances."""
    actual = actual if isinstance(actual, tuple) else (actual,)
    expected = expected if isinstance(expected, tuple) else (expected,)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(got, want, atol=atol, rtol=rtol)


@pytest.mark.parametrize("mode", ("uvu", "uvw"))
def test_streaming_uvu_and_scalar_uvw_match_standard_through_hvp(gpu_context, mode):
    """Match standard values and AD for UVU and exact [1,V,1] UVW lowering."""
    jax, jnp = gpu_context
    generated, standard, values, tangents, dout, rows, cols = _scalar_case(
        gpu_context, mode
    )

    def native(*args):
        return generated(*args, rows, cols)

    def standard_operator(*args):
        return standard(*args, rows, cols)

    _assert_tree_close(jax.jit(native)(*values), jax.jit(standard_operator)(*values))
    _assert_tree_close(
        jax.jvp(native, values, tangents)[1],
        jax.jvp(standard_operator, values, tangents)[1],
    )

    def energy(operator, *args):
        return jnp.vdot(operator(*args), dout)

    native_grad = jax.grad(lambda *args: energy(native, *args), (0, 1, 2))
    standard_grad = jax.grad(lambda *args: energy(standard_operator, *args), (0, 1, 2))
    _assert_tree_close(native_grad(*values), standard_grad(*values))
    _assert_tree_close(
        jax.jvp(native_grad, values, tangents)[1],
        jax.jvp(standard_grad, values, tangents)[1],
        atol=4e-3,
        rtol=4e-3,
    )


@pytest.mark.parametrize("active_input", range(3))
def test_streaming_single_input_derivatives_match_standard(gpu_context, active_input):
    """Match JVP and backward-JVP results with one active primal input."""
    jax, jnp = gpu_context
    generated, standard, values, tangents, dout, rows, cols = _scalar_case(
        gpu_context, "uvu"
    )

    def operator(implementation, arguments):
        return implementation(*arguments, rows, cols)

    def varying_forward(implementation, value):
        arguments = list(values)
        arguments[active_input] = value
        return operator(implementation, arguments)

    _assert_tree_close(
        jax.jvp(
            lambda value: varying_forward(generated, value),
            (values[active_input],),
            (tangents[active_input],),
        )[1],
        jax.jvp(
            lambda value: varying_forward(standard, value),
            (values[active_input],),
            (tangents[active_input],),
        )[1],
    )

    def gradients(implementation, arguments):
        return jax.grad(
            lambda *operands: jnp.vdot(operator(implementation, operands), dout),
            (0, 1, 2),
        )(*arguments)

    def varying_gradients(implementation, value):
        arguments = list(values)
        arguments[active_input] = value
        return gradients(implementation, arguments)

    _assert_tree_close(
        jax.jvp(
            lambda value: varying_gradients(generated, value),
            (values[active_input],),
            (tangents[active_input],),
        )[1],
        jax.jvp(
            lambda value: varying_gradients(standard, value),
            (values[active_input],),
            (tangents[active_input],),
        )[1],
        atol=4e-3,
        rtol=4e-3,
    )


def test_irreducible_uvw_uses_standard_path_and_ir_mul_layout(monkeypatch):
    """Delegate irreducible UVW through the established native layout."""
    import importlib
    import jax.numpy as jnp

    from openequivariance.core.e3nn_lite import TPProblem

    module = importlib.import_module("openequivariance.jax.TensorProductConv")

    class FakeLoopUnroll:
        """Record the native configuration without constructing a GPU kernel."""

        def __init__(self, config, **kwargs):
            del kwargs
            self.config = config

        def forward(self, X, Y, W, rows, cols, sender_perm, **kwargs):
            del Y, W, rows, cols, sender_perm, kwargs
            return X

        def forward_cpu(self, X, Y, W, output, graph):
            del Y, W, graph
            output[...] = X

    monkeypatch.setattr(module, "LoopUnrollTensorProductConv", FakeLoopUnroll)

    problem = TPProblem(
        "2x1e",
        "1x0e",
        "2x1e",
        [(0, 0, 0, "uvw", True)],
        shared_weights=False,
        internal_weights=False,
        layout="ir_mul",
    )
    operator = module.TensorProductConv(problem, mode="auto")
    assert operator.implementation == "standard"
    assert operator._impl.config.layout == "mul_ir"
    x = jnp.arange(6, dtype=jnp.float32).reshape(1, 6)
    np.testing.assert_array_equal(
        operator(
            x,
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, problem.weight_numel), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.zeros((1,), dtype=jnp.int32),
        ),
        x,
    )
    output = np.empty_like(np.asarray(x))
    operator.forward_cpu(
        np.asarray(x),
        np.ones((1, 1), dtype=np.float32),
        np.ones((1, problem.weight_numel), dtype=np.float32),
        output,
        object(),
    )
    np.testing.assert_array_equal(output, x)


def test_native_weight_permutation_round_trips_multiple_paths():
    """Round-trip canonical weights through nontrivial native UVU ordering."""
    import jax.numpy as jnp

    from openequivariance.core.e3nn_lite import TPProblem
    from openequivariance.jax.TensorProductConv import TensorProductConv

    problem = TPProblem(
        "2x0e + 2x1e",
        "2x0e + 2x1e",
        "2x0e + 2x1e",
        [(0, 0, 0, "uvu", True), (1, 1, 1, "uvu", True)],
        shared_weights=False,
        internal_weights=False,
    )
    operator = TensorProductConv(problem, mode="streaming")
    canonical = jnp.arange(3 * problem.weight_numel, dtype=jnp.float32).reshape(
        3, problem.weight_numel
    )
    np.testing.assert_array_equal(
        operator.reorder_weights_to_e3nn(operator.reorder_weights_from_e3nn(canonical)),
        canonical,
    )


def test_multipath_streaming_matches_standard_through_hvp(gpu_context):
    """Match forward, full VJP, and HVP results for multiple UVU paths."""
    jax, jnp = gpu_context
    from openequivariance.core.e3nn_lite import TPProblem
    from openequivariance.jax.TensorProductConv import TensorProductConv

    problem = TPProblem(
        "2x0e + 2x1e",
        "1x0e + 1x1e",
        "2x0e + 2x1e + 2x2e",
        [
            (0, 0, 0, "uvu", True),
            (0, 1, 1, "uvu", True),
            (1, 0, 1, "uvu", True),
            (1, 1, 0, "uvu", True),
            (1, 1, 2, "uvu", True),
        ],
        shared_weights=False,
        internal_weights=False,
    )
    streaming = TensorProductConv(problem, mode="streaming")
    standard = TensorProductConv(problem, mode="standard")
    keys = jax.random.split(jax.random.key(912), 8)
    rows = jnp.asarray([2, 0, 3, 1, 0, 2, 3, 1], dtype=jnp.int32)
    cols = jnp.asarray([0, 2, 1, 3, 1, 3, 2, 0], dtype=jnp.int32)
    x = jax.random.normal(keys[0], (4, problem.irreps_in1.dim), dtype=np.float32)
    sh = jax.random.normal(keys[1], (8, problem.irreps_in2.dim), dtype=np.float32)
    weights = jax.random.normal(keys[2], (8, problem.weight_numel), dtype=np.float32)
    tangent_weights = jax.random.normal(
        keys[5], (8, problem.weight_numel), dtype=np.float32
    )
    streaming_values = (x, sh, streaming.reorder_weights_from_e3nn(weights))
    standard_values = (x, sh, standard.reorder_weights_from_e3nn(weights))
    streaming_tangents = (
        jax.random.normal(keys[3], x.shape, dtype=np.float32),
        jax.random.normal(keys[4], sh.shape, dtype=np.float32),
        streaming.reorder_weights_from_e3nn(tangent_weights),
    )
    standard_tangents = (
        streaming_tangents[0],
        streaming_tangents[1],
        standard.reorder_weights_from_e3nn(tangent_weights),
    )
    dout = jax.random.normal(keys[6], (4, problem.irreps_out.dim), dtype=np.float32)

    def operator(implementation, *operands):
        return implementation(*operands, rows, cols)

    _assert_tree_close(
        operator(streaming, *streaming_values),
        operator(standard, *standard_values),
    )

    def gradient(implementation, operands):
        return jax.grad(
            lambda *values: jnp.vdot(operator(implementation, *values), dout),
            (0, 1, 2),
        )(*operands)

    streaming_gradient = gradient(streaming, streaming_values)
    standard_gradient = gradient(standard, standard_values)
    _assert_tree_close(
        (
            *streaming_gradient[:2],
            streaming.reorder_weights_to_e3nn(streaming_gradient[2]),
        ),
        (
            *standard_gradient[:2],
            standard.reorder_weights_to_e3nn(standard_gradient[2]),
        ),
    )
    streaming_hvp = jax.jvp(
        lambda *values: gradient(streaming, values),
        streaming_values,
        streaming_tangents,
    )[1]
    standard_hvp = jax.jvp(
        lambda *values: gradient(standard, values),
        standard_values,
        standard_tangents,
    )[1]
    _assert_tree_close(
        (*streaming_hvp[:2], streaming.reorder_weights_to_e3nn(streaming_hvp[2])),
        (*standard_hvp[:2], standard.reorder_weights_to_e3nn(standard_hvp[2])),
        atol=4e-3,
        rtol=4e-3,
    )


def test_shared_weights_use_the_loop_unroll_fallback(monkeypatch):
    """Keep the established shared-weight interface on the standard path."""
    import importlib
    import jax.numpy as jnp

    from openequivariance.core.e3nn_lite import TPProblem

    module = importlib.import_module("openequivariance.jax.TensorProductConv")

    class FakeLoopUnroll:
        """Record the fallback call without constructing a GPU kernel."""

        def __init__(self, config, **kwargs):
            del kwargs
            self.config = config
            self.weights = None

        def forward(self, X, Y, W, rows, cols, sender_perm, **kwargs):
            del X, Y, rows, cols, sender_perm, kwargs
            self.weights = W
            return W

    monkeypatch.setattr(module, "LoopUnrollTensorProductConv", FakeLoopUnroll)
    problem = TPProblem(
        "1x0e",
        "1x0e",
        "1x0e",
        [(0, 0, 0, "uvu", True)],
        shared_weights=True,
        internal_weights=False,
    )
    operator = module.TensorProductConv(problem)
    shared_weights = jnp.ones((problem.weight_numel,), dtype=jnp.float32)
    result = operator(
        jnp.ones((2, 1), dtype=jnp.float32),
        jnp.ones((3, 1), dtype=jnp.float32),
        shared_weights,
        jnp.array([0, 1, 0], dtype=jnp.int32),
        jnp.array([1, 0, 1], dtype=jnp.int32),
    )
    assert operator.implementation == "standard"
    assert operator._impl.config.shared_weights
    np.testing.assert_array_equal(result, shared_weights)
