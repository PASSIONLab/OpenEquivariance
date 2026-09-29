import pytest
import os


@pytest.fixture
def ctx(with_jax):
    if not with_jax:
        pytest.skip("Skipping JAX tests")
    os.environ["OEQ_NOTORCH"] = "1"
    import openequivariance as oeq
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(42)
    dim, n_nodes, n_nz = 10, 3, 4

    problem = oeq.TPProblem(
        oeq.Irreps("1x2e"),
        oeq.Irreps("1x3e"),
        oeq.Irreps("1x2e"),
        [(0, 0, 0, "uvu", True)],
        shared_weights=False,
        internal_weights=False,
    )
    edge = jnp.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=jnp.int32)

    X = jax.random.uniform(
        key, (dim, n_nodes, problem.irreps_in1.dim), dtype=jnp.float32
    )
    Y = jax.random.uniform(key, (dim, n_nz, problem.irreps_in2.dim), dtype=jnp.float32)
    W = jax.random.uniform(key, (dim, n_nz, problem.weight_numel), dtype=jnp.float32)

    return {
        "X": X,
        "Y": Y,
        "W": W,
        "r": edge[0],
        "c": edge[1],
        "conv": oeq.jax.TensorProductConv(problem, deterministic=False),
        "jax": jax,
        "jnp": jnp,
        "dim": dim,
    }


def verify(ctx, in_axes, args):
    jax, jnp = ctx["jax"], ctx["jnp"]
    res_vmap = jax.vmap(ctx["conv"].forward, in_axes)(*args)

    res_loop = []
    for i in range(ctx["dim"]):
        i_args = [a[i] if ax == 0 else a for a, ax in zip(args, in_axes)]
        res_loop.append(ctx["conv"].forward(*i_args))

    assert jnp.allclose(res_vmap, jnp.stack(res_loop), atol=1e-5)


def test_vmap_std(ctx):
    verify(
        ctx, (0, 0, 0, None, None), (ctx["X"], ctx["Y"], ctx["W"], ctx["r"], ctx["c"])
    )


def test_vmap_bcast_X(ctx):
    verify(
        ctx,
        (None, 0, 0, None, None),
        (ctx["X"][0], ctx["Y"], ctx["W"], ctx["r"], ctx["c"]),
    )


def test_vmap_bcast_XW(ctx):
    verify(
        ctx,
        (None, 0, None, None, None),
        (ctx["X"][0], ctx["Y"], ctx["W"][0], ctx["r"], ctx["c"]),
    )


def test_vmap_streaming_preserves_padded_sentinel(with_jax):
    """Match independent streaming calls with one padded batched launch."""
    if not with_jax:
        pytest.skip("Skipping JAX tests")
    os.environ["OEQ_NOTORCH"] = "1"
    import jax
    import jax.numpy as jnp
    import openequivariance as oeq

    problem = oeq.TPProblem(
        oeq.Irreps("2x0e"),
        oeq.Irreps("1x0e"),
        oeq.Irreps("2x0e"),
        [(0, 0, 0, "uvu", True)],
        shared_weights=False,
        internal_weights=False,
    )
    conv = oeq.jax.TensorProductConv(problem, mode="streaming")
    batch_size, nodes, edges = 3, 3, 6
    keys = jax.random.split(jax.random.key(17), 4)
    dtype = problem.irrep_dtype
    x = jax.random.normal(
        keys[0], (batch_size, nodes, problem.irreps_in1.dim), dtype=dtype
    )
    y = jax.random.normal(
        keys[1], (batch_size, edges, problem.irreps_in2.dim), dtype=dtype
    )
    weights = jax.random.normal(
        keys[2], (batch_size, edges, problem.weight_numel), dtype=dtype
    )
    receivers = jnp.asarray((0, 1, 1, 2, nodes, nodes), dtype=jnp.int32)
    senders = jnp.asarray((1, 0, 2, 1, nodes, nodes), dtype=jnp.int32)
    row_ptr = jnp.asarray((0, 1, 3, 4), dtype=jnp.int32)

    def apply(a, b, w):
        return conv(
            a,
            b,
            w,
            receivers,
            senders,
            indices_are_sorted=True,
            row_ptr=row_ptr,
        )

    mapped = jax.jit(jax.vmap(apply))(x, y, weights)
    expected = jnp.stack([apply(x[i], y[i], weights[i]) for i in range(batch_size)])
    assert jnp.allclose(mapped, expected, rtol=1e-5, atol=1e-5)

    cotangent = jax.random.normal(keys[3], mapped.shape, dtype=dtype)

    def loss(a, b, w, dout):
        return jnp.vdot(apply(a, b, w), dout)

    gradient = jax.grad(loss, argnums=(0, 1, 2))
    mapped_grad = jax.jit(jax.vmap(gradient))(x, y, weights, cotangent)
    expected_grad = tuple(
        jnp.stack(
            [
                gradient(x[i], y[i], weights[i], cotangent[i])[argument]
                for i in range(batch_size)
            ]
        )
        for argument in range(3)
    )
    for actual, reference in zip(mapped_grad, expected_grad):
        assert jnp.allclose(actual, reference, rtol=1e-5, atol=1e-5)
    assert jnp.all(mapped_grad[1][:, 4:] == 0)
    assert jnp.all(mapped_grad[2][:, 4:] == 0)

    def hvp(a, b, w, dout, da, db, dw):
        return jax.jvp(
            lambda pa, pb, pw: gradient(pa, pb, pw, dout),
            (a, b, w),
            (da, db, dw),
        )[1]

    tangents = (0.1 * x, 0.1 * y, 0.1 * weights)
    mapped_hvp = jax.jit(jax.vmap(hvp))(x, y, weights, cotangent, *tangents)
    expected_hvp = tuple(
        jnp.stack(
            [
                hvp(
                    x[i],
                    y[i],
                    weights[i],
                    cotangent[i],
                    tangents[0][i],
                    tangents[1][i],
                    tangents[2][i],
                )[argument]
                for i in range(batch_size)
            ]
        )
        for argument in range(3)
    )
    for actual, reference in zip(mapped_hvp, expected_hvp):
        assert jnp.allclose(actual, reference, rtol=1e-5, atol=1e-5)
