import pytest
import os
from functools import partial


@pytest.fixture(params=("standard", "streaming"))
def ctx(with_jax, request):
    if not with_jax:
        pytest.skip("Skipping JAX tests")
    os.environ["OEQ_NOTORCH"] = "1"
    import openequivariance as oeq
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(42)
    dim, n_nodes, n_nz = 10, 3, 4
    if request.param == "streaming":
        n_nz = 6

    problem = oeq.TPProblem(
        oeq.Irreps("1x2e"),
        oeq.Irreps("1x3e"),
        oeq.Irreps("1x2e"),
        [(0, 0, 0, "uvu", True)],
        shared_weights=False,
        internal_weights=False,
    )
    edge = jnp.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=jnp.int32)
    if request.param == "streaming":
        edge = jnp.pad(edge, ((0, 0), (0, 2)), constant_values=n_nodes)

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
        "conv": oeq.jax.TensorProductConv(
            problem, deterministic=False, mode=request.param
        ),
        "jax": jax,
        "jnp": jnp,
        "dim": dim,
    }


def verify(ctx, in_axes, args):
    jax, jnp = ctx["jax"], ctx["jnp"]
    for sorted_edges in (True, False):
        if not sorted_edges:
            x, y, weights, rows, cols = args
            args = (
                x, y[..., ::-1, :], weights[..., ::-1, :],
                rows[..., ::-1], cols[..., ::-1],
            )
        forward = partial(ctx["conv"].forward, indices_are_sorted=sorted_edges)
        res_vmap = jax.vmap(forward, in_axes)(*args)

        res_loop = []
        for i in range(ctx["dim"]):
            i_args = [a[i] if ax == 0 else a for a, ax in zip(args, in_axes)]
            res_loop.append(forward(*i_args))

        assert jnp.allclose(res_vmap, jnp.stack(res_loop), atol=1e-5)


def test_vmap_std(ctx):
    verify(
        ctx, (0, 0, 0, None, None), (ctx["X"], ctx["Y"], ctx["W"], ctx["r"], ctx["c"])
    )

    jnp = ctx["jnp"]
    rows = jnp.broadcast_to(ctx["r"], (ctx["dim"], ctx["r"].size))
    cols = jnp.broadcast_to(ctx["c"], (ctx["dim"], ctx["c"].size))
    verify(ctx, (0, 0, 0, 0, 0), (ctx["X"], ctx["Y"], ctx["W"], rows, cols))


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
