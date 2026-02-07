import pytest
import os

@pytest.fixture
def with_jax(request):
    return request.config.getoption("--jax")

def test_tutorial_vmap(with_jax):
    if not with_jax:
        pytest.skip("Skipping JAX VMAP when testing PyTorch")

    os.environ["OEQ_NOTORCH"] = "1"
    import openequivariance as oeq
    import jax
    import jax.numpy as jnp

    seed = 42
    key = jax.random.PRNGKey(seed)

    vmap_dim = 10
    batch_size = 1000
    X_ir, Y_ir, Z_ir = oeq.Irreps("1x2e"), oeq.Irreps("1x3e"), oeq.Irreps("1x2e")
    instructions = [(0, 0, 0, "uvu", True)]

    problem = oeq.TPProblem(
        X_ir, Y_ir, Z_ir, instructions, shared_weights=False, internal_weights=False
    )

    edge_index = jnp.array(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ],
        dtype=jnp.int32 
    )

    node_ct, nonzero_ct = 3, 4
    X = jax.random.uniform(
        key, shape=(node_ct, X_ir.dim), minval=0.0, maxval=1.0, dtype=jnp.float32
    )
    Y = jax.random.uniform(
        key,
        shape=(vmap_dim, nonzero_ct, Y_ir.dim),
        minval=0.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    W = jax.random.uniform(
        key,
        shape=(vmap_dim, nonzero_ct, problem.weight_numel),
        minval=0.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    tp_conv = oeq.jax.TensorProductConv(problem, deterministic=False)
    Z_vmap = jax.vmap(tp_conv.forward, in_axes=(None, 0, 0, None, None))(X, Y, W, edge_index[0], edge_index[1])

    Z_loop = jnp.empty_like(Z_vmap)
    for i in range(vmap_dim):
        Z_loop = tp_conv.forward(X, Y[i], W[i], edge_index[0], edge_index[1])
        Z_vmap_i = Z_vmap[i]

    assert jnp.allclose(Z_vmap, Z_loop, atol=1e-5), "vmap and loop results do not match"

