import pytest
import os


@pytest.fixture
def with_jax(request):
    return request.config.getoption("--jax")


def test_tutorial_torch(with_jax):
    if with_jax:
        pytest.skip("Skipping PyTorch tutorial when testing JAX")

    import torch
    import e3nn.o3 as o3

    gen = torch.Generator(device="cuda")

    batch_size = 1000
    X_ir, Y_ir, Z_ir = o3.Irreps("1x2e"), o3.Irreps("1x3e"), o3.Irreps("1x2e")
    X = torch.rand(batch_size, X_ir.dim, device="cuda", generator=gen)
    Y = torch.rand(batch_size, Y_ir.dim, device="cuda", generator=gen)

    instructions = [(0, 0, 0, "uvu", True)]

    tp_e3nn = o3.TensorProduct(
        X_ir, Y_ir, Z_ir, instructions, shared_weights=False, internal_weights=False
    ).to("cuda")
    W = torch.rand(batch_size, tp_e3nn.weight_numel, device="cuda", generator=gen)

    Z = tp_e3nn(X, Y, W)
    print(torch.norm(Z))
    # ===============================

    # ===============================
    import openequivariance as oeq

    problem = oeq.TPProblem(
        X_ir, Y_ir, Z_ir, instructions, shared_weights=False, internal_weights=False
    )
    tp_fast = oeq.TensorProduct(problem)

    Z = tp_fast(X, Y, W)  # Reuse X, Y, W from earlier
    print(torch.norm(Z))
    # ===============================

    # Graph Convolution
    # ===============================
    from torch_geometric import EdgeIndex

    node_ct, nonzero_ct = 3, 4

    # Receiver, sender indices for message passing GNN
    edge_index = EdgeIndex(
        [
            [0, 1, 1, 2],  # Receiver
            [1, 0, 2, 1],
        ],  # Sender
        device="cuda",
        dtype=torch.long,
    )

    X = torch.rand(node_ct, X_ir.dim, device="cuda", generator=gen)
    Y = torch.rand(nonzero_ct, Y_ir.dim, device="cuda", generator=gen)
    W = torch.rand(nonzero_ct, problem.weight_numel, device="cuda", generator=gen)

    tp_conv = oeq.TensorProductConv(
        problem, deterministic=False
    )  # Reuse problem from earlier
    Z = tp_conv.forward(
        X, Y, W, edge_index[0], edge_index[1]
    )  # Z has shape [node_ct, z_ir.dim]
    print(torch.norm(Z))
    # ===============================

    # ===============================
    _, sender_perm = edge_index.sort_by("col")  # Sort by sender index
    edge_index, receiver_perm = edge_index.sort_by("row")  # Sort by receiver index

    # Now we can use the faster deterministic algorithm
    tp_conv = oeq.TensorProductConv(problem, deterministic=True)
    Z = tp_conv.forward(
        X, Y[receiver_perm], W[receiver_perm], edge_index[0], edge_index[1], sender_perm
    )
    print(torch.norm(Z))
    # ===============================
    assert True


def test_tutorial_jax(with_jax):
    if not with_jax:
        pytest.skip("Skipping JAX tutorial when testing PyTorch")

    os.environ["OEQ_NOTORCH"] = "1"
    import openequivariance as oeq
    import jax

    seed = 42
    key = jax.random.PRNGKey(seed)

    batch_size = 1000
    X_ir, Y_ir, Z_ir = oeq.Irreps("1x2e"), oeq.Irreps("1x3e"), oeq.Irreps("1x2e")
    instructions = [(0, 0, 0, "uvu", True)]

    problem = oeq.TPProblem(
        X_ir, Y_ir, Z_ir, instructions, shared_weights=False, internal_weights=False
    )
    tp_fast = oeq.jax.TensorProduct(problem)

    X = jax.random.uniform(
        key,
        shape=(batch_size, X_ir.dim),
        minval=0.0,
        maxval=1.0,
        dtype=jax.numpy.float32,
    )
    Y = jax.random.uniform(
        key,
        shape=(batch_size, Y_ir.dim),
        minval=0.0,
        maxval=1.0,
        dtype=jax.numpy.float32,
    )
    W = jax.random.uniform(
        key,
        shape=(batch_size, tp_fast.weight_numel),
        minval=0.0,
        maxval=1.0,
        dtype=jax.numpy.float32,
    )

    Z = tp_fast(X, Y, W)
    print(jax.numpy.linalg.norm(Z))

    edge_index = jax.numpy.array(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ],
        dtype=jax.numpy.int32,  # NOTE: This int32, not int64
    )

    node_ct, nonzero_ct = 3, 4
    X = jax.random.uniform(
        key, shape=(node_ct, X_ir.dim), minval=0.0, maxval=1.0, dtype=jax.numpy.float32
    )
    Y = jax.random.uniform(
        key,
        shape=(nonzero_ct, Y_ir.dim),
        minval=0.0,
        maxval=1.0,
        dtype=jax.numpy.float32,
    )
    W = jax.random.uniform(
        key,
        shape=(nonzero_ct, problem.weight_numel),
        minval=0.0,
        maxval=1.0,
        dtype=jax.numpy.float32,
    )
    tp_conv = oeq.jax.TensorProductConv(problem, deterministic=False)
    Z = tp_conv.forward(X, Y, W, edge_index[0], edge_index[1])
    print(jax.numpy.linalg.norm(Z))

    jitted = jax.jit(lambda X, Y, W, e1, e2: tp_conv.forward(X, Y, W, e1, e2))
    print(jax.numpy.linalg.norm(jitted(X, Y, W, edge_index[0], edge_index[1])))
