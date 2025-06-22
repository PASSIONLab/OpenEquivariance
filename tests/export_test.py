import shutil
import torch
import pytest
import tempfile
import subprocess
import os
import sys

import numpy as np
import openequivariance as oeq
from torch_geometric import EdgeIndex
import importlib.resources

from openequivariance.implementations.E3NNTensorProduct import E3NNTensorProduct


@pytest.fixture(scope="session")
def problem_and_irreps():
    X_ir, Y_ir, Z_ir = oeq.Irreps("32x5e"), oeq.Irreps("1x3e"), oeq.Irreps("32x5e")
    problem = oeq.TPProblem(
        X_ir,
        Y_ir,
        Z_ir,
        [(0, 0, 0, "uvu", True)],
        shared_weights=False,
        internal_weights=False,
        irrep_dtype=np.float32,
        weight_dtype=np.float32,
    )

    gen = torch.Generator(device="cuda")
    gen.manual_seed(0)

    return (
        problem,
        X_ir,
        Y_ir,
        Z_ir,
    )


@pytest.fixture(params=["batch", "conv_det", "conv_atomic"], scope="session")
def tp_and_inputs(request, problem_and_irreps):
    problem, X_ir, Y_ir, _ = problem_and_irreps
    gen = torch.Generator(device="cuda")
    gen.manual_seed(0)

    if request.param == "batch":
        batch_size = 1000
        X = torch.rand(batch_size, X_ir.dim, device="cuda", generator=gen)
        Y = torch.rand(batch_size, Y_ir.dim, device="cuda", generator=gen)
        W = torch.rand(batch_size, problem.weight_numel, device="cuda", generator=gen)
        return oeq.TensorProduct(problem), (X, Y, W)
    else:
        node_ct, nonzero_ct = 3, 4

        # Receiver, sender indices for message passing GNN
        edge_index = EdgeIndex(
            [[0, 1, 1, 2], [1, 0, 2, 1]], device="cuda", dtype=torch.long
        )

        _, sender_perm = edge_index.sort_by("col")
        edge_index, _ = edge_index.sort_by("row")
        edge_index = [edge_index[0].detach(), edge_index[1].detach()]

        X = torch.rand(node_ct, X_ir.dim, device="cuda", generator=gen)
        Y = torch.rand(nonzero_ct, Y_ir.dim, device="cuda", generator=gen)
        W = torch.rand(nonzero_ct, problem.weight_numel, device="cuda", generator=gen)

        if request.param == "conv_atomic":
            return oeq.TensorProductConv(problem, torch_op=True, deterministic=False), (
                X,
                Y,
                W,
                edge_index[0],
                edge_index[1],
            )
        elif request.param == "conv_det":
            return oeq.TensorProductConv(problem, torch_op=True, deterministic=True), (
                X,
                Y,
                W,
                edge_index[0],
                edge_index[1],
                sender_perm,
            )


def test_torch_load(tp_and_inputs):
    tp, inputs = tp_and_inputs
    original_result = tp.forward(*inputs)

    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
        torch.save(tp, tmp_file.name)
        loaded_tp = torch.load(tmp_file.name)

    reloaded_result = loaded_tp(*inputs)
    assert torch.allclose(original_result, reloaded_result, atol=1e-5)


def test_jitscript(tp_and_inputs):
    tp, inputs = tp_and_inputs
    uncompiled_result = tp.forward(*inputs)

    scripted_tp = torch.jit.script(tp)
    loaded_tp = None
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
        scripted_tp.save(tmp_file.name)
        loaded_tp = torch.jit.load(tmp_file.name)

    compiled_result = loaded_tp(*inputs)
    assert torch.allclose(uncompiled_result, compiled_result, atol=1e-5)


def test_compile(tp_and_inputs):
    tp, inputs = tp_and_inputs
    uncompiled_result = tp.forward(*inputs)

    compiled_tp = torch.compile(tp)
    compiled_result = compiled_tp(*inputs)
    assert torch.allclose(uncompiled_result, compiled_result, atol=1e-5)


def test_export(tp_and_inputs):
    tp, inputs = tp_and_inputs
    uncompiled_result = tp.forward(*inputs)

    exported_tp = torch.export.export(tp, args=inputs, strict=False)
    exported_result = exported_tp.module()(*inputs)
    assert torch.allclose(uncompiled_result, exported_result, atol=1e-5)


def test_aoti(tp_and_inputs):
    tp, inputs = tp_and_inputs
    uncompiled_result = tp.forward(*inputs)

    exported_tp = torch.export.export(tp, args=inputs, strict=False)
    aoti_model = None
    with tempfile.NamedTemporaryFile(suffix=".pt2") as tmp_file:
        try:
            output_path = torch._inductor.aoti_compile_and_package(
                exported_tp, package_path=tmp_file.name
            )
        except Exception as e:
            err_msg = (
                "AOTI compile_and_package failed. NOTE: OpenEquivariance only supports AOTI for "
                + "PyTorch version >= 2.8.0.dev20250410+cu126 due to incomplete TorchBind support "
                + "in prior versions. "
                + f"{e}"
            )
            assert False, err_msg

        aoti_model = torch._inductor.aoti_load_package(output_path)

    aoti_result = aoti_model(*inputs)
    assert torch.allclose(uncompiled_result, aoti_result, atol=1e-5)


def test_jitscript_cpp_interface(problem_and_irreps):
    assert oeq.LINKED_LIBPYTHON, oeq.LINKED_LIBPYTHON_ERROR
    problem, X_ir, Y_ir, _ = problem_and_irreps
    cmake_prefix_path = torch.utils.cmake_prefix_path
    torch_ext_so_path = oeq.torch_ext_so_path()

    oeq_tp = oeq.TensorProduct(problem).to("cuda")
    scripted_oeq = torch.jit.script(oeq_tp)

    e3nn_tp = E3NNTensorProduct(problem).e3nn_tp.to("cuda")
    scripted_e3nn = torch.jit.script(e3nn_tp)

    batch_size = 1000

    with (
        tempfile.TemporaryDirectory() as tmpdir,
        tempfile.NamedTemporaryFile(suffix=".pt") as oeq_file,
        tempfile.NamedTemporaryFile(suffix=".pt") as e3nn_file,
    ):
        scripted_oeq.save(oeq_file.name)
        scripted_e3nn.save(e3nn_file.name)

        test_path = importlib.resources.files("openequivariance") / "extension" / "test"
        build_dir = os.path.join(tmpdir, "build")
        os.makedirs(build_dir, exist_ok=True)

        for item in test_path.iterdir():
            shutil.copy(item, tmpdir)

        try:
            subprocess.run(
                [
                    "cmake",
                    "..",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_PREFIX_PATH=" + cmake_prefix_path,
                    "-DOEQ_EXTLIB=" + torch_ext_so_path,
                ],
                cwd=build_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            subprocess.run(
                ["make"],
                cwd=build_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            subprocess.run(
                [
                    "./load_jitscript",
                    e3nn_file.name,
                    oeq_file.name,
                    str(X_ir.dim),
                    str(Y_ir.dim),
                    str(problem.weight_numel),
                    str(batch_size),
                ],
                cwd=build_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            print(e.stdout.decode(), file=sys.stderr)
            print(e.stderr.decode(), file=sys.stderr)
            assert False
