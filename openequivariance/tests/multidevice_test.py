import textwrap
import torch
import subprocess
import os


def test_multidevice():
    result = subprocess.run(
        [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=gpu",
            __file__,
        ],
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        error_string = f"""
        Invocation: {" ".join(result.args)}
        Test failed with return code {result.returncode}.
        \nOutput:\n\n{result.stdout.decode()}
        \nError:\n\n{result.stderr.decode()}
        """
        assert False, textwrap.dedent(error_string)

    assert True


if __name__ == "__main__":
    import openequivariance as oeq

    # Use MACE-large to test >64KB shared memory allocation
    from openequivariance.benchmark.problems import mace_problems

    problem = mace_problems()[0]

    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"
    torch.set_default_device(device)

    X_ir, Y_ir, Z_ir = problem.irreps_in1, problem.irreps_in2, problem.irreps_out
    tp = oeq.TensorProduct(problem)

    batch_size = 1000
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    X = torch.rand(batch_size, X_ir.dim, device=device, generator=gen)
    Y = torch.rand(batch_size, Y_ir.dim, device=device, generator=gen)
    W = torch.rand(batch_size, problem.weight_numel, device=device, generator=gen)

    with torch.cuda.device(device):
        result = tp.forward(X, Y, W)
