import pytest, torch, subprocess, argparse

#torchrun
#    --standalone
#    --nnodes=1
#    --nproc-per-node=$NUM_TRAINERS
#    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

def test_multidevice():
    subprocess.run([
        "python -m torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=gpu",
        __file__], 
        capture_output=True,
        capture_stderr=True, 
        check=False)
    
    if subprocess.returncode != 0:
        assert False, f"Test failed with return code {subprocess.returncode}. Output:\n\n {subprocess.stdout.decode()}. Error:\n\n {subprocess.stderr.decode()}"

    assert True

if __name__ == "__main__":
    print("Running test_multidevice")
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", "--local_rank", type=int)
    args = parser.parse_args()

    print(f"local_rank: {args.local_rank}")
    exit(1)