import torch


def test_autocast_enabled():
    n = 100
    x = torch.randn(size=(n, n), dtype=torch.float32)
    y = torch.randn(size=(n, n), dtype=torch.float32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        z = torch.matmul(x, y)
        assert z.dtype == torch.bfloat16


def test_python_disable_autocast():
    n = 100
    x = torch.randn(size=(n, n), dtype=torch.float32)
    y = torch.randn(size=(n, n), dtype=torch.float32)

    torch.library.register_autocast("aten::matmul", "cpu", None)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        z = torch.matmul(x, y)
        assert z.dtype == torch.float32
