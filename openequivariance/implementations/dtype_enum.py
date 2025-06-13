import numpy as np
import torch

PYTORCH_DTYPE_ENUM: dict[type[torch.dtype], int] = {
    torch.float32: 1,
    torch.float64: 2,
    torch.int32: 3,
}

NUMPY_DTYPE_ENUM: dict[type[np.generic], int] = {
    np.float32: 1,
    np.float64: 2,
    torch.int32: 3,
}
