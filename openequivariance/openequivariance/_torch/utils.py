import torch
import numpy as np
from types import MappingProxyType
from openequivariance.core.utils import DTypeEnum


def reorder_helper(schedule, weights_in, direction, has_batch_dim):
    assert direction in ["forward", "backward"]

    specs = schedule.weight_reordering_info(weights_in, has_batch_dim)
    weights_out = torch.zeros_like(weights_in)

    for spec in specs:
        parent_range = spec["parent_range"]
        parent_shape = spec["parent_shape"]
        weights_subrange = spec["weights_subrange"]
        child_range = spec["child_range"]
        transpose_perm = spec["transpose_perm"]

        if direction == "forward":
            reshape_size = spec["reshape_size"]

            sliced_weights = weights_in[parent_range].reshape(parent_shape)[
                weights_subrange
            ]

            weights_out[child_range] = sliced_weights.permute(transpose_perm).reshape(
                reshape_size
            )

        elif direction == "backward":
            transpose_child_shape = spec["transpose_child_shape"]
            child_shape = spec["child_shape"]

            sliced_weights = (
                weights_in[child_range]
                .reshape(transpose_child_shape)
                .permute(transpose_perm)
            )

            weights_out[parent_range].reshape(parent_shape)[weights_subrange] = (
                sliced_weights.flatten().reshape(child_shape)
            )

    return weights_out


def reorder_numpy_helper(schedule, weights_in, direction, has_batch_dim):
    weights_in = torch.from_numpy(weights_in.copy())
    result = reorder_helper(schedule, weights_in, direction, has_batch_dim)
    return result.detach().cpu().numpy().copy()


def reorder_torch(schedule, weights_in, direction, has_batch_dim):
    if isinstance(weights_in, torch.Tensor):
        return reorder_helper(schedule, weights_in, direction, has_batch_dim)
    else:
        return reorder_numpy_helper(schedule, weights_in, direction, has_batch_dim)


enum_to_torch_dtype = MappingProxyType(
    {
        DTypeEnum.FLOAT32: torch.float32,
        DTypeEnum.FLOAT64: torch.float64,
        DTypeEnum.INT32: torch.int32,
        DTypeEnum.INT64: torch.int64,
        DTypeEnum.UINT8: torch.uint8,
    }
)

def string_to_tensor(text: str) -> torch.Tensor:
    bytes_data = text.encode('utf-8')
    np_bytes = np.frombuffer(bytes_data, dtype=np.uint8)
    result = torch.tensor(np_bytes)
    result.requires_grad = False
    return result