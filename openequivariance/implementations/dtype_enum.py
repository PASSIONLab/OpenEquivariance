from enum import IntEnum
from types import MappingProxyType
import numpy as np
import torch


class DTypeEnum(IntEnum):
    FLOAT32 = 1
    FLOAT64 = 2
    INT32 = 3
    INT64 = 4


dtype_to_enum_mapping = MappingProxyType(
    {
        torch.float32: DTypeEnum.FLOAT32,
        torch.float64: DTypeEnum.FLOAT64,
        torch.int32: DTypeEnum.INT32,
        torch.int64: DTypeEnum.INT64,
        np.float32: DTypeEnum.FLOAT32,
        np.float64: DTypeEnum.FLOAT64,
        np.int32: DTypeEnum.INT32,
        np.int64: DTypeEnum.INT64,
        np.dtype(np.float32): DTypeEnum.FLOAT32,
        np.dtype(np.float64): DTypeEnum.FLOAT64,
        np.dtype(np.int32): DTypeEnum.INT32,
        np.dtype(np.int64): DTypeEnum.INT64,
    }
)


enum_to_torch_dtype_mapping = MappingProxyType(
    {
        DTypeEnum.FLOAT32: torch.float32,
        DTypeEnum.FLOAT64: torch.float64,
        DTypeEnum.INT32: torch.int32,
        DTypeEnum.INT64: torch.int64,
    }
)
