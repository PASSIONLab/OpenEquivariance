from enum import IntEnum
from typing import Mapping
from types import MappingProxyType
import numpy as np
import torch


class DTypeEnum(IntEnum):
    FLOAT32 = 1
    FLOAT64 = 2
    INT32 = 3
    INT64 = 4
    UINT8 = 5


dtype_to_enum = (
    MappingProxyType(
        {
            torch.float32: DTypeEnum.FLOAT32,
            torch.float64: DTypeEnum.FLOAT64,
            torch.int32: DTypeEnum.INT32,
            torch.int64: DTypeEnum.INT64,
            torch.uint8: DTypeEnum.UINT8,
            # torch
            np.float32: DTypeEnum.FLOAT32,
            np.float64: DTypeEnum.FLOAT64,
            np.int32: DTypeEnum.INT32,
            np.int64: DTypeEnum.INT64,
            np.uint8: DTypeEnum.UINT8,
            # numpy generic
            np.dtype(np.float32): DTypeEnum.FLOAT32,
            np.dtype(np.float64): DTypeEnum.FLOAT64,
            np.dtype(np.int32): DTypeEnum.INT32,
            np.dtype(np.int64): DTypeEnum.INT64,
            np.dtype(np.uint8): DTypeEnum.UINT8,
            # numpy dtype
        }
    )
)


enum_to_torch_dtype = MappingProxyType(
    {
        DTypeEnum.FLOAT32: torch.float32,
        DTypeEnum.FLOAT64: torch.float64,
        DTypeEnum.INT32: torch.int32,
        DTypeEnum.INT64: torch.int64,
        DTypeEnum.UINT8: torch.uint8,
    }
)
