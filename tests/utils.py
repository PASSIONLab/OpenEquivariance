from dataclasses import dataclass
from typing import Callable, Tuple, Any

import torch


@dataclass
class Executable:
    func: Callable[..., Any]
    buffers: Tuple[torch.Tensor, ...]

    def __call__(self) -> Any:
        return self.func(*self.buffers)
