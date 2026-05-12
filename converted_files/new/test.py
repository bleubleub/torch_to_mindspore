import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

from mindspore import nn


class ABC(nn.Cell):
    def __init__(
        activation:Optional[nn.Module] = None
    ):
        super().__init__()

    def construct(self, x:Tensor):
        return self.activation(x)

class ABC(nn.Cell):
    def __init__(
        activation:nn.Module
    ):
        super().__init__()

    def construct(self, x:Tensor):
        return self.activation(x)