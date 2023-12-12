from functools import partial
from collections import namedtuple
from typing import Optional, Tuple, Union

import torch
from torch.nn import Module, ModuleList
from torch import nn, einsum
import torch.nn.functional as F

from beartype import beartype

from einops import rearrange, repeat, reduce, pack, unpack

from colt5_attention import topk as maybe_differentiable_topk

import torch.distributed as dist


class MLPExpert(Module):
    def __init__(
        self,
        dim,
        hidden_mult = 4,
        mult_bias = True,
        prenorm = False
    ):
        super().__init__()
        dim_hidden = int(dim * hidden_mult * 2 / 3)

        self.net = Sequential(
            RMSNorm(dim) if prenorm else None,
            nn.Linear(dim, dim_hidden * 2),
            GEGLU(dim_hidden, mult_bias = mult_bias),
            nn.Linear(dim_hidden, dim)
        )

        self.apply(self.init_)

    def init_(self, module):
        if isinstance(module, nn.Linear):
            dim = module.weight.shape[0]
            std = dim ** -0.5

            module.weight.data.uniform_(-std, std)
            module.bias.data.uniform_(-std, std)

    def forward(self, x):
        return self.net(x)

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale

# expert class
# best performing was ff geglu with multiplicative bias (just after gating)

class GEGLU(Module):
    def __init__(
        self,
        dim,
        mult_bias = True
    ):
        super().__init__()
        self.mult_bias = nn.Parameter(torch.ones(dim)) if mult_bias else 1.

    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x * self.mult_bias

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

def exists(val):
    return val is not None