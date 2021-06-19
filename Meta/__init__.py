import torch
from .seed_cell_linear import SeedCellLinear as Linear
from .seed_cell_conv import SeedCellConv2D as Conv2D


class Sequential(torch.nn.Sequential):
    def forward(self, *input):
        input = list(input)
        for module in self._modules.values():
            if "SeedCell" in str(module):
                input = list(module(*input))
            else:
                input[0] = module(input[0])
        return input[0]