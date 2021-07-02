from .Layers.linear import Linear
from .seed_cell import SeedCell
from .evo import EvoSeedCell
import torch
import random


class Sequential(torch.nn.Sequential):
    def forward(self, *input):
        input = list(input)
        for module in self._modules.values():
            if "SeedCell" in str(module):
                input = list(module(*input))
            else:
                input[0] = module(input[0])
        return input[0]

    def reset_parameters(self) -> None:
        for module in self._modules.values():
            if "SeedCell" in str(module):
                module.reset_parameters()


class MultiSequential(torch.nn.Module):
    def __init__(self, *input, num_sequentials=1):
        super(MultiSequential, self).__init__()
        self.num_sequentials = num_sequentials

        def clone(inputs):
            return [inp.__class__(*inp.args) if hasattr(inp, 'seed_cell') else inp for inp in inputs]

        self.sequentials = torch.nn.ModuleList([Sequential(*clone(input)) for _ in range(num_sequentials)])

    def forward(self, *input):
        outputs = [sequential(*self.sub_range(input, i)) for i, sequential in enumerate(self.sequentials)]
        return torch.cat(outputs, 0)

    def sub_range(self, input, i):
        if len(input) == 0:
            return input
        interval = input[0].shape[0] // self.num_sequentials
        if i == self.num_sequentials - 1:
            return [inp[i * interval:] if isinstance(inp, torch.Tensor)
                    else self.sub_range(inp, i) if isinstance(inp, list) else inp for inp in input]
        else:
            return [inp[i * interval:(i + 1) * interval] if isinstance(inp, torch.Tensor)
                    else self.sub_range(inp, i) if isinstance(inp, list) else inp for inp in input]

    def reset_one(self, ind=None) -> None:
        if ind is None:
            ind = random.randint(0, self.num_sequentials - 1)
        with torch.no_grad():
            self.sequentials[ind].reset_parameters()

