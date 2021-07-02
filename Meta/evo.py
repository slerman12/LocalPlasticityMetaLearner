import math
import random
import torch
from .seed_cell import SeedCell


class EvoSeedCell(torch.nn.Module):
    r"""Evolutionary container for seed cell reproduction
    """
    __constants__ = ['num_layers', 'hidden_state_size', 'context_size']
    num_layers: int
    hidden_state_size: int
    context_size: int
    population: list = []
    cache: list = []
    _prev_error: list = [math.inf]
    _optimizer: list = [None]

    def __init__(self, num_layers, hidden_state_size, context_size, population_size=10, std=1) -> None:
        super(EvoSeedCell, self).__init__()

        self.num_layers = num_layers
        self.hidden_state_size = hidden_state_size
        self.context_size = context_size
        self.population_size = population_size
        self.std = std

        for i in range(population_size):
            seed_cell = SeedCell(num_layers, hidden_state_size, context_size)
            self.population.append(seed_cell)
            setattr(seed_cell, 'is_alive', True)
            for j, param in enumerate(seed_cell.parameters()):
                self.register_parameter("param_{}_{}".format(i, j), param)

    def forward(self, context: list, hiddens: tuple[torch.Tensor, torch.Tensor],
                prev_output: torch.Tensor, prev_input: torch.Tensor, hidden_in: torch.Tensor) \
            -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        # assumes error is last element of context
        error = context[-1].mean()
        if error > self.prev_error:
            for seed_cell in self.cache:
                seed_cell.is_alive = False
            living_cells = [seed_cell for seed_cell in self.population if seed_cell.is_alive]
            for i, seed_cell in enumerate(self.population):
                if not seed_cell.is_alive:
                    new_seed_cell = self.mutate(random.choice(living_cells))
                    self.population[i] = new_seed_cell
                    setattr(new_seed_cell, 'is_alive', True)
                    for j, param in enumerate(new_seed_cell.parameters()):
                        self.register_parameter("param_{}_{}".format(i, j), param)
                        if self.optimizer is not None:
                            # does this continue increasing in memory indefinitely?
                            self.optimizer.add_param_group({"params": param})

        if self.prev_error != error:
            self.cache.clear()
        self.prev_error = error
        seed_cell = random.choice(self.population)
        self.cache.append(seed_cell)

        return seed_cell(context, hiddens, prev_output, prev_input, hidden_in)

    def mutate(self, seed_cell):
        new_seed_cell = SeedCell(self.num_layers, self.hidden_state_size, self.context_size)
        new_seed_cell.load_state_dict(seed_cell.state_dict())
        for param in new_seed_cell.parameters():
            self.add_noise(param)

        return new_seed_cell

    def add_noise(self, param):
        with torch.no_grad():
            # noise = torch.randn(param.size()) * self.std
            noise = torch.normal(mean=0, std=torch.full_like(param, self.std))
            param.add_(noise)

    @property
    def prev_error(self):
        return self._prev_error[0]

    @prev_error.setter
    def prev_error(self, value):
        self._prev_error[0] = value

    @property
    def optimizer(self):
        return self._optimizer[0]

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer[0] = value

    def set_optim(self, optimizer):
        self.optimizer = optimizer

    def extra_repr(self) -> str:
        return 'num_layers={}, hidden_state_size={}, context_size={}, population_size={}, std={}'.format(
            self.num_layers, self.hidden_state_size, self.context_size, self.population_size, self.std
        )

