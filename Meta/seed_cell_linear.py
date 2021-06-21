import math
from typing import TypeVar
import torch
T = TypeVar('T', bound='Module')


class SeedCellLinear(torch.nn.Module):
    r"""Applies a linear transformation to the incoming data:
    :math:`y = xW^T + b
    using Seed Cells to self-optimize W.`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output[0]: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the projection weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        # hidden_state: (possibly persistent) hidden state of the module of shape
        #     :math:`(\text{out\_features}, hidden_state_size)`. The values are
        #     initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
        #     :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = SeedCellLinear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)[0]
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features', 'hidden_state_size']
    in_features: int
    out_features: int
    hidden_state_size: int = 5
    weight: torch.Tensor
    meta_net = torch.nn.Sequential(torch.nn.Linear(hidden_state_size + 6, 32),
                                   # torch.nn.Linear(2 * hidden_state_size + 6, 32),  # If hidden state persistent
                                   torch.nn.Tanh(),
                                   torch.nn.Linear(32, hidden_state_size + 1))

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(SeedCellLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.Tensor(out_features, in_features)
        # self.hidden_state = torch.Tensor(out_features, self.hidden_state_size)

        if bias:
            # Disable bias for now
            # self.bias = torch.nn.Parameter(torch.Tensor(out_features))
            self.bias = torch.zeros(out_features)
        else:
            self.register_parameter('bias', None)

        for i, param in enumerate(self.meta_net.parameters()):
            self.register_parameter('meta_net_{}'.format(i), param)

        self.prev_input = None
        self.prev_output = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # torch.nn.init.kaiming_uniform_(self.hidden_state, a=math.sqrt(5))
        # Disable bias for now
        # if self.bias is not None:
        #     fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, prev_y_label: torch.Tensor = None, prev_y_pred: torch.Tensor = None,
                prev_error: torch.Tensor = None, truncate: bool = False, hidden_state_in: torch.Tensor = None) \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, torch.Tensor]:
        # SELF-OPTIMIZATION
        hidden_state = None
        is_initialized = self.prev_input is not None and self.prev_output is not None
        self_optimizing = prev_y_label is not None and prev_y_pred is not None and prev_error is not None
        if self_optimizing and is_initialized and self.training:
            prev_batch_size = self.prev_input.shape[0]
            prev_input = self.prev_input
            prev_output = self.prev_output

            if hidden_state_in is None:
                hidden_state_in = torch.zeros(prev_batch_size, self.in_features, self.hidden_state_size)

            # Detach to truncate recursion
            if truncate:
                prev_input = self.prev_input.detach()
                prev_output = self.prev_output.detach()
                prev_y_label = prev_y_label.detach()
                prev_y_pred = prev_y_pred.detach()
                prev_error = prev_error.detach()
                # self.hidden_state = self.hidden_state.detach()
                self.weight = self.weight.detach()

            weight = torch.flatten(self.weight.unsqueeze(0).expand(prev_batch_size, -1, -1))[:, None]
            # hidden_state = self.hidden_state.unsqueeze(2).expand(-1, -1, self.in_features, -1).reshape(-1, self.hidden_state_size)
            hidden_state_in = hidden_state_in.unsqueeze(1).expand(-1, self.out_features, -1, -1).reshape(-1, self.hidden_state_size)
            y_label = torch.flatten(prev_y_label.unsqueeze(2).expand(-1, self.out_features, self.in_features))[:, None]
            y_pred = torch.flatten(prev_y_pred.unsqueeze(2).expand(-1, self.out_features, self.in_features))[:, None]
            error = torch.flatten(prev_error.unsqueeze(2).expand(-1, self.out_features, self.in_features))[:, None]
            # l = torch.full_like(error, self.l)
            # L = torch.full_like(error, self.L)
            input_i = torch.flatten(prev_input.unsqueeze(1).expand(-1, self.out_features, -1))[:, None]
            output_j = torch.flatten(prev_output.unsqueeze(2).expand(-1, -1, self.in_features))[:, None]

            composite = torch.cat([y_label, y_pred, error, weight, hidden_state_in, input_i, output_j], -1)

            updates = self.meta_net(composite)
            updates = updates.view(prev_batch_size, self.out_features, self.in_features, self.hidden_state_size + 1)
            # updates = torch.normal(mean=updates, std=0.01)

            # Would be faster to use two meta-nets
            self.weight = self.weight + updates.mean(0)[:, :, -1].squeeze()
            # Can make it persistent
            # self.hidden_state += updates.mean(1)[:, :-1]
            hidden_state = updates.mean(2)[:, :, :-1]

        # FORWARD PASS
        output = torch.nn.functional.linear(input, self.weight, self.bias)

        if self.training and (self_optimizing or not is_initialized):
            self.prev_input = input
            self.prev_output = output

        if not self.training:
            self.prev_input = None
            self.prev_output = None

        return output, prev_y_label, prev_y_pred, prev_error, truncate, hidden_state

    def eval(self: T) -> T:
        r"""Sets the module in evaluation mode.

        Also serves to reset self.prev_input and self.prev_output.
        TODO alternatively could check if self.prev_input matches batch size of prev_y_pred; reset otherwise

        Returns:
            Module: self
        """
        self.prev_input = None
        self.prev_output = None
        return super(SeedCellLinear, self).eval()

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, hidden_state_size={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.hidden_state_size
        )


