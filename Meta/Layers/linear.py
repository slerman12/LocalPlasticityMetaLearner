import math
import torch


class Linear(torch.nn.Module):
    r"""Applies a linear transformation to the incoming data:
    :math:`y = xW^T + b
    using Seed Cells to self-optimize.`

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

        >>> m = Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)[0]
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features', 'hidden_state_size']
    in_features: int
    out_features: int
    weight: torch.Tensor
    hidden_out: torch.Tensor
    hidden_state: torch.Tensor
    seed_cell: torch.nn.Module

    def __init__(self, seed_cell: torch.nn.Module, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.args = (seed_cell, in_features, out_features, bias)
        self.seed_cell = seed_cell
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.Tensor(out_features, in_features)

        if bias:
            # Disable bias for now
            # self.bias = torch.nn.Parameter(torch.Tensor(out_features))
            self.bias = torch.zeros(out_features)
        else:
            self.register_parameter('bias', None)

        for i, param in enumerate(self.seed_cell.parameters()):
            self.register_parameter("param_{}".format(i), param)

        self.hidden_out = torch.zeros(seed_cell.num_layers, 1, self.out_features, seed_cell.hidden_state_size)
        self.hidden_state = torch.zeros(seed_cell.num_layers, 1, self.out_features, seed_cell.hidden_state_size)

        self.prev_output = None
        self.prev_input = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # torch.nn.init.normal(self.weight)
        self.hidden_out = torch.zeros_like(self.hidden_out)
        self.hidden_state = torch.zeros_like(self.hidden_state)
        self.prev_output = None
        self.prev_input = None
        # Disable bias for now
        # if self.bias is not None:
        #     fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, context=None, truncate: bool = False, hidden_in: torch.Tensor = None) \
            -> tuple[torch.Tensor, list, bool, torch.Tensor]:
        # SELF-OPTIMIZATION
        if context is None:
            context = []
        # Detach to truncate recursion
        if truncate:
            self.prev_input = self.prev_input.detach()
            self.prev_output = self.prev_output.detach()
            context = [data.detach() for data in context]
            self.weight = self.weight.detach()
            self.hidden_state = self.hidden_state.detach()
            self.hidden_out = self.hidden_out.detach()
        is_initialized = self.prev_input is not None and self.prev_output is not None
        self_optimizing = len(context) > 0
        hidden_out = None
        if self_optimizing and is_initialized and self.training:
            hiddens = (self.hidden_out, self.hidden_state)

            (q, k), hiddens = self.seed_cell(context, hiddens, self.prev_output, self.prev_input, hidden_in)

            self.hidden_out, self.hidden_state = hiddens

            # [prev batch size * out, h]
            prev_batch_size = self.prev_input.shape[0]
            hidden_out = self.hidden_out[-1, :prev_batch_size].view(-1, self.seed_cell.hidden_state_size)

            # [prev batch size, out, in]
            delta = torch.matmul(q, torch.transpose(k, 1, 2))

            # [out, in]
            # self.weight = torch.normal(mean=self.weight, std=0.1) + delta.mean(0)
            self.weight = self.weight + delta.mean(0)

        # FORWARD PASS
        output = torch.nn.functional.linear(input, self.weight, self.bias)

        if self.training and (self_optimizing or not is_initialized):
            self.prev_output = output
            self.prev_input = input

        # TODO alternatively could check if self.prev_input matches batch size of prev_y_pred; reset otherwise
        # if not self.training:
        #     self.prev_output = None
        #     self.prev_input = None

        return output, context, truncate, hidden_out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


