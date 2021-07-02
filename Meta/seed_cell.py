import torch


class SeedCell(torch.nn.Module):
    r"""Seed cell core
    """
    __constants__ = ['num_layers', 'hidden_state_size', 'context_size']
    num_layers: int
    hidden_state_size: int
    context_size: int

    def __init__(self, num_layers, hidden_state_size, context_size) -> None:
        super(SeedCell, self).__init__()

        self.num_layers = num_layers
        self.hidden_state_size = hidden_state_size
        self.context_size = context_size

        self.q_1_linear = torch.nn.Linear(hidden_state_size + context_size + 1, hidden_state_size)
        self.k_1_linear = torch.nn.Linear(hidden_state_size, hidden_state_size)
        self.v_linear = torch.nn.Linear(hidden_state_size, hidden_state_size)

        self.attention = torch.nn.MultiheadAttention(hidden_state_size, 1)
        # later pytorch
        # self.attention = torch.nn.MultiheadAttention(hidden_state_size, 1, batch_first=True)

        self.lstm = torch.nn.LSTM(hidden_state_size + context_size + 1, hidden_state_size, num_layers, batch_first=True)

        self.k_2_linear = torch.nn.Linear(hidden_state_size, hidden_state_size)
        self.q_2_linear = torch.nn.Linear(hidden_state_size, hidden_state_size)

    def forward(self, context: list, hiddens: tuple[torch.Tensor, torch.Tensor],
                prev_output: torch.Tensor, prev_input: torch.Tensor, hidden_in: torch.Tensor) \
            -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        prev_batch_size = prev_input.shape[0]
        all_hidden_out, all_hidden_state = hiddens
        out_features = prev_output.shape[-1]
        in_features = prev_input.shape[-1]

        shape_mismatch = prev_batch_size - all_hidden_out.shape[1]
        if shape_mismatch > 0:
            hidden = torch.zeros(self.num_layers, shape_mismatch, out_features, self.hidden_state_size)
            # [l, batch size, out, h]
            all_hidden_out = torch.cat([all_hidden_out, hidden], dim=1)
            all_hidden_state = torch.cat([all_hidden_state, hidden], dim=1)

        # [l, prev batch size * out, h]
        hidden = all_hidden_out[:, :prev_batch_size].view(self.num_layers, -1, self.hidden_state_size)
        hidden_state = all_hidden_state[:, :prev_batch_size].view(hidden.shape)

        # [prev batch size * out, h]
        hidden_out = all_hidden_out[-1, :prev_batch_size].view(-1, self.hidden_state_size)  # last layer

        if hidden_in is None:
            # could put y_label, y_pred, error here instead
            # [prev batch size * in, h]
            hidden_in = prev_input.view(-1, 1).expand(-1, self.hidden_state_size)

        # [prev batch size, out, 1]
        context = [data.expand(-1, out_features).unsqueeze(2) for data in context]
        # [prev batch size, out, c]
        context = torch.cat([prev_output.unsqueeze(2), *context], -1)
        # [prev batch size * out, h + c]
        h_hat = torch.cat([hidden_out, context.view(-1, self.context_size + 1)], -1)

        # [prev batch size * out, h]
        q_1 = self.q_1_linear(h_hat)
        # [prev batch size * in, h]
        k_1 = self.k_1_linear(hidden_in)
        v = self.v_linear(hidden_in)

        # [prev batch size, out, h]
        q_1 = q_1.view(prev_batch_size, out_features, self.hidden_state_size)
        # [prev batch size, in, h]
        k_1 = k_1.view(prev_batch_size, in_features, self.hidden_state_size)
        v = v.view(k_1.shape)

        # [prev batch size, out, h]
        attention, _ = self.attention(q_1.transpose(0, 1), k_1.transpose(0, 1), v.transpose(0, 1))
        attention = attention.transpose(0, 1)
        # later pytorch
        # attention = self.attention(q_1, k_1, v, need_weights=False)
        # attention = attention.view(-1, 1, self.hidden_state_size)

        # [prev batch size * out, 1, h]
        attention = attention.reshape(-1, 1, self.hidden_state_size)
        # [prev batch size * out, 1, c]
        context = context.view(-1, 1, self.context_size + 1)

        _, (hidden, hidden_state) = self.lstm(torch.cat([attention, context], -1), (hidden, hidden_state))
        # [prev batch size * out, h]
        hidden_out = hidden[-1]
        # hidden_out = self.meta_net_0(torch.cat([attention, context], -1)).squeeze()

        shape = [self.num_layers, prev_batch_size, out_features, self.hidden_state_size]

        if shape_mismatch < 0:
            # [l, batch size, out, h]
            all_hidden_out = torch.cat([hidden.view(shape), all_hidden_out[:, prev_batch_size:]], 1)
            all_hidden_state = torch.cat([hidden_state.view(shape), all_hidden_state[:, prev_batch_size:]], 1)
        else:
            all_hidden_out = hidden.view(shape)
            all_hidden_state = hidden_state.view(shape)

        # [prev batch size, out, h]
        q_2 = self.q_2_linear(hidden_out).view(q_1.shape)
        # [prev batch size, in, h]
        k_2 = self.k_2_linear(hidden_in).view(k_1.shape)

        return (q_2, k_2), (all_hidden_out, all_hidden_state)

    def extra_repr(self) -> str:
        return 'num_layers={}, hidden_state_size={}, context_size={}'.format(
            self.num_layers, self.hidden_state_size, self.context_size
        )
