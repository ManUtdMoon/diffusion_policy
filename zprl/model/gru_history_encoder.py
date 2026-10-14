import torch
import torch.nn as nn


class GRUHistoryEncoder(nn.Module):
    def __init__(self, obs_dim, exec_dim, history_length, gru_input_dim=128,
                 gru_hidden_dim=128, gru_num_layers=2, sub_ratio=2, debug=False):
        super().__init__()
        for value in (obs_dim, exec_dim, history_length, gru_input_dim,
                      gru_hidden_dim, gru_num_layers, sub_ratio):
            assert type(value) is int and value > 0
        self.obs_dim = obs_dim
        self.exec_dim = exec_dim
        self.history_length = history_length
        self.sub_ratio = sub_ratio
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_num_layers = gru_num_layers
        self.output_dim = gru_hidden_dim
        self.debug = debug
        self.input_projection = nn.Linear(obs_dim + exec_dim, gru_input_dim)
        self.cells = nn.ModuleList([
            nn.GRUCell(gru_input_dim if layer == 0 else gru_hidden_dim, gru_hidden_dim)
            for layer in range(gru_num_layers)])
        self.output_projection = nn.Linear(gru_hidden_dim, gru_hidden_dim)
        for projection in (self.input_projection, self.output_projection):
            nn.init.xavier_uniform_(projection.weight, gain=1.0)
            nn.init.zeros_(projection.bias)

    def forward(self, obs, a_exec, obs_valid):
        batch_size = obs.shape[0]
        assert obs.shape == (batch_size, self.history_length, self.obs_dim)
        assert a_exec.shape == (batch_size, self.history_length, self.exec_dim)
        assert obs_valid.shape == (batch_size, self.history_length)
        assert obs_valid.dtype == torch.bool
        assert obs.device == a_exec.device == obs_valid.device
        assert obs.dtype == a_exec.dtype and obs.is_floating_point()
        if self.debug:
            assert torch.all(obs_valid[:, -1])
        start = (self.history_length - 1) % self.sub_ratio
        obs, a_exec, valid = (obs[:, start::self.sub_ratio],
                             a_exec[:, start::self.sub_ratio],
                             obs_valid[:, start::self.sub_ratio])
        a_exec = torch.cat([a_exec[:, :-1], torch.zeros_like(a_exec[:, -1:])], dim=1)
        obs = torch.where(valid[..., None], obs, torch.zeros_like(obs))
        a_exec = torch.where(valid[..., None], a_exec, torch.zeros_like(a_exec))
        x = self.input_projection(torch.cat([obs, a_exec], dim=-1))
        states = [x.new_zeros(batch_size, self.gru_hidden_dim)
                  for _ in range(self.gru_num_layers)]
        for index in range(x.shape[1]):
            layer_input = x[:, index]
            next_states = []
            for layer, cell in enumerate(self.cells):
                candidate = cell(layer_input, states[layer])
                updated = torch.where(valid[:, index, None], candidate, states[layer])
                next_states.append(updated)
                layer_input = updated
            states = next_states
        return torch.tanh(self.output_projection(states[-1]))
