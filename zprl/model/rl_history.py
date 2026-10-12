if __name__ == "__main__":
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import logging

import torch
import torch.nn as nn

from zprl.model.online import Actor, BatchedLayerNorm, BatchedLinear, BatchedSoftQNet, SquashedNormal, layer_init

logger = logging.getLogger(__name__)


def _parameters_m(module):
    return sum(p.numel() for p in module.parameters()) / 1e6


def _subsample_history(obs, a_exec, obs_valid, sub_ratio):
    # Anchor at the current observation and keep chronological oa pairs.
    start = (obs.shape[1] - 1) % sub_ratio
    return obs[:, start::sub_ratio], a_exec[:, start::sub_ratio], obs_valid[:, start::sub_ratio]


class HistoryEncoder(nn.Module):
    def __init__(self, obs_dim, exec_dim, history_length, d_model=256, nhead=4,
                 num_layers=4, dim_feedforward=1024, dropout=0.1, causal_mask=False,
                 debug=False, sub_ratio=1):
        super().__init__()
        assert history_length > 0
        assert isinstance(sub_ratio, int) and sub_ratio > 0
        assert d_model % nhead == 0
        self.obs_dim = obs_dim
        self.exec_dim = exec_dim
        self.history_length = history_length
        self.sub_ratio = sub_ratio
        self.nhead = nhead
        self.causal_mask = causal_mask
        self.debug = debug
        self.projection = nn.Linear(obs_dim + exec_dim, d_model)
        self.position = nn.Embedding((history_length - 1) // sub_ratio + 1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False)

    def forward(self, obs, a_exec, obs_valid):
        batch_size, history_length, obs_dim = obs.shape
        assert history_length == self.history_length and obs_dim == self.obs_dim
        assert a_exec.shape == (batch_size, history_length, self.exec_dim)
        assert obs_valid.shape == (batch_size, history_length)
        assert obs_valid.dtype == torch.bool
        if self.debug:
            assert torch.all(obs_valid[:, -1])
        assert obs.device == a_exec.device == obs_valid.device
        assert obs.dtype == a_exec.dtype

        obs, a_exec, obs_valid = _subsample_history(obs, a_exec, obs_valid, self.sub_ratio)
        history_length = obs.shape[1]
        # The action for the decision being evaluated is never part of its history.
        a_exec = torch.cat([a_exec[:, :-1], torch.zeros_like(a_exec[:, -1:])], dim=1)
        obs = torch.where(obs_valid.unsqueeze(-1), obs, torch.zeros_like(obs))
        a_exec = torch.where(obs_valid.unsqueeze(-1), a_exec, torch.zeros_like(a_exec))
        x = self.projection(torch.cat([obs, a_exec], dim=-1))
        positions = torch.arange(history_length, device=obs.device)
        x = x + self.position(positions).unsqueeze(0)

        # True means blocked. Padded queries may attend valid keys so no row is
        # fully masked, including when causal attention meets left padding.
        mask = (~obs_valid)[:, None, :].expand(batch_size, history_length, history_length)
        if self.causal_mask:
            future = torch.ones(history_length, history_length, device=obs.device,
                                dtype=torch.bool).triu(1)
            mask = mask | (obs_valid[:, :, None] & future)
        mask = mask.repeat_interleave(self.nhead, dim=0)
        return self.transformer(x, mask=mask)[:, -1]


class HistoryActor(nn.Module):
    def __init__(self, obs_dim, exec_dim, action_dim, history_length, hidden_dim=256,
                 d_model=256, nhead=4, num_layers=4, dim_feedforward=1024,
                 dropout=0.1, causal_mask=False, log_std_min=-10.0, log_std_max=2.0,
                 debug=False, sub_ratio=1):
        super().__init__()
        self.encoder = HistoryEncoder(obs_dim, exec_dim, history_length, d_model,
                                      nhead, num_layers, dim_feedforward, dropout,
                                      causal_mask, debug, sub_ratio)
        self.net = nn.Sequential(
            layer_init(nn.Linear(d_model + exec_dim, hidden_dim)),
            nn.GELU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.GELU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.GELU(),
        )
        self.mean = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.log_std = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        logger.info(
            "HistoryActor parameters: encoder=%.4f M, net=%.4f M",
            _parameters_m(self.encoder), _parameters_m(self.net))

    def forward(self, obs, a_exec, obs_valid, a_base):
        h = self.encoder(obs, a_exec, obs_valid)
        assert a_base.shape == (obs.shape[0], self.encoder.exec_dim)
        assert a_base.device == h.device and a_base.dtype == h.dtype
        x = self.net(torch.cat([h, a_base], dim=-1))
        mean = self.mean(x)
        logstd = torch.tanh(self.log_std(x))
        logstd = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (logstd + 1)
        return mean, logstd

    def get_eval_action(self, obs, a_exec, obs_valid, a_base):
        mean, _ = self.forward(obs, a_exec, obs_valid, a_base)
        return torch.tanh(mean)

    def get_action(self, obs, a_exec, obs_valid, a_base):
        mean, logstd = self.forward(obs, a_exec, obs_valid, a_base)
        dist = SquashedNormal(mean, logstd.exp(), validate_args=self.encoder.debug)
        sample = dist.rsample()
        log_prob = dist.log_prob(sample).sum(dim=-1, keepdim=True)
        if self.encoder.debug:
            assert torch.all(torch.isfinite(sample))
            assert torch.all(torch.isfinite(log_prob))
        return {
            'sample': sample,
            'mean': dist.mean,
            'log_prob': log_prob,
            'log_std': logstd,
        }


class HistoryQEnsemble(nn.Module):
    def __init__(self, obs_dim, exec_dim, num_qs, history_length, hidden_dim=256,
                 d_model=256, nhead=4, num_layers=4, dim_feedforward=1024,
                 dropout=0.1, causal_mask=False, debug=False,
                 critic_action_input='exec', sub_ratio=1):
        super().__init__()
        assert num_qs > 0
        if critic_action_input not in ('exec', 'base_res'):
            raise ValueError(f"Invalid critic_action_input: {critic_action_input}")
        self.critic_action_input = critic_action_input
        self.num_qs = num_qs
        self.encoder = HistoryEncoder(obs_dim, exec_dim, history_length, d_model,
                                      nhead, num_layers, dim_feedforward, dropout,
                                      causal_mask, debug, sub_ratio)
        critic_action_dim = exec_dim * (2 if critic_action_input == 'base_res' else 1)
        self.net = nn.Sequential(
            BatchedLinear(num_qs, d_model + critic_action_dim, hidden_dim),
            BatchedLayerNorm(num_qs, hidden_dim),
            nn.Tanh(),
            BatchedLinear(num_qs, hidden_dim, hidden_dim),
            BatchedLayerNorm(num_qs, hidden_dim),
            nn.Tanh(),
            BatchedLinear(num_qs, hidden_dim, hidden_dim),
            BatchedLayerNorm(num_qs, hidden_dim),
            nn.Tanh(),
            BatchedLinear(num_qs, hidden_dim, 1),
        )
        logger.info(
            "HistoryQEnsemble parameters: encoder=%.4f M, net=%.4f M, critic_action_input=%s",
            _parameters_m(self.encoder), _parameters_m(self.net), critic_action_input)

    def forward(self, obs, a_exec, obs_valid, candidate_action, candidate_res=None):
        h = self.encoder(obs, a_exec, obs_valid)
        assert candidate_action.shape == (obs.shape[0], self.encoder.exec_dim)
        assert candidate_action.device == h.device and candidate_action.dtype == h.dtype
        if self.critic_action_input == 'base_res':
            assert candidate_res is not None
            assert candidate_res.shape == candidate_action.shape
            assert candidate_res.device == h.device and candidate_res.dtype == h.dtype
            x = torch.cat([h, candidate_action, candidate_res], dim=-1)
        else:
            assert candidate_res is None
            x = torch.cat([h, candidate_action], dim=-1)
        x = x.unsqueeze(0).expand(self.num_qs, -1, -1)
        return self.net(x)


class FlattenHistoryEncoder(nn.Module):
    def __init__(self, obs_dim, exec_dim, history_length, debug=False, sub_ratio=1):
        super().__init__()
        assert history_length > 0
        assert isinstance(sub_ratio, int) and sub_ratio > 0
        self.obs_dim = obs_dim
        self.exec_dim = exec_dim
        self.history_length = history_length
        self.sub_ratio = sub_ratio
        self.debug = debug
        sampled_length = (history_length - 1) // sub_ratio + 1
        self.output_dim = sampled_length * obs_dim + (sampled_length - 1) * exec_dim

    def forward(self, obs, a_exec, obs_valid):
        batch_size = obs.shape[0]
        assert obs.shape == (batch_size, self.history_length, self.obs_dim)
        assert a_exec.shape == (batch_size, self.history_length, self.exec_dim)
        assert obs_valid.shape == (batch_size, self.history_length)
        assert obs_valid.dtype == torch.bool
        assert obs.device == a_exec.device == obs_valid.device
        assert obs.dtype == a_exec.dtype
        if self.debug:
            assert torch.all(obs_valid[:, -1])
        obs, a_exec, obs_valid = _subsample_history(obs, a_exec, obs_valid, self.sub_ratio)
        obs = torch.where(obs_valid.unsqueeze(-1), obs, torch.zeros_like(obs))
        past_exec = torch.where(obs_valid[:, :-1, None], a_exec[:, :-1],
                                torch.zeros_like(a_exec[:, :-1]))
        past = torch.cat([obs[:, :-1], past_exec], dim=-1).flatten(start_dim=1)
        return torch.cat([past, obs[:, -1]], dim=-1)


class HistoryMLPActor(HistoryActor):
    def __init__(self, obs_dim, exec_dim, action_dim, history_length, hidden_dim=256,
                 log_std_min=-10.0, log_std_max=2.0, debug=False, sub_ratio=1):
        nn.Module.__init__(self)
        self.encoder = FlattenHistoryEncoder(obs_dim, exec_dim, history_length, debug, sub_ratio)
        actor = Actor(self.encoder.output_dim + exec_dim, action_dim, hidden_dim,
                      log_std_min, log_std_max)
        self.net = actor.net
        self.mean = actor.mean
        self.log_std = actor.log_std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        logger.info(
            "HistoryMLPActor parameters: %.4f M", _parameters_m(self)
        )


class HistoryMLPQEnsemble(HistoryQEnsemble):
    def __init__(self, obs_dim, exec_dim, num_qs, history_length, hidden_dim=256,
                 debug=False, critic_action_input='exec', sub_ratio=1):
        nn.Module.__init__(self)
        assert num_qs > 0
        if critic_action_input not in ('exec', 'base_res'):
            raise ValueError(f"Invalid critic_action_input: {critic_action_input}")
        self.critic_action_input = critic_action_input
        self.num_qs = num_qs
        self.encoder = FlattenHistoryEncoder(obs_dim, exec_dim, history_length, debug, sub_ratio)
        critic_action_dim = exec_dim * (2 if critic_action_input == 'base_res' else 1)
        self.net = BatchedSoftQNet(self.encoder.output_dim, critic_action_dim, num_qs, hidden_dim).net
        logger.info(
            "HistoryMLPQEnsemble parameters: %.4f M", _parameters_m(self)
        )
        logger.info("Critic action input: %s", critic_action_input)


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    common = dict(
        obs_dim=512,
        exec_dim=20,
        history_length=4,
        hidden_dim=512,
        d_model=256,
        nhead=4,
        num_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
        causal_mask=False,
    )
    HistoryActor(**common, action_dim=common['exec_dim'])
    HistoryQEnsemble(**common, num_qs=10)


if __name__ == "__main__":
    main()
