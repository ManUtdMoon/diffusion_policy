"""Fresh-process GPU checks for OneLeg collection and both train.py entries."""

import argparse
import gc
import hashlib
import json
import os
import pathlib
import resource
import runpy
import sys
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-gpu', action='store_true')
    parser.add_argument('--mode', choices=('resrl', 'dsrl', 'train', 'train-noise'), required=True)
    parser.add_argument('--n-envs', choices=(2, 20), type=int, required=True)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--max-steps', type=int, default=18, help='Collection episode limit; training smoke uses 16')
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--snapshot', type=pathlib.Path, help='Manually serialize an evaluation snapshot after training')
    args = parser.parse_args()
    if not args.run_gpu:
        parser.error('Pass --run-gpu in a fresh oneleg process')

    import isaacgym
    import gymnasium
    import numpy as np
    import torch
    from zprl.common.oneleg_online import load_oneleg_base
    from zprl.common.oneleg_episode_replay_buffer import OneLegEpisodeReplayBuffer, OneLegEpisodeCollector
    from zprl.env.furniturebench.oneleg_env import OneLegEnv
    from zprl.env.furniturebench.oneleg_online_env import OneLegOnlineEnv
    from zprl.policy.oneleg_history_policy import OneLegHistorySumPolicy, OneLegHistoryNoiseSumPolicy
    from zprl.policy.reactive_history_policy import HistoryResiduePolicy
    from zprl.policy.reactive_history_noise_policy import HistoryNoisePolicy

    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1,0,2,3'
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    np.random.seed(42)
    started = time.perf_counter()
    checkpoint = 'data/upload/oneleg/checkpoints/epoch_0055-mse_0.020349.ckpt'
    result = dict(mode=args.mode, n_envs=args.n_envs, compile=args.compile,
                  checkpoint=checkpoint, interface='full_batch', policy_seed=42)

    def fingerprint(module):
        digest = hashlib.sha256()
        for key, value in module.state_dict().items():
            digest.update(key.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def check_base(base, before):
        assert fingerprint(base) == before
        assert all(not m.training for m in base.modules())
        assert all(not p.requires_grad and p.grad is None for p in base.parameters())
        result['frozen_base_sha256'] = before

    if args.mode in ('train', 'train-noise'):
        if args.mode == 'train-noise':
            import zprl.workspace.train_online_reactive_history_noise_oneleg_workspace as module
            workspace_class = module.TrainOnlineReactiveHistoryNoiseOneLegWorkspace
            policy_key = 'noise_policy'
        else:
            import zprl.workspace.train_online_reactive_history_oneleg_workspace as module
            workspace_class = module.TrainOnlineReactiveHistoryOneLegWorkspace
            policy_key = 'res_policy'
        original_run = workspace_class.run
        original_update = workspace_class._update
        state = {}

        def observed_load(*a, **kw):
            base, cfg = load_oneleg_base(*a, **kw)
            state['base_before'] = fingerprint(base)
            return base, cfg

        def observed_env(*a, **kw):
            env = OneLegOnlineEnv(*a, **kw)
            original_reset = env.reset

            def observed_reset(*reset_args, **reset_kwargs):
                assert not reset_args and not reset_kwargs
                obs = original_reset()
                state.setdefault('layouts', []).append(
                    env.env.env.unwrapped.root_pos.detach().cpu().clone())
                return obs

            env.reset = observed_reset
            return env

        def observed_update(workspace, cfg, optimizers):
            if 'policy_before' not in state:
                policy = getattr(workspace, policy_key)
                state['policy_before'] = {name: fingerprint(getattr(policy, name))
                                          for name in ('actor', 'qs', 'q_targets')}
                state['alpha_before'] = policy.log_alpha.item()
                state['encoder_lrs'] = {name: optimizer.param_groups[-1]['lr']
                    for name, optimizer in optimizers.items() if name != 'alpha_optimizer'}
            metrics = original_update(workspace, cfg, optimizers)
            assert all(np.isfinite(value) for value in metrics.values())
            return metrics

        def observed_run(workspace):
            original_run(workspace)
            policy = getattr(workspace, policy_key)
            check_base(workspace.base_policy, state['base_before'])
            assert workspace.global_update > 1
            assert all(fingerprint(getattr(policy, name)) != state['policy_before'][name]
                       for name in ('actor', 'qs', 'q_targets'))
            assert policy.log_alpha.item() != state['alpha_before']
            assert not policy.q_targets.training
            assert all(not p.requires_grad and p.grad is None for p in policy.q_targets.parameters())
            assert state['encoder_lrs'] == {'actor_optimizer': 1e-5, 'q_optimizer': 1e-5}
            assert workspace.global_step >= workspace.replay.committed_transitions
            assert workspace.num_episodes == workspace.replay.committed_episodes
            assert workspace.collector.pending_transitions == 0
            assert workspace.global_step - workspace.cfg.training.num_steps < args.n_envs
            assert workspace.last_log['info/global_step'] == workspace.global_step * 4
            assert workspace.last_log['info/chunk_step'] == workspace.global_step
            assert 'info/res_ratio' not in workspace.last_log
            metrics = workspace.last_log
            assert metrics['info/idle_ratio'] == metrics['info/idle_steps'] / (
                metrics['info/solid_steps'] + metrics['info/idle_steps'])
            assert len(state['layouts']) >= 2
            assert (state['layouts'][0] != state['layouts'][1]).flatten(1).any(1).all()
            result.update(metrics=workspace.last_log, encoder_lrs=state['encoder_lrs'],
                          q_actor_alpha_target_changed=True, reset_without_seed=True,
                          layouts_changed_for_all_rows=True,
                          injected_success=False)
            if args.snapshot:
                import dill
                snapshot = workspace.get_snapshot()
                assert set(snapshot) == {'cfg', 'method', policy_key, 'base_ckpt', 'base_sha256',
                                         'global_step', 'global_update', 'num_episodes'}
                assert not any('base_policy' in name for name in snapshot[policy_key])
                torch.save(snapshot, args.snapshot, pickle_module=dill)
                result['manual_snapshot'] = str(args.snapshot)
                result['steering_sha256'] = fingerprint(policy)

        budget = 26 if args.n_envs == 2 else 140
        output_dir = args.output.with_suffix('').as_posix() + '-run'
        sys.argv = ['train.py', '--config-name=train_online_reactive_history_oneleg_workspace',
            f'online_task.n_envs={args.n_envs}', 'online_task.max_steps=16',
            'n_action_steps=8', 'n_rl_steps=4', 'res_policy.history_length=5', 'res_policy.sub_ratio=2',
            'res_policy.hidden_dim=32', 'res_policy.gru_hidden_dim=16', 'res_policy.gru_input_dim=16',
            'res_policy.num_qs=2', f'training.num_steps={budget}', f'training.learning_start={args.n_envs * 4 + 1}',
            'training.training_freq=7', 'training.utd=0.5', 'training.batch_size=4',
            'training.buffer_size=256', 'training.n_step=3', 'training.log_every=7',
            f'training.compile={str(args.compile).lower()}', 'training.debug=True',
            'logging.mode=offline', f'hydra.run.dir={output_dir}']
        if args.mode == 'train-noise':
            sys.argv = [arg.replace('train_online_reactive_history_oneleg_workspace',
                                   'train_online_reactive_history_noise_oneleg_workspace')
                        .replace('res_policy.', 'noise_policy.')
                        .replace('n_action_steps=8', 'n_action_steps=4')
                        for arg in sys.argv if not arg.startswith('n_rl_steps=')]
            sys.argv.append('n_noise_steps=2')
        with patch.object(module, 'load_oneleg_base', side_effect=observed_load), \
             patch.object(module, 'OneLegOnlineEnv', side_effect=observed_env), \
             patch.object(workspace_class, '_update', observed_update), \
             patch.object(workspace_class, 'run', observed_run):
            runpy.run_path('train.py', run_name='__main__')
        assert not list(pathlib.Path(output_dir).glob('**/*.ckpt'))
        result['train_output'] = output_dir
    else:
        base, _ = load_oneleg_base(checkpoint, device='cuda:0')
        before = fingerprint(base)

        class InjectedSuccess:
            """Only the success signal is injected; all actions/observations are native."""
            def __init__(self, adapter):
                self.adapter = adapter
                self.num_envs = adapter.num_envs
                self.device = adapter.device
                self.steps = 0
                self.natural_success = set()
                self.hold_checks = 0

            def reset(self, seed=None):
                self.steps = 0
                return self.adapter.reset(seed=seed)

            def step(self, action):
                if self.steps >= 3:
                    torch.testing.assert_close(action[0], env.hold_action, atol=0, rtol=0)
                    self.hold_checks += 1
                obs, reward, done, info = self.adapter.step(action)
                self.steps += 1
                self.natural_success.update(self.adapter.is_success().nonzero().flatten().tolist())
                return obs, reward, done, info

            def is_success(self):
                success = self.adapter.is_success().clone()
                if self.steps >= 3:
                    success[0] = True
                return success

            def close(self):
                self.adapter.close()

        adapter = InjectedSuccess(OneLegEnv(num_envs=args.n_envs, max_env_steps=args.max_steps))
        env = OneLegOnlineEnv(4, max_steps=args.max_steps, env=adapter)
        common = dict(obs_dim=base.obs_feature_dim, history_length=5, sub_ratio=2,
                      hidden_dim=32, gru_input_dim=16, gru_hidden_dim=16, gru_num_layers=1,
                      num_qs=2, num_subset=2, gamma=.9881)
        if args.mode == 'resrl':
            policy = HistoryResiduePolicy(action_dim=40, **common).to('cuda:0')
            session = OneLegHistorySumPolicy(.05, base.obs_feature_dim, 0, 10, 8, base, policy, 4)
        else:
            policy = HistoryNoisePolicy(action_dim=20, exec_dim=40, **common).to('cuda:0')
            session = OneLegHistoryNoiseSumPolicy(1.5, base.obs_feature_dim, 0, 10, 4, 16, 1, base, policy, 2)
        replay = OneLegEpisodeReplayBuffer(max(256, args.n_envs * ((args.max_steps + 3) // 4)),
            gymnasium.spaces.Box(-np.inf, np.inf, (base.obs_feature_dim,), dtype=np.float32),
            gymnasium.spaces.Box(-np.inf, np.inf, (policy.action_dim,), dtype=np.float32),
            method=args.mode, device='cuda:0', n_steps=3, gamma=policy.gamma,
            history_length=5, action_step_dim=10,
            **({'base_action_dim': 40} if args.mode == 'resrl' else {'exec_dim': 40}))
        collector = OneLegEpisodeCollector(replay, args.n_envs)
        rows = []
        traces = [[] for _ in range(args.n_envs)]
        completed_rows = []
        global_step = executed_steps = 0
        try:
            obs = env.reset(seed=42)
            collector.begin_batch()
            session.reset(obs)
            while not env.batch_done:
                history = session.history.get()
                action = session.predict_action(obs, deterministic=True)['action']
                rows.append(action.shape[0])
                assert action.shape == (args.n_envs, 4, 10)
                obs, rewards, dones, infos = env.step(action)
                transition = session.on_env_step(obs, dones, infos)
                endpoint = session.history.get()
                for row, info in enumerate(infos):
                    if info['transition_valid']:
                        traces[row].append((tuple(x[row].clone() for x in history),
                                            tuple(x[row].clone() for x in endpoint),
                                            transition['next_a_base'][row].clone() if args.mode == 'resrl' else None))
                global_step += sum(info['transition_valid'] for info in infos)
                executed_steps += sum(info['executed_action_steps'] for info in infos)
                completed_rows.extend(dones.nonzero().flatten().tolist())
                collector.collect(transition, rewards, dones, infos)
            assert replay.committed_transitions == global_step
            assert replay.committed_episodes == args.n_envs
            storage = replay.storage
            for row, trace in enumerate(traces):
                indices = np.flatnonzero(storage.episode_ids[:storage.pos, 0] == completed_rows.index(row))
                for start, index in enumerate(indices):
                    batch = storage._get_samples(np.array([index]))
                    end = min(start + 3, len(trace)) - 1
                    for actual, expected in zip((batch.obs, batch.a_exec, batch.obs_valid, batch.exec_steps), trace[start][0]):
                        torch.testing.assert_close(actual[0], expected)
                    for actual, expected in zip((batch.next_obs, batch.next_a_exec, batch.next_obs_valid, batch.next_exec_steps), trace[end][1]):
                        torch.testing.assert_close(actual[0], expected)
                    if args.mode == 'resrl':
                        torch.testing.assert_close(batch.next_a_base[0], trace[end][2])
            assert adapter.hold_checks > 0
            assert env.solid_steps == executed_steps
            assert (env.solid_steps + env.idle_steps) // args.n_envs <= args.max_steps
            for episode in env.results:
                if episode['termination_reason'] == 'timeout':
                    assert episode['episode_steps'] == args.max_steps
            check_base(base, before)
            result.update(global_step=global_step, solid_steps=env.solid_steps,
                idle_steps=env.idle_steps, idle_ratio=env.idle_steps / (env.solid_steps + env.idle_steps),
                max_steps=args.max_steps,
                committed_transitions=replay.committed_transitions,
                committed_episodes=replay.committed_episodes, policy_rows=rows,
                natural_successes=sorted(adapter.natural_success), injected_success={'env_index': 0, 'step': 3},
                native_success_episodes=[item['env_index'] for item in env.results
                    if item['success'] and item['env_index'] != 0],
                episode_results=env.results, online_replay_history_equal=True)
        finally:
            env.close()
        assert adapter.adapter.env is None
    gc.collect()
    result.update(close_ok=True, wall_seconds=time.perf_counter() - started,
        rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        torch_peak_bytes=torch.cuda.max_memory_allocated(0))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
