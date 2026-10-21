"""Instrument one real train.py job without changing its training loop or data."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import resource
import runpy
import sys
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('resrl', 'dsrl'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, training_args = parser.parse_known_args()

    import isaacgym
    import numpy as np
    import torch
    from zprl.common.oneleg_episode_replay_buffer import OneLegEpisodeCollector

    if args.method == 'resrl':
        import zprl.workspace.train_online_reactive_history_oneleg_workspace as module
        workspace_class = module.TrainOnlineReactiveHistoryOneLegWorkspace
        policy_key = 'res_policy'
    else:
        import zprl.workspace.train_online_reactive_history_noise_oneleg_workspace as module
        workspace_class = module.TrainOnlineReactiveHistoryNoiseOneLegWorkspace
        policy_key = 'noise_policy'

    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 2)))
    timings = defaultdict(lambda: dict(seconds=0., calls=0))
    state = dict(episodes=[], updates=[], effective_decisions=0, reset_calls=0)
    result = {}

    def fingerprint(model):
        digest = hashlib.sha256()
        for key, value in model.state_dict().items():
            digest.update(key.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def timed(name, function):
        def call(*a, **kw):
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                return function(*a, **kw)
            finally:
                torch.cuda.synchronize()
                timings[name]['seconds'] += time.perf_counter() - started
                timings[name]['calls'] += 1
        return call

    load_base, make_env = module.load_oneleg_base, module.OneLegOnlineEnv
    run, update = workspace_class.run, workspace_class._update
    discard = OneLegEpisodeCollector.discard_pending
    started = time.perf_counter()
    with ExitStack() as stack:
        def observed_load(*a, **kw):
            base, cfg = load_base(*a, **kw)
            state['base_before'] = fingerprint(base)
            encode = base.obs_encoder.forward

            def full_batch_encode(obs):
                assert all(value.shape[0] == state['env'].num_envs for value in obs.values())
                return encode(obs)

            stack.enter_context(patch.object(base.obs_encoder, 'forward', timed('encode', full_batch_encode)))
            decoder = 'conditional_sample' if args.method == 'resrl' else 'conditional_predict_from_noise'
            stack.enter_context(patch.object(base, decoder, timed('decode', getattr(base, decoder))))
            return base, cfg

        def observed_env(*a, **kw):
            env = state['env'] = make_env(*a, **kw)
            state['clamp_counts'] = torch.zeros(2, device=env.device, dtype=torch.long)
            reset, step, native_step = env.reset, env.step, env.env.step

            def observed_reset(*a, **kw):
                assert not a and not kw  # Training resets must not reseed layouts.
                state['reset_calls'] += 1
                return reset()

            def observed_step(action):
                assert action.shape == (env.num_envs, env.n_action_steps, 10)
                obs, rewards, dones, infos = step(action)
                state['effective_decisions'] += sum(info['transition_valid'] for info in infos)
                state['episodes'].extend(dict(env.results[row]) for row in dones.nonzero().flatten().tolist())
                return obs, rewards, dones, infos

            def observed_native_step(action):
                assert action.shape == (env.num_envs, 10)
                active = env.active.clone()
                assert torch.equal(action[~active], env.hold_action.expand(int((~active).sum()), -1))
                output = native_step(action)
                state['clamp_counts'][0] += (action[active].abs() > 1).sum()
                state['clamp_counts'][1] += active.sum() * 10
                return output

            stack.enter_context(patch.object(env, 'reset', timed('reset', observed_reset)))
            stack.enter_context(patch.object(env, 'step', timed('collect', observed_step)))
            stack.enter_context(patch.object(env.env, 'step', observed_native_step))
            return env

        def observed_discard(collector):
            state['uncommitted_at_stop'] = collector.pending_transitions
            discard(collector)

        def observed_update(workspace, cfg, optimizers):
            policy = getattr(workspace, policy_key)
            if not state['updates']:
                state['before'] = {key: fingerprint(getattr(policy, key)) for key in ('actor', 'qs', 'q_targets')}
                state['alpha_before'] = policy.log_alpha.item()
                for key in ('actor_optimizer', 'q_optimizer'):
                    assert optimizers[key].param_groups[1]['lr'] == cfg.training.enc_lr
                optimized = {id(p) for optimizer in optimizers.values()
                             for group in optimizer.param_groups for p in group['params']}
                assert not optimized.intersection(id(p) for p in workspace.base_policy.parameters())
                assert not optimized.intersection(id(p) for p in policy.q_targets.parameters())
            metrics = update(workspace, cfg, optimizers)
            assert all(np.isfinite(value) for value in metrics.values())
            state['updates'].append(dict(decision=workspace.global_step, global_update=workspace.global_update))
            return metrics

        def observed_run(workspace):
            torch.cuda.set_device(workspace.cfg.training.device)
            run(workspace)
            cfg, env = workspace.cfg, state['env']
            policy = getattr(workspace, policy_key)
            tr = cfg.n_rl_steps if args.method == 'resrl' else cfg.n_action_steps
            assert state['updates'] and state['episodes']
            assert timings['encode']['calls'] > 0 and timings['decode']['calls'] > 0
            assert fingerprint(workspace.base_policy) == state['base_before']
            assert all(not m.training for m in workspace.base_policy.modules())
            assert all(not p.requires_grad and p.grad is None for p in workspace.base_policy.parameters())
            assert all(fingerprint(getattr(policy, key)) != state['before'][key] for key in state['before'])
            assert policy.log_alpha.item() != state['alpha_before']
            assert not policy.q_targets.training
            assert all(not p.requires_grad and p.grad is None for p in policy.q_targets.parameters())
            assert workspace.global_step == state['effective_decisions']
            assert workspace.global_step == workspace.replay.committed_transitions + state['uncommitted_at_stop']
            assert workspace.replay.committed_episodes == len(state['episodes']) == workspace.num_episodes
            assert workspace.collector.pending_transitions == 0 and env.env is None
            assert 0 <= workspace.global_step - cfg.training.num_steps < env.num_envs
            assert 0 <= state['updates'][0]['decision'] - cfg.training.learning_start < env.num_envs
            assert workspace.global_update == len(state['updates']) * int(cfg.training.training_freq * cfg.training.utd)
            assert workspace.last_log['info/global_step'] == workspace.global_step * tr
            assert all(np.isfinite(value) for value in workspace.last_log.values())
            assert not list(Path(workspace.output_dir).glob('**/*.ckpt'))
            clipped, components = state['clamp_counts'].tolist()
            assert components == env.solid_steps * 10
            parameters = dict(n_envs=env.num_envs, max_steps=env.max_steps,
                history_length=policy.history_length, sub_ratio=policy.sub_ratio,
                ta=cfg.n_action_steps, tr=tr, tn=cfg.get('n_noise_steps'),
                hidden_dim=cfg[policy_key].hidden_dim, num_qs=policy.num_qs,
                batch_size=cfg.training.batch_size, training_freq=cfg.training.training_freq,
                utd=cfg.training.utd, n_step=cfg.training.n_step,
                learning_start=cfg.training.learning_start, num_steps=cfg.training.num_steps,
                buffer_size=cfg.training.buffer_size, seed=cfg.training.seed,
                compile=cfg.training.compile, base_ckpt=cfg.online_task.base_ckpt)
            result.update(method=args.method, parameters=parameters,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                metrics=workspace.last_log, episodes=state['episodes'], updates=state['updates'],
                committed_transitions=workspace.replay.committed_transitions,
                uncommitted_at_stop=state['uncommitted_at_stop'], reset_calls=state['reset_calls'],
                success_count=sum(episode['success'] for episode in state['episodes']),
                timeout_count=sum(episode['termination_reason'] == 'timeout' for episode in state['episodes']),
                clamp_components=clipped, active_action_components=components,
                clamp_ratio=clipped / max(components, 1),
                frozen_base_sha256=state['base_before'], q_actor_alpha_target_changed=True,
                full_batch_inference=True, reset_without_seed=True, injected_events=False,
                close_ok=True, automatic_checkpoint=False)

        stack.enter_context(patch.object(module, 'load_oneleg_base', observed_load))
        stack.enter_context(patch.object(module, 'OneLegOnlineEnv', observed_env))
        stack.enter_context(patch.object(OneLegEpisodeCollector, 'discard_pending', observed_discard))
        stack.enter_context(patch.object(workspace_class, '_update', timed('update', observed_update)))
        stack.enter_context(patch.object(workspace_class, 'run', observed_run))
        sys.argv = ['train.py', *training_args]
        runpy.run_path('train.py', run_name='__main__')
    if result:  # Hydra --cfg only renders the job; no simulation or summary.
        result.update(timings=dict(timings), wall_seconds=time.perf_counter() - started,
            rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            torch_peak_bytes=torch.cuda.max_memory_allocated())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
