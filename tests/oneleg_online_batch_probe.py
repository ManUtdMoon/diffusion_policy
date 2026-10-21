"""Opt-in native GPU acceptance: python -m tests.oneleg_online_batch_probe --help."""

import argparse
import gc
import hashlib
import json
import os
import pathlib
import resource
import time
import weakref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-gpu', action='store_true', help='Explicitly opt in to Isaac Gym')
    parser.add_argument('--n-envs', type=int, choices=(2, 20), required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--checkpoint', default='data/upload/oneleg/checkpoints/epoch_0055-mse_0.020349.ckpt')
    args = parser.parse_args()
    if not args.run_gpu:
        parser.error('Pass --run-gpu in a fresh oneleg process')

    import isaacgym
    import torch
    from furniture_bench.sim_config import sim_config
    from zprl.common.oneleg_online import load_oneleg_base
    from zprl.env.furniturebench.oneleg_env import OneLegEnv, pack_oneleg_observation
    from zprl.env.furniturebench.oneleg_online_env import OneLegOnlineEnv

    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1,0,2,3'
    torch.set_num_threads(2)
    torch.cuda.set_device(0)

    def memory():
        free, total = torch.cuda.mem_get_info(0)
        return dict(rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            cuda_device_used_bytes=total - free, torch_allocated_bytes=torch.cuda.memory_allocated(0),
            torch_reserved_bytes=torch.cuda.memory_reserved(0),
            torch_peak_bytes=torch.cuda.max_memory_allocated(0))

    class ObservedAdapter:
        """Observe unchanged native calls; inject only labelled test signals."""

        def __init__(self, adapter):
            self.adapter = adapter
            self.num_envs = adapter.num_envs
            self.device = adapter.device
            self.owner = None
            self.reset_calls = 0
            self.step_calls = 0
            self.batch_steps = 0
            self.inject_success = {}
            self.fail_at = None
            self.natural_success = set()
            self.hold_rows_checked = 0
            self.reset_costs = []
            self.last_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        def reset(self, seed=None):
            native = self.adapter.env.unwrapped
            start_sim = native.isaac_gym.get_sim_time(native.sim)
            start_wall = time.perf_counter()
            obs = self.adapter.reset(seed=seed)
            elapsed = time.perf_counter() - start_wall
            self.reset_costs.append(dict(seed=seed, wall_seconds=elapsed,
                sim_seconds=native.isaac_gym.get_sim_time(native.sim) - start_sim))
            self.reset_calls += 1
            self.batch_steps = 0
            self.natural_success = set()
            self.last_success.zero_()
            assert not self.adapter.is_success().any()
            assert not native.env_steps.any()
            # Reading the existing observation does not advance physics.
            packed = pack_oneleg_observation(native.get_observation())
            for key in obs:
                torch.testing.assert_close(obs[key], packed[key], rtol=0, atol=0)
            return obs

        def step(self, action):
            if self.batch_steps + 1 == self.fail_at:
                raise RuntimeError('injected probe failure; native simulator is healthy')
            inactive = ~self.owner.active
            native = self.adapter.env.unwrapped
            previous_grasp = native.last_grasp.clone()
            if inactive.any():
                torch.testing.assert_close(action[inactive],
                    self.owner.hold_action.expand(int(inactive.sum()), -1), rtol=0, atol=0)
            obs, reward, done, info = self.adapter.step(action)
            self.step_calls += 1
            self.batch_steps += 1
            self.hold_rows_checked += int(inactive.sum())
            torch.testing.assert_close(native.last_grasp[inactive], previous_grasp[inactive], rtol=0, atol=0)
            self.last_success = self.adapter.is_success()
            natural = self.last_success | (reward.flatten() == 1)
            self.natural_success.update(natural.nonzero().flatten().tolist())
            if self.inject_success:
                reward = reward.clone()
                for index, step in self.inject_success.items():
                    if self.batch_steps >= step:
                        reward[index] = 1
            return obs, reward, done, info

        def is_success(self):
            success = self.last_success.clone()
            for index, step in self.inject_success.items():
                if self.batch_steps >= step:
                    success[index] = True
            return success

        def close(self):
            self.adapter.close()

    report = dict(interface='full_batch', n_envs=args.n_envs, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        compute_device_id=0, graphics_device_id=0, checkpoint=args.checkpoint,
        torch_version=torch.__version__, memory_before=memory(), stages=[])
    start_wall = time.perf_counter()
    adapter = OneLegEnv(num_envs=args.n_envs, max_env_steps=700,
                       compute_device_id=0, graphics_device_id=0)
    native = adapter.env.unwrapped
    native_ref = weakref.ref(native)
    observed = ObservedAdapter(adapter)
    env = OneLegOnlineEnv(env=observed)
    observed.owner = env
    base = None

    def reset(seed):
        assert env.batch_done
        before = observed.reset_calls
        obs = env.reset(seed=seed)
        assert observed.reset_calls == before + 1
        assert env.active.all()
        assert obs['robot_state'].shape == (args.n_envs, 1, 16)
        for key in ('color_image1', 'color_image2'):
            assert obs[key].shape == (args.n_envs, 1, 3, 240, 320)
            assert obs[key].std() > 0
        return obs

    def run_batch(label, obs, use_base=False):
        before = dict(solid_steps=env.solid_steps, idle_steps=env.idle_steps)
        native_before = observed.step_calls
        sim_start = native.isaac_gym.get_sim_time(native.sim)
        wall_start = time.perf_counter()
        reset_count = observed.reset_calls
        expected_decisions = 0
        executed_steps = 0
        chunks = []
        terminal = reference = None
        while not env.batch_done:
            entry_active = env.active.clone()
            count = int(entry_active.sum())
            expected_decisions += count
            assert obs['robot_state'].shape == (args.n_envs, 1, 16)
            if use_base:
                with torch.no_grad():
                    action = base.predict_action(obs)['action']
            else:
                action = env.hold_action.repeat(args.n_envs, env.n_action_steps, 1)
                action[:, :, -1] = -1
                if observed.inject_success:
                    action[0, :, -1] = 1
            assert action.shape == (args.n_envs, 8, 10)
            obs, reward, done, infos = env.step(action)
            assert observed.reset_calls == reset_count, 'Hidden reset inside step'
            assert len(infos) == args.n_envs
            assert [row['env_index'] for row in infos] == list(range(args.n_envs))
            for index, row in enumerate(infos):
                assert row['transition_valid'] == bool(entry_active[index])
                if entry_active[index]:
                    assert 1 <= row['executed_action_steps'] <= 8
                else:
                    assert row['executed_action_steps'] == 0 and not done[index] and reward[index] == 0
            executed_steps += sum(row['executed_action_steps'] for row in infos)
            chunks.append(dict(active=count, observation_rows=len(obs['robot_state']), action_rows=len(action),
                policy_rows=len(action) if use_base else None, lengths=[row['executed_action_steps'] for row in infos]))
            for row in infos:
                if row['env_index'] == 0 and 'terminal_observation' in row and row['success']:
                    terminal = row['terminal_observation']
                    reference = {key: value.clone() for key, value in terminal.items()}
            if terminal is not None:
                for key in terminal:
                    torch.testing.assert_close(terminal[key], reference[key], rtol=0, atol=0)
            assert obs['robot_state'].shape == (args.n_envs, 1, 16)
        assert env.last_chunk_valid
        deltas = {key: getattr(env, key) - value for key, value in before.items()}
        deltas.update(global_step=expected_decisions, native_steps=observed.step_calls - native_before)
        assert deltas['solid_steps'] == executed_steps
        assert deltas['native_steps'] * args.n_envs == deltas['solid_steps'] + deltas['idle_steps']
        sim_elapsed = native.isaac_gym.get_sim_time(native.sim) - sim_start
        expected_sim = deltas['native_steps'] * native.sim_steps * sim_config['sim_params'].dt
        assert abs(sim_elapsed - expected_sim) < 1e-4, (sim_elapsed, expected_sim)
        stage = dict(label=label, max_steps=env.max_steps, **deltas,
            nominal_steps=deltas['global_step'] * 8, physical_sim_seconds=sim_elapsed,
            wall_seconds=time.perf_counter() - wall_start, chunks=chunks,
            results=[dict(row) for row in env.results],
            natural_success_indices=sorted(observed.natural_success),
            injected_success_steps=dict(observed.inject_success), memory=memory())
        report['stages'].append(stage)
        print('GPU_BATCH_STAGE_OK', args.n_envs, label, deltas, flush=True)
        return stage, terminal, reference

    def fingerprint():
        digest = hashlib.sha256()
        for key, value in base.state_dict().items():
            digest.update(key.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    try:
        obs = reset(41000)
        stage, _, _ = run_batch('real_700_step_timeout', obs)
        timeouts = [row for row in stage['results'] if row['termination_reason'] == 'timeout']
        assert timeouts and all(row['episode_steps'] == 700 for row in timeouts)
        assert stage['native_steps'] == 700 and all(length == 4 for length in stage['chunks'][-1]['lengths'])

        base, _ = load_oneleg_base(args.checkpoint)
        original_base = fingerprint()
        base = base.to(env.device)
        env.max_steps = 16
        observed.inject_success = {0: 3}
        obs = reset(42000)
        stage, terminal, reference = run_batch('injected_success_inactive_hold', obs, use_base=True)
        assert stage['chunks'][0]['lengths'] == [3] + [8] * (args.n_envs - 1)
        assert stage['chunks'][1]['active'] == args.n_envs - 1
        assert stage['chunks'][1]['lengths'] == [0] + [8] * (args.n_envs - 1)
        assert all(chunk['policy_rows'] == args.n_envs for chunk in stage['chunks'])
        assert stage['results'][0]['episode_return'] == 1
        assert stage['global_step'] == 2 * args.n_envs - 1

        observed.inject_success = {}
        for batch in range(2):
            obs = reset(43000 + batch * args.n_envs)
            if terminal is not None:
                for key in terminal:
                    torch.testing.assert_close(terminal[key], reference[key], rtol=0, atol=0)
            run_batch('real_base_actions_' + str(batch), obs, use_base=True)
        assert fingerprint() == original_base
        assert not base.training and not any(parameter.requires_grad for parameter in base.parameters())
        report['base_unchanged'] = True

        env.mode = 'eval'
        observed.inject_success = {0: 1}
        observed.fail_at = 3
        reset(44000)
        try:
            env.step(env.hold_action.repeat(args.n_envs, 8, 1))
        except RuntimeError as error:
            assert 'injected probe failure' in str(error)
        else:
            raise AssertionError('Missing injected failure')
        assert env.results[0]['termination_reason'] == 'success'
        assert all(row['termination_reason'] == 'failure' for row in env.results[1:])
        assert not env.last_chunk_valid
        report['injected_eval_failure_results'] = [dict(row) for row in env.results]
        report['reset_costs'] = observed.reset_costs
        report['hold_rows_checked'] = observed.hold_rows_checked
        report['native_calls'] = observed.step_calls
        report['memory_before_close'] = memory()
    finally:
        del native
        env.close()
        env.close()
        base = None
        gc.collect()
        torch.cuda.empty_cache()
    assert native_ref() is None, 'Native simulator retained after close'
    report.update(close_ok=True, wall_seconds=time.perf_counter() - start_wall, memory_after_close=memory())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print('GPU_BATCH_PROBE_OK', args.n_envs, str(args.output), flush=True)


if __name__ == '__main__':
    main()
