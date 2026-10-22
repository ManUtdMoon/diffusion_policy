import ast
import io
import os
import pathlib
import tempfile
import unittest
from typing import Optional
from unittest import mock

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader

from zprl.dataset.oneleg_image_dataset import OneLegImageDataset
from zprl.model.common.normalizer import LinearNormalizer


SHAPE_META = {
    'obs': {
        'color_image1': {'shape': [3, 240, 320], 'type': 'rgb'},
        'color_image2': {'shape': [3, 240, 320], 'type': 'rgb'},
        'robot_state': {'shape': [16], 'type': 'low_dim'},
    },
    'action': {'shape': [10]},
}


class OneLegDatasetTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.paths = []
        self.actions = []
        self.states = []
        for i, ends in enumerate(([3, 7, 12], [2, 5, 9, 14])):
            path = str(pathlib.Path(self.directory.name) / f'{i}.zarr')
            root = zarr.open_group(path, mode='w')
            root.create_dataset('meta/episode_ends', data=np.array(ends))
            n = ends[-1]
            action = np.broadcast_to(
                (100 * i + np.arange(n, dtype=np.float32))[:, None], (n, 10)).copy()
            state = np.broadcast_to(action[:, :1], (n, 16)).copy()
            state[:, 3:9] = [1, 0, 0, 0, 1, 0]
            self.states.append(state)
            root.create_dataset('data/robot_state', data=state)
            root.create_dataset('data/action', data=np.zeros_like(action))
            action[:, :3] = action[:, :3] * [0.01, 0.02, 0.03] + [0.3, -0.1, 0.05]
            angle = 0.3 + np.arange(n) * 0.07 + i
            action[:, 3:9] = np.stack([np.cos(angle), -np.sin(angle), np.zeros(n),
                np.sin(angle), np.cos(angle), np.zeros(n)], axis=-1)
            # Preserve stored rot6d even when it is not exactly orthonormal.
            action[:, 3:6] *= 0.97
            action[:, 9] = np.where(np.arange(n) % 2, 1., -1.)
            self.actions.append(action)
            root.create_dataset('data/action_pos', data=action)
            # No reward, success, reset or simulator metadata is needed.
            for key, pixel in (('color_image1', 255), ('color_image2', 128)):
                array = root.create_dataset('data/' + key,
                    shape=(n, 240, 320, 3), chunks=(1, 240, 320, 3), dtype='u1')
                array[:] = pixel
            self.paths.append(path)
        self.dataset = OneLegImageDataset(SHAPE_META, self.paths,
            horizon=16, seed=42, val_ratio=0.25)

    def test_splits_weights_and_episode_boundaries(self):
        dataset = self.dataset
        np.testing.assert_allclose(dataset.sample_probabilities, [2 / 5, 3 / 5])
        again = OneLegImageDataset(SHAPE_META, self.paths,
            horizon=16, seed=42, val_ratio=0.25)
        for i, (train, val, ends) in enumerate(zip(
                dataset.train_masks, dataset.val_masks, dataset.episode_ends)):
            np.testing.assert_array_equal(val, again.val_masks[i])
            self.assertFalse(np.any(train & val))
            self.assertTrue(np.all(train | val))
            for rows, mask in ((dataset.indices[i], train),
                    (dataset.get_validation_dataset().indices[i], val)):
                lengths = np.diff(np.r_[0, ends])
                self.assertEqual(len(rows), lengths[mask].sum())
                for start, end, sample_start, sample_end in rows:
                    episode = np.searchsorted(ends, start, side='right')
                    self.assertTrue(mask[episode])
                    self.assertLessEqual(end, ends[episode])
                    self.assertEqual(sample_start, 0)
                    self.assertEqual(sample_end, end - start)

    def test_continuous_actions_tail_padding_and_source_selection(self):
        dataset = self.dataset
        for source_index, rows in enumerate(dataset.indices):
            with mock.patch('numpy.random.choice', return_value=source_index) as choose:
                for idx, (start, end, _, _) in enumerate(rows):
                    sample = dataset[idx]
                    expected = self.actions[source_index][
                        np.minimum(np.arange(start, start + 16), end - 1)]
                    np.testing.assert_array_equal(sample['action'], expected)
                    self.assertEqual(set(sample), {'obs', 'action'})
                    self.assertEqual(set(sample['obs']), set(SHAPE_META['obs']))
                    np.testing.assert_array_equal(sample['obs']['robot_state'],
                        self.states[source_index][start:start + 1])
                self.assertEqual(choose.call_args.kwargs['p'].tolist(),
                    dataset.sample_probabilities.tolist())
                torch.testing.assert_close(dataset[len(rows)]['action'], dataset[0]['action'])

    def test_batch_and_image_scaling(self):
        batch = next(iter(DataLoader(self.dataset, batch_size=2, num_workers=0)))
        self.assertEqual(tuple(batch['action'].shape), (2, 16, 10))
        self.assertEqual(tuple(batch['obs']['robot_state'].shape), (2, 1, 16))
        for key, value in (('color_image1', 1.), ('color_image2', 128 / 255.)):
            image = batch['obs'][key]
            self.assertEqual(tuple(image.shape), (2, 1, 3, 240, 320))
            self.assertEqual(image.dtype, torch.float32)
            torch.testing.assert_close(image, torch.full_like(image, value))
            torch.testing.assert_close(self.dataset.get_normalizer()[key].normalize(image), image)

    def test_validation_visits_every_window_once_and_supports_source_index(self):
        dataset = self.dataset.get_validation_dataset()
        samples = []
        with mock.patch('numpy.random.choice', side_effect=AssertionError('random validation')):
            for i in range(len(dataset)):
                samples.append(dataset[i]['action'].numpy())
        starts = [sample[0, 0].item() for sample in samples]
        offset = 0
        for i, rows in enumerate(dataset.indices):
            for start, end, _, _ in rows:
                np.testing.assert_array_equal(samples[offset], self.actions[i][
                    np.minimum(np.arange(start, start + 16), end - 1)])
                offset += 1
        expected = [self.actions[i][rows[:, 0], 0].tolist()
                    for i, rows in enumerate(dataset.indices)]
        self.assertEqual(starts, sum(expected, []))
        for i in range(2):
            val = self.dataset.get_validation_dataset(index=i)
            self.assertEqual(len(val), len(expected[i]))
            self.assertEqual([val[j]['action'][0, 0].item() for j in range(len(val))], expected[i])

    def test_normalizer_uses_both_complete_sources_and_roundtrips(self):
        normalizer = self.dataset.get_normalizer()
        data = {'action': torch.from_numpy(np.concatenate(self.actions)),
                'robot_state': torch.from_numpy(np.concatenate(self.states))}
        limits = LinearNormalizer()
        limits.fit(data, mode='limits')
        for key, stop in (('action', 10), ('robot_state', 9)):
            values = data[key]
            params = normalizer[key].params_dict
            self.assertEqual(tuple(params['scale'].shape), (values.shape[-1],))
            self.assertEqual(params['scale'].dtype, torch.float32)
            for stat, expected in dict(min=values.amin(0), max=values.amax(0),
                    mean=values.mean(0), std=values.std(0)).items():
                torch.testing.assert_close(normalizer[key].get_input_stats()[stat], expected)
            normalized = normalizer[key].normalize(values)
            torch.testing.assert_close(normalized[:, 3:stop], values[:, 3:stop], rtol=0, atol=0)
            torch.testing.assert_close(params['scale'][3:stop], torch.ones(stop - 3))
            torch.testing.assert_close(params['offset'][3:stop], torch.zeros(stop - 3))
            for section in (slice(0, 3), slice(stop, values.shape[-1])):
                torch.testing.assert_close(normalized[:, section],
                    limits[key].normalize(values)[:, section], rtol=0, atol=0)
            torch.testing.assert_close(normalizer[key].unnormalize(normalized),
                values, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(normalizer['action'].normalize(data['action'])[:, :3].amin(0),
            -torch.ones(3))
        torch.testing.assert_close(normalizer['action'].normalize(data['action'])[:, :3].amax(0),
            torch.ones(3))
        for dataset in (self.dataset, self.dataset.get_validation_dataset(),
                self.dataset.get_validation_dataset(index=1)):
            torch.testing.assert_close(dataset.get_all_actions(), data['action'], rtol=0, atol=0)
            for key, value in normalizer.state_dict().items():
                torch.testing.assert_close(dataset.get_normalizer().state_dict()[key], value, rtol=0, atol=0)
        stream = io.BytesIO()
        torch.save(normalizer.state_dict(), stream)
        stream.seek(0)
        restored = LinearNormalizer()
        restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        for key, value in normalizer.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
        for key, values in data.items():
            torch.testing.assert_close(restored[key].normalize(values),
                normalizer[key].normalize(values), rtol=0, atol=0)
        self.assertTrue(self.dataset.sources[0].read_only)

    def test_constant_gripper_command_remains_identity(self):
        for path in self.paths:
            zarr.open_group(path, mode='a')['data/action_pos'][:, 9] = 1.
        dataset = OneLegImageDataset(SHAPE_META, self.paths, horizon=16)
        normalizer = dataset.get_normalizer()['action']
        self.assertEqual(normalizer.get_input_stats()['min'][9].item(), 1.)
        self.assertEqual(normalizer.get_input_stats()['max'][9].item(), 1.)
        self.assertTrue(torch.all(normalizer.normalize(dataset.get_all_actions())[:, 9] == 1))

    def test_original_action_column_is_not_required(self):
        for path in self.paths:
            del zarr.open_group(path, mode='a')['data/action']
        dataset = OneLegImageDataset(SHAPE_META, self.paths, horizon=16)
        np.testing.assert_array_equal(dataset.get_all_actions(), np.concatenate(self.actions))
        for source_index, rows in enumerate(dataset.indices):
            start, end, _, _ = rows[0]
            with mock.patch('numpy.random.choice', return_value=source_index):
                np.testing.assert_array_equal(dataset[0]['action'], self.actions[source_index][
                    np.minimum(np.arange(start, start + 16), end - 1)])

    def test_invalid_action_pos_is_rejected(self):
        for path, valid in zip(self.paths, self.actions):
            cases = [('missing', None), ('length', valid[:-1]),
                     ('width', valid[:, :-1]), ('dtype', valid.astype(np.float64))]
            for name, values in cases:
                with self.subTest(source=path, invalid=name):
                    source = zarr.open_group(path, mode='a')
                    del source['data/action_pos']
                    if values is not None:
                        source.create_dataset('data/action_pos', data=values)
                    error = {'missing': KeyError, 'length': AssertionError}.get(name, ValueError)
                    with self.assertRaises(error):
                        OneLegImageDataset(SHAPE_META, self.paths, horizon=16)
                    source.create_dataset('data/action_pos', data=valid, overwrite=True)

    def test_invalid_contract_is_rejected(self):
        for kwargs in ({'n_obs_steps': 2}, {'pad_after': 3}, {'pad_before': 1}):
            with self.assertRaises(ValueError):
                OneLegImageDataset(SHAPE_META, self.paths, horizon=16, **kwargs)
        with self.assertRaises(IndexError):
            self.dataset[len(self.dataset)]

    def test_compressed_memory_cache_is_independent_of_source_files(self):
        dataset = self.dataset
        validation = dataset.get_validation_dataset()
        self.assertIs(validation.sources, dataset.sources)
        for path, cached in zip(self.paths, dataset.sources):
            original = zarr.open_group(path, mode='r')
            self.assertIsInstance(cached.store, zarr.MemoryStore)
            self.assertTrue(cached.read_only)
            self.assertEqual(set(cached['data'].keys()), set(SHAPE_META['obs']) | {'action_pos'})
            for key in cached['data']:
                self.assertEqual(cached['data'][key].chunks, original['data'][key].chunks)
                self.assertEqual(cached['data'][key].compressor, original['data'][key].compressor)
            chunk = 'data/color_image1/0.0.0.0'
            self.assertEqual(cached.store[chunk], original.store[chunk])
            # Temporary fixture only: cached samples must survive loss of disk access.
            os.rename(path, path + '.unavailable')
        self.assertEqual(tuple(dataset[0]['action'].shape), (16, 10))
        self.assertEqual(tuple(validation[0]['action'].shape), (16, 10))
        self.assertTrue(torch.isfinite(dataset.get_normalizer()['action'].normalize(
            dataset.get_all_actions())).all())
        batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=2)))
        self.assertEqual(tuple(batch['action'].shape), (2, 16, 10))

    @unittest.skipUnless(os.environ.get('ONELEG_AUTHOR_REPO'),
        'set ONELEG_AUTHOR_REPO to compare with the author sampler')
    def test_author_sampler_matches_active_oneleg_contract(self):
        path = pathlib.Path(os.environ['ONELEG_AUTHOR_REPO']) / 'diffusion_policy/common/sampler.py'
        module = ast.parse(path.read_text())
        module.body = [node for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in ('create_indices', 'ImprovedDatasetSampler')]
        for node in module.body:
            node.decorator_list = []
        namespace = {'np': np, 'Optional': Optional, 'ReplayBuffer': object}
        exec(compile(module, str(path), 'exec'), namespace)
        sampler_cls = namespace['ImprovedDatasetSampler']

        class AuthorBuffer:
            def __init__(self, source, ends):
                self.data = source['data']
                self.episode_ends = ends

            def keys(self):
                return ['color_image1', 'color_image2', 'robot_state', 'action']

            def __getitem__(self, key):
                return self.data['action_pos' if key == 'action' else key]

        for source_index, source in enumerate(self.dataset.sources):
            for validation in (False, True):
                dataset = (self.dataset.get_validation_dataset() if validation
                           else self.dataset)
                masks = dataset.val_masks if validation else dataset.train_masks
                sampler = sampler_cls(
                    replay_buffer=AuthorBuffer(source, dataset.episode_ends[source_index]),
                    sequence_length=16, shape_meta=SHAPE_META,
                    pad_before=0, pad_after=15, episode_mask=masks[source_index],
                    key_first_k={'color_image1': 1, 'color_image2': 1,
                        'robot_state': 1, 'action': 16})
                np.testing.assert_array_equal(dataset.indices[source_index], sampler.indices)
                for idx in range(len(sampler)):
                    # The author leaves unread obs slots as placeholders; only
                    # the first observation is consumed by the memory-less policy.
                    with np.errstate(invalid='ignore'):
                        expected = sampler.sample_data(idx)
                    dataset_idx = idx + (sum(len(rows) for rows in
                        dataset.indices[:source_index]) if validation else 0)
                    with mock.patch('numpy.random.choice', return_value=source_index):
                        sample = dataset[dataset_idx]
                    np.testing.assert_array_equal(sample['action'], expected['action'])
                    np.testing.assert_array_equal(sample['obs']['robot_state'],
                        expected['obs']['robot_state'][:1])
                    for key in dataset.rgb_keys:
                        image = np.moveaxis(expected['obs'][key][:1], -1, 1).astype(np.float32) / 255.
                        np.testing.assert_array_equal(sample['obs'][key], image)


if __name__ == '__main__':
    unittest.main()
