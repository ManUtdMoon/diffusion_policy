import json
import pathlib
import tempfile
import unittest

import numpy as np
import zarr

from soe.build_soe_success_dataset import build_success_dataset
from zprl.dataset.adroit_image_dataset import AdroitImageDataset
from zprl.dataset.metaworld_image_dataset import MetaworldImageDataset


class SoeDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)

    def episode(self, task, round, index, success=True, length=4, marker=0.1):
        directory = self.root / task / 'seed_0' / f'round_{round}' / 'rollouts'
        (directory / 'episodes').mkdir(parents=True, exist_ok=True)
        path = pathlib.Path('episodes') / f'ep_{index:06d}.npz'
        if success:
            state_dim, action_dim = (24, 26) if task == 'adroit_hammer' else (9, 4)
            np.savez_compressed(directory / path,
                img=np.full((length, 84, 84, 3), int(marker*100), dtype=np.uint8),
                state=np.full((length, state_dim), marker, dtype=np.float32),
                action=np.full((length, action_dim), marker, dtype=np.float32))
        return {'episode_uid': f'{task}/0/{round}/{index}',
            'task': task, 'seed': 0, 'round': round, 'episode_index': index,
            'success': success, 'length': length, 'trajectory_path': str(path)}

    def manifest(self, task, round, episodes):
        directory = self.root / task / 'seed_0' / f'round_{round}' / 'rollouts'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'manifest.jsonl').write_text(''.join(
            json.dumps(episode)+'\n' for episode in episodes))

    def test_cumulative_successes_preserve_boundaries_and_load(self):
        for task, cls in [('metaworld_box-close', MetaworldImageDataset),
                          ('adroit_hammer', AdroitImageDataset)]:
            with self.subTest(task=task):
                first = self.episode(task, 1, 0, length=2, marker=0.1)
                failure = self.episode(task, 1, 1, success=False)
                second = self.episode(task, 2, 0, length=5, marker=0.2)
                self.manifest(task, 1, [failure, first])
                self.manifest(task, 2, [second])
                root = self.root / task / 'seed_0'
                (root / 'eval').mkdir()
                (root / 'eval/manifest.jsonl').write_text('not a manifest')
                self.manifest(task, 3, [{'not': 'a collected episode'}])
                initial = build_success_dataset(task, 0, 1, rollout_root=root)
                summary = build_success_dataset(task, 0, 2, rollout_root=root)
                self.assertEqual(initial['cumulative_success_episodes'], 1)
                self.assertEqual(summary['cumulative_attempted_episodes'], 3)
                self.assertEqual(summary['cumulative_success_episodes'], 2)
                self.assertEqual(summary['cumulative_transitions'], 7)
                self.assertEqual(summary['rounds'][0]['success_episodes'], 1)
                output = pathlib.Path(summary['dataset_path'])
                data = zarr.open_group(str(output), mode='r')
                np.testing.assert_array_equal(data['meta/episode_ends'][:], [2, 7])
                self.assertEqual(data['data/img'].dtype, np.uint8)
                self.assertEqual(data['data/action'].dtype, np.float32)
                lineage = json.loads((output.parent/'lineage.json').read_text())
                self.assertEqual([e['episode_uid'] for e in lineage],
                    [first['episode_uid'], second['episode_uid']])
                self.assertEqual([e['dataset_episode_index'] for e in lineage], [0, 1])
                self.assertEqual(json.loads((output.parent/'dataset_summary.json').read_text()), summary)
                dataset = cls(str(output), horizon=4, n_obs_steps=1,
                    pad_before=0, pad_after=3, val_ratio=0, max_train_episodes=None)
                for i in range(len(dataset)):
                    sample = dataset[i]
                    action = sample['action'].numpy()
                    self.assertTrue(np.all(action == action[0, 0]))
                    self.assertEqual(tuple(sample['obs']['image'].shape), (1, 3, 84, 84))
                dataset.get_normalizer()
                self.assertEqual(zarr.open_group(initial['dataset_path'], mode='r')
                    ['meta/episode_ends'][:].tolist(), [2])

    def test_no_success_raises_without_building_empty_dataset(self):
        task = 'metaworld_box-close'
        self.manifest(task, 1, [self.episode(task, 1, 0, success=False)])
        root = self.root / task / 'seed_0'
        with self.assertRaisesRegex(ValueError, 'No successful episodes'):
            build_success_dataset(task, 0, 1, rollout_root=root)
        self.assertFalse((root/'round_1/success_dataset').exists())

    def test_foreign_seed_raises(self):
        task = 'metaworld_box-close'
        episode = self.episode(task, 1, 0)
        root = self.root / task / 'seed_0'
        self.manifest(task, 1, [{**episode, 'seed': 1}])
        with self.assertRaisesRegex(ValueError, 'task/seed/round'):
            build_success_dataset(task, 0, 1, rollout_root=root)

    def test_missing_round_and_existing_output_raise(self):
        task = 'metaworld_box-close'
        episode = self.episode(task, 1, 0)
        self.manifest(task, 1, [episode])
        root = self.root / task / 'seed_0'
        with self.assertRaises(FileNotFoundError):
            build_success_dataset(task, 0, 2, rollout_root=root)
        build_success_dataset(task, 0, 1, rollout_root=root)
        with self.assertRaises(FileExistsError):
            build_success_dataset(task, 0, 1, rollout_root=root)


if __name__ == '__main__':
    unittest.main()
