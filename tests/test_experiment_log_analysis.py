import csv
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch
from pathlib import Path

import numpy as np
import yaml

from visualize.analyze_logs import analyze, config_value, discover_logs, read_run
from visualize import analyze_logs


class ExperimentLogAnalysisTest(unittest.TestCase):
    def make_run(self, root, name, seed, points, history=1):
        run = root / name
        (run / '.hydra').mkdir(parents=True)
        cfg = dict(task_name='${online_task.name}', online_task=dict(name='square'),
                   exp_name='same-name', n_action_steps=4, n_rl_steps=2,
                   res_policy=dict(history_length=history),
                   training=dict(seed=seed, device=f'cuda:{seed}', num_steps=100),
                   logging=dict(name=f'run-{seed}'))
        (run / '.hydra/config.yaml').write_text(yaml.safe_dump(cfg))
        rows = [dict(**{'info/global_step': step, 'info/chunk_step': step / 2,
                        'info/recent_done_sr': value}) for step, value in points]
        (run / 'logs.json.txt').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return run

    def test_recursive_discovery_deduplicates_overlapping_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_run(root, 'grid/group/seed_40', 40, [(0, 0)])
            other = self.make_run(root, 'timestamp_run', 50, [(0, 0)])
            found = discover_logs([root, run, run / 'logs.json.txt'])
            self.assertEqual(set(found), {run / 'logs.json.txt', other / 'logs.json.txt'})

    def test_groups_ignore_seed_and_device_but_split_changed_hyperparameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_run(root, 'runs/a', 40, [(0, 0), (100, 1)])
            self.make_run(root, 'runs/b', 50, [(0, 0), (100, 1)])
            self.make_run(root, 'runs/c', 60, [(0, 0), (100, 1)], history=4)
            manifest = analyze([root / 'runs'], root / 'analysis', plots=False)
            groups = {}
            for run in manifest['runs']:
                groups.setdefault(run['group'], []).append(run['seed'])
            self.assertEqual(sorted(map(len, groups.values())), [1, 2])
            comparison = json.loads((root / 'analysis/config_comparison.json').read_text())
            self.assertIn('res_policy.history_length', comparison['square']['differences'])

    def test_equal_seed_weight_real_steps_and_common_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_run(root, 'runs/a', 40, [(0, 0), (50, .2), (100, .4), (150, 999)])
            self.make_run(root, 'runs/b', 50, [(0, 1), (25, 1), (50, 1), (75, 1), (100, 1)])
            manifest = analyze([root / 'runs'], root / 'analysis', metrics=[],
                               tail_window=100, point_count=5, plots=False)
            self.assertEqual(manifest['common_end_by_task'], {'square': 100})
            with (root / 'analysis/aggregate.csv').open() as handle:
                aggregate = list(csv.DictReader(handle))[0]
            self.assertAlmostEqual(float(aggregate['tail_mean']), .65)
            self.assertAlmostEqual(float(aggregate['endpoint_mean']), .7)
            self.assertAlmostEqual(float(aggregate['tail_sd']), np.std([.3, 1], ddof=1))
            with (root / 'analysis/curves.csv').open() as handle:
                curves = list(csv.DictReader(handle))
            self.assertEqual(max(float(row['step']) for row in curves), 100)
            self.assertAlmostEqual(float(curves[-1]['mean']), .7)
            self.assertTrue(all(row['n_seeds'] == '2' for row in curves))

    def test_log_snapshot_reports_corruption_and_retains_last_finite_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_run(root, 'run', 40, [(100, .2), (100, .7)])
            with (run / 'logs.json.txt').open('a') as handle:
                handle.write('{"info/global_step": 200, "info/recent_done_sr": NaN}\n')
                handle.write('{"info/global_step":')
            parsed = read_run(run / 'logs.json.txt', ['info/recent_done_sr'], 'info/global_step')
            self.assertEqual(parsed['series']['info/recent_done_sr'], [(100, .7)])
            self.assertEqual(parsed['last_step'], 200)
            self.assertEqual(len(parsed['warnings']), 3)

    def test_duplicate_seed_is_not_counted_as_independent_replication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_run(root, 'runs/a', 40, [(100, 1)])
            self.make_run(root, 'runs/b', 40, [(100, 1)])
            with self.assertRaisesRegex(ValueError, 'Duplicate seed'):
                analyze([root / 'runs'], root / 'analysis', plots=False)

    def test_tasks_have_separate_budgets_and_no_config_eval_executes(self):
        config = dict(gamma='${eval:dangerous()}', other='${gamma}')
        self.assertEqual(config_value(config, 'other'), '${eval:dangerous()}')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_run(root, 'runs/a', 40, [(0, 0), (100, 1)])
            run = self.make_run(root, 'runs/b', 50, [(0, 0), (200, 1)])
            cfg = yaml.safe_load((run / '.hydra/config.yaml').read_text())
            cfg['online_task']['name'] = 'tool_hang'
            (run / '.hydra/config.yaml').write_text(yaml.safe_dump(cfg))
            manifest = analyze([root / 'runs'], root / 'analysis', plots=False)
            self.assertEqual(manifest['common_end_by_task'], {'square': 100, 'tool_hang': 200})

    def test_source_runs_cannot_be_output_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_run(root, 'run', 40, [(0, 0)])
            with self.assertRaisesRegex(ValueError, 'source run'):
                analyze([run], run / 'analysis', plots=False)

    def test_missing_primary_is_reported_as_missing_instead_of_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_run(root, 'run', 40, [(0, 0)])
            (run / 'logs.json.txt').write_text('{"info/global_step":100,"info/q_predicted":0.5}\n')
            manifest = analyze([run], root / 'analysis', plots=False)
            self.assertTrue(any('Primary metric missing' in warning
                                for warning in manifest['runs'][0]['warnings']))
            with (root / 'analysis/aggregate.csv').open() as handle:
                primary = next(row for row in csv.DictReader(handle)
                               if row['metric'] == 'info/recent_done_sr')
            self.assertEqual(primary['tail_n'], '0')
            self.assertEqual(primary['tail_mean'], '')

    def test_automatic_output_reuses_cache_and_refreshes_after_log_growth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_run(root, 'runs/a', 40, [(0, 0), (100, 1)])
            with patch('visualize.analyze_logs.ROOT', root):
                first = analyze([root / 'runs'], plots=False)
                output = Path(first['output'])
                self.assertFalse((output / 'series.csv').exists())
                modified = (output / 'summary.md').stat().st_mtime_ns
                with patch('visualize.analyze_logs.read_run', side_effect=AssertionError('cache missed')):
                    second = analyze([run, root / 'runs'], plots=False, tail_window=100000.0)
                self.assertTrue(second['reused'])
                self.assertEqual(first['output'], second['output'])
                self.assertEqual((output / 'summary.md').stat().st_mtime_ns, modified)
                with (run / 'logs.json.txt').open('a') as handle:
                    handle.write('{"info/global_step":200,"info/recent_done_sr":0.5}\n')
                updated = analyze([root / 'runs'], plots=False)
                self.assertNotIn('reused', updated)
                self.assertEqual(updated['output'], first['output'])
                self.assertEqual(updated['common_end_by_task']['square'], 200)
                self.assertEqual(len(list((root / '.doc/experiment-analysis').iterdir())), 1)

    def test_different_comparisons_do_not_overwrite_each_other_and_series_is_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_run(root, 'runs/a', 40, [(0, 0), (100, 1)])
            with patch('visualize.analyze_logs.ROOT', root):
                first = analyze([run], plots=False)
                second = analyze([run], plots=False, export_series=True)
                self.assertNotEqual(first['output'], second['output'])
                self.assertTrue((Path(second['output']) / 'series.csv').exists())
                with self.assertRaisesRegex(ValueError, 'different analysis'):
                    analyze([run], first['output'], plots=False, tail_window=50)


    def test_simultaneous_requests_share_one_writer_and_one_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_run(root, 'runs/a', 40, [(0, 0), (100, 1)])
            barrier = Barrier(2)

            def request():
                barrier.wait(timeout=5)
                return analyze([run], plots=False)

            with patch('visualize.analyze_logs.ROOT', root), \
                 patch('visualize.analyze_logs._analyze', wraps=analyze_logs._analyze) as writer, \
                 ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(request) for _ in range(2)]
                results = [future.result(timeout=10) for future in futures]
                self.assertEqual(writer.call_count, 1)
            self.assertEqual(results[0]['output'], results[1]['output'])
            self.assertEqual(sum(result.get('reused', False) for result in results), 1)


if __name__ == '__main__':
    unittest.main()
