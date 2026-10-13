"""Discover local RL runs, extract selected metrics, then aggregate equally by seed."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = [
    'info/recent_done_sr', 'test/mean_score', 'info/q_predicted', 'info/q_target',
    'info/actor_entropy', 'info/res_n_rms', 'loss/alpha', 'loss/critic_loss',
    'loss/actor_grad_norm', 'loss/q1_grad_norm',
]


def discover_logs(paths):
    found = set()
    for path in paths:
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise ValueError(f'Input does not exist: {path}')
        if path.is_file():
            if path.name != 'logs.json.txt':
                raise ValueError(f'Expected logs.json.txt: {path}')
            found.add(path)
        else:
            found.update(log.resolve() for log in path.rglob('logs.json.txt'))
    if not found:
        raise ValueError('No logs.json.txt found under the supplied paths')
    return sorted(found)


def config_value(config, key, default=None):
    # Resolve plain references only; never execute Hydra ${eval:...} expressions.
    for _ in range(10):
        value = config
        for part in key.split('.'):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        match = re.fullmatch(r'\$\{([\w.]+)\}', value) if isinstance(value, str) else None
        if not match:
            return value
        key = match.group(1)
    return default


def experiment_config(config):
    ignored = {'seed', 'device', 'logging', 'hydra', 'output_dir', 'exp_name'}
    return {key: experiment_config(value) if isinstance(value, dict) else value
            for key, value in config.items() if key not in ignored}


def flatten_config(config, prefix=''):
    result = {}
    for key, value in config.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            result.update(flatten_config(value, name))
        else:
            result[name] = value
    return result


def read_run(log, metrics, step_key):
    config_path = log.parent / '.hydra/config.yaml'
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    config = config or {}
    warnings = []
    if not config:
        warnings.append('Missing Hydra config: grouping cannot infer equivalent runs')
    seed = config_value(config, 'training.seed')
    if seed is None:
        match = re.fullmatch(r'seed_(\d+)', log.parent.name)
        seed = int(match.group(1)) if match else None
        warnings.append('Seed inferred from directory or unknown; check manifest')
    task = config_value(config, 'task_name', config_value(config, 'online_task.name', 'unknown'))
    name = config_value(config, 'exp_name', log.parent.name)
    comparable = experiment_config(config) if config else {'run': str(log.parent)}
    signature = hashlib.sha256(json.dumps(comparable, sort_keys=True).encode()).hexdigest()[:10]
    values = {metric: {} for metric in metrics}
    available, invalid_lines, nonfinite = set(), [], 0
    last_step, last_chunk, last_update, previous_step = None, None, None, None
    backwards, repeated = 0, 0
    snapshot = log.stat()
    # Bound the read to the file size at discovery so an active writer cannot
    # extend this analysis indefinitely. A partial final record is reported.
    with log.open('rb') as handle:
        remaining = snapshot.st_size
        line_number = 0
        while remaining:
            raw = handle.readline(remaining)
            remaining -= len(raw)
            line_number += 1
            try:
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError('Expected JSON object')
            except (ValueError, UnicodeDecodeError):
                invalid_lines.append(line_number)
                continue
            available.update(row)
            step = row.get(step_key)
            if not isinstance(step, (int, float)) or not math.isfinite(step):
                continue
            backwards += previous_step is not None and step < previous_step
            repeated += previous_step is not None and step == previous_step
            previous_step = step
            if last_step is None or step >= last_step:
                last_step = step
                last_chunk = row.get('info/chunk_step')
                last_update = row.get('info/global_update')
            for metric in metrics:
                value = row.get(metric)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if math.isfinite(value):
                        values[metric][step] = value
                    else:
                        nonfinite += 1
    if last_step is None:
        raise ValueError(f'No finite {step_key} values in {log}')
    if invalid_lines:
        warnings.append(f'Invalid/partial JSON records: {len(invalid_lines)}; first lines {invalid_lines[:5]}')
    if backwards or repeated:
        warnings.append(f'Step resets/repeats: {backwards}/{repeated}; last finite value per metric/step retained')
    if nonfinite:
        warnings.append(f'Nonfinite selected metric values skipped: {nonfinite}')
    if log.stat().st_size != snapshot.st_size:
        warnings.append('Log grew during reading; analysis uses the initial byte-size snapshot')
    configured_steps = config_value(config, 'training.num_steps')
    complete = (last_chunk >= configured_steps if isinstance(last_chunk, (int, float))
                and isinstance(configured_steps, (int, float)) else None)
    if complete is False:
        warnings.append('Run has not reached configured training.num_steps')
    return dict(path=str(log.parent), task=task, experiment=name, signature=signature,
                seed=seed, config=config, comparable_config=comparable,
                last_step=last_step, last_chunk=last_chunk, last_update=last_update,
                complete=complete, source_bytes=snapshot.st_size,
                available_metrics=sorted(available), warnings=warnings,
                series={metric: sorted(points.items()) for metric, points in values.items()})


def assign_groups(runs):
    keys = sorted({(run['task'], run['experiment'], run['signature']) for run in runs})
    labels = {key: f'G{index + 1}' for index, key in enumerate(keys)}
    seen = {}
    for index, run in enumerate(runs):
        run['run_id'] = f'R{index + 1}'
        run['group'] = labels[(run['task'], run['experiment'], run['signature'])]
        key = run['group'], run['seed']
        if run['seed'] is not None and key in seen:
            raise ValueError(f'Duplicate seed {run["seed"]} in {run["group"]}: '
                             f'{seen[key]} and {run["path"]}. Select distinct runs before aggregating.')
        seen[key] = run['path']


def seed_statistics(points, end, tail_window):
    points = [(step, value) for step, value in points if step <= end]
    tail = [value for step, value in points if end - tail_window < step <= end]
    return dict(endpoint=points[-1][1] if points else None,
                endpoint_step=points[-1][0] if points else None,
                tail_mean=float(np.mean(tail)) if tail else None, tail_points=len(tail))


def mean_sd(values):
    values = [value for value in values if value is not None]
    return (float(np.mean(values)) if values else None,
            float(np.std(values, ddof=1)) if len(values) > 1 else None, len(values))


def aligned_curves(runs, metrics, ends, point_count):
    rows = []
    for task, end in ends.items():
        for metric in metrics:
            available = [run for run in runs if run['task'] == task
                         and run['series'][metric] and run['series'][metric][0][0] <= end]
            if not available:
                continue
            start = max(run['series'][metric][0][0] for run in available)
            stop = min(end, *(run['series'][metric][-1][0] for run in available))
            if stop < start:
                continue
            grid = np.linspace(start, stop, point_count if stop > start else 1)
            groups = defaultdict(list)
            for run in available:
                x, y = np.asarray(run['series'][metric]).T
                groups[run['group']].append(np.interp(grid, x, y))
            for group, values in groups.items():
                values = np.asarray(values)
                means = values.mean(axis=0)
                stds = values.std(axis=0, ddof=1) if len(values) > 1 else [None] * len(grid)
                rows.extend(dict(task=task, group=group, metric=metric, step=float(step),
                                 mean=float(mean), sd=float(sd) if sd is not None else None,
                                 n_seeds=len(values))
                            for step, mean, sd in zip(grid, means, stds))
    return rows


def write_csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(output, curves, runs, step_key):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    by_panel = defaultdict(list)
    for row in curves:
        by_panel[row['task'], row['metric'], row['group']].append(row)
    names = {run['group']: run['experiment'] for run in runs}
    colors = {group: f'C{index % 10}' for index, group in enumerate(sorted(names))}
    files = []
    for task in sorted({row['task'] for row in curves}):
        metrics = sorted({row['metric'] for row in curves if row['task'] == task})
        columns = min(2, len(metrics))
        rows = math.ceil(len(metrics) / columns)
        fig, axes = plt.subplots(rows, columns,
                                 figsize=(13 if columns == 2 else 10, max(4, 3.2 * rows)), squeeze=False)
        for ax, metric in zip(axes.flat, metrics):
            for group in sorted(names):
                panel = by_panel.get((task, metric, group), [])
                if not panel:
                    continue
                x = np.array([row['step'] for row in panel]) / 1e6
                y = np.array([row['mean'] for row in panel])
                line, = ax.plot(x, y, color=colors[group],
                               label=f'{group}: {names[group]} (n={panel[0]["n_seeds"]})')
                if panel[0]['sd'] is not None:
                    sd = np.array([row['sd'] for row in panel])
                    ax.fill_between(x, y - sd, y + sd, color=line.get_color(), alpha=0.15)
            ax.set_title(metric)
            ax.set_xlabel(f'{step_key} / 1e6')
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        for ax in list(axes.flat)[len(metrics):]:
            ax.set_visible(False)
        fig.suptitle(f'{task}: equal seed weight, mean +/- sample SD', fontsize=11)
        fig.tight_layout()
        stem = re.sub(r'[^\w.-]', '_', str(task)) + '_curves'
        for suffix in ('png', 'pdf'):
            path = output / f'{stem}.{suffix}'
            fig.savefig(path, dpi=150)
            files.append(path.name)
        plt.close(fig)
    return files


def _analyze(paths, output, metrics=DEFAULT_METRICS, primary='info/recent_done_sr',
             step_key='info/global_step', end_step=None, tail_window=100000,
             point_count=501, plots=True, export_series=False):
    if tail_window <= 0 or point_count < 2:
        raise ValueError('tail_window must be positive and point_count at least 2')
    logs = discover_logs(paths)
    metrics = list(dict.fromkeys([primary, *metrics]))
    runs = [read_run(log, metrics, step_key) for log in logs]
    assign_groups(runs)
    ends = {task: min(run['last_step'] for run in runs if run['task'] == task)
            for task in {run['task'] for run in runs}}
    if end_step is not None:
        ends = {task: min(end, end_step) for task, end in ends.items()}
    statistics, series_rows = [], []
    for run in runs:
        for metric in metrics:
            stats = seed_statistics(run['series'][metric], ends[run['task']], tail_window)
            statistics.append(dict(run=run['run_id'], group=run['group'], seed=run['seed'],
                                   task=run['task'], metric=metric, **stats))
            if export_series:
                series_rows.extend(dict(run=run['run_id'], group=run['group'], seed=run['seed'],
                                        task=run['task'], metric=metric, step=step, value=value)
                                   for step, value in run['series'][metric])
        if not run['series'][primary]:
            run['warnings'].append(f'Primary metric missing: {primary}')
        else:
            stats = seed_statistics(run['series'][primary], ends[run['task']], tail_window)
            if stats['endpoint_step'] != ends[run['task']]:
                run['warnings'].append(f'Primary endpoint uses step {stats["endpoint_step"]}, '
                                       f'not the common end {ends[run["task"]]}')
            if not stats['tail_points']:
                run['warnings'].append('Primary metric has no records in the common tail window')
    aggregates = []
    for group in sorted({run['group'] for run in runs}):
        members = [run for run in runs if run['group'] == group]
        for metric in metrics:
            selected = [row for row in statistics if row['group'] == group and row['metric'] == metric]
            endpoint_mean, endpoint_sd, endpoint_n = mean_sd([row['endpoint'] for row in selected])
            tail_mean, tail_sd, tail_n = mean_sd([row['tail_mean'] for row in selected])
            aggregates.append(dict(group=group, task=members[0]['task'], experiment=members[0]['experiment'],
                                   metric=metric, n_runs=len(members), endpoint_n=endpoint_n,
                                   endpoint_mean=endpoint_mean, endpoint_sd=endpoint_sd,
                                   tail_n=tail_n, tail_mean=tail_mean, tail_sd=tail_sd))
    output.mkdir(parents=True, exist_ok=True)
    comparisons = {}
    for task in ends:
        configs = {run['group']: flatten_config(run['comparable_config'])
                   for run in runs if run['task'] == task}
        keys = set().union(*(config.keys() for config in configs.values()))
        shared, differences = {}, {}
        for key in sorted(keys):
            values = {group: config.get(key, '<missing>') for group, config in configs.items()}
            if len({json.dumps(value, sort_keys=True) for value in values.values()}) == 1:
                shared[key] = next(iter(values.values()))
            else:
                differences[key] = values
        comparisons[task] = dict(shared=shared, differences=differences)
    (output / 'config_comparison.json').write_text(json.dumps(comparisons, indent=2, ensure_ascii=False) + '\n')
    curves = aligned_curves(runs, metrics, ends, point_count)
    if export_series:
        write_csv(output / 'series.csv', series_rows,
                  ['run', 'group', 'seed', 'task', 'metric', 'step', 'value'])
    write_csv(output / 'per_seed.csv', statistics,
              ['run', 'group', 'seed', 'task', 'metric', 'endpoint', 'endpoint_step', 'tail_mean', 'tail_points'])
    write_csv(output / 'aggregate.csv', aggregates, list(aggregates[0]))
    write_csv(output / 'curves.csv', curves, ['task', 'group', 'metric', 'step', 'mean', 'sd', 'n_seeds'])
    figures = plot_curves(output, curves, runs, step_key) if plots else []
    manifest = dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                    inputs=[str(Path(path).resolve()) for path in paths], primary=primary,
                    metrics=metrics, step_key=step_key, common_end_by_task=ends,
                    tail_window=tail_window, figures=figures,
                    runs=[{key: value for key, value in run.items() if key != 'series'} for run in runs])
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    report = [
        '# Experiment log summary', '', f'Primary metric: `{primary}`; axis: `{step_key}`.',
        f'Common end per task: `{json.dumps(ends)}`; tail window: {tail_window:g} steps.',
        'Read each run separately; select metrics, summarize each seed, then aggregate seeds equally.',
        'Tail = arithmetic mean of records in (end-window, end]; endpoint = latest record <= end.',
        'Curves use linear interpolation only inside the shared observed range; no extrapolation.',
        'SD = sample standard deviation (ddof=1), not a confidence interval. One seed has no SD.',
        'Training recent_done_sr uses overlapping recent-episode windows and exploration actions;',
        'it is not independent deterministic evaluation. global_step is nominal environment steps.', '',
        '| Group | Task / experiment | Seeds | Tail mean +/- SD (n) | Endpoint mean +/- SD (n) |',
        '| --- | --- | --- | --- | --- |',
    ]

    def formatted(mean, sd, count):
        return f'{mean:.3f} +/- {sd:.3f} ({count})' if sd is not None else (
            f'{mean:.3f} ({count})' if mean is not None else 'missing (0)')

    for row in aggregates:
        if row['metric'] != primary:
            continue
        members = [run for run in runs if run['group'] == row['group']]
        seeds = ', '.join(str(run['seed']) for run in members)
        label = f'{row["task"]} / {row["experiment"]}'.replace('|', '\\|')
        report.append(f'| {row["group"]} | {label} | {seeds} | '
                      f'{formatted(row["tail_mean"], row["tail_sd"], row["tail_n"])} | '
                      f'{formatted(row["endpoint_mean"], row["endpoint_sd"], row["endpoint_n"])} |')
    report.extend(['', '## Configurations', '',
                   '| Group | Ta / Tr | H / ratio / dropout | n_step / UTD | buffer / num_steps |',
                   '| --- | --- | --- | --- | --- |'])
    for group in sorted({run['group'] for run in runs}):
        config = next(run['config'] for run in runs if run['group'] == group)
        def value(key):
            return config_value(config, key, 'missing')
        report.append(f'| {group} | {value("n_action_steps")} / {value("n_rl_steps")} | '
                      f'{value("res_policy.history_length")} / {value("res_policy.sub_ratio")} / '
                      f'{value("res_policy.dropout")} | {value("training.n_step")} / '
                      f'{value("training.utd")} | {value("training.buffer_size")} / '
                      f'{value("training.num_steps")} |')
    for task, comparison in comparisons.items():
        if not comparison['differences']:
            continue
        report.extend(['', f'### Config differences: {task}', '', '| Field | Values by group |',
                       '| --- | --- |'])
        for key, values in comparison['differences'].items():
            rendered = json.dumps(values, ensure_ascii=False).replace('|', '\\|')
            report.append(f'| {key} | `{rendered}` |')
    report.extend(['', '## Runs and checks', ''])
    for run in runs:
        report.append(f'- {run["run_id"]}: {run["group"]}, seed={run["seed"]}, '
                      f'last_step={run["last_step"]}, Q updates={run["last_update"]}, '
                      f'complete={run["complete"]}; `{run["path"]}`')
        report.extend(f'  - {warning}' for warning in run['warnings'])
    report.extend(['', 'Full configs and metric inventory: `manifest.json`; shared/differing config fields: `config_comparison.json`.',
                   ('Selected per-run points: `series.csv`.' if export_series else
                    'Dense per-run series not exported; use --export-series when needed.'),
                   'Seed statistics: `per_seed.csv`;',
                   'group statistics: `aggregate.csv`; aligned plot data: `curves.csv`.',
                   'Equal configs do not establish equal historical source code or checkpoint bytes.',
                   *[f'Figure: [{name}]({name})' for name in figures], ''])
    (output / 'summary.md').write_text('\n'.join(report))
    return manifest


def analyze(paths, output=None, metrics=DEFAULT_METRICS, primary='info/recent_done_sr',
            step_key='info/global_step', end_step=None, tail_window=100000,
            point_count=501, plots=True, export_series=False):
    logs = discover_logs(paths)
    options = dict(metrics=sorted(set([primary, *metrics])), primary=primary,
                   step_key=step_key, end_step=float(end_step) if end_step is not None else None,
                   tail_window=float(tail_window),
                   point_count=point_count, plots=plots, export_series=export_series)
    identity = dict(logs=[str(log) for log in logs], options=options)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    output = Path(output).resolve() if output is not None else ROOT / '.doc/experiment-analysis' / key
    for protected in (ROOT / 'data', ROOT / 'data_local'):
        if output == protected.resolve() or protected.resolve() in output.parents:
            raise ValueError('Write analysis artifacts outside data/ and data_local/')
    if any(output == log.parent or log.parent in output.parents for log in logs):
        raise ValueError('Output must not be inside a source run')
    source_state = {}
    for log in logs:
        for source in (log, log.parent / '.hydra/config.yaml'):
            stat = source.stat() if source.exists() else None
            source_state[str(source)] = [stat.st_size, stat.st_mtime_ns] if stat else None
    source_state['helper_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    # Serialize writers to the same analysis across conversations/processes.
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = output / 'manifest.json'
        cached = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if cached and cached.get('analysis_key') not in (None, key):
            raise ValueError('Output belongs to a different analysis; omit --output or choose another directory')
        if cached and cached.get('analysis_key') == key and cached.get('source_state') == source_state:
            artifacts = ['summary.md', 'config_comparison.json', 'per_seed.csv',
                         'aggregate.csv', 'curves.csv', *cached['figures']]
            if export_series:
                artifacts.append('series.csv')
            if all((output / name).exists() for name in artifacts):
                return dict(cached, reused=True)
        manifest = _analyze(logs, output, **options)
        manifest.update(analysis_key=key, source_state=source_state, output=str(output))
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', help='Run directories, ancestors, or logs.json.txt files')
    parser.add_argument('--output', help='Optional artifact directory; default reuses an analysis fingerprint')
    parser.add_argument('--metrics', nargs='+', default=DEFAULT_METRICS)
    parser.add_argument('--primary', default='info/recent_done_sr')
    parser.add_argument('--step-key', default='info/global_step')
    parser.add_argument('--end-step', type=float, help='Cap the common observed budget')
    parser.add_argument('--tail-window', type=float, default=100000)
    parser.add_argument('--points', type=int, default=501)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--export-series', action='store_true', help='Also export the dense selected per-run series CSV')
    args = parser.parse_args()
    try:
        manifest = analyze(args.paths, args.output, args.metrics, args.primary, args.step_key,
                           args.end_step, args.tail_window, args.points, not args.no_plots, args.export_series)
    except ValueError as error:
        parser.error(str(error))
    print(f'{"Reused" if manifest.get("reused") else "Analyzed"}: {len(manifest["runs"])} runs; '
          f'{len({run["group"] for run in manifest["runs"]})} groups; '
          f'common end={manifest["common_end_by_task"]}')
    print(f'Read {Path(manifest["output"]) / "summary.md"}')


if __name__ == '__main__':
    main()
