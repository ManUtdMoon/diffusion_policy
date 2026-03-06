#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively collect logs.json.txt files under an input directory, "
            "merge them into one JSONL file, verify info/global_step is strictly increasing, "
            "and export eval metrics at a fixed step interval."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root directory to search recursively (e.g. data/flip_online/).",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=500,
        help="Step interval for exporting eval.txt (default: 500).",
    )
    return parser.parse_args()


def load_records(input_dir: Path) -> List[Tuple[int, Dict, str]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    log_files = sorted(input_dir.rglob("logs.json.txt"))
    if not log_files:
        alt_files = sorted(input_dir.rglob("*.json.txt"))
        hint = ""
        if alt_files:
            preview = "\n".join(f"  - {p}" for p in alt_files[:10])
            hint = f"\nFound other *.json.txt files:\n{preview}"
        raise FileNotFoundError(
            f"No logs.json.txt found under: {input_dir}{hint}"
        )

    records: List[Tuple[int, Dict, str]] = []
    for log_path in log_files:
        with log_path.open("r", encoding="utf-8") as f:
            for line_idx, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {log_path}:{line_idx}: {exc}"
                    ) from exc

                if "info/global_step" not in obj:
                    raise KeyError(
                        f'Missing key "info/global_step" at {log_path}:{line_idx}'
                    )

                try:
                    step = int(obj["info/global_step"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f'Invalid "info/global_step" value at {log_path}:{line_idx}: '
                        f'{obj["info/global_step"]}'
                    ) from exc

                source = f"{log_path}:{line_idx}"
                records.append((step, obj, source))

    return records


def ensure_strictly_increasing(records: List[Tuple[int, Dict, str]]) -> None:
    prev_step = None
    prev_source = None
    for step, _, source in records:
        if prev_step is not None and step <= prev_step:
            raise ValueError(
                "info/global_step is not strictly increasing: "
                f"previous={prev_step} ({prev_source}), current={step} ({source})"
            )
        prev_step = step
        prev_source = source


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_path = input_dir / "all_logs.json.txt"
    eval_output_path = input_dir / "eval.txt"
    if args.eval_interval <= 0:
        raise ValueError(f"--eval-interval must be > 0, got {args.eval_interval}")

    records = load_records(input_dir)
    records.sort(key=lambda x: x[0])
    ensure_strictly_increasing(records)

    input_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for _, obj, _ in records:
            f.write(json.dumps(obj, ensure_ascii=True) + "\n")

    eval_count = 0
    with eval_output_path.open("w", encoding="utf-8") as f:
        for step, obj, source in records:
            if step % args.eval_interval != 0:
                continue
            if "info/recent_done_sr" not in obj:
                raise KeyError(
                    f'Missing key "info/recent_done_sr" at {source}'
                )
            if "info/recent_done_avg_len" not in obj:
                raise KeyError(
                    f'Missing key "info/recent_done_avg_len" at {source}'
                )
            sr = float(obj["info/recent_done_sr"])
            avg_len = float(obj["info/recent_done_avg_len"])
            f.write(f"{step} {sr} {avg_len}\n")
            eval_count += 1

    print(f"Merged {len(records)} lines from {input_dir} -> {output_path}")
    print(
        f"Wrote {eval_count} eval lines (interval={args.eval_interval}) -> {eval_output_path}"
    )


if __name__ == "__main__":
    main()
