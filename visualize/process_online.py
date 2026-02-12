#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively collect logs.json.txt files under an input directory, "
            "merge them into one JSONL file, and verify info/global_step is strictly increasing."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root directory to search recursively (e.g. data/flip_online/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output merged JSONL file path (e.g. data/flip_online/all_logs.json.txt).",
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
    output_path = args.output.expanduser().resolve()

    records = load_records(input_dir)
    records.sort(key=lambda x: x[0])
    ensure_strictly_increasing(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for _, obj, _ in records:
            f.write(json.dumps(obj, ensure_ascii=True) + "\n")

    print(f"Merged {len(records)} lines from {input_dir} -> {output_path}")


if __name__ == "__main__":
    main()
