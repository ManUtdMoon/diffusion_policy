#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import dill
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively scan .ckpt files under one or more directories, read "
            "global_step / recent_done_success(es) / recent_done_epi_len, and "
            "export mean successful episode length vs. global_step."
        )
    )
    parser.add_argument(
        "input_dirs",
        nargs="+",
        type=Path,
        help="One or more root directories to search recursively for .ckpt files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output text file path.",
    )
    return parser.parse_args()


def find_ckpt_files(input_dirs: Iterable[Path]) -> List[Path]:
    ckpt_paths: List[Path] = []
    for input_dir in input_dirs:
        input_dir = input_dir.expanduser().resolve()
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

        workspace_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
        for workspace_dir in workspace_dirs:
            checkpoints_dir = workspace_dir / "checkpoints"
            if not checkpoints_dir.is_dir():
                continue
            ckpt_paths.extend(sorted(checkpoints_dir.glob("*.ckpt")))
    if not ckpt_paths:
        raise FileNotFoundError("No .ckpt files found.")
    return ckpt_paths


def load_ckpt_row(ckpt_path: Path) -> Dict:
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu")

    if "global_step" not in payload:
        raise KeyError(f"Missing key 'global_step' in {ckpt_path}")

    if "recent_done_successes" in payload:
        successes = [bool(x) for x in payload["recent_done_successes"]]
    elif "recent_done_success" in payload:
        successes = [bool(x) for x in payload["recent_done_success"]]
    else:
        raise KeyError(
            f"Missing 'recent_done_successes'/'recent_done_success' in {ckpt_path}"
        )

    if "recent_done_epi_len" not in payload:
        raise KeyError(f"Missing key 'recent_done_epi_len' in {ckpt_path}")
    epi_lens = [float(x) for x in payload["recent_done_epi_len"]]

    if len(successes) != len(epi_lens):
        raise ValueError(
            f"Mismatched lengths in {ckpt_path}: "
            f"success={len(successes)} vs epi_len={len(epi_lens)}"
        )

    success_epi_lens = [epi_len for success, epi_len in zip(successes, epi_lens) if success]
    mean_success_epi_len = None
    if success_epi_lens:
        mean_success_epi_len = sum(success_epi_lens) / len(success_epi_lens)

    return {
        "global_step": int(payload["global_step"]),
        "success_epi_lens": success_epi_lens,
        "num_success": len(success_epi_lens),
    }


def aggregate_rows(rows: List[Dict]) -> List[Dict]:
    step_to_success_lens = defaultdict(list)
    step_to_num_success = defaultdict(list)
    for row in rows:
        step_to_success_lens[row["global_step"]].extend(row["success_epi_lens"])
        step_to_num_success[row["global_step"]].append(row["num_success"])

    result = []
    for step in sorted(step_to_success_lens.keys()):
        success_epi_lens = step_to_success_lens[step]
        if not success_epi_lens:
            continue
        num_success_values = step_to_num_success[step]
        result.append(
            {
                "global_step": step,
                "mean_success_epi_len": sum(success_epi_lens) / len(success_epi_lens),
                "mean_num_success": sum(num_success_values) / len(num_success_values),
            }
        )
    return result


def write_output(output_path: Path, rows: List[Dict]) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# global_step mean_success_epi_len mean_num_success\n")
        for row in rows:
            f.write(
                f"{row['global_step']} "
                f"{row['mean_success_epi_len']:.6f} "
                f"{row['mean_num_success']:.6f}\n"
            )


def main() -> None:
    args = parse_args()
    ckpt_paths = find_ckpt_files(args.input_dirs)
    rows = [load_ckpt_row(ckpt_path) for ckpt_path in ckpt_paths]
    rows = aggregate_rows(rows)
    write_output(args.output, rows)

    print(f"Scanned ckpts: {len(ckpt_paths)}")
    print(f"Written rows: {len(rows)}")
    print(f"Saved to: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
