#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import dill
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan workspace/checkpoints/*.ckpt under one or more directories, read "
            "global_step / recent_done_success(es) / recent_done_epi_len, and "
            "export mean successful episode length vs. global_step."
        )
    )
    parser.add_argument(
        "input_dirs",
        nargs="+",
        type=Path,
        help="One or more root directories to search for workspace/checkpoints/*.ckpt files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output text file path.",
    )
    return parser.parse_args()


def read_existing_rows(output_path: Path) -> Tuple[List[Dict], Optional[int]]:
    output_path = output_path.expanduser().resolve()
    if not output_path.exists():
        return [], None

    rows = []
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            rows.append(
                {
                    "global_step": int(parts[0]),
                    "mean_success_epi_len": float(parts[1]),
                    "mean_num_success": float(parts[2]),
                }
            )

    if not rows:
        return [], None
    return rows, rows[-1]["global_step"]


def read_workspace_global_step(checkpoints_dir: Path) -> Optional[int]:
    step_path = checkpoints_dir / "global_step.txt"
    if not step_path.exists():
        return None
    text = step_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return int(text)


def find_ckpt_files(input_dirs: Iterable[Path], min_global_step: Optional[int] = None) -> Tuple[List[Path], int]:
    ckpt_paths: List[Path] = []
    skipped_workspaces = 0
    for input_dir in input_dirs:
        input_dir = input_dir.expanduser().resolve()
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

        checkpoints_dirs = set()
        if input_dir.name == "checkpoints":
            checkpoints_dirs.add(input_dir)
        if (input_dir / "checkpoints").is_dir():
            checkpoints_dirs.add(input_dir / "checkpoints")
        checkpoints_dirs.update(path for path in input_dir.rglob("checkpoints") if path.is_dir())

        for checkpoints_dir in sorted(checkpoints_dirs):
            if not checkpoints_dir.is_dir():
                continue
            workspace_global_step = read_workspace_global_step(checkpoints_dir)
            if (
                min_global_step is not None
                and workspace_global_step is not None
                and workspace_global_step <= min_global_step
            ):
                skipped_workspaces += 1
                continue
            ckpt_paths.extend(sorted(checkpoints_dir.glob("*.ckpt")))
    if not ckpt_paths and min_global_step is None:
        raise FileNotFoundError("No .ckpt files found.")
    return ckpt_paths, skipped_workspaces


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
    existing_rows, last_output_step = read_existing_rows(args.output)
    ckpt_paths, skipped_workspaces = find_ckpt_files(
        args.input_dirs,
        min_global_step=last_output_step,
    )
    new_rows = []
    for ckpt_path in ckpt_paths:
        row = load_ckpt_row(ckpt_path)
        if last_output_step is not None and row["global_step"] <= last_output_step:
            continue
        new_rows.append(row)
    new_rows = aggregate_rows(new_rows)
    rows = existing_rows + new_rows
    write_output(args.output, rows)

    print(f"Last output step: {last_output_step}")
    print(f"Skipped workspaces: {skipped_workspaces}")
    print(f"Scanned ckpts: {len(ckpt_paths)}")
    print(f"Written rows: {len(rows)} ({len(new_rows)} new)")
    print(f"Saved to: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
