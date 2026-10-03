"""Select a reproducible subset from the GSM8K validation split.

The input is expected to be a Hugging Face ``DatasetDict`` saved with
``DatasetDict.save_to_disk``.  The script does not silently fall back to a
different split: a validation split is required by the experiment protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from datasets import Dataset, DatasetDict, load_from_disk


DEFAULT_NUM_SAMPLES = 372
DEFAULT_DIAGNOSTIC_SIZE = 128
DEFAULT_SEED = 42


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select GSM8K validation questions reproducibly."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/gsm8k"),
        help="Path passed to datasets.load_from_disk (default: data/gsm8k).",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split to sample from (default: validation).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of rows to select (default: {DEFAULT_NUM_SAMPLES}).",
    )
    parser.add_argument(
        "--diagnostic-size",
        type=int,
        default=DEFAULT_DIAGNOSTIC_SIZE,
        help=(
            "Number of selected rows assigned to the diagnostic pool; the "
            "remaining rows are assigned to Critic training."
        ),
    )
    parser.add_argument(
        "--pilot-size",
        type=int,
        default=32,
        help="Number of diagnostic rows saved for the first formal run (default: 32).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for sampling (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/gsm8k_validation_selection"),
        help="Directory in which the selected datasets and manifest are saved.",
    )
    return parser.parse_args()


def _load_split(dataset_path: Path, split: str) -> Dataset:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist: {dataset_path}. "
            "Prepare the GSM8K DatasetDict first."
        )

    loaded = load_from_disk(str(dataset_path))
    if not isinstance(loaded, DatasetDict):
        raise TypeError(
            f"Expected a DatasetDict at {dataset_path}, got {type(loaded).__name__}."
        )
    if split not in loaded:
        available = ", ".join(sorted(loaded.keys()))
        raise KeyError(
            f"Split '{split}' is not present in {dataset_path}. "
            f"Available splits: {available}. "
            "The script will not substitute train or test for validation."
        )
    return loaded[split]


def _ensure_source_index(dataset: Dataset) -> Dataset:
    if "source_index" in dataset.column_names:
        return dataset
    # _treetune__idx is useful when it exists, but source_index is always
    # present and unambiguous even for a raw Dataset without that column.
    source_indices = list(range(len(dataset)))
    return dataset.add_column("source_index", source_indices)


def _write_jsonl(dataset: Dataset, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_dataset(dataset: Dataset, directory: Path, name: str) -> None:
    output = directory / name
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output}. "
            "Choose another --output-dir."
        )
    dataset.save_to_disk(str(output))
    _write_jsonl(dataset, directory / f"{name}.jsonl")


def select_validation_data(
    dataset_path: Path,
    output_dir: Path,
    split: str = "validation",
    num_samples: int = DEFAULT_NUM_SAMPLES,
    diagnostic_size: int = DEFAULT_DIAGNOSTIC_SIZE,
    pilot_size: int = 32,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Select and persist the experiment's validation subsets.

    Returns the manifest as a dictionary so callers can use this function
    without going through the command line entry point.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if diagnostic_size < 0 or diagnostic_size > num_samples:
        raise ValueError("diagnostic_size must be between 0 and num_samples")
    if pilot_size < 0 or pilot_size > diagnostic_size:
        raise ValueError("pilot_size must be between 0 and diagnostic_size")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose another --output-dir."
        )

    source = _ensure_source_index(_load_split(dataset_path, split))
    if len(source) < num_samples:
        raise ValueError(
            f"Requested {num_samples} rows from split '{split}', but it only has "
            f"{len(source)} rows."
        )

    sampled = source.shuffle(seed=seed).select(range(num_samples))
    diagnostic = sampled.select(range(diagnostic_size))
    pilot = diagnostic.select(range(pilot_size))
    critic = sampled.select(range(diagnostic_size, num_samples))

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_dataset(sampled, output_dir, "selected")
    _save_dataset(diagnostic, output_dir, "diagnostic")
    _save_dataset(pilot, output_dir, "pilot")
    _save_dataset(critic, output_dir, "critic")

    manifest: Dict[str, Any] = {
        "dataset_path": str(dataset_path.resolve()),
        "source_split": split,
        "source_size": len(source),
        "num_samples": num_samples,
        "diagnostic_size": diagnostic_size,
        "pilot_size": pilot_size,
        "critic_size": len(critic),
        "seed": seed,
        "selected_source_indices": sampled["source_index"],
        "diagnostic_source_indices": diagnostic["source_index"],
        "pilot_source_indices": pilot["source_index"],
        "critic_source_indices": critic["source_index"],
        "outputs": {
            "selected_dataset": "selected",
            "diagnostic_dataset": "diagnostic",
            "pilot_dataset": "pilot",
            "critic_dataset": "critic",
            "jsonl_suffix": ".jsonl",
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    args = _parse_args()
    manifest = select_validation_data(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        split=args.split,
        num_samples=args.num_samples,
        diagnostic_size=args.diagnostic_size,
        pilot_size=args.pilot_size,
        seed=args.seed,
    )
    print(
        f"Selected {manifest['num_samples']} rows from "
        f"{manifest['source_split']} with seed {manifest['seed']}."
    )
    print(
        f"Pilot: {manifest['pilot_size']}; "
        f"Diagnostic pool: {manifest['diagnostic_size']}; "
        f"Critic: {manifest['critic_size']}"
    )
    print(f"Saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
