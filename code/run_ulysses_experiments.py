import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_ENCODERS = ["qwen3embedding06b", "bidirlm1bembedding", "bgem3", "embeddinggemma300m"]
DEFAULT_DIMES = ["oracle", "rel", "prf"]
COLLECTION = "ulysses-rfcorpus"


def memmap_files(basepath, encoder):
    output_dir = Path(basepath) / "data" / "memmap" / COLLECTION / encoder
    return output_dir / f"{encoder}.dat", output_dir / f"{encoder}_map.csv"


def run_command(command):
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def ensure_memmap(args, encoder):
    memmap_path, mapping_path = memmap_files(args.basepath, encoder)
    if not args.overwrite_memmaps and memmap_path.exists() and mapping_path.exists():
        print(f"skipping existing memmap for {encoder}", flush=True)
        return

    command = [
        sys.executable,
        "code/encode_memmap.py",
        "--corpus",
        COLLECTION,
        "--encoder",
        encoder,
        "--basepath",
        args.basepath,
        "--batch-size",
        str(args.batch_size),
    ]

    if args.limit is not None:
        command += ["--limit", str(args.limit)]
    if args.overwrite_memmaps:
        command.append("--overwrite")

    run_command(command)


def run_evaluation(args, encoder, dime):
    command = [
        sys.executable,
        "code/main.py",
        "--collection",
        COLLECTION,
        "--encoder",
        encoder,
        "--dime",
        dime,
        "--basepath",
        args.basepath,
        "--output-dir",
        args.output_dir,
    ]
    run_command(command)


def collate_outputs(output_dir):
    output_path = Path(output_dir)
    mean_files = sorted(path for path in output_path.glob("*_mean.csv") if not path.name.startswith("all_"))
    per_query_files = sorted(path for path in output_path.glob("*_per_query.csv") if not path.name.startswith("all_"))

    if mean_files:
        pd.concat([pd.read_csv(path) for path in mean_files], ignore_index=True).to_csv(
            output_path / "all_mean_metrics.csv", index=False
        )

    if per_query_files:
        pd.concat([pd.read_csv(path) for path in per_query_files], ignore_index=True).to_csv(
            output_path / "all_per_query_metrics.csv", index=False
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basepath", default=".")
    parser.add_argument("--output-dir", default="results/ulysses-rfcorpus")
    parser.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS)
    parser.add_argument("--dimes", nargs="+", default=DEFAULT_DIMES, choices=DEFAULT_DIMES)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite-memmaps", action="store_true")
    parser.add_argument("--skip-memmap", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for encoder in args.encoders:
        if not args.skip_memmap:
            ensure_memmap(args, encoder)

        for dime in args.dimes:
            run_evaluation(args, encoder, dime)

    collate_outputs(args.output_dir)
