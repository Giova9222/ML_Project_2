"""Command-line front-end for the Cascade ground-truth helpers.

    python src/cascade_data/cli.py list
    python src/cascade_data/cli.py list --files DS28-XCaMPgf-m-V1
    python src/cascade_data/cli.py download DS28-XCaMPgf-m-V1 --limit 1
    python src/cascade_data/cli.py index
    python src/cascade_data/cli.py show
    python src/cascade_data/cli.py noise --dataset DS28-XCaMPgf-m-V1
    python src/cascade_data/cli.py models
    python src/cascade_data/cli.py download-models OGB_zf_pDp_7.5Hz_smoothing200ms
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow both `python src/cascade_data/cli.py ...` and `python -m cascade_data.cli ...`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cascade_data.index import DatasetIndex
from cascade_data.noise import (
    frame_rate_from_time,
    noise_level,
    plot_noise_levels,
)
from cascade_data.pretrained import DEFAULT_MODELS, PretrainedModels
from cascade_data.repo import CascadeRepository
from cascade_data.two_photon import load_sessions

DEFAULT_ROOT = Path(__file__).resolve().parent.parent.parent / "data"
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "pretrained"


def build_parser():
    parser = argparse.ArgumentParser(description="Cascade ground-truth helper")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="local data root")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list remote datasets or files")
    p_list.add_argument("--files", metavar="DATASET", help="list files of one dataset")

    p_dl = sub.add_parser("download", help="download a subset of the data")
    p_dl.add_argument("datasets", nargs="*", help="dataset names (default: all)")
    p_dl.add_argument("--limit", type=int, default=1,
                      help="max files per dataset, smallest first (default: 1)")
    p_dl.add_argument("--dry-run", action="store_true")

    p_idx = sub.add_parser("index", help="(re)build the local index")
    p_idx.add_argument("--force", action="store_true")

    p_show = sub.add_parser("show", help="show the local index")
    p_show.add_argument("--dataset")

    p_noise = sub.add_parser("noise",
                             help="characterise the noise level of indexed sessions")
    p_noise.add_argument("--dataset", help="only this dataset folder")
    p_noise.add_argument("--limit", type=int, help="max files to process")
    p_noise.add_argument("--out", default=None,
                         help="save the noise-level histogram here (PNG)")
    p_noise.add_argument("--no-plot", action="store_true",
                         help="do not write the histogram")

    p_models = sub.add_parser("models", help="list pretrained Cascade models")

    p_dlm = sub.add_parser("download-models",
                           help="download pretrained Cascade models")
    p_dlm.add_argument("models", nargs="*",
                       help="model names (default: the curated subset)")
    p_dlm.add_argument("--out", default=DEFAULT_MODELS_DIR,
                       help="destination directory")
    p_dlm.add_argument("--all", action="store_true",
                       help="download every pretrained model")
    p_dlm.add_argument("--overwrite", action="store_true",
                       help="re-download models that are already present")
    p_dlm.add_argument("--dry-run", action="store_true")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    repo = CascadeRepository(cache_dir=root / ".cache")

    if args.cmd == "list":
        if args.files:
            for f in repo.files(args.files):
                print(f"{f.size:>9}  {f.name}")
        else:
            for d in repo.datasets():
                print(d)
    elif args.cmd == "download":
        datasets = args.datasets or repo.datasets()
        repo.download_many(datasets, root / "raw", limit=args.limit,
                           dry_run=args.dry_run)
    elif args.cmd == "index":
        DatasetIndex(root).build(force=args.force)
    elif args.cmd == "show":
        for e in DatasetIndex(root).entries(args.dataset):
            print(f"{e.dataset:45s} {e.name:55s} {e.n_samples:>7d} samples "
                  f"{e.duration_s:7.1f}s {e.n_aps:>4d} APs  {e.cell_type}")
    elif args.cmd == "noise":
        index = DatasetIndex(root)
        if not index.index_path.exists():
            index.build()
        entries = index.entries(args.dataset)
        if args.limit is not None:
            entries = entries[:args.limit]

        levels = []
        for entry in entries:
            for session in load_sessions(root / entry.path):
                rate = frame_rate_from_time(session.time)
                level = noise_level(session.dff, rate)
                levels.append(level)
                print(f"{entry.dataset:45s} {session.name:55s} "
                      f"{rate:6.1f} Hz  noise {level:6.2f} % s^(1/2)")

        if levels:
            arr = np.asarray(levels)
            print(f"{arr.size} session(s): median {np.median(arr):.2f}, "
                  f"mean {arr.mean():.2f}, max {arr.max():.2f} % s^(1/2)")
            if not args.no_plot:
                out = args.out or (root / "noise_levels.png")
                plot_noise_levels(arr, out)
                print(f"histogram -> {out}")
    elif args.cmd == "models":
        models = PretrainedModels()
        for m in models.models():
            print(m)
    elif args.cmd == "download-models":
        models = PretrainedModels()
        if args.all:
            names = models.models()
        else:
            names = args.models or list(DEFAULT_MODELS)
        models.download_many(names, Path(args.out), dry_run=args.dry_run,
                             overwrite=args.overwrite)


if __name__ == "__main__":
    main()
