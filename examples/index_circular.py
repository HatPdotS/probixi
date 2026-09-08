"""Index event lists with the validated circular-integration configuration."""

import argparse
from pathlib import Path

import torch

from probixi import DataOffloader, DuckDBOffloader, IntegrateConfig, Probixi, SeedConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("cell", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop", type=int)
    parser.add_argument("--recalibrate-every", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(20260907)
    p = Probixi(
        args.input,
        args.geometry,
        args.cell,
        device=torch.device(args.device),
        seed=SeedConfig(
            top_directions=128, max_index_peaks=128, rank_sigma=0.003, max_lattices=3
        ),
        integrate=IntegrateConfig(
            radii=(7, 9, 11), ewald_cutoff=0.0068, adu_per_photon=1
        ),
    )
    p.calibrate(n_seed=32)
    p.peak_size_max = p.finder.size_max = 200
    stream = p.index_frame_stream(
        p.frames(stop=args.stop),
        batch_size=8,
        recalibrate_every=args.recalibrate_every,
        enrich_alpha=0.001,
    )
    kwargs = dict(
        geometry=p.geometry,
        cell=p.target_cell,
        geometry_file=args.geometry,
        files=p.metadata.files,
    )
    if args.output.suffix in (".duckdb", ".db"):
        writer = DuckDBOffloader(
            args.output,
            **kwargs,
            multi_lattice=True,
            frame_range=(0, args.stop or p.metadata.n_frames),
        )
    else:
        writer = DataOffloader(args.output, **kwargs)
    with writer as out:
        for frame in stream:
            out.write(frame)
            if stream.stats.frames % 1000 == 0:
                print(stream.stats, flush=True)
    print(stream.stats, flush=True)


if __name__ == "__main__":
    main()
