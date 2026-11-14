#!/usr/bin/env python3
"""Repeat SIFT128 coordinates so the same ground truth works at higher dimensions."""

import argparse
import os
from contextlib import ExitStack
from pathlib import Path
from struct import Struct
from tempfile import TemporaryDirectory

SIFT_DIM = 128
DIMENSIONS = (256, 512, 1024)
HEADER = Struct("<i")


def expand_vectors(source, target, filename):
    with ExitStack() as stack:
        original = stack.enter_context((source / filename).open("rb"))
        outputs = [
            (stack.enter_context((target / str(dim) / filename).open("wb")),
             HEADER.pack(dim), dim // SIFT_DIM)
            for dim in DIMENSIONS
        ]
        expected_header = HEADER.pack(SIFT_DIM)
        row_bytes = 4 * SIFT_DIM
        rows = 0
        while header := original.read(HEADER.size):
            if header != expected_header:
                raise ValueError(f"{filename}: expected dimension {SIFT_DIM} at row {rows + 1}")
            vector = original.read(row_bytes)
            if len(vector) != row_bytes:
                raise ValueError(f"{filename}: truncated vector at row {rows + 1}")
            for output, output_header, copies in outputs:
                output.write(output_header + vector * copies)
            rows += 1
        if not rows:
            raise ValueError(f"{filename}: no vectors")
        return rows


def check_ground_truth(path, query_count):
    with path.open("rb") as original:
        header = original.read(HEADER.size)
        if len(header) != HEADER.size or HEADER.unpack(header)[0] <= 0:
            raise ValueError(f"{path.name}: invalid ivecs header")
        width = HEADER.unpack(header)[0]
        if path.stat().st_size != query_count * (HEADER.size + 4 * width):
            raise ValueError(f"{path.name}: row count does not match {query_count} queries")


def expand(source, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".sift-expand-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        for dim in DIMENSIONS:
            (staging / str(dim)).mkdir(parents=True)
        base_count = expand_vectors(source, staging, "base.fvecs")
        query_count = expand_vectors(source, staging, "query.fvecs")
        ground_truth = source / "groundtruth.ivecs"
        check_ground_truth(ground_truth, query_count)
        for dim in DIMENSIONS:
            link = staging / str(dim) / ground_truth.name
            relative = os.path.relpath(ground_truth, start=output / str(dim))
            link.symlink_to(relative)
        staging.rename(output)
    print(f"Expanded {base_count:,} base vectors and {query_count:,} queries into {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="SIFT128 directory containing base.fvecs, query.fvecs and groundtruth.ivecs")
    parser.add_argument("output", type=Path, help="new directory for 256/, 512/, and 1024/")
    args = parser.parse_args()
    try:
        output = args.output.absolute()
        expand(args.source.resolve(), output.parent.resolve() / output.name)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
