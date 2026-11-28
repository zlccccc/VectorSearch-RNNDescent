#!/usr/bin/env python3
"""Measure QPS and Recall@10 on SIFT with repeated, independent index builds."""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
from pathlib import Path
from statistics import median

from expand_sift import DIMENSIONS, HEADER, SIFT_DIM, check_ground_truth, expand

FILES = ("base.fvecs", "query.fvecs", "groundtruth.ivecs")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path, dim):
    with path.open("rb") as vectors:
        header = vectors.read(HEADER.size)
        if header != HEADER.pack(dim):
            raise ValueError(f"{path}: expected {dim}-dimensional fvecs")
        rows, remainder = divmod(os.fstat(vectors.fileno()).st_size, 4 * (1 + dim))
        if remainder or not rows:
            raise ValueError(f"{path}: invalid fvecs file size")
        return rows


def prepare_datasets(source, expanded):
    base_rows, query_rows = (read_rows(source / name, SIFT_DIM) for name in FILES[:2])
    ground_truth = source / FILES[2]
    check_ground_truth(ground_truth, query_rows)
    datasets = {SIFT_DIM: source}
    if expanded is not None:
        if not expanded.exists():
            expand(source, expanded)
        for dim in DIMENSIONS:
            dataset = expanded / str(dim)
            counts = tuple(read_rows(dataset / name, dim) for name in FILES[:2])
            if counts != (base_rows, query_rows) or not (dataset / FILES[2]).samefile(ground_truth):
                raise ValueError(f"{dataset}: vectors or ground truth do not match {source}")
            datasets[dim] = dataset
    return datasets, base_rows, query_rows


def run_case(binary, dataset, run, dim, budget, queries, repeat, pool_gib, environment):
    run.mkdir()
    command = [
        str(binary), "--dataset-dir", str(dataset), "--dataset-name", "sift",
        "--K0", "32", "--num-initialize", str(max(32, budget)), "--search-L", str(budget),
        "--rerank-candidates", str(budget), "--beam-size", "8", "--pca-out", "0", "--topk", "10",
        "--neighbor-pool-gb", str(pool_gib), "--repeat", str(repeat), "--output-iter", "0",
    ]
    log_path = run / "run.log"
    with log_path.open("w") as log:
        process = subprocess.run(command, env=dict(environment, RESULT_DIR=str(run)),
                                 stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"benchmark exited with code {process.returncode}; see {log_path}")

    report = dict(line.split("=", 1) for line in (run / "latest.txt").read_text().splitlines())
    if (int(report["dim"]), int(report["checked_queries"])) != (dim, queries):
        raise ValueError(f"{run}: unexpected vector dimension or checked query count")
    threads = int(environment["OMP_NUM_THREADS"])
    if any(int(report[f"{phase}_config.num_threads"]) != threads for phase in ("search", "build")):
        raise ValueError(f"{run}: unexpected thread count")
    latencies = [float(report[f"latency_ms[{i}]"]) for i in range(repeat)]
    recall = float(report["recall@10"])
    if any(not math.isfinite(ms) or ms <= 0 for ms in latencies) or not 0 <= recall <= 1:
        raise ValueError(f"{run}: invalid latency or recall")
    cache = re.search(r"([\d.]+)% neighbors can be cached", log_path.read_text())
    if cache is None:
        raise ValueError(f"{log_path}: missing neighbor cache coverage")
    return median(1000 / ms for ms in latencies), recall * 100, float(cache[1]), float(report["build_seconds"])


def benchmark(args):
    source = args.source.resolve()
    binary = args.binary.resolve()
    output = args.output.parent.resolve() / args.output.name
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"benchmark executable not found or not executable: {binary}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output directory already exists: {output}")

    expanded = None
    if args.profile == "dimensions":
        expanded = (args.expanded or source.with_name(source.name + "-expanded")).resolve()
        if output == expanded or output in expanded.parents:
            raise ValueError("the results directory cannot contain the expanded dataset directory")
    datasets, base_rows, queries = prepare_datasets(source, expanded)
    if args.profile == "dimensions":
        cases = [(dim, 16) for dim in datasets]
    else:
        cases = [(SIFT_DIM, budget) for budget in (16, 32, 64, 128)]

    openmp_inputs = {
        "OMP_NUM_THREADS": str(args.threads),
        "OMP_WAIT_POLICY": os.environ.get("OMP_WAIT_POLICY") or "PASSIVE",
        "GOMP_SPINCOUNT": os.environ.get("GOMP_SPINCOUNT") or "0",
    }
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_DYNAMIC="FALSE", **openmp_inputs)
    output.mkdir(parents=True)
    metadata = {
        "profile": args.profile, "binary": str(binary),
        "binary_sha256": file_sha256(binary),
        "source_sha256": {name: file_sha256(source / name) for name in FILES},
        "machine": platform.machine(), "source": str(source),
        "expanded": str(expanded) if expanded is not None else None,
        "base_rows": base_rows, "query_rows": queries, "threads": args.threads, "openmp": openmp_inputs,
        "builds": args.builds, "repeat": args.repeat, "pool_gib": args.pool_gib,
    }
    (output / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")

    results = {case: [] for case in cases}
    for batch in range(1, args.builds + 1):
        order = cases if batch % 2 else reversed(cases)
        for dim, budget in order:
            print(f"Build {batch}/{args.builds}, {dim}D, search_L={budget}", flush=True)
            run = output / f"build-{batch}-{dim}d-l{budget}"
            result = run_case(binary, datasets[dim], run, dim, budget, queries, args.repeat, args.pool_gib, environment)
            results[dim, budget].append(result)
            print(f"  {result[0]:,.0f} QPS, Recall@10 {result[1]:.2f}%, cache {result[2]:.2f}%", flush=True)

    with (output / "summary.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("dimension", "search_L", "qps_median", "qps_min", "qps_max",
                         "recall_at_10_pct", "cache_pct", "build_seconds"))
        print("Dimension  search_L  Recall@10  QPS median (build range)  Neighbor cache")
        for (dim, budget), rows in results.items():
            rates, recalls, caches, builds = zip(*rows)
            writer.writerow((dim, budget, f"{median(rates):.1f}", f"{min(rates):.1f}", f"{max(rates):.1f}",
                             f"{median(recalls):.2f}", f"{median(caches):.2f}", f"{median(builds):.3f}"))
            print(f"{dim:9}  {budget:8}  {median(recalls):8.2f}%  {median(rates):,.0f} "
                  f"({min(rates):,.0f}–{max(rates):,.0f})  {median(caches):.2f}%")
    print(f"Results saved to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("dimensions", "recall"))
    parser.add_argument("source", type=Path, help="128D SIFT directory with matching ground truth")
    parser.add_argument("output", type=Path, help="new directory for reports and logs")
    parser.add_argument("--binary", type=Path, default=Path("build/algorithm"))
    parser.add_argument("--expanded", type=Path, help="directory to generate or reuse expanded vectors (dimensions only)")
    parser.add_argument("--threads", type=int, default=16, choices=range(1, 17), metavar="1-16")
    parser.add_argument("--builds", type=int, default=3, help="independent builds per case (default: 3)")
    parser.add_argument("--repeat", type=int, default=7, help="query passes per build (default: 7)")
    parser.add_argument("--pool-gib", type=int, default=4, help="neighbor pool budget in GiB (default: 4)")
    args = parser.parse_args()
    if min(args.builds, args.repeat) < 1 or args.pool_gib < 0:
        parser.error("builds and repeat must be positive, and pool-gib must be non-negative")
    if args.expanded is not None and args.profile != "dimensions":
        parser.error("--expanded requires the dimensions profile")
    try:
        benchmark(args)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
