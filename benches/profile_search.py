#!/usr/bin/env python3
"""Capture and render a Linux perf profile for the benchmark query window."""

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def query_window(log):
    markers = re.findall(r"^query_window (begin|end) (\d+\.\d+)$", log.read_text(), re.MULTILINE)
    if [kind for kind, _ in markers] != ["begin", "end"]:
        raise ValueError(f"{log}: expected one query window from the benchmark executable")
    start, end = [value for _, value in markers]
    if Decimal(start) >= Decimal(end):
        raise ValueError(f"{log}: invalid query window")
    return start, end


def write_command(command, output, **kwargs):
    with output.open("w") as stream:
        subprocess.run(command, stdout=stream, check=True, **kwargs)


def profile(args, command):
    perf = shutil.which(args.perf)
    if not perf:
        raise ValueError(f"perf executable not found: {args.perf}")
    if args.flamegraph_dir:
        collapse = args.flamegraph_dir / "stackcollapse-perf.pl"
        render = args.flamegraph_dir / "flamegraph.pl"
        if not collapse.is_file() or not render.is_file():
            raise ValueError(f"FlameGraph scripts not found in {args.flamegraph_dir}")
        if not shutil.which('perl'):
            raise ValueError("FlameGraph requires perl")

    output = args.output.resolve()
    output.mkdir(parents=True)
    data = output / "perf.data"
    log = output / "benchmark.log"
    record = [perf, "record", "--clockid", "mono", "--user-callchains", "--no-bpf-event",
              "-e", args.event, "-F", str(args.frequency), "--call-graph", args.call_graph,
              "-o", str(data), "--", *command]
    environment = dict(os.environ, RNNDESCENT_PROFILE_WINDOW="1")
    environment.setdefault("RESULT_DIR", str(output / "benchmark-report"))
    try:
        write_command(record, log, stderr=subprocess.STDOUT, env=environment)
    except subprocess.CalledProcessError as error:
        raise ValueError(f"perf record or benchmark failed with status {error.returncode}; see {log}") from error

    start, end = query_window(log)
    interval = f"{start},{end}"
    stacks = output / "perf.script"
    write_command([perf, "script", "-i", str(data), "--ns", "--time", interval, "-F", "-period"], stacks)
    if not stacks.stat().st_size:
        raise ValueError(f"no samples in query window; increase the benchmark --repeat count (see {log})")
    write_command([perf, "report", "-i", str(data), "--time", interval, "--stdio", "--no-children"],
                  output / "perf-report.txt")
    (output / "settings.json").write_text(json.dumps({
        "command": command, "perf": perf, "event": args.event, "frequency": args.frequency,
        "call_graph": args.call_graph, "start_monotonic": start, "end_monotonic": end,
    }, indent=2) + "\n")
    if args.flamegraph_dir:
        folded = output / "cpu.folded"
        write_command(["perl", str(collapse), str(stacks)], folded)
        write_command(["perl", str(render), "--title", "RNN-Descent query CPU profile", str(folded)],
                      output / "cpu.svg")
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new directory for the generated profile")
    parser.add_argument("--perf", default="perf", help="perf executable or path")
    parser.add_argument("--event", default="cpu-clock:u")
    parser.add_argument("--frequency", type=int, default=199)
    parser.add_argument("--call-graph", default="dwarf,8192", help="perf call graph mode")
    parser.add_argument("--flamegraph-dir", type=Path, help="directory containing the FlameGraph Perl scripts")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="benchmark command after --")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.frequency < 1:
        parser.error("benchmark command and positive sampling frequency are required")
    try:
        profile(args, command)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
