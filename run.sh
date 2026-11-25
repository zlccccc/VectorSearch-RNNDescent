#!/bin/sh
set -eu

cd "$(dirname "$0")"
build_dir="${BUILD_DIR:-build}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-ACTIVE}"
case "$OMP_WAIT_POLICY" in
    [Aa][Cc][Tt][Ii][Vv][Ee])
        export GOMP_SPINCOUNT="${GOMP_SPINCOUNT:-16000}"
        ;;
esac

cmake -S . -B "$build_dir"
cmake --build "$build_dir" --target vector_search_app --parallel

if [ "$#" -eq 0 ]; then
    set -- --mode dataset --dataset-dir "benches/datasets/${DATASET:-sift}"
fi
exec "$build_dir/algorithm" "$@"
