#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
DATASETS_DIR="${ROOT_DIR}/benches/datasets"

usage() {
    cat <<'USAGE'
Usage:
  sh benches/download_dataset.sh <dataset> [output_dir]

Supported datasets:
  siftsmall
  sift
  gist

Examples:
  sh benches/download_dataset.sh siftsmall
  sh benches/download_dataset.sh sift benches/datasets

Downloaded datasets follow the upstream layout:
  benches/datasets/<dataset>/

This script also creates standard aliases for the local benchmark entry:
  base.fvecs|bvecs
  query.fvecs|bvecs
  groundtruth.ivecs
  learn.fvecs|bvecs   (if provided by the dataset)
USAGE
}

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "missing required command: $1" >&2
        exit 1
    fi
}

download_file() {
    url="$1"
    output="$2"
    if command -v curl >/dev/null 2>&1; then
        if [ -f "$output" ]; then
            curl -L --fail --retry 3 --retry-delay 1 -C - -o "$output" "$url"
        else
            curl -L --fail --retry 3 --retry-delay 1 -o "$output" "$url"
        fi
    elif command -v wget >/dev/null 2>&1; then
        wget -c -O "$output" "$url"
    else
        echo "curl or wget is required to download datasets" >&2
        exit 1
    fi
}

safe_link() {
    source_path="$1"
    target_path="$2"
    if [ -f "$source_path" ]; then
        rm -f "$target_path"
        ln -s "$(basename "$source_path")" "$target_path"
    fi
}

link_standard_names() {
    dataset_name="$1"
    dataset_dir="$2"
    case "$dataset_name" in
        siftsmall)
            safe_link "${dataset_dir}/siftsmall_base.fvecs" "${dataset_dir}/base.fvecs"
            safe_link "${dataset_dir}/siftsmall_query.fvecs" "${dataset_dir}/query.fvecs"
            safe_link "${dataset_dir}/siftsmall_groundtruth.ivecs" "${dataset_dir}/groundtruth.ivecs"
            safe_link "${dataset_dir}/siftsmall_learn.fvecs" "${dataset_dir}/learn.fvecs"
            ;;
        sift)
            safe_link "${dataset_dir}/sift_base.fvecs" "${dataset_dir}/base.fvecs"
            safe_link "${dataset_dir}/sift_query.fvecs" "${dataset_dir}/query.fvecs"
            safe_link "${dataset_dir}/sift_groundtruth.ivecs" "${dataset_dir}/groundtruth.ivecs"
            safe_link "${dataset_dir}/sift_learn.fvecs" "${dataset_dir}/learn.fvecs"
            ;;
        gist)
            safe_link "${dataset_dir}/gist_base.fvecs" "${dataset_dir}/base.fvecs"
            safe_link "${dataset_dir}/gist_query.fvecs" "${dataset_dir}/query.fvecs"
            safe_link "${dataset_dir}/gist_groundtruth.ivecs" "${dataset_dir}/groundtruth.ivecs"
            safe_link "${dataset_dir}/gist_learn.fvecs" "${dataset_dir}/learn.fvecs"
            ;;
        *)
            echo "unsupported dataset for alias creation: ${dataset_name}" >&2
            exit 1
            ;;
    esac
}

resolve_url() {
    dataset_name="$1"
    case "$dataset_name" in
        siftsmall) echo "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz" ;;
        sift) echo "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz" ;;
        gist) echo "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz" ;;
        *)
            echo "unsupported dataset: ${dataset_name}" >&2
            usage
            exit 1
            ;;
    esac
}

main() {
    if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
        usage
        exit 1
    fi

    need_cmd tar
    dataset_name="$1"
    output_root="${2:-${DATASETS_DIR}}"
    dataset_dir="${output_root}/${dataset_name}"
    archive_name="${dataset_name}.tar.gz"
    archive_path="${output_root}/${archive_name}"
    url=$(resolve_url "${dataset_name}")

    mkdir -p "${output_root}"
    if [ -d "${dataset_dir}" ] && [ -f "${dataset_dir}/groundtruth.ivecs" ]; then
        echo "dataset already prepared: ${dataset_dir}"
        exit 0
    fi

    echo "downloading ${dataset_name} from ${url}"
    download_file "${url}" "${archive_path}"

    rm -rf "${dataset_dir}"
    mkdir -p "${dataset_dir}"
    tar -xzf "${archive_path}" -C "${dataset_dir}"

    nested_dir="${dataset_dir}/${dataset_name}"
    if [ -d "${nested_dir}" ]; then
        find "${nested_dir}" -mindepth 1 -maxdepth 1 -exec mv {} "${dataset_dir}"/ \;
        rmdir "${nested_dir}"
    fi

    link_standard_names "${dataset_name}" "${dataset_dir}"

    echo "dataset ready at ${dataset_dir}"
    echo "run example:"
    echo "  ./build/algorithm --mode dataset --dataset-dir ${dataset_dir} --topk 10 --repeat 5"
}

main "$@"
