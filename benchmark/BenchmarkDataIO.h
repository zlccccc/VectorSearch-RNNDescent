#pragma once

#include "BenchmarkTypes.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <vector>

namespace benchmark {

namespace fs = std::filesystem;

using std::ifstream;
using std::ios;
using std::runtime_error;
using std::string;
using std::transform;
using std::vector;

inline string get_ext_lower(const string &path) {
    string ext = fs::path(path).extension().string();
    transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)tolower(c); });
    return ext;
}

inline DenseVectors load_dense_vectors(const string &path) {
    ifstream fin(path, ios::binary);
    if (!fin) {
        throw runtime_error("failed to open vector file: " + path);
    }

    const string ext = get_ext_lower(path);
    DenseVectors out;
    int first_dim = 0;
    fin.read(reinterpret_cast<char *>(&first_dim), sizeof(int));
    if (!fin || first_dim <= 0) {
        throw runtime_error("invalid vector file header: " + path);
    }
    fin.seekg(0, ios::end);
    const size_t file_size = (size_t)fin.tellg();
    fin.seekg(0, ios::beg);

    if (ext == ".fvecs") {
        const size_t record_size = sizeof(int) + sizeof(float) * (size_t)first_dim;
        if (file_size % record_size != 0) {
            throw runtime_error("fvecs file size is not aligned: " + path);
        }
        out.rows = (int)(file_size / record_size);
        out.dim = first_dim;
        out.values.resize((size_t)out.rows * out.dim);
        for (int i = 0; i < out.rows; i++) {
            int dim = 0;
            fin.read(reinterpret_cast<char *>(&dim), sizeof(int));
            if (!fin || dim != out.dim) {
                throw runtime_error("inconsistent fvecs dimension in: " + path);
            }
            fin.read(reinterpret_cast<char *>(out.values.data() + (size_t)i * out.dim), sizeof(float) * out.dim);
            if (!fin) {
                throw runtime_error("failed to read fvecs payload from: " + path);
            }
        }
        return out;
    }

    if (ext == ".bvecs") {
        const size_t record_size = sizeof(int) + (size_t)first_dim;
        if (file_size % record_size != 0) {
            throw runtime_error("bvecs file size is not aligned: " + path);
        }
        out.rows = (int)(file_size / record_size);
        out.dim = first_dim;
        out.values.resize((size_t)out.rows * out.dim);
        vector<unsigned char> buffer(out.dim);
        for (int i = 0; i < out.rows; i++) {
            int dim = 0;
            fin.read(reinterpret_cast<char *>(&dim), sizeof(int));
            if (!fin || dim != out.dim) {
                throw runtime_error("inconsistent bvecs dimension in: " + path);
            }
            fin.read(reinterpret_cast<char *>(buffer.data()), out.dim);
            if (!fin) {
                throw runtime_error("failed to read bvecs payload from: " + path);
            }
            for (int j = 0; j < out.dim; j++) {
                out.values[(size_t)i * out.dim + j] = (float)buffer[j];
            }
        }
        return out;
    }

    throw runtime_error("unsupported dense vector format: " + path);
}

inline GroundTruth load_ground_truth(const string &path) {
    ifstream fin(path, ios::binary);
    if (!fin) {
        throw runtime_error("failed to open ground truth file: " + path);
    }
    const string ext = get_ext_lower(path);
    if (ext != ".ivecs") {
        throw runtime_error("unsupported ground truth format: " + path);
    }

    int first_dim = 0;
    fin.read(reinterpret_cast<char *>(&first_dim), sizeof(int));
    if (!fin || first_dim <= 0) {
        throw runtime_error("invalid ivecs header: " + path);
    }
    fin.seekg(0, ios::end);
    const size_t file_size = (size_t)fin.tellg();
    fin.seekg(0, ios::beg);

    const size_t record_size = sizeof(int) + sizeof(int) * (size_t)first_dim;
    if (file_size % record_size != 0) {
        throw runtime_error("ivecs file size is not aligned: " + path);
    }

    GroundTruth out;
    out.rows = (int)(file_size / record_size);
    out.width = first_dim;
    out.ids.resize((size_t)out.rows * out.width);
    for (int i = 0; i < out.rows; i++) {
        int dim = 0;
        fin.read(reinterpret_cast<char *>(&dim), sizeof(int));
        if (!fin || dim != out.width) {
            throw runtime_error("inconsistent ivecs width in: " + path);
        }
        fin.read(reinterpret_cast<char *>(out.ids.data() + (size_t)i * out.width), sizeof(int) * out.width);
        if (!fin) {
            throw runtime_error("failed to read ivecs payload from: " + path);
        }
    }
    return out;
}

inline bool try_pick_existing(const fs::path &root, const vector<string> &candidates, string &picked) {
    for (const auto &candidate : candidates) {
        fs::path path = root / candidate;
        if (fs::exists(path)) {
            picked = path.string();
            return true;
        }
    }
    return false;
}

inline void resolve_dataset_paths(BenchmarkConfig &config) {
    if (config.mode != "dataset")
        return;

    if (!config.dataset_dir.empty()) {
        const fs::path root(config.dataset_dir);
        if (config.base_path.empty()) {
            if (!try_pick_existing(root, {"base.fvecs", "base.bvecs", "learn.fvecs", "learn.bvecs"}, config.base_path)) {
                throw runtime_error("failed to resolve base vectors under dataset dir: " + config.dataset_dir);
            }
        }
        if (config.query_path.empty()) {
            if (!try_pick_existing(root, {"query.fvecs", "query.bvecs", "queries.fvecs", "queries.bvecs"}, config.query_path)) {
                throw runtime_error("failed to resolve query vectors under dataset dir: " + config.dataset_dir);
            }
        }
        if (config.gt_path.empty()) {
            try_pick_existing(root, {"groundtruth.ivecs", "gt.ivecs", "gnd.ivecs"}, config.gt_path);
        }
        if (config.dataset_name.empty()) {
            config.dataset_name = root.filename().string();
        }
    }

    if (config.base_path.empty() || config.query_path.empty()) {
        throw runtime_error("dataset mode requires base/query paths or a dataset directory");
    }
}

} // namespace benchmark
