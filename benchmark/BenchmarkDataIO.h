#pragma once

#include "BenchmarkTypes.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
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

namespace detail {

template <class Value>
struct VectorRecords {
    vector<Value> values;
    int rows = 0;
    int width = 0;
};

template <class Stored, class Value>
VectorRecords<Value> read_vector_records(ifstream &input, const string &path, const char *format) {
    VectorRecords<Value> records;
    input.read(reinterpret_cast<char *>(&records.width), sizeof(int));
    if (!input || records.width <= 0) {
        constexpr auto header = std::is_same_v<Value, int> ? "invalid ivecs header: " : "invalid vector file header: ";
        throw runtime_error(header + path);
    }
    input.seekg(0, ios::end);
    const auto file_size = static_cast<size_t>(input.tellg());
    input.seekg(0, ios::beg);
    const size_t payload_size = static_cast<size_t>(records.width) * sizeof(Stored);
    const size_t record_size = sizeof(int) + payload_size;
    if (file_size % record_size != 0)
        throw runtime_error(string(format) + " file size is not aligned: " + path);

    records.rows = static_cast<int>(file_size / record_size);
    records.values.resize(static_cast<size_t>(records.rows) * records.width);
    vector<Stored> buffer;
    if constexpr (!std::is_same_v<Stored, Value>)
        buffer.resize(records.width);
    for (int i = 0; i < records.rows; ++i) {
        int width = 0;
        input.read(reinterpret_cast<char *>(&width), sizeof(int));
        if (!input || width != records.width) {
            constexpr auto field = std::is_same_v<Value, int> ? " width in: " : " dimension in: ";
            throw runtime_error("inconsistent " + string(format) + field + path);
        }

        auto *row = records.values.data() + static_cast<size_t>(i) * records.width;
        auto *payload = buffer.data();
        if constexpr (std::is_same_v<Stored, Value>)
            payload = row;
        input.read(reinterpret_cast<char *>(payload), payload_size);
        if (!input)
            throw runtime_error("failed to read " + string(format) + " payload from: " + path);
        if constexpr (!std::is_same_v<Stored, Value>)
            std::copy_n(payload, records.width, row);
    }
    return records;
}

} // namespace detail

inline DenseVectors load_dense_vectors(const string &path) {
    ifstream fin(path, ios::binary);
    if (!fin)
        throw runtime_error("failed to open vector file: " + path);

    const string ext = get_ext_lower(path);
    if (ext != ".fvecs" && ext != ".bvecs")
        throw runtime_error("unsupported dense vector format: " + path);
    auto records = ext == ".fvecs" ? detail::read_vector_records<float, float>(fin, path, "fvecs")
                                  : detail::read_vector_records<unsigned char, float>(fin, path, "bvecs");
    return {std::move(records.values), records.rows, records.width};
}

inline GroundTruth load_ground_truth(const string &path) {
    ifstream fin(path, ios::binary);
    if (!fin)
        throw runtime_error("failed to open ground truth file: " + path);
    if (get_ext_lower(path) != ".ivecs")
        throw runtime_error("unsupported ground truth format: " + path);

    auto records = detail::read_vector_records<int, int>(fin, path, "ivecs");
    return {std::move(records.values), records.rows, records.width};
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
            config.dataset_name = (root.has_filename() ? root : root.parent_path()).filename().string();
        }
    }

    if (config.base_path.empty() || config.query_path.empty()) {
        throw runtime_error("dataset mode requires base/query paths or a dataset directory");
    }
}

} // namespace benchmark
