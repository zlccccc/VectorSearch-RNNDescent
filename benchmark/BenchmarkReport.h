#pragma once

#include "BenchmarkTypes.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

namespace benchmark {

namespace fs = std::filesystem;
namespace chrono = std::chrono;

using std::accumulate;
using std::max_element;
using std::min_element;
using std::ofstream;
using std::ostringstream;
using std::string;
using std::time_t;
using std::to_string;

inline string sanitize_name(string name) {
    if (name.empty())
        return "random";
    for (char &c : name) {
        const bool ok = isalnum((unsigned char)c) || c == '-' || c == '_';
        if (!ok)
            c = '_';
    }
    return name;
}

inline string timestamp_string() {
    const auto now = chrono::system_clock::now();
    const time_t tt = chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&tt);
    ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d-%H%M%S");
    return oss.str();
}

inline fs::path benchmark_results_root() {
    if (const char *result_dir = std::getenv("RESULT_DIR"); result_dir != nullptr && *result_dir != '\0') {
        return fs::path(result_dir);
    }

    const fs::path cwd = fs::current_path();
    if (fs::exists(cwd / "CMakeCache.txt") && fs::exists(cwd.parent_path() / "CMakeLists.txt")) {
        return cwd.parent_path() / "benches" / "results";
    }
    return cwd / "benches" / "results";
}

inline void write_benchmark_report(const BenchmarkConfig &config, const BenchmarkStats &stats) {
    const string dataset_key = sanitize_name(config.mode == "dataset" ? config.dataset_name : (string("random-") + to_string(config.dim) + "d"));
    const char *result_dir = std::getenv("RESULT_DIR");
    const bool has_result_dir = result_dir != nullptr && *result_dir != char(0);
    const fs::path out_dir = has_result_dir ? fs::path(result_dir) : (benchmark_results_root() / dataset_key);
    fs::create_directories(out_dir);

    const string stamp = timestamp_string();
    const fs::path report_path = out_dir / (stamp + ".txt");
    const fs::path latest_path = out_dir / "latest.txt";

    ostringstream oss;
    oss << "mode=" << config.mode << char(10);
    if (!config.dataset_name.empty())
        oss << "dataset_name=" << config.dataset_name << char(10);
    if (!config.dataset_dir.empty())
        oss << "dataset_dir=" << config.dataset_dir << char(10);
    if (!config.base_path.empty())
        oss << "base_path=" << config.base_path << char(10);
    if (!config.query_path.empty())
        oss << "query_path=" << config.query_path << char(10);
    if (!config.gt_path.empty())
        oss << "gt_path=" << config.gt_path << char(10);
    oss << "dim=" << config.dim << char(10);
    oss << "topk=" << config.topk << char(10);
    oss << "repeat=" << config.repeat << char(10);
    oss << "warmup_topk=" << config.warmup_topk << char(10);
    oss << "build_config.T1=" << config.build_config.T1 << char(10);
    oss << "build_config.T2=" << config.build_config.T2 << char(10);
    oss << "build_config.S=" << config.build_config.S << char(10);
    oss << "build_config.R=" << config.build_config.R << char(10);
    oss << "build_config.K0=" << config.build_config.K0 << char(10);
    oss << "build_config.num_threads=" << config.build_config.num_threads << char(10);
    oss << "build_config.neighbor_pool_size_limit_bytes=" << config.build_config.neighbor_pool_size_limit_bytes << char(10);
    oss << "search_config.num_initialize=" << config.search_config.num_initialize << char(10);
    oss << "search_config.search_L=" << config.search_config.search_L << char(10);
    oss << "search_config.beam_size=" << config.search_config.beam_size << char(10);
    oss << "search_config.refine_max=" << config.search_config.refine_max << char(10);
    oss << "search_config.num_threads=" << config.search_config.num_threads << char(10);
    oss << "pca_config.enabled=" << (config.pca_config.enabled ? 1 : 0) << char(10);
    oss << "pca_config.out_dim=" << config.pca_config.out_dim << char(10);
    oss << "build_seconds=" << stats.build_seconds << char(10);
    for (size_t i = 0; i < stats.latency_ms.size(); i++) {
        oss << "latency_ms[" << i << "]=" << stats.latency_ms[i] << char(10);
    }
    if (!stats.latency_ms.empty()) {
        const float min_latency = *min_element(stats.latency_ms.begin(), stats.latency_ms.end());
        const float max_latency = *max_element(stats.latency_ms.begin(), stats.latency_ms.end());
        const double avg_latency = accumulate(stats.latency_ms.begin(), stats.latency_ms.end(), 0.0) / stats.latency_ms.size();
        oss << "latency_ms_avg=" << avg_latency << char(10);
        oss << "latency_ms_min=" << min_latency << char(10);
        oss << "latency_ms_max=" << max_latency << char(10);
    }
    if (stats.accuracy.checked_queries > 0) {
        oss << "checked_queries=" << stats.accuracy.checked_queries << char(10);
        oss << "recall@1=" << stats.accuracy.recall1 << char(10);
        oss << "recall@" << stats.accuracy.recall_k << "=" << stats.accuracy.recallk << char(10);
        oss << "avg_distance_error=" << stats.accuracy.avg_distance_error << char(10);
        oss << "max_distance_error=" << stats.accuracy.max_distance_error << char(10);
        oss << "avg_top1_gap(" << stats.accuracy.top1_gap_label << ")=" << stats.accuracy.avg_top1_gap << char(10);
    }

    const string report = oss.str();
    ofstream report_file(report_path);
    ofstream latest_file(latest_path);
    if (!report_file.good() || !latest_file.good()) {
        throw std::runtime_error("failed to open benchmark report output under: " + out_dir.string());
    }
    report_file << report;
    latest_file << report;
    std::cout << "saved benchmark report: " << report_path << std::endl;
}

} // namespace benchmark
