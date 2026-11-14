#include "Checks.h"
#include "benchmark/BenchmarkCLI.h"
#include "benchmark/BenchmarkDataIO.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace tests {
namespace {

struct TempDirectory {
    std::filesystem::path path;

    TempDirectory() {
        std::string pattern = (std::filesystem::temp_directory_path() / "rnndescent-test-XXXXXX").string();
        const char *directory = mkdtemp(pattern.data());
        require(directory != nullptr, "could not create temporary test directory");
        path = directory;
    }

    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

template <class Value, size_t Size>
void write_record(std::ofstream &out, int32_t width, const std::array<Value, Size> &values) {
    out.write(reinterpret_cast<const char *>(&width), sizeof(width));
    out.write(reinterpret_cast<const char *>(values.data()), sizeof(Value) * values.size());
    require(bool(out), "could not write test record");
}

benchmark::BenchmarkConfig parse(std::vector<std::string> args) {
    args.insert(args.begin(), "algorithm");
    std::vector<char *> argv;
    for (auto &arg : args)
        argv.push_back(arg.data());
    return benchmark::parse_args(static_cast<int>(argv.size()), argv.data());
}

} // namespace

void test_benchmark() {
    auto config = parse({"--dim", "256", "--neighbor-pool-gb", "2", "--pca-out", "65"});
    benchmark::apply_solution_preset(config);
    require(config.dim_explicit && config.dim == 256, "explicit dimension");
    require(config.build_config.neighbor_pool_size_limit_bytes == 2LL * 1024 * 1024 * 1024, "GiB budget conversion");
    require(config.pca_config.enabled && config.pca_config.out_dim == 65, "PCA configuration");
    auto defaults = parse({});
    benchmark::apply_solution_preset(defaults);
    require(defaults.build_config.neighbor_pool_size_limit_bytes == 0, "neighbor cache must be off by default");
    require_error([] { parse({"--neighbor-pool-gb", "-1"}); }, "non-negative integer");
    require_error([] { parse({"--neighbor-pool-gb", "8589934592"}); }, "fits in bytes");
    require_error([] { parse({"--neighbor-pool-gb", "2.5"}); }, "non-negative integer");
    require_error([] { parse({"--dim"}); }, "missing value");

    const TempDirectory tmp;
    const auto previous_path = std::filesystem::current_path();
    struct RestorePath {
        std::filesystem::path path;
        ~RestorePath() {
            std::error_code error;
            std::filesystem::current_path(path, error);
        }
    } restore{previous_path};
    std::filesystem::current_path(tmp.path);
    std::filesystem::create_directories("benches/datasets/sift");
    require(setenv("DATASET", "sift", 1) == 0, "could not set dataset name");
    auto auto_dataset = parse({});
    benchmark::apply_default_dataset_mode(auto_dataset);
    require(auto_dataset.mode == "dataset" && auto_dataset.dataset_name == "sift", "implicit dataset selection");
    auto explicit_random = parse({"--mode", "random", "--dim", "256"});
    benchmark::apply_default_dataset_mode(explicit_random);
    require(explicit_random.mode == "random", "explicit random mode must override dataset auto-detection");

    const auto base = (tmp.path / "base.fvecs").string();
    const auto query = (tmp.path / "query.bvecs").string();
    const auto gt = (tmp.path / "groundtruth.ivecs").string();
    {
        std::ofstream out(base, std::ios::binary);
        write_record(out, 2, std::array<float, 2>{1.25f, -7.5f});
        write_record(out, 2, std::array<float, 2>{3.0f, 4.0f});
    }
    {
        std::ofstream out(query, std::ios::binary);
        write_record(out, 2, std::array<uint8_t, 2>{0, 255});
    }
    {
        std::ofstream out(gt, std::ios::binary);
        write_record(out, 2, std::array<int32_t, 2>{1, 0});
    }

    auto dataset = parse({"--dataset-dir", tmp.path.string()});
    benchmark::resolve_dataset_paths(dataset);
    require(dataset.base_path == base && dataset.query_path == query && dataset.gt_path == gt, "dataset file aliases");
    const auto base_values = benchmark::load_dense_vectors(dataset.base_path);
    const auto query_values = benchmark::load_dense_vectors(dataset.query_path);
    const auto truth = benchmark::load_ground_truth(dataset.gt_path);
    require(base_values.rows == 2 && base_values.dim == 2 && base_values.values == std::vector<float>{1.25f, -7.5f, 3.0f, 4.0f},
            "fvecs row order and values");
    require(query_values.rows == 1 && query_values.dim == 2 && query_values.values == std::vector<float>{0, 255},
            "bvecs bytes must be unsigned");
    require(truth.rows == 1 && truth.width == 2 && truth.ids == std::vector<int>{1, 0}, "ivecs IDs");

    {
        std::ofstream out(base, std::ios::binary | std::ios::app);
        write_record(out, 3, std::array<float, 2>{5.0f, 6.0f});
    }
    require_error([&] { benchmark::load_dense_vectors(base); }, "inconsistent fvecs dimension");
    {
        std::ofstream out(gt, std::ios::binary | std::ios::app);
        write_record(out, 2, std::array<int32_t, 1>{0});
    }
    require_error([&] { benchmark::load_ground_truth(gt); }, "file size is not aligned");
}

} // namespace tests
