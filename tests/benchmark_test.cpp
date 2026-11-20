#include "benchmark/BenchmarkCLI.h"
#include "benchmark/BenchmarkDataIO.h"
#include "benchmark/BenchmarkRunner.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <gmock/gmock.h>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace tests {
namespace {

struct TempDirectory {
    std::filesystem::path path;

    TempDirectory() {
        std::string pattern = (std::filesystem::temp_directory_path() / "rnndescent-test-XXXXXX").string();
        const char *directory = mkdtemp(pattern.data());
        if (directory == nullptr)
            throw std::runtime_error("could not create temporary test directory");
        path = directory;
    }

    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

struct ScopedEnvVar {
    ScopedEnvVar(const char *name, const char *value) : name(name) {
        if (const char *previous = std::getenv(name))
            previous_value = previous;
        if (setenv(name, value, 1) != 0)
            throw std::runtime_error("could not set test environment variable");
    }

    ~ScopedEnvVar() {
        if (previous_value)
            setenv(name, previous_value->c_str(), 1);
        else
            unsetenv(name);
    }

    const char *name;
    std::optional<std::string> previous_value;
};

template <class Value, size_t Size>
void write_record(std::ofstream &out, int32_t width, const std::array<Value, Size> &values) {
    out.write(reinterpret_cast<const char *>(&width), sizeof(width));
    out.write(reinterpret_cast<const char *>(values.data()), sizeof(Value) * values.size());
    ASSERT_TRUE(out) << "could not write test record";
}

benchmark::BenchmarkConfig parse(std::vector<std::string> args) {
    args.insert(args.begin(), "algorithm");
    std::vector<char *> argv;
    for (auto &arg : args)
        argv.push_back(arg.data());
    return benchmark::parse_args(static_cast<int>(argv.size()), argv.data());
}

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

TEST(BenchmarkCliTest, AppliesExplicitSettingsAndCacheDefaults) {
    auto config = parse({"--dim", "256", "--neighbor-pool-gb", "2", "--pca-out", "65"});
    benchmark::apply_solution_preset(config);
    EXPECT_TRUE(config.dim_explicit);
    EXPECT_EQ(config.dim, 256);
    EXPECT_EQ(config.build_config.neighbor_pool_size_limit_bytes, 2LL * 1024 * 1024 * 1024);
    EXPECT_TRUE(config.pca_config.enabled);
    EXPECT_EQ(config.pca_config.out_dim, 65);
    auto defaults = parse({});
    benchmark::apply_solution_preset(defaults);
    EXPECT_EQ(defaults.build_config.neighbor_pool_size_limit_bytes, 0);
}

TEST(BenchmarkCliTest, RejectsInvalidBudgetsAndMissingValues) {
    EXPECT_THAT([] { parse({"--neighbor-pool-gb", "-1"}); },
                ThrowsMessage<std::runtime_error>(HasSubstr("non-negative integer")));
    EXPECT_THAT([] { parse({"--neighbor-pool-gb", "8589934592"}); },
                ThrowsMessage<std::runtime_error>(HasSubstr("fits in bytes")));
    EXPECT_THAT([] { parse({"--neighbor-pool-gb", "2.5"}); },
                ThrowsMessage<std::runtime_error>(HasSubstr("non-negative integer")));
    EXPECT_THAT([] { parse({"--dim"}); }, ThrowsMessage<std::runtime_error>(HasSubstr("missing value")));
}

TEST(BenchmarkCliTest, GraphBuildOptionsOverrideDatasetPreset) {
    auto config = parse({"--mode", "dataset", "--dataset-name", "gist", "--S", "192", "--R", "512", "--K0", "96",
                         "--local-three-hop-prune"});
    benchmark::apply_solution_preset(config);
    EXPECT_EQ(config.build_config.S, 192);
    EXPECT_EQ(config.build_config.R, 512);
    EXPECT_EQ(config.build_config.K0, 96);
    EXPECT_TRUE(config.build_config.local_three_hop_pruning);
    EXPECT_THAT([] { parse({"--R", "0"}); }, ThrowsMessage<std::runtime_error>(HasSubstr("positive integer")));
    EXPECT_THAT([] { parse({"--S", "32.5"}); }, ThrowsMessage<std::runtime_error>(HasSubstr("positive integer")));
}

TEST(BenchmarkCliTest, AppliesSift100kPresetByDatasetName) {
    auto small = parse({"--mode", "dataset", "--dataset-name", "SIFT100K"});
    benchmark::apply_solution_preset(small);
    EXPECT_EQ(small.build_config.S, 64);
    EXPECT_EQ(small.build_config.R, 96);
    EXPECT_EQ(small.build_config.K0, 24);
    EXPECT_EQ(small.search_config.num_initialize, 80);
    EXPECT_EQ(small.search_config.search_L, 72);
    EXPECT_EQ(small.search_config.refine_max, 24);
    EXPECT_EQ(small.search_config.beam_size, 4);

    auto full = parse({"--mode", "dataset", "--dataset-name", "sift"});
    benchmark::apply_solution_preset(full);
    EXPECT_EQ(full.search_config.search_L, 128);
}

TEST(BenchmarkCliTest, ConfiguresFp32ShortlistAndFullRefinement) {
    for (const int limit : {0, 16, 32, 64}) {
        auto config = parse({"--mode", "dataset", "--dataset-name", "gist", "--fp32-refine-max", std::to_string(limit)});
        benchmark::apply_solution_preset(config);
        EXPECT_EQ(config.search_config.fp32_refine_max, limit);
        EXPECT_EQ(config.search_config.refine_max, 1024);
    }
    auto defaults = parse({});
    benchmark::apply_solution_preset(defaults);
    EXPECT_EQ(defaults.search_config.fp32_refine_max, 32);
    EXPECT_THAT([] { parse({"--fp32-refine-max", "-1"}); },
                ThrowsMessage<std::runtime_error>(HasSubstr("non-negative integer")));
    EXPECT_THAT([] { parse({"--fp32-refine-max", "32.5"}); },
                ThrowsMessage<std::runtime_error>(HasSubstr("non-negative integer")));
    EXPECT_THAT([] { parse({"--fp32-refine-max"}); }, ThrowsMessage<std::runtime_error>(HasSubstr("missing value")));
}

TEST(BenchmarkCliTest, ExplicitRandomModeOverridesDatasetDetection) {
    const TempDirectory tmp;
    struct RestorePath {
        std::filesystem::path cwd = std::filesystem::current_path();
        ~RestorePath() {
            std::error_code error;
            std::filesystem::current_path(cwd, error);
        }
    } restore;
    const ScopedEnvVar dataset_env("DATASET", "sift");
    std::filesystem::current_path(tmp.path);
    std::filesystem::create_directories("benches/datasets/sift");
    auto auto_dataset = parse({});
    benchmark::apply_default_dataset_mode(auto_dataset);
    EXPECT_EQ(auto_dataset.mode, "dataset");
    EXPECT_EQ(auto_dataset.dataset_name, "sift");
    auto explicit_random = parse({"--mode", "random", "--dim", "256"});
    benchmark::apply_default_dataset_mode(explicit_random);
    EXPECT_EQ(explicit_random.mode, "random");
}

TEST(BenchmarkDataTest, ResolvesAliasesAndReadsVectorRecords) {
    const TempDirectory tmp;
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
    EXPECT_EQ(dataset.base_path, base);
    EXPECT_EQ(dataset.query_path, query);
    EXPECT_EQ(dataset.gt_path, gt);
    const auto base_values = benchmark::load_dense_vectors(dataset.base_path);
    const auto query_values = benchmark::load_dense_vectors(dataset.query_path);
    const auto truth = benchmark::load_ground_truth(dataset.gt_path);
    EXPECT_EQ(base_values.rows, 2);
    EXPECT_EQ(base_values.dim, 2);
    EXPECT_EQ(base_values.values, (std::vector<float>{1.25f, -7.5f, 3.0f, 4.0f}));
    EXPECT_EQ(query_values.rows, 1);
    EXPECT_EQ(query_values.dim, 2);
    EXPECT_EQ(query_values.values, (std::vector<float>{0, 255}));
    EXPECT_EQ(truth.rows, 1);
    EXPECT_EQ(truth.width, 2);
    EXPECT_EQ(truth.ids, (std::vector<int>{1, 0}));
}

TEST(BenchmarkDataTest, RejectsInconsistentDimensionsAndTruncatedGroundTruth) {
    const TempDirectory tmp;
    const auto base = (tmp.path / "base.fvecs").string();
    const auto gt = (tmp.path / "groundtruth.ivecs").string();
    {
        std::ofstream out(base, std::ios::binary);
        write_record(out, 2, std::array<float, 2>{1.25f, -7.5f});
        write_record(out, 3, std::array<float, 2>{5.0f, 6.0f});
    }
    EXPECT_THAT([&] { benchmark::load_dense_vectors(base); },
                ThrowsMessage<std::runtime_error>(HasSubstr("inconsistent fvecs dimension")));
    {
        std::ofstream out(gt, std::ios::binary);
        write_record(out, 2, std::array<int32_t, 2>{1, 0});
        write_record(out, 2, std::array<int32_t, 1>{0});
    }
    EXPECT_THAT([&] { benchmark::load_ground_truth(gt); },
                ThrowsMessage<std::runtime_error>(HasSubstr("file size is not aligned")));
}

TEST(BenchmarkDataTest, UsesCommonSearchSettingsForCustomDatasets) {
    const TempDirectory tmp;
    const auto base = (tmp.path / "base.fvecs").string();
    const auto query = (tmp.path / "query.fvecs").string();
    const auto results = (tmp.path / "results").string();
    const ScopedEnvVar result_env("RESULT_DIR", results.c_str());

    for (const int32_t dim : {1024, 65}) {
        SCOPED_TRACE(dim);
        std::mt19937 rng(71);
        std::vector<float> row(dim);
        {
            std::ofstream base_file(base, std::ios::binary);
            std::ofstream query_file(query, std::ios::binary);
            for (int i = 0; i < 200; ++i) {
                for (float &coordinate : row)
                    coordinate = static_cast<float>(rng() % 1000) / 100.0f;
                base_file.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
                base_file.write(reinterpret_cast<const char *>(row.data()), dim * sizeof(float));
            }
            query_file.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
            query_file.write(reinterpret_cast<const char *>(row.data()), dim * sizeof(float));
            ASSERT_TRUE(base_file && query_file);
        }

        auto config = parse({"--base", base, "--query", query, "--topk", "5", "--repeat", "1",
                             "--warmup-topk", "0", "--output-iter", "0", "--refine-max", "20"});
        ASSERT_NO_THROW(benchmark::run_dataset_benchmark(config));
        std::ifstream report(tmp.path / "results" / "latest.txt");
        ASSERT_TRUE(report);
        const std::string text((std::istreambuf_iterator<char>(report)), std::istreambuf_iterator<char>());
        const rnndescent::RNNDescent::BuildConfig build_defaults;
        const rnndescent::RNNDescent::SearchConfig search_defaults;
        EXPECT_THAT(text, HasSubstr("dim=" + std::to_string(dim) + "\n"));
        EXPECT_THAT(text, HasSubstr("search_config.search_L=" + std::to_string(search_defaults.search_L) + "\n"));
        EXPECT_THAT(text, HasSubstr("build_config.K0=" + std::to_string(build_defaults.K0) + "\n"));
        EXPECT_THAT(text, HasSubstr("search_config.num_initialize=" + std::to_string(search_defaults.num_initialize) + "\n"));
        EXPECT_THAT(text, HasSubstr("search_config.refine_max=20\n"));
        EXPECT_THAT(text, HasSubstr("search_config.fp32_refine_max=32\n"));
    }
}

} // namespace
} // namespace tests
