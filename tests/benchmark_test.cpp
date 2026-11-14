#include "benchmark/BenchmarkCLI.h"
#include "benchmark/BenchmarkDataIO.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gmock/gmock.h>
#include <optional>
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

TEST(BenchmarkCliTest, ExplicitRandomModeOverridesDatasetDetection) {
    const TempDirectory tmp;
    struct RestoreEnvironment {
        std::filesystem::path cwd = std::filesystem::current_path();
        std::optional<std::string> dataset;
        RestoreEnvironment() {
            if (const char *value = std::getenv("DATASET"))
                dataset = value;
        }
        ~RestoreEnvironment() {
            std::error_code error;
            std::filesystem::current_path(cwd, error);
            if (dataset)
                setenv("DATASET", dataset->c_str(), 1);
            else
                unsetenv("DATASET");
        }
    } restore;
    std::filesystem::current_path(tmp.path);
    std::filesystem::create_directories("benches/datasets/sift");
    ASSERT_EQ(setenv("DATASET", "sift", 1), 0);
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

} // namespace
} // namespace tests
