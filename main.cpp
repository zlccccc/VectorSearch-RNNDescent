#include "solution/solution.h"
#include <bits/stdc++.h>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

namespace {

struct DenseVectors {
    vector<float> values;
    int rows = 0;
    int dim = 0;
};

struct GroundTruth {
    vector<int> ids;
    int rows = 0;
    int width = 0;
};

struct BenchmarkConfig {
    string mode = "random";
    string dataset_name;
    string dataset_dir;
    string base_path;
    string query_path;
    string gt_path;
    int data_size = 100000;
    int dim = 256;
    int test_iter = 10000;
    int exact_check_queries = 20;
    int output_iter = 5;
    int topk = 10;
    int repeat = 10;
    int warmup_topk = 10;
    int override_k0 = -1;
    int override_num_initialize = -1;
    int override_search_l = -1;
    int override_refine_max = -1;
    int override_beam_size = -1;
    long long override_neighbor_pool_gb = -1;
    bool dim_explicit = false;
    bool override_pca_enabled = false;
    bool pca_enabled = false;
    int pca_out_dim = 0;
    rnndescent::RNNDescent::BuildConfig build_config;
    rnndescent::RNNDescent::SearchConfig search_config;
    rnndescent::IndexRNNDescent::PCAConfig pca_config;
};

struct AccuracyStats {
    int checked_queries = 0;
    int recall_k = 0;
    double recall1 = 0.0;
    double recallk = 0.0;
    double avg_distance_error = 0.0;
    double max_distance_error = 0.0;
    double avg_top1_gap = 0.0;
    string top1_gap_label = "exact";
};

struct BenchmarkStats {
    double build_seconds = 0.0;
    vector<float> latency_ms;
    AccuracyStats accuracy;
};

string normalize_dataset_name(string dataset_name) {
    transform(dataset_name.begin(), dataset_name.end(), dataset_name.begin(), [](unsigned char c) { return static_cast<char>(tolower(c)); });
    return dataset_name;
}

void apply_common_config(BenchmarkConfig &config) {
    const int thread_limit = std::min(16, omp_get_max_threads());
    config.build_config.num_threads = thread_limit;
    config.search_config.num_threads = thread_limit;
    config.search_config.beam_size = 8;
    config.build_config.S = 196;
    config.build_config.R = 2048;
    config.build_config.T1 = 4;
    config.build_config.T2 = 15;
    config.pca_config.enabled = false;
    config.pca_config.out_dim = 0;
}

void apply_dimension_default_preset(BenchmarkConfig &config) {
    if (config.dim == 256) {
        config.search_config.num_initialize = 1024;
        config.search_config.search_L = 512;
        config.search_config.refine_max = 384;
        config.build_config.K0 = 48;
    } else if (config.dim == 512) {
        config.search_config.num_initialize = 1024;
        config.search_config.search_L = 256;
        config.search_config.refine_max = 64;
        config.build_config.K0 = 64;
    } else if (config.dim == 1024) {
        config.search_config.num_initialize = 160;
        config.search_config.search_L = 128;
        config.search_config.refine_max = 128;
        config.build_config.K0 = 32;
    } else if (config.dim == 1536) {
        config.search_config.num_initialize = 512;
        config.search_config.search_L = 192;
        config.search_config.refine_max = 128;
        config.build_config.K0 = 48;
    } else {
        throw runtime_error("Unsupported dimension");
    }
}

bool apply_dataset_preset(BenchmarkConfig &config) {
    const string key = normalize_dataset_name(config.dataset_name);
    if (key == "siftsmall" || key == "sift") {
        config.search_config.num_initialize = 192;
        config.search_config.search_L = 128;
        config.search_config.refine_max = 64;
        config.build_config.K0 = 32;
        return true;
    }
    if (key == "gist") {
        config.search_config.num_initialize = 768;
        config.search_config.search_L = 320;
        config.search_config.refine_max = 256;
        config.build_config.K0 = 80;
        return true;
    }
    return false;
}

void apply_config_overrides(BenchmarkConfig &config) {
    if (config.override_k0 > 0)
        config.build_config.K0 = config.override_k0;
    if (config.override_num_initialize > 0)
        config.search_config.num_initialize = config.override_num_initialize;
    if (config.override_search_l > 0)
        config.search_config.search_L = config.override_search_l;
    if (config.override_refine_max > 0)
        config.search_config.refine_max = config.override_refine_max;
    if (config.override_beam_size > 0)
        config.search_config.beam_size = config.override_beam_size;
    if (config.override_neighbor_pool_gb >= 0)
        config.build_config.neighbor_pool_size_limit_bytes = config.override_neighbor_pool_gb * 1024ll * 1024 * 1024;
    if (config.override_pca_enabled) {
        config.pca_config.enabled = config.pca_enabled;
        config.pca_config.out_dim = config.pca_enabled ? config.pca_out_dim : 0;
    }
}

void apply_solution_preset(BenchmarkConfig &config) {
    apply_common_config(config);
    if (config.mode == "random" || normalize_dataset_name(config.dataset_name) == "random") {
        apply_dimension_default_preset(config);
    } else if (!apply_dataset_preset(config)) {
        apply_dimension_default_preset(config);
    }
    apply_config_overrides(config);
}

void print_solution_config(const BenchmarkConfig &config) {
    cout << "build config:"
         << " T1=" << config.build_config.T1 << " T2=" << config.build_config.T2 << " S=" << config.build_config.S << " R=" << config.build_config.R
         << " K0=" << config.build_config.K0 << " num_threads=" << config.build_config.num_threads
         << " neighbor_pool_limit_gb=" << (double)config.build_config.neighbor_pool_size_limit_bytes / 1024 / 1024 / 1024 << endl;
    cout << "search config:"
         << " num_initialize=" << config.search_config.num_initialize << " search_L=" << config.search_config.search_L
         << " beam_size=" << config.search_config.beam_size << " refine_max=" << config.search_config.refine_max
         << " num_threads=" << config.search_config.num_threads << endl;
    cout << "pca config:"
         << " enabled=" << (config.pca_config.enabled ? 1 : 0) << " out_dim=" << config.pca_config.out_dim << endl;
}

string sanitize_name(string name);
string timestamp_string();
fs::path benchmark_results_root();
void write_benchmark_report(const BenchmarkConfig &config, const BenchmarkStats &stats);

float calc_l2_sqr(const vector<float> &lhs, int lhs_idx, const vector<float> &rhs, int rhs_idx, int dim) {
    float res = 0.0f;
    const size_t lhs_offset = 1ULL * lhs_idx * dim;
    const size_t rhs_offset = 1ULL * rhs_idx * dim;
    for (int i = 0; i < dim; i++) {
        const float diff = lhs[lhs_offset + i] - rhs[rhs_offset + i];
        res += diff * diff;
    }
    return res;
}

void randvector(vector<float> &data, int row_idx, int dim) {
    float sum = 0.0f;
    const size_t offset = 1ULL * row_idx * dim;
    for (int i = 0; i < dim; i++) {
        data[offset + i] = rand() * 1.0f / RAND_MAX;
        sum += data[offset + i] * data[offset + i];
    }
    sum = sqrt(sum);
    for (int i = 0; i < dim; i++) {
        data[offset + i] /= sum;
    }
}

vector<pair<float, int>> exact_topk(const vector<float> &dataset, const vector<float> &query, int query_idx, int data_size, int dim, int topk) {
    priority_queue<pair<float, int>> heap;
    for (int i = 0; i < data_size; i++) {
        const float dist = calc_l2_sqr(dataset, i, query, query_idx, dim);
        if ((int)heap.size() < topk) {
            heap.push({dist, i});
        } else if (dist < heap.top().first || (dist == heap.top().first && i < heap.top().second)) {
            heap.pop();
            heap.push({dist, i});
        }
    }

    vector<pair<float, int>> result(heap.size());
    for (int i = (int)result.size() - 1; i >= 0; i--) {
        result[i] = heap.top();
        heap.pop();
    }
    sort(result.begin(), result.end(), [](const pair<float, int> &lhs, const pair<float, int> &rhs) {
        if (lhs.first == rhs.first) {
            return lhs.second < rhs.second;
        }
        return lhs.first < rhs.first;
    });
    return result;
}

string get_ext_lower(const string &path) {
    string ext = fs::path(path).extension().string();
    transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)tolower(c); });
    return ext;
}

DenseVectors load_dense_vectors(const string &path) {
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

GroundTruth load_ground_truth(const string &path) {
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

bool try_pick_existing(const fs::path &root, const vector<string> &candidates, string &picked) {
    for (const auto &candidate : candidates) {
        fs::path path = root / candidate;
        if (fs::exists(path)) {
            picked = path.string();
            return true;
        }
    }
    return false;
}

void resolve_dataset_paths(BenchmarkConfig &config) {
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

void print_sample_results(const vector<float> &dataset, const vector<float> &query, const vector<int> &result, const vector<float> &solution_distances, int dim,
                          int topk, int output_iter) {
    const int query_count = (int)query.size() / dim;
    for (int i = 0; i < min(output_iter, query_count); i++) {
        for (int k = 0; k < topk; k++)
            cout << result[i * topk + k] << " ";
        puts(" <- result id");
        for (int k = 0; k < topk; k++)
            cout << solution_distances[i * topk + k] << " ";
        puts(" <- solution distance");
        for (int k = 0; k < topk; k++) {
            const float real = calc_l2_sqr(dataset, result[i * topk + k], query, i, dim);
            cout << real << " ";
        }
        puts(" <- real distance");
    }
}

AccuracyStats evaluate_accuracy_exact(const vector<float> &dataset, const vector<float> &query, const vector<int> &result,
                                      const vector<float> &solution_distances, int data_size, int dim, int topk, int checked_queries) {
    const int query_count = (int)query.size() / dim;
    checked_queries = min(query_count, checked_queries);
    double total_recall1 = 0.0;
    double total_recall10 = 0.0;
    double max_distance_error = 0.0;
    double avg_distance_error = 0.0;
    double avg_top1_gap = 0.0;

    for (int qi = 0; qi < checked_queries; qi++) {
        auto gt = exact_topk(dataset, query, qi, data_size, dim, topk);
        unordered_set<int> gt_ids;
        gt_ids.reserve(gt.size() * 2);
        for (const auto &item : gt)
            gt_ids.insert(item.second);

        int hit = 0;
        for (int k = 0; k < topk; k++) {
            const int id = result[qi * topk + k];
            if (gt_ids.count(id))
                hit++;
            const float real = calc_l2_sqr(dataset, id, query, qi, dim);
            const double err = fabs(real - solution_distances[qi * topk + k]);
            max_distance_error = max(max_distance_error, err);
            avg_distance_error += err;
        }
        total_recall1 += result[qi * topk] == gt[0].second;
        total_recall10 += hit * 1.0 / topk;
        avg_top1_gap += solution_distances[qi * topk] - gt[0].first;
    }

    AccuracyStats stats;
    stats.checked_queries = checked_queries;
    stats.recall_k = topk;
    stats.recall1 = total_recall1 / checked_queries;
    stats.recallk = total_recall10 / checked_queries;
    stats.avg_distance_error = avg_distance_error / (checked_queries * topk);
    stats.max_distance_error = max_distance_error;
    stats.avg_top1_gap = avg_top1_gap / checked_queries;
    stats.top1_gap_label = "exact";
    cout << "exact-check queries: " << stats.checked_queries << endl;
    cout << "recall@1: " << stats.recall1 << endl;
    cout << "recall@" << topk << ": " << stats.recallk << endl;
    cout << "avg returned distance abs error: " << stats.avg_distance_error << endl;
    cout << "max returned distance abs error: " << stats.max_distance_error << endl;
    cout << "avg top1 distance gap vs exact: " << stats.avg_top1_gap << endl;
    return stats;
}

AccuracyStats evaluate_accuracy_with_gt(const vector<float> &dataset, const vector<float> &query, const GroundTruth &gt, const vector<int> &result,
                                        const vector<float> &solution_distances, int dim, int topk) {
    const int query_count = min((int)query.size() / dim, gt.rows);
    const int recall_k = min(topk, gt.width);
    double total_recall1 = 0.0;
    double total_recallk = 0.0;
    double max_distance_error = 0.0;
    double avg_distance_error = 0.0;
    double avg_top1_gap = 0.0;

    for (int qi = 0; qi < query_count; qi++) {
        unordered_set<int> gt_ids;
        gt_ids.reserve(recall_k * 2);
        for (int k = 0; k < recall_k; k++) {
            gt_ids.insert(gt.ids[(size_t)qi * gt.width + k]);
        }

        int hit = 0;
        for (int k = 0; k < topk; k++) {
            const int id = result[qi * topk + k];
            if (gt_ids.count(id))
                hit++;
            const float real = calc_l2_sqr(dataset, id, query, qi, dim);
            const double err = fabs(real - solution_distances[qi * topk + k]);
            max_distance_error = max(max_distance_error, err);
            avg_distance_error += err;
        }

        const int gt_top1 = gt.ids[(size_t)qi * gt.width];
        total_recall1 += result[qi * topk] == gt_top1;
        total_recallk += hit * 1.0 / recall_k;
        const float gt_top1_dist = calc_l2_sqr(dataset, gt_top1, query, qi, dim);
        avg_top1_gap += solution_distances[qi * topk] - gt_top1_dist;
    }

    AccuracyStats stats;
    stats.checked_queries = query_count;
    stats.recall_k = recall_k;
    stats.recall1 = total_recall1 / query_count;
    stats.recallk = total_recallk / query_count;
    stats.avg_distance_error = avg_distance_error / (query_count * topk);
    stats.max_distance_error = max_distance_error;
    stats.avg_top1_gap = avg_top1_gap / query_count;
    stats.top1_gap_label = "gt top1";
    cout << "gt-check queries: " << stats.checked_queries << endl;
    cout << "recall@1: " << stats.recall1 << endl;
    cout << "recall@" << recall_k << ": " << stats.recallk << endl;
    cout << "avg returned distance abs error: " << stats.avg_distance_error << endl;
    cout << "max returned distance abs error: " << stats.max_distance_error << endl;
    cout << "avg top1 distance gap vs gt top1: " << stats.avg_top1_gap << endl;
    return stats;
}

void run_random_benchmark(BenchmarkConfig config) {
    Solution solution;
    BenchmarkStats stats;
    puts("start random benchmark");
    vector<float> dataset(1ULL * config.data_size * config.dim);
    srand(0);
    for (int i = 0; i < config.data_size; i++)
        randvector(dataset, i, config.dim);

    vector<float> query(1ULL * config.test_iter * config.dim);
    for (int i = 0; i < config.test_iter; i++)
        randvector(query, i, config.dim);

    vector<int> result((size_t)config.topk * config.test_iter);
    apply_solution_preset(config);
    config.dataset_name = "random";
    print_solution_config(config);
    const auto before_build = chrono::high_resolution_clock::now();
    solution.build(config.dim, dataset, config.warmup_topk, config.build_config, config.search_config, config.pca_config);
    const auto after_build = chrono::high_resolution_clock::now();
    stats.build_seconds = chrono::duration<double>(after_build - before_build).count();
    cout << "build time: " << stats.build_seconds << endl;

    for (int i = 0; i < config.repeat; i++) {
        const auto before_test = chrono::high_resolution_clock::now();
        solution.search(query, result, config.topk);
        const auto after_test = chrono::high_resolution_clock::now();
        const float t = chrono::duration<double, milli>(after_test - before_test).count() / config.test_iter;
        stats.latency_ms.push_back(t);
        cout << "[" << i << "] average test time: " << t << " ms; " << 1000. / t << " offline score" << endl;
    }

    stats.accuracy = evaluate_accuracy_exact(dataset, query, result, solution.distances, config.data_size, config.dim, config.topk, config.exact_check_queries);
    print_sample_results(dataset, query, result, solution.distances, config.dim, config.topk, config.output_iter);
    write_benchmark_report(config, stats);
    solution.reset();
}

void run_dataset_benchmark(BenchmarkConfig config) {
    BenchmarkStats stats;
    resolve_dataset_paths(config);
    cout << "dataset mode: " << (config.dataset_name.empty() ? string("custom") : config.dataset_name) << endl;
    cout << "base: " << config.base_path << endl;
    cout << "query: " << config.query_path << endl;
    if (!config.gt_path.empty())
        cout << "groundtruth: " << config.gt_path << endl;

    const DenseVectors dataset = load_dense_vectors(config.base_path);
    const DenseVectors query = load_dense_vectors(config.query_path);
    if (dataset.dim != query.dim) {
        throw runtime_error("dataset/query dimension mismatch");
    }

    GroundTruth gt;
    if (!config.gt_path.empty()) {
        gt = load_ground_truth(config.gt_path);
        if (gt.rows != query.rows) {
            throw runtime_error("ground truth row count does not match query count");
        }
    }

    cout << "loaded base vectors: " << dataset.rows << " x " << dataset.dim << endl;
    cout << "loaded query vectors: " << query.rows << " x " << query.dim << endl;
    if (gt.rows > 0)
        cout << "loaded ground truth: " << gt.rows << " x " << gt.width << endl;

    Solution solution;
    vector<int> result((size_t)config.topk * query.rows);
    apply_solution_preset(config);
    print_solution_config(config);
    const auto before_build = chrono::high_resolution_clock::now();
    solution.build(dataset.dim, dataset.values, config.warmup_topk, config.build_config, config.search_config, config.pca_config);
    const auto after_build = chrono::high_resolution_clock::now();
    stats.build_seconds = chrono::duration<double>(after_build - before_build).count();
    cout << "build time: " << stats.build_seconds << endl;

    for (int i = 0; i < config.repeat; i++) {
        const auto before_test = chrono::high_resolution_clock::now();
        solution.search(query.values, result, config.topk);
        const auto after_test = chrono::high_resolution_clock::now();
        const float t = chrono::duration<double, milli>(after_test - before_test).count() / query.rows;
        stats.latency_ms.push_back(t);
        cout << "[" << i << "] average test time: " << t << " ms/query; " << 1000. / t << " qps(k)" << endl;
    }

    if (gt.rows > 0) {
        stats.accuracy = evaluate_accuracy_with_gt(dataset.values, query.values, gt, result, solution.distances, dataset.dim, config.topk);
    }
    print_sample_results(dataset.values, query.values, result, solution.distances, dataset.dim, config.topk, config.output_iter);
    write_benchmark_report(config, stats);
    solution.reset();
}

string sanitize_name(string name) {
    if (name.empty())
        return "random";
    for (char &c : name) {
        const bool ok = isalnum((unsigned char)c) || c == '-' || c == '_';
        if (!ok)
            c = '_';
    }
    return name;
}

string timestamp_string() {
    const auto now = chrono::system_clock::now();
    const time_t tt = chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&tt);
    ostringstream oss;
    oss << put_time(&tm, "%Y%m%d-%H%M%S");
    return oss.str();
}

fs::path benchmark_results_root() {
    if (const char *result_dir = std::getenv("RESULT_DIR"); result_dir != nullptr && *result_dir != '\0') {
        return fs::path(result_dir);
    }

    const fs::path cwd = fs::current_path();
    if (fs::exists(cwd / "CMakeCache.txt") && fs::exists(cwd.parent_path() / "CMakeLists.txt")) {
        return cwd.parent_path() / "benches" / "results";
    }
    return cwd / "benches" / "results";
}

void write_benchmark_report(const BenchmarkConfig &config, const BenchmarkStats &stats) {
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
    if (config.dataset_name.empty() == false)
        oss << "dataset_name=" << config.dataset_name << char(10);
    if (config.dataset_dir.empty() == false)
        oss << "dataset_dir=" << config.dataset_dir << char(10);
    if (config.base_path.empty() == false)
        oss << "base_path=" << config.base_path << char(10);
    if (config.query_path.empty() == false)
        oss << "query_path=" << config.query_path << char(10);
    if (config.gt_path.empty() == false)
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
    if (stats.latency_ms.empty() == false) {
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
    if (report_file.good() == false || latest_file.good() == false) {
        throw runtime_error("failed to open benchmark report output under: " + out_dir.string());
    }
    report_file << report;
    latest_file << report;
    cout << "saved benchmark report: " << report_path << endl;
}

void print_usage(const char *prog) {
    cout << "Usage:\n";
    cout << "  " << prog << "                      # random benchmark\n";
    cout << "  " << prog << " --mode dataset --dataset-dir <dir> [--topk 10] [--repeat 10]\n";
    cout << "  " << prog << " --mode dataset --base <base.fvecs|bvecs> --query <query.fvecs|bvecs> [--gt <groundtruth.ivecs>]\n";
    cout << "Optional args:\n";
    cout << "  --data-size <n> --dim <d> --test-iter <n> --topk <k> --repeat <n>\n";
    cout << "  --output-iter <n> --exact-check-queries <n> --warmup-topk <k>\n";
    cout << "  --K0 <n> --num-initialize <n> --search-L <n> --refine-max <n> [--beam-size <n>]\n";
    cout << "  --neighbor-pool-gb <n>\n";
    cout << "  --enable-pca --disable-pca --pca-out <n>\n";
}

BenchmarkConfig parse_args(int argc, char **argv) {
    BenchmarkConfig config;
    for (int i = 1; i < argc; i++) {
        const string arg = argv[i];
        auto require_value = [&](const string &name) -> string {
            if (i + 1 >= argc)
                throw runtime_error("missing value for argument: " + name);
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "--mode") {
            config.mode = require_value(arg);
        } else if (arg == "--dataset-name") {
            config.dataset_name = require_value(arg);
        } else if (arg == "--dataset-dir") {
            config.dataset_dir = require_value(arg);
            config.mode = "dataset";
        } else if (arg == "--base") {
            config.base_path = require_value(arg);
            config.mode = "dataset";
        } else if (arg == "--query") {
            config.query_path = require_value(arg);
            config.mode = "dataset";
        } else if (arg == "--gt") {
            config.gt_path = require_value(arg);
            config.mode = "dataset";
        } else if (arg == "--data-size") {
            config.data_size = stoi(require_value(arg));
        } else if (arg == "--dim") {
            config.dim = stoi(require_value(arg));
            config.dim_explicit = true;
        } else if (arg == "--test-iter") {
            config.test_iter = stoi(require_value(arg));
        } else if (arg == "--topk") {
            config.topk = stoi(require_value(arg));
        } else if (arg == "--repeat") {
            config.repeat = stoi(require_value(arg));
        } else if (arg == "--output-iter") {
            config.output_iter = stoi(require_value(arg));
        } else if (arg == "--exact-check-queries") {
            config.exact_check_queries = stoi(require_value(arg));
        } else if (arg == "--warmup-topk") {
            config.warmup_topk = stoi(require_value(arg));
        } else if (arg == "--K0") {
            config.override_k0 = stoi(require_value(arg));
        } else if (arg == "--num-initialize") {
            config.override_num_initialize = stoi(require_value(arg));
        } else if (arg == "--search-L") {
            config.override_search_l = stoi(require_value(arg));
        } else if (arg == "--refine-max") {
            config.override_refine_max = stoi(require_value(arg));
        } else if (arg == "--beam-size") {
            config.override_beam_size = stoi(require_value(arg));
        } else if (arg == "--neighbor-pool-gb") {
            config.override_neighbor_pool_gb = stoll(require_value(arg));
        } else if (arg == "--enable-pca") {
            config.override_pca_enabled = true;
            config.pca_enabled = true;
        } else if (arg == "--disable-pca") {
            config.override_pca_enabled = true;
            config.pca_enabled = false;
            config.pca_out_dim = 0;
        } else if (arg == "--pca-out") {
            config.override_pca_enabled = true;
            config.pca_enabled = true;
            config.pca_out_dim = stoi(require_value(arg));
        } else {
            throw runtime_error("unknown argument: " + arg);
        }
    }

    if (config.topk <= 0 || config.repeat <= 0 || config.warmup_topk <= 0) {
        throw runtime_error("topk/repeat/warmup-topk must be positive");
    }
    return config;
}

} // namespace

int main(int argc, char **argv) {
    try {
        BenchmarkConfig config = parse_args(argc, argv);
        if (config.mode == "dataset") {
            run_dataset_benchmark(config);
        } else if (config.mode == "random") {
            if (config.dim_explicit) {
                run_random_benchmark(config);
            } else {
                for (int dim : {256, 512, 1024}) {
                    BenchmarkConfig current = config;
                    current.dim = dim;
                    cout << "\n=== random benchmark dim=" << dim << " ===" << endl;
                    run_random_benchmark(current);
                }
            }
        } else {
            throw runtime_error("unsupported mode: " + config.mode);
        }
        return 0;
    } catch (const exception &e) {
        cerr << "Error: " << e.what() << endl;
        print_usage(argv[0]);
        return 1;
    }
}
