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
};

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

void print_sample_results(const vector<float> &dataset, const vector<float> &query, const vector<int> &result, const vector<float> &solution_distances, int dim, int topk,
                          int output_iter) {
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

void evaluate_accuracy_exact(const vector<float> &dataset, const vector<float> &query, const vector<int> &result, const vector<float> &solution_distances, int data_size,
                             int dim, int topk, int checked_queries) {
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

    avg_distance_error /= checked_queries * topk;
    avg_top1_gap /= checked_queries;
    cout << "exact-check queries: " << checked_queries << endl;
    cout << "recall@1: " << total_recall1 / checked_queries << endl;
    cout << "recall@" << topk << ": " << total_recall10 / checked_queries << endl;
    cout << "avg returned distance abs error: " << avg_distance_error << endl;
    cout << "max returned distance abs error: " << max_distance_error << endl;
    cout << "avg top1 distance gap vs exact: " << avg_top1_gap << endl;
}

void evaluate_accuracy_with_gt(const vector<float> &dataset, const vector<float> &query, const GroundTruth &gt, const vector<int> &result,
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

    avg_distance_error /= query_count * topk;
    avg_top1_gap /= query_count;
    cout << "gt-check queries: " << query_count << endl;
    cout << "recall@1: " << total_recall1 / query_count << endl;
    cout << "recall@" << recall_k << ": " << total_recallk / query_count << endl;
    cout << "avg returned distance abs error: " << avg_distance_error << endl;
    cout << "max returned distance abs error: " << max_distance_error << endl;
    cout << "avg top1 distance gap vs gt top1: " << avg_top1_gap << endl;
}

void run_random_benchmark(const BenchmarkConfig &config) {
    Solution solution;
    puts("start random benchmark");
    vector<float> dataset(1ULL * config.data_size * config.dim);
    srand(0);
    for (int i = 0; i < config.data_size; i++)
        randvector(dataset, i, config.dim);

    vector<float> query(1ULL * config.test_iter * config.dim);
    for (int i = 0; i < config.test_iter; i++)
        randvector(query, i, config.dim);

    vector<int> result((size_t)config.topk * config.test_iter);
    const auto before_build = chrono::high_resolution_clock::now();
    solution.build(config.dim, dataset, config.warmup_topk);
    const auto after_build = chrono::high_resolution_clock::now();
    cout << "build time: " << chrono::duration<double>(after_build - before_build).count() << endl;

    for (int i = 0; i < config.repeat; i++) {
        const auto before_test = chrono::high_resolution_clock::now();
        solution.search(query, result, config.topk);
        const auto after_test = chrono::high_resolution_clock::now();
        const float t = chrono::duration<double, milli>(after_test - before_test).count() / config.test_iter;
        cout << "[" << i << "] average test time: " << t << " ms; " << 1000. / t << " offline score" << endl;
    }

    evaluate_accuracy_exact(dataset, query, result, solution.distances, config.data_size, config.dim, config.topk, config.exact_check_queries);
    print_sample_results(dataset, query, result, solution.distances, config.dim, config.topk, config.output_iter);
    solution.reset();
}

void run_dataset_benchmark(BenchmarkConfig config) {
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
    const auto before_build = chrono::high_resolution_clock::now();
    solution.build(dataset.dim, dataset.values, config.warmup_topk);
    const auto after_build = chrono::high_resolution_clock::now();
    cout << "build time: " << chrono::duration<double>(after_build - before_build).count() << endl;

    for (int i = 0; i < config.repeat; i++) {
        const auto before_test = chrono::high_resolution_clock::now();
        solution.search(query.values, result, config.topk);
        const auto after_test = chrono::high_resolution_clock::now();
        const float t = chrono::duration<double, milli>(after_test - before_test).count() / query.rows;
        cout << "[" << i << "] average test time: " << t << " ms/query; " << 1000. / t << " qps(k)" << endl;
    }

    if (gt.rows > 0) {
        evaluate_accuracy_with_gt(dataset.values, query.values, gt, result, solution.distances, dataset.dim, config.topk);
    }
    print_sample_results(dataset.values, query.values, result, solution.distances, dataset.dim, config.topk, config.output_iter);
    solution.reset();
}

void print_usage(const char *prog) {
    cout << "Usage:\n";
    cout << "  " << prog << "                      # random benchmark\n";
    cout << "  " << prog << " --mode dataset --dataset-dir <dir> [--topk 10] [--repeat 10]\n";
    cout << "  " << prog << " --mode dataset --base <base.fvecs|bvecs> --query <query.fvecs|bvecs> [--gt <groundtruth.ivecs>]\n";
    cout << "Optional args:\n";
    cout << "  --data-size <n> --dim <d> --test-iter <n> --topk <k> --repeat <n>\n";
    cout << "  --output-iter <n> --exact-check-queries <n> --warmup-topk <k>\n";
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
            run_random_benchmark(config);
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
