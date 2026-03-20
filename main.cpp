#include "solution/solution.h"
#include <bits/stdc++.h>
using namespace std;

namespace {

int output_iter = 5;
int test_iter = 10000;
int exact_check_queries = 20;

float calc_l2_sqr(const float *a, const float *b, int d) {
    float res = 0.0f;
    for (int i = 0; i < d; i++) {
        float diff = a[i] - b[i];
        res += diff * diff;
    }
    return res;
}

void randvector(float *data, int d) {
    float sum = 0.0f;
    for (int i = 0; i < d; i++) {
        data[i] = rand() * 1.0f / RAND_MAX;
        sum += data[i] * data[i];
    }
    sum = sqrt(sum);
    for (int k = 0; k < d; k++) {
        data[k] = data[k] / sum;
    }
}

vector<pair<float, int>> exact_topk(const vector<float> &dataset, const float *query, int data_size, int d, int k) {
    priority_queue<pair<float, int>> heap;
    for (int i = 0; i < data_size; i++) {
        float dist = calc_l2_sqr(dataset.data() + 1LL * i * d, query, d);
        if ((int)heap.size() < k) {
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

void evaluate_accuracy(const vector<float> &dataset, const vector<float> &query, const vector<int> &result, const float *solution_distances, int data_size,
                       int d) {
    const int checked_queries = min((int)query.size() / d, exact_check_queries);
    double total_recall1 = 0.0;
    double total_recall10 = 0.0;
    double max_distance_error = 0.0;
    double avg_distance_error = 0.0;
    double avg_top1_gap = 0.0;

    for (int qi = 0; qi < checked_queries; qi++) {
        auto gt = exact_topk(dataset, query.data() + 1LL * qi * d, data_size, d, topk);
        unordered_set<int> gt_ids;
        gt_ids.reserve(gt.size() * 2);
        for (auto &item : gt) {
            gt_ids.insert(item.second);
        }

        int hit = 0;
        for (int k = 0; k < topk; k++) {
            int id = result[qi * topk + k];
            if (gt_ids.count(id)) {
                hit++;
            }
            float real = calc_l2_sqr(dataset.data() + 1LL * id * d, query.data() + 1LL * qi * d, d);
            double err = fabs(real - solution_distances[qi * topk + k]);
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
    cout << "recall@10: " << total_recall10 / checked_queries << endl;
    cout << "avg returned distance abs error: " << avg_distance_error << endl;
    cout << "max returned distance abs error: " << max_distance_error << endl;
    cout << "avg top1 distance gap vs exact: " << avg_top1_gap << endl;
}

void print_sample_results(const vector<float> &dataset, const vector<float> &query, const vector<int> &result, const float *solution_distances, int d) {
    for (int i = 0; i < min(output_iter, (int)query.size() / d); i++) {
        for (int k = 0; k < topk; k++)
            cout << result[i * topk + k] << " ";
        puts(" <- result id");
        for (int k = 0; k < topk; k++)
            cout << solution_distances[i * topk + k] << " ";
        puts(" <- solution distance");
        for (int k = 0; k < topk; k++) {
            float real = calc_l2_sqr(dataset.data() + 1LL * result[i * topk + k] * d, query.data() + 1LL * i * d, d);
            cout << real << " ";
        }
        puts(" <- real distance");
    }
}

void solve(int data_size, int d) {
    Solution solution;
    puts("start to solve the problem");
    vector<float> dataset(1LL * data_size * d);
    srand(0);
    for (int i = 0; i < data_size; i++)
        randvector(dataset.data() + 1LL * i * d, d);
    vector<float> query(1LL * test_iter * d);
    for (int i = 0; i < test_iter; i++)
        randvector(query.data() + 1LL * i * d, d);
    puts("start solve");
    vector<int> result(topk * test_iter);
    auto before_build = std::chrono::high_resolution_clock::now();
    solution.build(d, dataset);
    auto after_build = std::chrono::high_resolution_clock::now();
    cout << "build time: " << std::chrono::duration<double>(after_build - before_build).count() << endl;

    for (int i = 0; i < 10; i++) {
        auto before_test = std::chrono::high_resolution_clock::now();
        solution.search(query, result.data());
        auto after_test = std::chrono::high_resolution_clock::now();
        float t = std::chrono::duration<double, std::milli>(after_test - before_test).count() / test_iter;
        cout << "[" << i << "] average test time: " << t << " ms; " << 1000. / t << " offline score" << endl;
    }

    evaluate_accuracy(dataset, query, result, solution.distances, data_size, d);
    print_sample_results(dataset, query, result, solution.distances, d);

    solution.index->reset();
}

} // namespace

int main(int argc, char **argv) {
    solve(100000, 256);
    // solve(1000000, 512);
    // solve(1000000, 1024);
    // solve(1000000, 1536);
    return 0;
}
