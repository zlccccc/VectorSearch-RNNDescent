#pragma once

#include "BenchmarkTypes.h"
#include "../solution/rnndescent/Logger.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

namespace benchmark {

using std::cout;
using std::endl;
using std::max;
using std::min;
using std::pair;
using std::priority_queue;
using std::sort;
using std::unordered_set;
using std::vector;

inline float calc_l2_sqr(const vector<float> &lhs, int lhs_idx, const vector<float> &rhs, int rhs_idx, int dim) {
    float res = 0.0f;
    const size_t lhs_offset = 1ULL * lhs_idx * dim;
    const size_t rhs_offset = 1ULL * rhs_idx * dim;
    for (int i = 0; i < dim; i++) {
        const float diff = lhs[lhs_offset + i] - rhs[rhs_offset + i];
        res += diff * diff;
    }
    return res;
}

inline void randvector(std::mt19937 &rng, vector<float> &data, int row_idx, int dim) {
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float sum = 0.0f;
    const size_t offset = 1ULL * row_idx * dim;
    for (int i = 0; i < dim; i++) {
        data[offset + i] = dist(rng);
        sum += data[offset + i] * data[offset + i];
    }
    sum = std::sqrt(sum);
    for (int i = 0; i < dim; i++) {
        data[offset + i] /= sum;
    }
}

inline vector<pair<float, int>> exact_topk(const vector<float> &dataset, const vector<float> &query, int query_idx, int data_size, int dim, int topk) {
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

inline void print_sample_results(const vector<float> &dataset, const vector<float> &query, const vector<int> &result,
                                 const vector<float> &solution_distances, int dim, int topk, int output_iter) {
    const int query_count = (int)query.size() / dim;
    for (int i = 0; i < min(output_iter, query_count); i++) {
        for (int k = 0; k < topk; k++)
            cout << result[i * topk + k] << " ";
        rnndescent::Logger::line(" <- result id");
        for (int k = 0; k < topk; k++)
            cout << solution_distances[i * topk + k] << " ";
        rnndescent::Logger::line(" <- solution distance");
        for (int k = 0; k < topk; k++) {
            const float real = calc_l2_sqr(dataset, result[i * topk + k], query, i, dim);
            cout << real << " ";
        }
        rnndescent::Logger::line(" <- real distance");
    }
}

inline AccuracyStats evaluate_accuracy_exact(const vector<float> &dataset, const vector<float> &query, const vector<int> &result,
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
            const double err = std::fabs(real - solution_distances[qi * topk + k]);
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

inline AccuracyStats evaluate_accuracy_with_gt(const vector<float> &dataset, const vector<float> &query, const GroundTruth &gt, const vector<int> &result,
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
            const double err = std::fabs(real - solution_distances[qi * topk + k]);
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

} // namespace benchmark
