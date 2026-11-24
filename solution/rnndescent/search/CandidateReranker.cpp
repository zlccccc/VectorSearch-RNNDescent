#include "CandidateReranker.h"

#include <algorithm>
#include <utility>

#include "../common/Assertions.h"
#include "../discomputer/PlatformSimdDistanceComputer.h"

namespace rnndescent {

CandidateReranker::CandidateReranker(const FloatMatrixView &base, int graph_dimension, int candidate_count,
                                     const config::SearchConfig &config,
                                     const int *graph_row_order)
    : CandidateReranker(SelectedDistanceComputerFactory::create_refine_search(base.data_ptr(), base.row_count(), base.dimension()), graph_dimension) {
    shortlist_enabled_ = config.rerank_quantization.has_value();
    // Byte navigation scores are already the original-dimensional byte
    // scores when PCA is disabled. Packed navigation uses a different
    // encoding from the affine packed reranker even in the original space.
    reuse_graph_scores_ = graph_dimension == base.dimension() &&
                         config.graph_quantization == QuantizedFormat::Byte &&
                         config.rerank_quantization == QuantizedFormat::Byte;
    if (shortlist_enabled_ && !reuse_graph_scores_ && config.fp32_refine_max > 0 &&
        config.fp32_refine_max < candidate_count) {
        graph_ordered_codes_ = graph_row_order != nullptr;
        quantized_ = SelectedDistanceComputerFactory::create_quantized_refine(base.data_ptr(), base.row_count(),
                                                                             base.dimension(), graph_row_order,
                                                                             *config.rerank_quantization);
    }
}

CandidateReranker::CandidateReranker(std::unique_ptr<MyDistanceComputer> fp32, int graph_dimension,
                                     std::unique_ptr<MyDistanceComputer> quantized)
    : fp32_(std::move(fp32)), quantized_(std::move(quantized)), reuse_graph_scores_(fp32_->dimension() == graph_dimension) {}

void CandidateReranker::set_queries(const FloatMatrixView &queries) {
    fp32_->set_query(queries.data_ptr(), queries.row_count());
    if (quantized_)
        quantized_->set_query(queries.data_ptr(), queries.row_count());
}

void CandidateReranker::rerank(int query_id, std::vector<SingleNeighbor> &candidates, int fp32_limit,
                               const SearchResultView &result, const int *original_ids) const {
    const size_t fp32_count = prepare_shortlist(candidates, fp32_limit, result.topk());
    score_for_rerank(query_id, candidates, fp32_count, original_ids, 0, 1);
    write_topk(candidates, result);
}

size_t CandidateReranker::prepare_shortlist(std::vector<SingleNeighbor> &candidates, int fp32_limit, int topk) const {
    if (!shortlist_enabled_ || fp32_limit <= 0)
        return candidates.size();

    const size_t fp32_count = std::min(candidates.size(), static_cast<size_t>(std::max(fp32_limit, topk)));
    if (fp32_count < candidates.size() && !quantized_) {
        RNNDESCENT_ASSERT_MSG(reuse_graph_scores_, "shortlisting requires compatible original-dimensional scores");
        std::nth_element(candidates.begin(), candidates.begin() + fp32_count, candidates.end());
        candidates.resize(fp32_count);
    }
    return fp32_count;
}

void CandidateReranker::score_for_rerank(int query_id, std::vector<SingleNeighbor> &candidates, size_t fp32_count,
                                        const int *original_ids, int worker, int workers) const {
    const auto score = [&](MyDistanceComputer &computer, const int *row_order) {
        // Assign whole four-candidate groups. The last worker also scores the tail.
        const size_t groups = candidates.size() / 4;
        const size_t begin = 4 * (groups * worker / workers);
        const size_t end = worker + 1 == workers ? candidates.size() : 4 * (groups * (worker + 1) / workers);
        if (begin == end)
            return;
        if (row_order) {
            for (size_t i = begin; i < end; ++i)
                candidates[i].id = row_order[candidates[i].id];
        }
        computer.score_candidates(query_id, candidates.data() + begin, end - begin);
    };

    if (fp32_count < candidates.size()) {
        // BFS-ordered codes keep graph IDs until the shortlist is selected.
        score(*quantized_, graph_ordered_codes_ ? nullptr : original_ids);
        if (!graph_ordered_codes_)
            original_ids = nullptr;
        if (workers > 1) {
            // Finish all coarse scores before selecting the FP32 shortlist.
#pragma omp barrier
        }
        if (worker == 0) {
            std::nth_element(candidates.begin(), candidates.begin() + fp32_count, candidates.end());
            candidates.resize(fp32_count);
        }
        if (workers > 1) {
            // Publish the shortlist and its new size before splitting FP32 work.
#pragma omp barrier
        }
    }

    score(*fp32_, original_ids);
}

void CandidateReranker::write_topk(std::vector<SingleNeighbor> &candidates, const SearchResultView &result) {
    const int topk = result.topk();
    const auto top_end = candidates.begin() + topk;
    if (top_end != candidates.end())
        std::nth_element(candidates.begin(), top_end, candidates.end());
    std::sort(candidates.begin(), top_end);

    for (int i = 0; i < topk; ++i) {
        result.indices_ptr()[i] = candidates[i].id;
        result.distances_ptr()[i] = candidates[i].distance;
    }
}

} // namespace rnndescent
