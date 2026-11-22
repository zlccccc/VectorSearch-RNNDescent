#include "CandidateReranker.h"

#include <algorithm>
#include <utility>

#include "../common/Assertions.h"
#include "../discomputer/PlatformSimdDistanceComputer.h"

namespace rnndescent {

CandidateReranker::CandidateReranker(const FloatMatrixView &base, int graph_dimension, int candidate_count, int fp32_limit,
                                     const int *graph_row_order)
    : CandidateReranker(SelectedDistanceComputerFactory::create_refine_search(base.data_ptr(), base.row_count(), base.dimension()), graph_dimension) {
    if (graph_dimension != base.dimension() && fp32_limit > 0 && fp32_limit < candidate_count) {
        graph_ordered_codes_ = graph_row_order != nullptr;
        quantized_ = SelectedDistanceComputerFactory::create_quantized_refine(base.data_ptr(), base.row_count(),
                                                                             base.dimension(), graph_row_order, fp32_limit);
    }
}

CandidateReranker::CandidateReranker(std::unique_ptr<MyDistanceComputer> fp32, int graph_dimension,
                                     std::unique_ptr<MyDistanceComputer> quantized)
    : fp32_(std::move(fp32)), quantized_(std::move(quantized)), graph_dimension_(graph_dimension) {}

void CandidateReranker::set_queries(const FloatMatrixView &queries) {
    fp32_->set_query(queries.data_ptr(), queries.row_count());
    if (quantized_)
        quantized_->set_query(queries.data_ptr(), queries.row_count());
}

void CandidateReranker::rerank(int query_id, std::vector<SingleNeighbor> &candidates, int fp32_limit,
                               const SearchResultView &result, const int *original_ids) const {
    const int topk = result.topk();
    const int fp32_count = std::max(fp32_limit, topk);
    if (fp32_limit > 0 && static_cast<size_t>(fp32_count) < candidates.size()) {
        if (quantized_) {
            quantized_->score_candidates(query_id, candidates);
        } else {
            // The graph already scored the original-dimensional codes when PCA is off.
            RNNDESCENT_ASSERT_MSG(fp32_->dimension() == graph_dimension_,
                                  "PCA refinement requires original-dimensional quantized distances");
        }
        std::nth_element(candidates.begin(), candidates.begin() + fp32_count, candidates.end());
        candidates.resize(fp32_count);
    }

    if (original_ids) {
        for (auto &candidate : candidates)
            candidate.id = original_ids[candidate.id];
    }
    fp32_->score_candidates(query_id, candidates);
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
