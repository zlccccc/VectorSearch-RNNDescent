#include "CandidateReranker.h"

#include <algorithm>
#include <utility>

#include "../common/Assertions.h"
#include "../discomputer/PlatformSimdDistanceComputer.h"

namespace rnndescent {
namespace {

void score_candidates(MyDistanceComputer &distance, int query_id, std::vector<SingleNeighbor> &candidates) {
    const size_t end4 = candidates.size() / 4 * 4;
    for (size_t i = 0; i < end4; i += 4) {
        auto *group = candidates.data() + i;
        distance.distances_batch_4(query_id, group[0].id, group[1].id, group[2].id, group[3].id,
                                   group[0].distance, group[1].distance, group[2].distance, group[3].distance);
    }
    for (size_t i = end4; i < candidates.size(); ++i)
        candidates[i].distance = distance(query_id, candidates[i].id);
}

} // namespace

CandidateReranker::CandidateReranker(const FloatMatrixView &base, int graph_dimension, int candidate_count, int fp32_limit,
                                     const int *graph_row_order)
    : CandidateReranker(SelectedDistanceComputerFactory::create_refine_search(base.data_ptr(), base.row_count(), base.dimension()), graph_dimension) {
    if (graph_dimension != base.dimension() && fp32_limit > 0 && fp32_limit < candidate_count) {
        quantized_ = SelectedDistanceComputerFactory::create_quantized_refine(base.data_ptr(), base.row_count(), base.dimension(), graph_row_order);
        graph_ordered_codes_ = graph_row_order != nullptr;
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
            score_candidates(*quantized_, query_id, candidates);
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
    score_candidates(*fp32_, query_id, candidates);
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
