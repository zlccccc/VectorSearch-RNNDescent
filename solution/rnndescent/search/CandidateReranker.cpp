#include "CandidateReranker.h"

#include <algorithm>
#include <utility>

#include "../common/Assertions.h"
#include "../discomputer/PlatformSimdDistanceComputer.h"

namespace rnndescent {
namespace {

void keep_closest(std::vector<SingleNeighbor> &candidates, size_t count) {
    if (count < candidates.size()) {
        std::nth_element(candidates.begin(), candidates.begin() + count, candidates.end());
        candidates.resize(count);
    }
}

} // namespace

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
    if (!reuse_graph_scores_ && config.uses_coarse_shortlist(candidate_count)) {
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

void CandidateReranker::rerank(int query_id, std::vector<SingleNeighbor> &candidates, int fp32_limit, int topk,
                               const int *original_ids) const {
    const auto score = [&](MyDistanceComputer &computer, const int *row_order) {
        if (row_order) {
            for (auto &candidate : candidates)
                candidate.id = row_order[candidate.id];
        }
        computer.score_candidates(query_id, candidates.data(), candidates.size());
    };

    const size_t fp32_count = std::max(fp32_limit, topk);
    if (shortlist_enabled_ && fp32_limit > 0 && fp32_count < candidates.size()) {
        if (quantized_) {
            // BFS-ordered codes keep graph IDs until the shortlist is selected.
            score(*quantized_, graph_ordered_codes_ ? nullptr : original_ids);
            if (!graph_ordered_codes_)
                original_ids = nullptr;
        } else {
            RNNDESCENT_ASSERT_MSG(reuse_graph_scores_, "shortlisting requires compatible original-dimensional scores");
        }
        keep_closest(candidates, fp32_count);
    }

    score(*fp32_, original_ids);
    keep_closest(candidates, topk);
}

void CandidateReranker::write_topk(std::vector<SingleNeighbor> &candidates, const SearchResultView &result) {
    const int topk = result.topk();
    keep_closest(candidates, topk);
    std::sort(candidates.begin(), candidates.end());

    for (int i = 0; i < topk; ++i) {
        result.indices_ptr()[i] = candidates[i].id;
        result.distances_ptr()[i] = candidates[i].distance;
    }
}

} // namespace rnndescent
