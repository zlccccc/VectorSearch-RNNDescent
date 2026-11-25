#pragma once

#include <memory>
#include <vector>

#include "../build/GraphConfig.h"
#include "../common/Views.h"
#include "../discomputer/MyDistanceComputer.h"
#include "../discomputer/utils.h"

namespace rnndescent {

class CandidateReranker {
  public:
    CandidateReranker(const FloatMatrixView &base, int graph_dimension, int candidate_count, const config::SearchConfig &config,
                      const int *graph_row_order = nullptr);
    CandidateReranker(std::unique_ptr<MyDistanceComputer> fp32, int graph_dimension,
                      std::unique_ptr<MyDistanceComputer> quantized = nullptr);

    void set_queries(const FloatMatrixView &queries);
    bool uses_graph_ordered_codes() const { return graph_ordered_codes_; }

    // Retain Top K with original IDs and FP32 distances. Sort only when writing results.
    void rerank(int query_id, std::vector<SingleNeighbor> &candidates, int fp32_limit, int topk,
                const int *original_ids = nullptr) const;
    static void write_topk(std::vector<SingleNeighbor> &candidates, const SearchResultView &result);

  private:
    std::unique_ptr<MyDistanceComputer> fp32_;
    std::unique_ptr<MyDistanceComputer> quantized_;
    bool shortlist_enabled_ = true;
    bool reuse_graph_scores_;
    bool graph_ordered_codes_ = false;
};

} // namespace rnndescent
