#pragma once

#include <memory>
#include <vector>

#include "Views.h"
#include "discomputer/MyDistanceComputer.h"
#include "discomputer/utils.h"

namespace rnndescent {

class CandidateReranker {
  public:
    CandidateReranker(const FloatMatrixView &base, int graph_dimension, int candidate_count, int fp32_limit,
                      const int *graph_row_order = nullptr);
    CandidateReranker(std::unique_ptr<MyDistanceComputer> fp32, int graph_dimension,
                      std::unique_ptr<MyDistanceComputer> quantized = nullptr);

    void set_queries(const FloatMatrixView &queries);
    bool uses_graph_ordered_codes() const { return graph_ordered_codes_; }

    // If original_ids is provided, candidates start in graph ID space and are mapped after shortlisting.
    void rerank(int query_id, std::vector<SingleNeighbor> &candidates, int fp32_limit,
                const SearchResultView &result, const int *original_ids = nullptr) const;

  private:
    std::unique_ptr<MyDistanceComputer> fp32_;
    std::unique_ptr<MyDistanceComputer> quantized_;
    int graph_dimension_;
    bool graph_ordered_codes_ = false;
};

} // namespace rnndescent
