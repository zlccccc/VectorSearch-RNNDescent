#pragma once

#include <memory>
#include <vector>

#include "Views.h"
#include "discomputer/MyDistanceComputer.h"
#include "discomputer/utils.h"

namespace rnndescent {

class CandidateReranker {
  public:
    CandidateReranker(const FloatMatrixView &base, int graph_dimension, int candidate_count, int fp32_limit);
    CandidateReranker(std::unique_ptr<MyDistanceComputer> fp32, int graph_dimension,
                      std::unique_ptr<MyDistanceComputer> quantized = nullptr);

    void set_queries(const FloatMatrixView &queries);

    // Candidates use original row IDs and are stored in a per-thread search buffer.
    void rerank(int query_id, std::vector<SingleNeighbor> &candidates, int fp32_limit,
                const SearchResultView &result) const;

  private:
    std::unique_ptr<MyDistanceComputer> fp32_;
    std::unique_ptr<MyDistanceComputer> quantized_;
    int graph_dimension_;
};

} // namespace rnndescent
