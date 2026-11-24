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

    // original_ids maps graph IDs to original rows when the input candidates use graph IDs.
    void rerank(int query_id, std::vector<SingleNeighbor> &candidates, int fp32_limit,
                const SearchResultView &result, const int *original_ids = nullptr) const;

    // The caller prepares the shortlist once and passes the returned FP32 count to each worker.
    size_t prepare_shortlist(std::vector<SingleNeighbor> &candidates, int fp32_limit, int topk) const;
    // All workers in the current team call this before the caller writes Top K.
    void score_for_rerank(int query_id, std::vector<SingleNeighbor> &candidates, size_t fp32_count,
                          const int *original_ids, int worker, int workers) const;
    static void write_topk(std::vector<SingleNeighbor> &candidates, const SearchResultView &result);

  private:
    static void select_fp32_candidates(std::vector<SingleNeighbor> &candidates, size_t fp32_count);

    std::unique_ptr<MyDistanceComputer> fp32_;
    std::unique_ptr<MyDistanceComputer> quantized_;
    bool shortlist_enabled_ = true;
    bool reuse_graph_scores_;
    bool graph_ordered_codes_ = false;
};

} // namespace rnndescent
