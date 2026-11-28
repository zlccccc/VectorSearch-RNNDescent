#pragma once

#include <cstddef>

#ifdef INTERNAL_CLOCK_TEST
#include <array>
#include <chrono>
#include <vector>
#endif

namespace rnndescent::search {

enum class QueryStage { RerankPrepare, Projection, GraphEncode, Ordering, Search, Count };
enum class SearchTeam { Batch, Single };

#ifdef INTERNAL_CLOCK_TEST

using ProfileClock = std::chrono::steady_clock;

inline double profile_ms(ProfileClock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

struct ProfileTimer {
    ProfileClock::time_point start = ProfileClock::now();

    double lap() {
        const auto now = ProfileClock::now();
        const auto elapsed = profile_ms(now - start);
        start = now;
        return elapsed;
    }
};

class QueryPhaseProfile {
  public:
    class Sample {
      public:
        explicit Sample(QueryPhaseProfile &profile) : profile_(profile) {}

        void stage(QueryStage stage) {
            profile_.times_[static_cast<std::size_t>(stage)] += timer_.lap();
        }

        void finish(int queries) {
            stage(QueryStage::Search);
            ++profile_.batches_;
            profile_.queries_ += queries;
        }

      private:
        QueryPhaseProfile &profile_;
        ProfileTimer timer_;
    };

    Sample measure() { return Sample(*this); }

    void flush();

  private:
    long long batches_ = 0;
    long long queries_ = 0;
    std::array<double, static_cast<std::size_t>(QueryStage::Count)> times_{};
};

struct alignas(64) PerfStats {
    float init_time = 0;
    float getneighbor_time = 0;
    float calculate_time = 0;
    float update_time = 0;
    float recalculate_time = 0;
    double single_setup_ms = 0;
    double single_merge_ms = 0;
    long long query_count = 0;
    long long calculate_distance_count = 0;
    long long bfs_length = 0;
    long long useful_count = 0;
    long long bfs_items = 0;

    void accumulate(const PerfStats &other);
    void flush();
    void reset() { *this = {}; }
};

class GraphQuerySample {
  public:
    explicit GraphQuerySample(PerfStats &stats) : stats_(stats) {}

    void initialized() { local_.init_time = timer_.lap(); }
    void adjacency_ready() { local_.getneighbor_time += timer_.lap(); }

    void neighbors_scored(int count) {
        local_.calculate_time += timer_.lap();
        ++local_.bfs_items;
        local_.calculate_distance_count += count;
    }

    void beam_finished() { ++local_.bfs_length; }

    void candidates_updated(std::ptrdiff_t count) {
        local_.update_time += timer_.lap();
        local_.useful_count += count;
    }

    void rerank_started() { timer_ = {}; }

    void finish(bool count_query) {
        local_.recalculate_time = timer_.lap();
        local_.query_count = count_query;
        stats_.accumulate(local_);
    }

  private:
    PerfStats &stats_;
    ProfileTimer timer_;
    PerfStats local_;
};

struct alignas(64) WorkerProfile {
    ProfileClock::time_point entered;
    ProfileClock::time_point finished;
    long long cpu_start = 0;
    double cpu_ms = 0;
    int jobs = 0;

    void start();
    void job_finished() { ++jobs; }
    void stop();
};

struct TeamProfile {
    long long regions = 0;
    long long worker_slots = 0;
    long long unused_slots = 0;
    long long cpu_slots = 0;
    long long cpu_overrun_slots = 0;
    double wall_ms = 0;
    double entry_ms = 0;
    double work_window_ms = 0;
    double join_ms = 0;
    double tail_worker_ms = 0;
    double worker_cpu_ms = 0;
    double worker_off_cpu_ms = 0;

    void record(ProfileClock::time_point launch, ProfileClock::time_point returned,
                const std::vector<WorkerProfile> &samples, int workers);
    void flush(const char *kind);
};

class GraphSearchProfile {
  public:
    GraphSearchProfile(std::size_t workers, PerfStats &stats)
        : stats_(stats), graph_stats_(workers), worker_profiles_(workers) {}

    GraphQuerySample query(int worker) { return GraphQuerySample(graph_stats_[worker]); }
    WorkerProfile &start_worker(int worker);
    void finish_team(SearchTeam kind, const ProfileTimer &timer);
    void single_setup_finished(ProfileTimer &timer) { stats_.single_setup_ms += timer.lap(); }
    void single_merge_finished(ProfileTimer &timer) { stats_.single_merge_ms += timer.lap(); }

    void flush();

  private:
    PerfStats &stats_;
    std::vector<PerfStats> graph_stats_;
    std::vector<WorkerProfile> worker_profiles_;
    int active_workers_ = 0;
    TeamProfile batch_;
    TeamProfile single_;
};

#else

struct ProfileTimer {};

struct QueryPhaseProfile {
    struct Sample {
        void stage(QueryStage) {}
        void finish(int) {}
    };

    Sample measure() { return {}; }
    void flush() {}
};

struct PerfStats {
    void reset() {}
};

struct GraphQuerySample {
    void initialized() {}
    void adjacency_ready() {}
    void neighbors_scored(int) {}
    void beam_finished() {}
    void candidates_updated(std::ptrdiff_t) {}
    void rerank_started() {}
    void finish(bool) {}
};

struct WorkerProfile {
    void job_finished() {}
    void stop() {}
};

class GraphSearchProfile {
  public:
    GraphSearchProfile(std::size_t, PerfStats &) {}

    GraphQuerySample query(int) { return {}; }
    WorkerProfile &start_worker(int) { return worker_; }
    void finish_team(SearchTeam, const ProfileTimer &) {}
    void single_setup_finished(ProfileTimer &) {}
    void single_merge_finished(ProfileTimer &) {}
    void flush() {}

  private:
    WorkerProfile worker_;
};

#endif

} // namespace rnndescent::search
