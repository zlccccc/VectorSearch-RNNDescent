#include "SearchProfile.h"

#include "../common/Logger.h"

#include <algorithm>
#include <ctime>
#include <numeric>
#include <omp.h>

namespace rnndescent::search {

namespace {

long long thread_cpu_ns() {
    timespec time{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &time) != 0)
        return -1;
    return 1000000000LL * time.tv_sec + time.tv_nsec;
}

} // namespace

void QueryPhaseProfile::flush() {
    if (!batches_)
        return;
    const auto total_ms = std::accumulate(times_.begin(), times_.end(), 0.0);
    const auto &[rerank_prepare, projection, graph_encode, ordering, search] = times_;
    Logger::info("query_profile batches=%lld queries=%lld total_ms=%.6f rerank_prepare_ms=%.6f projection_ms=%.6f "
                 "graph_encode_ms=%.6f ordering_ms=%.6f search_ms=%.6f\n",
                 batches_, queries_, total_ms, rerank_prepare, projection, graph_encode, ordering, search);
    *this = {};
}

void PerfStats::accumulate(const PerfStats &other) {
    init_time += other.init_time;
    getneighbor_time += other.getneighbor_time;
    calculate_time += other.calculate_time;
    update_time += other.update_time;
    recalculate_time += other.recalculate_time;
    single_setup_ms += other.single_setup_ms;
    single_merge_ms += other.single_merge_ms;
    query_count += other.query_count;
    calculate_distance_count += other.calculate_distance_count;
    bfs_length += other.bfs_length;
    useful_count += other.useful_count;
    bfs_items += other.bfs_items;
}

void PerfStats::flush() {
    if (!query_count)
        return;
    Logger::info("graph_profile queries=%lld init_worker_ms=%.6f adjacency_worker_ms=%.6f distance_worker_ms=%.6f "
                 "candidate_worker_ms=%.6f rerank_worker_ms=%.6f single_setup_ms=%.6f single_merge_ms=%.6f\n",
                 query_count, init_time, getneighbor_time, calculate_time, update_time, recalculate_time,
                 single_setup_ms, single_merge_ms);
    Logger::info("RNN Descent Running Time: mean bfs %f times; %f nodes; init_time = "
                 "%f, getneighbor_time = %f, calculate_time = %f, update_time = %f; "
                 "recalculate_time = %f; alltime = %f; calculate_dist_count = %f; "
                 "useful_count = %f\n",
                 static_cast<double>(bfs_length) / query_count, static_cast<double>(bfs_items) / query_count,
                 init_time / query_count, getneighbor_time / query_count, calculate_time / query_count,
                 update_time / query_count, recalculate_time / query_count,
                 (init_time + getneighbor_time + calculate_time + update_time + recalculate_time) / query_count,
                 static_cast<double>(calculate_distance_count) / query_count, static_cast<double>(useful_count) / query_count);
    reset();
}

void WorkerProfile::start() {
    entered = ProfileClock::now();
    cpu_start = thread_cpu_ns();
    jobs = 0;
}

void WorkerProfile::stop() {
    const auto cpu_end = thread_cpu_ns();
    cpu_ms = cpu_start >= 0 && cpu_end >= cpu_start ? (cpu_end - cpu_start) / 1e6 : -1;
    finished = ProfileClock::now();
}

void TeamProfile::record(ProfileClock::time_point launch, ProfileClock::time_point returned,
                         const std::vector<WorkerProfile> &samples, int workers) {
    auto first = samples.front().entered;
    auto last = samples.front().finished;
    for (int i = 0; i < workers; ++i) {
        first = std::min(first, samples[i].entered);
        last = std::max(last, samples[i].finished);
    }
    ++regions;
    worker_slots += workers;
    wall_ms += profile_ms(returned - launch);
    entry_ms += profile_ms(first - launch);
    work_window_ms += profile_ms(last - first);
    join_ms += profile_ms(returned - last);
    for (int i = 0; i < workers; ++i) {
        const auto &sample = samples[i];
        unused_slots += sample.jobs == 0;
        tail_worker_ms += profile_ms(last - sample.finished);
        if (sample.cpu_ms < 0)
            continue;
        const double elapsed = profile_ms(sample.finished - sample.entered);
        if (sample.cpu_ms > elapsed) {
            ++cpu_overrun_slots;
            continue;
        }
        ++cpu_slots;
        worker_cpu_ms += sample.cpu_ms;
        worker_off_cpu_ms += elapsed - sample.cpu_ms;
    }
}

void TeamProfile::flush(const char *kind) {
    if (!regions)
        return;
    Logger::info(true, "team_profile kind=%s regions=%lld worker_slots=%lld unused_slots=%lld cpu_slots=%lld cpu_overrun_slots=%lld "
                 "wall_ms=%.6f entry_ms=%.6f work_window_ms=%.6f join_ms=%.6f tail_worker_ms=%.6f "
                 "worker_cpu_ms=%.6f worker_off_cpu_ms=%.6f\n",
                 kind, regions, worker_slots, unused_slots, cpu_slots, cpu_overrun_slots, wall_ms, entry_ms, work_window_ms, join_ms,
                 tail_worker_ms, worker_cpu_ms, worker_off_cpu_ms);
    *this = {};
}

WorkerProfile &GraphSearchProfile::start_worker(int worker) {
    auto &sample = worker_profiles_[worker];
    sample.start();
    if (worker == 0)
        active_workers_ = omp_get_num_threads();
    return sample;
}

void GraphSearchProfile::finish_team(SearchTeam kind, const ProfileTimer &timer) {
    const auto returned = ProfileClock::now();
    auto &team = kind == SearchTeam::Batch ? batch_ : single_;
    team.record(timer.start, returned, worker_profiles_, active_workers_);
}

void GraphSearchProfile::flush() {
    batch_.flush("batch");
    single_.flush("single");
    for (auto &stats : graph_stats_) {
        stats_.accumulate(stats);
        stats.reset();
    }
    stats_.flush();
}

} // namespace rnndescent::search
