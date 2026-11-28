#include "QueryWorkers.h"

#include <algorithm>
#include <climits>
#include <linux/futex.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <system_error>
#include <unistd.h>

namespace rnndescent::search {

static_assert(std::atomic<std::uint32_t>::is_always_lock_free &&
              sizeof(std::atomic<std::uint32_t>) == sizeof(std::uint32_t));

bool QueryWorkers::can_run(int active) {
    return active > 1 && active <= max_workers && omp_get_level() == 0 && !omp_get_dynamic() &&
           omp_get_thread_limit() >= active;
}

std::unique_ptr<QueryWorkers> QueryWorkers::create(int active, int capacity) {
    const auto binding = omp_get_proc_bind();
    if (binding != omp_proc_bind_close && binding != omp_proc_bind_false)
        return {};
    const int caller_place = omp_get_place_num();
    const int places = omp_get_num_places();
    capacity = std::min({capacity, max_workers, omp_get_thread_limit()});
    if (binding == omp_proc_bind_close)
        capacity = std::min(capacity, places);
    if (active < 2 || capacity < active || (binding == omp_proc_bind_close && caller_place < 0))
        return {};

    try {
        auto team = std::unique_ptr<QueryWorkers>(new QueryWorkers(capacity, caller_place));
        team->start_workers();
        if (binding == omp_proc_bind_close) {
            for (int worker = 1; worker < capacity; ++worker) {
                const int place = (caller_place + worker) % places;
                std::vector<int> cpus(omp_get_place_num_procs(place));
                omp_get_place_proc_ids(place, cpus.data());
                cpu_set_t affinity{};
                for (const int cpu : cpus) {
                    if (cpu < 0 || cpu >= CPU_SETSIZE)
                        return {};
                    CPU_SET(cpu, &affinity);
                }
                if (CPU_COUNT(&affinity) &&
                    pthread_setaffinity_np(team->threads_[worker - 1].native_handle(), sizeof(affinity), &affinity) != 0)
                    return {};
            }
        }
        return team;
    } catch (const std::system_error &) {
        return {};
    }
}

bool QueryWorkers::matches(int active) const {
    return static_cast<std::size_t>(active) <= errors_.size() &&
           caller_ == std::this_thread::get_id() && caller_place_ == omp_get_place_num();
}

void QueryWorkers::wait_for_work(int worker, std::uint32_t generation) {
    const std::uint32_t mask = std::uint32_t{1} << (worker - 1);
    ::syscall(SYS_futex, &generation_, FUTEX_WAIT_BITSET_PRIVATE, generation, nullptr, nullptr, mask);
}

void QueryWorkers::wake_workers(std::uint32_t mask) {
    ::syscall(SYS_futex, &generation_, FUTEX_WAKE_BITSET_PRIVATE, INT_MAX, nullptr, nullptr, mask);
}

void QueryWorkers::wait_for_completion() {
    for (auto pending = pending_.load(std::memory_order_acquire); pending != 0;
         pending = pending_.load(std::memory_order_acquire))
        ::syscall(SYS_futex, &pending_, FUTEX_WAIT_PRIVATE, pending, nullptr, nullptr, 0);
}

void QueryWorkers::wake_caller() {
    ::syscall(SYS_futex, &pending_, FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
}

} // namespace rnndescent::search
