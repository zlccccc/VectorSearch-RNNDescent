#include "QueryWorkers.h"

#include <algorithm>
#include <system_error>

namespace rnndescent::search {

bool QueryWorkers::can_run(int active) {
    return active > 1 && active <= max_workers;
}

std::unique_ptr<QueryWorkers> QueryWorkers::create(int active, int capacity) {
    capacity = std::min(capacity, max_workers);
    if (!can_run(active) || capacity < active)
        return {};

    try {
        auto team = std::unique_ptr<QueryWorkers>(new QueryWorkers(capacity, -1));
        team->waits_ = std::make_unique<WaitState>();
        team->start_workers();
        return team;
    } catch (const std::system_error &) {
        return {};
    }
}

bool QueryWorkers::matches(int active) const {
    return static_cast<std::size_t>(active) <= errors_.size() && caller_ == std::this_thread::get_id();
}

void QueryWorkers::wait_for_work(int worker, std::uint32_t) {
    const std::uint32_t mask = std::uint32_t{1} << (worker - 1);
    std::unique_lock lock(waits_->mutex);
    waits_->workers[worker - 1].wait(lock, [&] {
        return stopping_.load(std::memory_order_relaxed) || (pending_.load(std::memory_order_acquire) & mask) != 0;
    });
}

void QueryWorkers::wake_workers(std::uint32_t mask) {
    // Hold the wait mutex to serialize notification with the predicate check and sleep.
    const std::lock_guard lock(waits_->mutex);
    for (std::size_t worker = 0; worker < threads_.size(); ++worker)
        if (mask & (std::uint32_t{1} << worker))
            waits_->workers[worker].notify_one();
}

void QueryWorkers::wait_for_completion() {
    std::unique_lock lock(waits_->mutex);
    waits_->done.wait(lock, [&] { return pending_.load(std::memory_order_acquire) == 0; });
}

void QueryWorkers::wake_caller() {
    const std::lock_guard lock(waits_->mutex);
    waits_->done.notify_one();
}

} // namespace rnndescent::search
