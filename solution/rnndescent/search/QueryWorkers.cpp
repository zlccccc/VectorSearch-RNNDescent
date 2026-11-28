#include "QueryWorkers.h"

#include <algorithm>

namespace rnndescent::search {

QueryWorkers::QueryWorkers(int capacity, int caller_place) : caller_place_(caller_place), errors_(capacity) {
    threads_.reserve(capacity - 1);
}

void QueryWorkers::start_workers() {
    for (int worker = 1; worker < static_cast<int>(errors_.size()); ++worker)
        threads_.emplace_back([this, worker] { worker_loop(worker); });
}

QueryWorkers::~QueryWorkers() {
    if (threads_.empty())
        return;
    stopping_.store(true, std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_release);
    wake_workers(~std::uint32_t{0});
    for (auto &thread : threads_)
        thread.join();
}

void QueryWorkers::run_impl(int active, void *context, void (*execute)(void *, int)) {
    std::fill_n(errors_.begin(), active, nullptr);
    context_ = context;
    execute_ = execute;
    const std::uint32_t mask = (std::uint32_t{1} << (active - 1)) - 1;
    // Each pending bit both publishes the job to its worker and selects who to wake.
    pending_.store(mask, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_release);
    wake_workers(mask);
    execute_worker(0);

    if (pending_.load(std::memory_order_acquire))
        wait_for_completion();

    for (int worker = 0; worker < active; ++worker)
        if (errors_[worker])
            std::rethrow_exception(errors_[worker]);
}

void QueryWorkers::execute_worker(int worker) noexcept {
    try {
        execute_(context_, worker);
    } catch (...) {
        errors_[worker] = std::current_exception();
    }
}

void QueryWorkers::worker_loop(int worker) {
    const std::uint32_t mask = std::uint32_t{1} << (worker - 1);
    for (;;) {
        const auto generation = generation_.load(std::memory_order_acquire);
        if (stopping_.load(std::memory_order_relaxed))
            return;
        if (pending_.load(std::memory_order_acquire) & mask) {
            execute_worker(worker);
            // The last worker publishes all results and wakes the caller.
            if (pending_.fetch_and(~mask, std::memory_order_acq_rel) == mask)
                wake_caller();
        } else {
            wait_for_work(worker, generation);
        }
    }
}

} // namespace rnndescent::search
