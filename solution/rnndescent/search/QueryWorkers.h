#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace rnndescent::search {

// A sleeping team shared by serially submitted search requests from one caller.
class QueryWorkers {
  public:
    static bool can_run(int active);
    static std::unique_ptr<QueryWorkers> create(int active, int capacity);
    ~QueryWorkers();

    bool matches(int active) const;

    template <class Work> void run(int active, Work &work) {
        run_impl(active, &work, [](void *context, int worker) { (*static_cast<Work *>(context))(worker); });
    }

  private:
    static constexpr int max_workers = 32;

    struct WaitState {
        std::mutex mutex;
        std::array<std::condition_variable, max_workers - 1> workers;
        std::condition_variable done;
    };

    explicit QueryWorkers(int capacity, int caller_place);
    void start_workers();
    void run_impl(int active, void *context, void (*execute)(void *, int));
    void execute_worker(int worker) noexcept;
    void worker_loop(int worker);

    void wait_for_work(int worker, std::uint32_t generation);
    void wake_workers(std::uint32_t mask);
    void wait_for_completion();
    void wake_caller();

    const std::thread::id caller_ = std::this_thread::get_id();
    const int caller_place_;
    std::atomic<bool> stopping_{false};
    void *context_ = nullptr;
    void (*execute_)(void *, int) = nullptr;
    alignas(64) std::atomic<std::uint32_t> generation_{0};
    alignas(64) std::atomic<std::uint32_t> pending_{0};
    std::vector<std::exception_ptr> errors_;
    std::vector<std::thread> threads_;
    std::unique_ptr<WaitState> waits_;
};

} // namespace rnndescent::search
