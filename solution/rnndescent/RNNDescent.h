#pragma once

#include <bits/stdc++.h>
#include <memory_resource>

// #include "discomputer/CblasDistComputerIP.h"
#include "discomputer/CblasDistComputerFP32.h"
// #include "discomputer/FaissDistComputerIP.h"
// #include "discomputer/FaissDistComputerL2.h"

#include "discomputer/PlatformSimdDistanceComputer.h"
#include "Logger.h"
// #include "discomputer/Avx512SimdDistanceComputerFP32.h"
// #define INTERNAL_CLOCK_TEST

// #include "discomputer/Avx512SimdDistanceComputerUInt8.h"
// #include "discomputer/NeonSimdDistanceComputerInt8.h"

namespace rnndescent {

struct MutexWarpper : std::mutex {
    MutexWarpper() = default;
    MutexWarpper(MutexWarpper const &) noexcept : std::mutex() {}
    bool operator==(MutexWarpper const &other) noexcept { return this == &other; }
};

struct XNeighbor {
    int id_;
    float distance;
    XNeighbor() = default;
    XNeighbor(int id, float distance, bool flag) : id_(id), distance(distance) {
        if (flag) {
            id_ |= 0x80000000;
        }
    }
    int id() { return id_ & 0x7fffffff; }
    int flag() {
        if (id_ & 0x80000000)
            return true;
        else
            return false;
        // return flag_;
    }
    void setflag(bool flag) {
        // flag_ = flag;
        if (flag)
            id_ |= 0x80000000;
        else
            id_ &= 0x7fffffff;
    }
    inline bool operator<(const XNeighbor &other) const {
        if (distance == other.distance) {
            return (id_ & 0x7fffffff) < (other.id_ & 0x7fffffff);
        }
        return distance < other.distance;
    }
};

struct RNNDescent {
    struct FloatMatrixView {
      public:
        FloatMatrixView() = default;
        FloatMatrixView(const float *data, int rows, int dim) : data_(data), rows(rows), dim(dim) {}

        static FloatMatrixView from_buffer(const float *data, int rows, int dim) { return FloatMatrixView(data, rows, dim); }

        static FloatMatrixView from_vector(const std::vector<float> &data, int dim) {
            FAISS_THROW_IF_NOT_MSG(dim > 0, "matrix view requires positive dimension");
            FAISS_THROW_IF_NOT_MSG(data.size() % dim == 0, "matrix view vector size must be divisible by dimension");
            return FloatMatrixView(data.data(), static_cast<int>(data.size() / dim), dim);
        }

        void validate(const char *name) const {
            (void)name;
            FAISS_THROW_IF_NOT_MSG(data_ != nullptr, "matrix view data pointer is null");
            FAISS_THROW_IF_NOT_MSG(rows > 0, "matrix view requires positive row count");
            FAISS_THROW_IF_NOT_MSG(dim > 0, "matrix view requires positive dimension");
        }

        const float *data_ptr() const { return data_; }
        int row_count() const { return rows; }
        int dimension() const { return dim; }

        const float *row_ptr(int row) const { return data_ + (size_t)row * dim; }

      private:
        const float *data_ = nullptr;
        int rows = 0;
        int dim = 0;
    };

    struct MutableFloatMatrixView {
      public:
        MutableFloatMatrixView() = default;
        MutableFloatMatrixView(float *data, int rows, int dim) : data_(data), rows(rows), dim(dim) {}

        static MutableFloatMatrixView from_buffer(float *data, int rows, int dim) { return MutableFloatMatrixView(data, rows, dim); }

        static MutableFloatMatrixView from_vector(std::vector<float> &data, int dim) {
            FAISS_THROW_IF_NOT_MSG(dim > 0, "mutable matrix view requires positive dimension");
            FAISS_THROW_IF_NOT_MSG(data.size() % dim == 0, "mutable matrix view vector size must be divisible by dimension");
            return MutableFloatMatrixView(data.data(), static_cast<int>(data.size() / dim), dim);
        }

        void validate(const char *name) const {
            (void)name;
            FAISS_THROW_IF_NOT_MSG(data_ != nullptr, "mutable matrix view data pointer is null");
            FAISS_THROW_IF_NOT_MSG(rows > 0, "mutable matrix view requires positive row count");
            FAISS_THROW_IF_NOT_MSG(dim > 0, "mutable matrix view requires positive dimension");
        }

        float *data_ptr() const { return data_; }
        int row_count() const { return rows; }
        int dimension() const { return dim; }

        float *row_ptr(int row) const { return data_ + (size_t)row * dim; }

      private:
        float *data_ = nullptr;
        int rows = 0;
        int dim = 0;
    };

    struct SearchResultView {
      public:
        SearchResultView() = default;
        SearchResultView(int *indices, float *distances, int topk) : indices_(indices), distances_(distances), topk_(topk) {}

        static SearchResultView from_buffers(int *indices, float *distances, int topk) { return SearchResultView(indices, distances, topk); }

        static SearchResultView from_vectors(std::vector<int> &indices, std::vector<float> &distances, int topk) {
            FAISS_THROW_IF_NOT_MSG(topk > 0, "search topk must be positive");
            FAISS_THROW_IF_NOT_MSG(indices.size() == distances.size(), "search result vector sizes must match");
            return SearchResultView(indices.data(), distances.data(), topk);
        }

        void validate() const {
            FAISS_THROW_IF_NOT_MSG(indices_ != nullptr, "search result indices buffer is null");
            FAISS_THROW_IF_NOT_MSG(distances_ != nullptr, "search result distance buffer is null");
            FAISS_THROW_IF_NOT_MSG(topk_ > 0, "search topk must be positive");
        }

        int *indices_ptr() const { return indices_; }
        float *distances_ptr() const { return distances_; }
        int topk() const { return topk_; }

        SearchResultView slice(int queryid) const {
            validate();
            return SearchResultView(indices_ + queryid * topk_, distances_ + queryid * topk_, topk_);
        }

      private:
        int *indices_ = nullptr;
        float *distances_ = nullptr;
        int topk_ = 0;
    };

    struct BuildConfig {
        int T1 = 4;
        int T2 = 15;
        int S = 16;
        int R = 96;
        int K0 = 48;
        int random_seed = 2021;
        int num_threads = 16;
        long long neighbor_pool_size_limit_bytes = 16ll * 1024 * 1024 * 1024;
        bool save_neighbor = true;
        bool random_init = true;
    };

    struct SearchConfig {
        int beam_size = 8;
        int num_threads = 16;
        int search_L = 96;
        int num_initialize = 64;
        int refine_max = 64;
    };

    static BuildConfig sanitize_build_config(BuildConfig config) {
        config.T1 = std::max(config.T1, 1);
        config.T2 = std::max(config.T2, 1);
        config.S = std::max(config.S, 1);
        config.R = std::max(config.R, 1);
        config.K0 = std::max(config.K0, 1);
        config.num_threads = std::max(config.num_threads, 1);
        config.neighbor_pool_size_limit_bytes = std::max(config.neighbor_pool_size_limit_bytes, 0ll);
        return config;
    }

    static SearchConfig sanitize_search_config(SearchConfig config, int topk = 1) {
        config.beam_size = std::max(config.beam_size, 1);
        config.num_threads = std::max(config.num_threads, 1);
        config.search_L = std::max(config.search_L, std::max(1, topk));
        config.num_initialize = std::max(config.num_initialize, config.search_L);
        if (config.num_initialize % 4 != 0)
            config.num_initialize = (config.num_initialize + 3) / 4 * 4;
        config.refine_max = std::max(config.refine_max, std::max(topk, 1));
        return config;
    };

    static void gen_random(std::mt19937 &rng, std::vector<int> &addr, const int N) {
        const int size = addr.size();
        FAISS_THROW_IF_NOT_MSG(0 <= size && size <= N, "gen_random requires 0 <= size <= N");
        if (size == 0)
            return;

        // Partial Fisher-Yates shuffle with sparse swaps: sample `size` unique ids from [0, N)
        // without materializing the full range.
        std::unordered_map<int, int> swapped;
        swapped.reserve(size * 2);
        for (int i = 0; i < size; ++i) {
            std::uniform_int_distribution<int> dist(i, N - 1);
            const int j = dist(rng);

            const auto it_i = swapped.find(i);
            const int value_i = (it_i == swapped.end()) ? i : it_i->second;
            const auto it_j = swapped.find(j);
            const int value_j = (it_j == swapped.end()) ? j : it_j->second;

            swapped[j] = value_i;
            addr[i] = value_j;
        }
    }

    using KNNGraph = std::vector<std::vector<XNeighbor>>;

    RNNDescent(const FloatMatrixView &data_view, bool verbose, const BuildConfig &build_config, const SearchConfig &search_config) {
        build(data_view, verbose, build_config, search_config);
    }

#ifdef INTERNAL_CLOCK_TEST
    struct PerfStats {
        float init_time{0};
        float getneighbor_time{0};
        float calculate_time{0};
        float update_time{0};
        float recalculate_time{0};
        long long query_count{0};
        long long calculate_distance_count{0};
        long long for_count{0};
        long long bfs_length{0};
        long long useful_count{0};
        long long bfs_items{0};

        void reset() {
            init_time = 0;
            getneighbor_time = 0;
            calculate_time = 0;
            update_time = 0;
            recalculate_time = 0;
            query_count = 0;
            calculate_distance_count = 0;
            for_count = 0;
            bfs_length = 0;
            useful_count = 0;
            bfs_items = 0;
        }
    } perf_stats;
#endif
    ~RNNDescent() { reset(); }

    void generate_graph(KNNGraph &graph, const FloatMatrixView &data_view, int n, bool verbose, const BuildConfig &build_config) {
        Logger::info(verbose, "generte graph ntotal = %d\n", n);
        auto qdis = SelectedDistanceComputerFactory::create_build_graph(data_view.data_ptr(), data_view.row_count(), data_view.dimension());
        init_graph(graph, n, *qdis, build_config);
        for (int t1 = 0; t1 < build_config.T1; ++t1) {
            if (verbose)
                std::cout << "Iter " << t1 << " : " << std::flush;
            for (int t2 = 0; t2 < build_config.T2; ++t2) {
                update_neighbors(graph, n, *qdis, build_config);
                if (verbose)
                    std::cout << "#" << std::flush;
            }
            if (verbose)
                Logger::line(verbose, "");

            if (t1 != build_config.T1 - 1)
                add_reverse_edges(graph, n, build_config);
        }

#pragma omp parallel for num_threads(build_config.num_threads)
        for (int u = 0; u < n; ++u) { // remove edges
            auto &pool = graph[u];
            std::sort(pool.begin(), pool.end());
            pool.erase(std::unique(pool.begin(), pool.end(), [](XNeighbor &a, XNeighbor &b) { return a.id() == b.id(); }), pool.end());
        }

        // 这里的resize可能会导致图不联通; 这件事情需要在上面build的时候就考虑到
        int all_edges_size = 0;
        for (int u = 0; u < n; ++u) {
            // 清理内存
            if (graph[u].size() > build_config.K0)
                graph[u].resize(build_config.K0);
            all_edges_size += graph[u].size();
            graph[u].shrink_to_fit();
        }
        Logger::info(verbose, "graph edges size = %d\n", all_edges_size);
    }

    void build(const FloatMatrixView &data_view, bool verbose, const BuildConfig &build_config, const SearchConfig &search_config) {
        data_view.validate("build data");
        const BuildConfig safe_build_config = sanitize_build_config(build_config);
        const SearchConfig safe_search_config = sanitize_search_config(search_config);
        const int n = data_view.row_count();
        const int dim = data_view.dimension();

        reset();
        node_locks_.clear();

        if (verbose)
            Logger::info(verbose, "Parameters: S=%d, R=%d, T1=%d, T2=%d; Point=%d; numThreadsMax=%d\n", safe_build_config.S, safe_build_config.R,
                         safe_build_config.T1, safe_build_config.T2, n, safe_search_config.num_threads);
        SelectedNeighborsContainerType::clear_memory();

        std::vector<std::vector<int>> edges; // distance并不重要
        edges.resize(n);
        {
            KNNGraph graph;
            generate_graph(graph, data_view, n, verbose, safe_build_config); // highest level
                                                                             // #pragma omp parallel for
            std::set<int> S;                                                 // 不能重复
            for (int i = 0; i < n; i++) {
                auto &pool = graph[i];
                sort(pool.begin(), pool.end());
                edges[i].reserve((pool.size() + 3) / 4 * 4);
                // 需要清理下内存; 变成4的倍数方便后面reorder
                S.clear();
                for (auto &edge : pool) {
                    if (!S.count(edge.id())) { // 这个地方感觉不太应该写成这样; 如果写得好的话, 理论上刚开始的edge id不会重复
                        S.insert(edge.id());
                        edges[i].emplace_back(edge.id());
                    }
                }
                while (edges[i].size() % 4 != 0) { // 补到差不多
                    int id = random() % n;
                    while (S.count(id))
                        id = random() % n;
                    S.insert(id);
                    edges[i].emplace_back(id);
                }
                std::vector<XNeighbor>().swap(pool);
            }
        }
        // 确定全局入口点
        Logger::info(verbose, "n = %d; initialize = %d\n", n, safe_search_config.num_initialize);
        {
            std::mt19937 rng(safe_build_config.random_seed);
            search_from_ids.reserve(n);
            search_from_ids.resize(safe_search_config.num_initialize);
            if (safe_build_config.random_init)
                gen_random(rng, search_from_ids, n);
            else
                iota(search_from_ids.begin(), search_from_ids.end(), 0);
        }

        // 重排ID, 加速运行
        std::vector<int> rollback_ids(n); // 连graph的时候边id需更新

        { // ID重排
            Logger::line(verbose, "Start Reordering The Graph.");
            // bfs from those indexes
            // search_from_ids.resize(numSearchInitializeItem); // range
            // for (int i = 0; i < search_from_ids.size(); i++)
            //     printf("%d ",search_from_ids[i]); puts("<");
            MyVisitedTable vis(n);

            // cluster_id: 重排ID以后, 从哪个cluster可以bfs到当前点
            std::vector<int> cluster_id(n);
            for (int i = 0; i < search_from_ids.size(); i++) {
                vis.set(search_from_ids[i]);
                cluster_id[search_from_ids[i]] = i;
            }
            std::vector<int> bfs_distance(n); // 只是用来记录一下
            int all_have_next = 0;
            // std::vector<int> current_layer;
            for (int i = 0; i < search_from_ids.size(); i++) {
                int u = search_from_ids[i];
                // index_to_result_mapping[i] = u;  // same as search_from_ids
                bool have_next = false;
                for (int v : edges[u]) {
                    if (!vis.get(v)) {
                        vis.set(v);
                        search_from_ids.push_back(v);
                        have_next = true;
                        bfs_distance[v] = bfs_distance[u] + 1;
                        cluster_id[v] = cluster_id[u];
                    }
                }
                all_have_next += have_next;
            }

            { // bfs重排后的cluster ID
                std::vector<int> cluster_size(safe_search_config.num_initialize);
                for (int i = 0; i < cluster_id.size(); i++)
                    cluster_size[cluster_id[i]]++;
                for (int v : cluster_size)
                    Logger::info(verbose, "%d ", v);
                Logger::line(verbose, " <<- initial cluster size");
                std::vector<int> bfs_length_count;
                for (int i = 0; i < search_from_ids.size(); i++) {
                    int v = bfs_distance[search_from_ids[i]];
                    if (v >= bfs_length_count.size())
                        bfs_length_count.resize(v + 1);
                    bfs_length_count[v]++;
                }
                for (int i = 0; i < bfs_length_count.size(); i++)
                    Logger::info(verbose, "bfs_length %d: %d\n", i, bfs_length_count[i]);
            }

            int cannot_search = 0; // 这里先暂时不处理这种情况; 会有概率有的点搜不到
            for (int i = 0; i < n; i++) {
                if (!vis.get(i))
                    search_from_ids.push_back(i), cannot_search++;
            }
#ifdef INTERNAL_CLOCK_TEST
            Logger::info(verbose, "search from ids = %d; have_next = %d; cannot_search = %d\n", (int)search_from_ids.size(), all_have_next, cannot_search);
            assert(n == search_from_ids.size());
#endif

            for (int i = 0; i < n; i++) // 连graph的时候边id需更新
                rollback_ids[search_from_ids[i]] = i;

            std::vector<float> fastsearch_pool;
            fastsearch_pool.resize(n * dim);
#pragma omp parallel for num_threads(safe_build_config.num_threads)
            for (int i = 0; i < n; i++) {
                int u = search_from_ids[i];
                // printf("u = %d; matrix.size() = %d\n",u, matrix.size());
                memcpy(fastsearch_pool.data() + i * dim, data_view.row_ptr(u), dim * sizeof(float));
            }
            graph_distance_computer = SelectedDistanceComputerFactory::create_cached_graph(fastsearch_pool.data(), n, dim);
        }
        { // 空间局部性优化
            SelectedNeighborsContainerType::init_neighbors_pool(dim, edges, safe_build_config.neighbor_pool_size_limit_bytes, safe_build_config.save_neighbor);
            for (int i = 0; i < n; i++) {
                int u = search_from_ids[i];
                auto &pool = edges[u];
                final_graph_neighbors.emplace_back(
                    SelectedNeighborsContainerType(dim, pool, graph_distance_computer.get(), rollback_ids, safe_build_config.save_neighbor)); // 全部save
            }
        }
        threadVt.resize(safe_search_config.num_threads);
        threadRetset.resize(safe_search_config.num_threads);
        neighborDistance.resize(safe_search_config.num_threads);
        threadUsefulset.resize(safe_search_config.num_threads);
        threadFinalset.resize(safe_search_config.num_threads);
        for (int i = 0; i < safe_search_config.num_threads; i++) {
            threadVt[i].init(n);
            threadRetset[i].reserve(std::max(safe_search_config.num_initialize, safe_search_config.search_L));
            neighborDistance[i].reserve(safe_search_config.search_L);
            threadFinalset[i].reserve(safe_search_config.search_L);
            threadUsefulset[i].reserve(safe_build_config.K0 * safe_search_config.beam_size + 1);
        }
    }

    std::vector<int> search_from_ids;

    std::vector<MyVisitedTable> threadVt;                  // threadRetset和之前的retset起到的价值差不多
    std::vector<std::vector<SingleNeighbor>> threadRetset; // threadRetset和之前的retset起到的价值差不多
    std::vector<std::vector<SingleNeighbor>> threadUsefulset;
    std::vector<std::vector<SingleNeighbor>> threadFinalset;
    std::vector<std::vector<float>> neighborDistance;

    void searchSingle(int threadid, int queryid, MyDistanceComputer &realqdis, const SearchConfig &search_config, int max_degree,
                      const SearchResultView &result) {
        result.validate();
        FAISS_THROW_IF_NOT_MSG(max_degree > 0, "search max_degree must be positive");
        const int topk = result.topk();

        const SearchConfig safe_search_config = sanitize_search_config(search_config, topk);
        const int num_threads = safe_search_config.num_threads;
        const int beam_size = safe_search_config.beam_size;
        const int search_L = safe_search_config.search_L;
        const int num_initialize = safe_search_config.num_initialize;
        const int refine_max = safe_search_config.refine_max;
        assert(0 <= threadid && threadid < num_threads);
#ifdef INTERNAL_CLOCK_TEST
        auto prevtime = std::chrono::high_resolution_clock::now(), nowtime = prevtime;
        float init_time = 0, getneighbor_time = 0, calculate_time = 0, update_time = 0;
        assert(num_initialize >= topk);
        assert(num_initialize >= search_L);
        assert(max_degree * beam_size >= max_degree);
#endif
        FAISS_THROW_IF_NOT_MSG(graph_distance_computer != nullptr, "The index is not build yet.");

        auto &retset = threadRetset[threadid];
        auto &usefulset = threadUsefulset[threadid];
        auto &finalset = threadFinalset[threadid];
        auto &vt = threadVt[threadid];
        // Initialize
        assert(num_initialize % 4 == 0);
        retset.resize(num_initialize);
        usefulset.resize(max_degree * beam_size + 1);
        finalset.resize(search_L);
        // for (int i = 0; i < numSearchInitializeItem; i++)
        //   retset[i].distance = qdis(i);
        // memset(vt.visited.data(), vt.visno, numSearchInitializeItem * sizeof(uint8_t));
        for (int i = 0; i < num_initialize; i += 4) {
            vt.set(i + 0);
            vt.set(i + 1);
            vt.set(i + 2);
            vt.set(i + 3);
            retset[i + 0].id = i + 0; // have flag
            retset[i + 1].id = i + 1; // have flag
            retset[i + 2].id = i + 2; // have flag
            retset[i + 3].id = i + 3; // have flag
            graph_distance_computer->distances_batch_4(queryid, i + 0, i + 1, i + 2, i + 3, retset[i].distance, retset[i + 1].distance, retset[i + 2].distance,
                                                       retset[i + 3].distance);
        }

        std::nth_element(retset.begin(), retset.begin() + search_L, retset.end());
        // Maintain the candidate pool in ascending order
        std::sort(retset.data(), retset.data() + search_L);
        // retset.resize(search_L); // remove other items; 不过没有用
        // retset[search_L].distance = INFINITY;

#ifdef INTERNAL_CLOCK_TEST
        int bfs_length = 0, bfs_items = 0;
        int calcdis = 0;
        nowtime = std::chrono::high_resolution_clock::now();
        init_time = std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
        prevtime = nowtime;
        int for_count = 0, calculate_distance_count = 0, useful_count = 0;
#endif
        // puts("Start search");

        // printf("push %d %f\n", retset[ID].id, retset[ID].distance);

        int smallest_pos = 0;
        SingleNeighbor *usefulsetStart = usefulset.data();
        // Stop until the smallest position updated is >= L
        while (smallest_pos < search_L) {
            SingleNeighbor *usefulsetEnd = usefulsetStart;
            // for (int beam = 0; smallest_pos < search_L && (beam < beamSizeMax || usefulsetEnd - usefulsetStart + K0 < K0 * beamSizeMax / 4); smallest_pos++)
            // {
            for (int beam = 0; smallest_pos < search_L && beam < beam_size; smallest_pos++) {
                // for (int beam = 0; smallest_pos < search_L && (beam < beamSizeMax || usefulsetEnd - usefulsetStart < K0 * beamSizeMax / 2); smallest_pos++) {
                // for (int beam = 0; smallest_pos < search_L && usefulsetEnd - usefulsetStart + K0 < beamSizeMax * K0; smallest_pos++) {
                if (retset[smallest_pos].id & 0x80000000)
                    continue;
                int n = retset[smallest_pos].id;
                retset[smallest_pos].id |= 0x80000000;
                beam++;

#ifdef INTERNAL_CLOCK_TEST
                nowtime = std::chrono::high_resolution_clock::now();
                getneighbor_time += std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
                prevtime = nowtime;
#endif
                auto &neighbors = final_graph_neighbors[n];

                // 这里会把所有距离都算出来; 不过其实吧 可以不算之前搜过的, 大概占比10%
                // vt全部assign比多算10%时间要长
                neighbors.compute_distance(queryid, neighborDistance[threadid]);

#ifdef INTERNAL_CLOCK_TEST
                nowtime = std::chrono::high_resolution_clock::now();
                calculate_time += std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
                prevtime = nowtime;
                // 这里其实有点慢
                bfs_items++;
                for_count += neighbors.size; // 这句话非常非常慢
                calculate_distance_count += neighbors.size;
#endif

                float limit = retset[search_L - 1].distance;
#pragma unroll
                for (int k = 0; k < neighbors.size; k++) {
                    // if (threadHeap[threadid].n == 1) {
                    //     printf("%f ", neighborDistance[threadid][k].distance); puts("<- first distance");
                    // }
                    if (neighborDistance[threadid][k] < limit) {
                        if (vt.get(neighbors.edges[k]))
                            continue;
                        vt.set(neighbors.edges[k]);
                        // usefulset.push_back({neighbors.edges[k], neighborDistance[threadid][k]});
                        usefulsetEnd->id = neighbors.edges[k];
                        usefulsetEnd->distance = neighborDistance[threadid][k];
                        usefulsetEnd++;
                    }
                }
#ifdef INTERNAL_CLOCK_TEST
                nowtime = std::chrono::high_resolution_clock::now();
                update_time += std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
                prevtime = nowtime;
#endif
            }
#ifdef INTERNAL_CLOCK_TEST
            bfs_length++;
            useful_count += usefulsetEnd - usefulsetStart;
#endif
            if (usefulsetEnd == usefulsetStart)
                continue;
            {
                // int usefulset_limit = search_L * 1.5; // 搜几次之后存在重复的情况; 超参设置这个resize; 不过感觉上不应该有啥区别
                // int usefulset_limit = search_L / 2;
                // int usefulset_limit = 128; // 这个优化后面再说
                // if (usefulset.size() > usefulset_limit) {
                //     std::nth_element(usefulsetStart, usefulsetStart + usefulset_limit, usefulsetEnd);
                //     usefulset.resize(usefulset_limit);
                // }
                std::sort(usefulsetStart, usefulsetEnd); // search_L的判断感觉没有必要去加; 新数据 K0 * beam 比 search_L 不会大太多
            }

#ifdef INTERNAL_CLOCK_TEST
            nowtime = std::chrono::high_resolution_clock::now();
            update_time += std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
            prevtime = nowtime;
#endif

            // O(n)复杂度; 不处理same的情况
            auto it_new = usefulsetStart, it_initial = retset.data(), it_finalset = finalset.data();
            usefulsetEnd->id = 0x3f3f3f3f;
            usefulsetEnd->distance = INFINITY;
            while (it_finalset != finalset.data() + search_L) {
                if (it_new->distance < it_initial->distance) {
                    int size = it_finalset - finalset.data();
                    if (size < smallest_pos)
                        smallest_pos = size;
                    // smallest_pos = std::min(smallest_pos, size);
                    *it_finalset++ = *it_new++;
                } else
                    *it_finalset++ = *it_initial++;
            }

            memcpy(&retset[0], &finalset[0], sizeof(SingleNeighbor) * search_L);

#ifdef INTERNAL_CLOCK_TEST
            // nowtime = std::chrono::high_resolution_clock::now();
            // update_time += std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
            // prevtime = nowtime;
#endif
        }

        // for (int i = 0; i < threadHeap[threadid].k; i++)
        //     threadRetset[threadid].push_back({threadHeap[threadid].ids[i] & 0x7fffffff, threadHeap[threadid].dis[i]});

        // printf("threadid = %d; ID = %d; search size = %d\n", threadid, ID, threadRetset[threadid].size());
        // assert(threadRetset[threadid].size() >= threadSearchMaxMergeCount);
        // // 输出一下计算的ID~
        // for (int i = 0; i < numThreadsMax; i++)
        //     printf("%d ", threadRetset[i].size());
        // puts("<- bfs_items");
        // for (int i = 0; i < numThreadsMax; i++)
        //     printf("%d ", calculate_distance_count[i]);
        // puts("<- calculate");

        int refinemax = std::min(refine_max, (int)retset.size());
        retset.resize(refinemax);

        for (size_t i = 0; i < retset.size(); i++)
            retset[i].id = search_from_ids[retset[i].id & 0x7fffffff];

        size_t exact_end = retset.size() / 4 * 4;
        for (size_t i = 0; i < exact_end; i += 4)
            realqdis.distances_batch_4(queryid, retset[i].id, retset[i + 1].id, retset[i + 2].id, retset[i + 3].id, retset[i].distance, retset[i + 1].distance,
                                       retset[i + 2].distance, retset[i + 3].distance);
        for (size_t i = exact_end; i < retset.size(); i++)
            retset[i].distance = realqdis(queryid, retset[i].id);
        if ((size_t)topk < retset.size()) {
            std::nth_element(retset.begin(), retset.begin() + topk, retset.end());
            std::sort(retset.begin(), retset.begin() + topk);
        } else {
            std::sort(retset.begin(), retset.end());
        }

        for (size_t i = 0; i < topk; i++) {
            result.indices_ptr()[i] = retset[i].id;
            result.distances_ptr()[i] = retset[i].distance;
        }

        vt.advance();

#ifdef INTERNAL_CLOCK_TEST
        nowtime = std::chrono::high_resolution_clock::now();
        float recalculate_time = std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
        prevtime = nowtime;
        {
            // 需要atomic加锁
            std::lock_guard<std::mutex> lock(m_mutex);
            perf_stats.query_count++;
            perf_stats.init_time += init_time;
            perf_stats.getneighbor_time += getneighbor_time;
            perf_stats.update_time += update_time;
            perf_stats.calculate_time += calculate_time;
            perf_stats.recalculate_time += recalculate_time;
            perf_stats.calculate_distance_count += calcdis;
            perf_stats.bfs_length += bfs_length;
            perf_stats.for_count += for_count;
            perf_stats.bfs_items += bfs_items;
            perf_stats.calculate_distance_count += calculate_distance_count;
            perf_stats.useful_count += useful_count;
        }
#endif
    }
#ifdef INTERNAL_CLOCK_TEST
    std::mutex m_mutex;
#endif

    void flush_perf_stats() {
#ifdef INTERNAL_CLOCK_TEST
        if (perf_stats.query_count == 0) {
            return;
        }
        Logger::info("RNN Descent Running Time: mean bfs %f times; %f nodes; init_time = "
                     "%f, getneighbor_time = %f, calculate_time = %f, update_time = %f; "
                     "recalculate_time = %f; alltime = %f; calculate_dist_count = %f; "
                     "for_count = %f; useful_count = %f\n",
                     (float)perf_stats.bfs_length / perf_stats.query_count, (float)perf_stats.bfs_items / perf_stats.query_count,
                     perf_stats.init_time / perf_stats.query_count, perf_stats.getneighbor_time / perf_stats.query_count,
                     perf_stats.calculate_time / perf_stats.query_count, perf_stats.update_time / perf_stats.query_count,
                     perf_stats.recalculate_time / perf_stats.query_count,
                     (perf_stats.init_time + perf_stats.getneighbor_time + perf_stats.calculate_time + perf_stats.update_time + perf_stats.recalculate_time) /
                         perf_stats.query_count,
                     (float)perf_stats.calculate_distance_count / perf_stats.query_count, (float)perf_stats.for_count / perf_stats.query_count,
                     (float)perf_stats.useful_count / perf_stats.query_count);
        perf_stats.reset();
#endif
    }

    void reset() {
        Logger::line(true, "Resetting RNNDescent index.");
        std::vector<SelectedNeighborsContainerType>().swap(final_graph_neighbors);
        // final_graph_neighbors.resize(0);
        // final_graph.resize(0);
        search_from_ids.resize(0);
        // std::vector<MyVisitedTable>().swap(threadVt); // 这个比较大
        graph_distance_computer.reset();
        flush_perf_stats();
    }

    /// Initialize the KNN graph randomly
    void init_graph(KNNGraph &graph, int n, MyDistanceComputer &qdis, const BuildConfig &build_config) {
        node_locks_.resize(n);
        graph.reserve(n);
        graph.resize(n);
        for (int i = 0; i < n; i++) {
            graph[i].reserve(build_config.R * 2);
        }

#pragma omp parallel num_threads(build_config.num_threads)
        {
            std::mt19937 rng(build_config.random_seed * 7741 + omp_get_thread_num());
#pragma omp for
            for (int i = 0; i < n; i++) {
                std::vector<int> tmp(build_config.S);

                gen_random(rng, tmp, n);

                for (int j = 0; j < build_config.S; j++) {
                    int id = tmp[j];
                    if (id == i)
                        continue;
                    float dist = qdis.symmetric_dis(i, id);

                    graph[i].push_back(XNeighbor(id, dist, true));
                }
                std::make_heap(graph[i].begin(), graph[i].end());
            }
        }
    }

    void update_neighbors(KNNGraph &graph, int n, MyDistanceComputer &qdis, const BuildConfig &build_config) {
        KNNGraph new_pools(build_config.num_threads);
        KNNGraph old_pools(build_config.num_threads);
#pragma omp parallel for num_threads(build_config.num_threads) schedule(dynamic, 16)
        for (int u = 0; u < n; ++u) {
            auto &nhood = graph[u];
            auto &pool = nhood;
            auto &new_pool = new_pools[omp_get_thread_num()];
            auto &old_pool = old_pools[omp_get_thread_num()];
            new_pool.clear();
            old_pool.clear();
            {
                std::lock_guard<std::mutex> guard(node_locks_[u]);
                old_pool.resize(pool.size());
                // std::copy(pool.begin(), pool.end(), old_pool.begin());
                memcpy(old_pool.data(), pool.data(), pool.size() * sizeof(XNeighbor));
                pool.clear();
            }
            std::sort(old_pool.begin(), old_pool.end());
            old_pool.erase(std::unique(old_pool.begin(), old_pool.end(), [](XNeighbor &a, XNeighbor &b) { return a.id() == b.id(); }), old_pool.end());

            for (auto &nn : old_pool) {
                bool ok = true;
                for (auto &other_nn : new_pool) {
                    if (!nn.flag() && !other_nn.flag()) {
                        continue;
                    }
                    if (nn.id() == other_nn.id()) {
                        ok = false;
                        break;
                    }
                    float distance = qdis.symmetric_dis(nn.id(), other_nn.id());
                    if (distance < nn.distance) {
                        ok = false;
                        insert_nn(graph, other_nn.id(), nn.id(), distance, true);
                        break;
                    }
                }
                if (ok) {
                    new_pool.emplace_back(nn);
                }
            }

            for (auto &nn : new_pool) {
                nn.setflag(false);
            }
            {
                std::lock_guard<std::mutex> guard(node_locks_[u]);
                pool.insert(pool.end(), new_pool.begin(), new_pool.end());
            }
        }
    }
    void add_reverse_edges(KNNGraph &graph, int n, const BuildConfig &build_config) {
        std::vector<std::vector<XNeighbor>> reverse_pools(n);

#pragma omp parallel for num_threads(build_config.num_threads)
        for (int u = 0; u < n; ++u) {
            for (auto &nn : graph[u]) {
                std::lock_guard<std::mutex> guard(node_locks_[nn.id()]);
                reverse_pools[nn.id()].emplace_back(XNeighbor(u, nn.distance, nn.flag()));
            }
        }

#pragma omp parallel for num_threads(build_config.num_threads)
        for (int u = 0; u < n; ++u) {
            auto &pool = graph[u];
            for (auto &nn : pool) {
                nn.setflag(true);
            }
            auto &rpool = reverse_pools[u];
            rpool.insert(rpool.end(), pool.begin(), pool.end());
            pool.clear();
            std::sort(rpool.begin(), rpool.end()); // 这里sort可能需要考虑度数了
            rpool.erase(std::unique(rpool.begin(), rpool.end(), [](XNeighbor &a, XNeighbor &b) { return a.id() == b.id(); }), rpool.end());
            if (rpool.size() > build_config.R) {
                rpool.resize(build_config.R);
            }
        }

#pragma omp parallel for num_threads(build_config.num_threads)
        for (int u = 0; u < n; ++u) {
            for (auto &nn : reverse_pools[u]) {
                std::lock_guard<std::mutex> guard(node_locks_[nn.id()]);
                graph[nn.id()].emplace_back(u, nn.distance, nn.flag()); // 所有edge
            }
        }

#pragma omp parallel for num_threads(build_config.num_threads)
        for (int u = 0; u < n; ++u) {
            auto &pool = graph[u];
            std::sort(pool.begin(), pool.end()); // 这里sort可能需要考虑度数了
            if (pool.size() > build_config.R) {
                pool.resize(build_config.R);
            }
        }
    }

    void insert_nn(KNNGraph &graph, int id, int nn_id, float distance, bool flag) {
        {
            std::lock_guard<std::mutex> guard(node_locks_[id]);
            graph[id].emplace_back(nn_id, distance, flag);

            // if (distance > graph[id].front().distance)
            //     return;
            // for (int i = 0; i < graph[id].size(); i++) {
            //     if (id == graph[id][i].id())
            //         return;
            // }
            // if (graph[id].size() < graph[id].capacity()) {
            //     graph[id].push_back(XNeighbor(id, distance, flag));
            //     std::push_heap(graph[id].begin(), graph[id].end());
            // } else {
            //     std::pop_heap(graph[id].begin(), graph[id].end());
            //     graph[id][graph[id].size() - 1] = XNeighbor(id, distance, flag);
            //     std::push_heap(graph[id].begin(), graph[id].end());
            // }

            // for (int i = 0; i < graph[id].size(); i++) {
            //     if (id == graph[id][i].id())
            //         return;
            // }
            // graph[id].emplace_back(nn_id, distance, flag);
        }
    }

    std::vector<SelectedNeighborsContainerType> final_graph_neighbors;
    std::unique_ptr<MyDistanceComputer> graph_distance_computer;

  private:
    std::vector<MutexWarpper> node_locks_;
};

} // namespace rnndescent

namespace rnndescent {} // namespace rnndescent
