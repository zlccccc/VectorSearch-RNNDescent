#pragma once

#include <bits/stdc++.h>
#include <memory_resource>

// #include "discomputer/CblasDistComputerIP.h"
#include "discomputer/CblasDistComputerFP32.h"
// #include "discomputer/FaissDistComputerIP.h"
// #include "discomputer/FaissDistComputerL2.h"

#include "discomputer/Avx2SimdDistanceComputerFP32.h"
// #include "discomputer/Avx512SimdDistanceComputerFP32.h"
// #define INTERNAL_CLOCK_TEST

#include "discomputer/Avx2SimdDistanceComputerInt8.h"
// #include "discomputer/Avx512SimdDistanceComputerUInt8.h"
// #include "discomputer/NeonSimdDistanceComputerInt8.h"

namespace {
struct MutexWarpper : std::mutex {
    MutexWarpper() = default;
    MutexWarpper(MutexWarpper const &) noexcept : std::mutex() {}
    bool operator==(MutexWarpper const &other) noexcept { return this == &other; }
};

std::vector<MutexWarpper> lockwarppers;

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
} // namespace

namespace rnndescent {

// using NeighborsContainerType = UInt4Neighbors;
using NeighborsContainerType = Int8Neighbors;
// using NeighborsContainerType = FP16Neighbors;
// using NeighborsContainerType = CblasNeighbors;

// using SaveneighborDiscomputer = SimdDistanceComputerInt4L2;
using SaveneighborDiscomputer = SimdDistanceComputerInt8L2;
// using SaveneighborDiscomputer = SimdDistanceComputerFP16L2;
// using SaveneighborDiscomputer = CblasDistanceComputerFP32L2;

struct RNNDescent {
    struct BuildConfig {
        int T1 = 4;
        int T2 = 15;
        int S = 16;
        int R = 96;
        int K0 = 48;
        int random_seed = 2021;
        int num_threads = 16;
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

    static void gen_random(std::mt19937 &rng, int *addr, const int size, const int N) { // 好像这个有极小概率random出错啊...shuffle最简单
        for (int i = 0; i < size; ++i) {
            addr[i] = rng() % (N - size);
        }
        std::sort(addr, addr + size);
        for (int i = 1; i < size; ++i) {
            if (addr[i] <= addr[i - 1]) {
                addr[i] = addr[i - 1] + 1;
            }
        }
        int off = rng() % N;
        for (int i = 0; i < size; ++i) {
            addr[i] = (addr[i] + off) % N;
        }
    }

    using KNNGraph = std::vector<std::vector<XNeighbor>>;

    explicit RNNDescent(const int d) : d(d) {}


#ifdef INTERNAL_CLOCK_TEST
    float init_time{0}, getneighbor_time{0}, calculate_time{0}, update_time{0}, recalculate_time{0};
    long long query_count{0}, calculate_distance_count{0}, for_count{0}, bfs_length{0}, useful_count{0}, bfs_items{0};
#endif
    ~RNNDescent() { reset(); }

    MyDistanceComputer *GenerateDistanceComputer(const float *data, int n) {
        // return new FaissDistanceComputerL2(data, n, dim);
        // return new CblasDistanceComputerFP32L2(data, n, d);
        // return new SimdDistanceComputerFP16L2(data, n, dim);
        return new SimdDistanceComputerInt8L2(data, n, d);
        // return new SimdDistanceComputerInt8L2Norm(data, n, dim);
        throw std::runtime_error("Invalid metric type");
    }

    void generate_graph(KNNGraph &graph, const float *x, bool verbose, const BuildConfig &build_config) {
        printf("generte graph ntotal = %d\n", ntotal);
        auto qdis = GenerateDistanceComputer(x, ntotal);
        init_graph(graph, *qdis, build_config);
        for (int t1 = 0; t1 < build_config.T1; ++t1) {
            if (verbose)
                std::cout << "Iter " << t1 << " : " << std::flush;
            for (int t2 = 0; t2 < build_config.T2; ++t2) {
                update_neighbors(graph, *qdis, build_config);
                if (verbose)
                    std::cout << "#" << std::flush;
            }
            if (verbose)
                printf("\n");

            if (t1 != build_config.T1 - 1)
                add_reverse_edges(graph, build_config);
        }
        delete qdis;

#pragma omp parallel for
        for (int u = 0; u < ntotal; ++u) { // remove edges
            auto &pool = graph[u];
            std::sort(pool.begin(), pool.end());
            pool.erase(std::unique(pool.begin(), pool.end(), [](XNeighbor &a, XNeighbor &b) { return a.id() == b.id(); }), pool.end());
        }

        // 这里的resize可能会导致图不联通; 这件事情需要在上面build的时候就考虑到
        int all_edges_size = 0;
        for (int u = 0; u < ntotal; ++u) {
            // 清理内存
            if (graph[u].size() > build_config.K0)
                graph[u].resize(build_config.K0);
            all_edges_size += graph[u].size();
            graph[u].shrink_to_fit();
        }
        printf("graph edges size = %d\n", all_edges_size);
    }

    void build(const int n, bool verbose, const float *x, const BuildConfig &build_config, const SearchConfig &search_config) {
        if (verbose)
            printf("Parameters: S=%d, R=%d, T1=%d, T2=%d; Point=%d; numThreadsMax=%d\n", build_config.S, build_config.R, build_config.T1, build_config.T2, n,
                   search_config.num_threads);
        NeighborsContainerType::clear_memory();

        std::vector<std::vector<int>> edges; // distance并不重要
        edges.resize(n);
        {
            KNNGraph graph;
            ntotal = n;
            generate_graph(graph, x, verbose, build_config); // highest level
                                               // #pragma omp parallel for
            std::set<int> S; // 不能重复
            for (int i = 0; i < n; i++) {
                auto &pool = graph[i];
                sort(pool.begin(), pool.end());
                edges[i].reserve((pool.size() + 3) / 4 * 4);
                // 需要清理下内存; 变成4的倍数方便后面reorder
                S.clear();
                for (auto &edge : pool) {
                    if (!S.count(edge.id())) {  // 这个地方感觉不太应该写成这样; 如果写得好的话, 理论上刚开始的edge id不会重复
                        S.insert(edge.id());
                        edges[i].emplace_back(edge.id());
                    }
                }
                while (edges[i].size() % 4 != 0) { // 补到差不多
                    int id = random() % ntotal;
                    while (S.count(id))
                        id = random() % ntotal;
                    S.insert(id);
                    edges[i].emplace_back(id);
                }
                std::vector<XNeighbor>().swap(pool);
            }
        }
        // 确定全局入口点
        printf("n = %d; initialize = %d\n", n, search_config.num_initialize);
        {
            std::mt19937 rng(build_config.random_seed);
            search_from_ids.reserve(ntotal);
            search_from_ids.resize(search_config.num_initialize);
            if (build_config.random_init)
                gen_random(rng, search_from_ids.data(), search_config.num_initialize, n);
            else
                iota(search_from_ids.begin(), search_from_ids.end(), 0);
        }
        ntotal = n;

        // 重排ID, 加速运行
        std::vector<int> rollback_ids(ntotal); // 连graph的时候边id需更新

        { // ID重排
            puts("Start Reordering The Graph.");
            // bfs from those indexes
            // search_from_ids.resize(numSearchInitializeItem); // range
            // for (int i = 0; i < search_from_ids.size(); i++)
            //     printf("%d ",search_from_ids[i]); puts("<");
            MyVisitedTable vis(ntotal);

            // cluster_id: 重排ID以后, 从哪个cluster可以bfs到当前点
            std::vector<int> cluster_id(ntotal);
            for (int i = 0; i < search_from_ids.size(); i++) {
                vis.set(search_from_ids[i]);
                cluster_id[search_from_ids[i]] = i;
            }
            std::vector<int> bfs_distance(ntotal); // 只是用来记录一下
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
                std::vector<int> cluster_size(search_config.num_initialize);
                for (int i = 0; i < cluster_id.size(); i++)
                    cluster_size[cluster_id[i]]++;
                for (int v : cluster_size)
                    printf("%d ", v);
                puts(" <<- initial cluster size");
                std::vector<int> bfs_length_count;
                for (int i = 0; i < search_from_ids.size(); i++) {
                    int v = bfs_distance[search_from_ids[i]];
                    if (v >= bfs_length_count.size())
                        bfs_length_count.resize(v + 1);
                    bfs_length_count[v]++;
                }
                for (int i = 0; i < bfs_length_count.size(); i++)
                    printf("bfs_length %d: %d\n", i, bfs_length_count[i]);
            }

            int cannot_search = 0; // 这里先暂时不处理这种情况; 会有概率有的点搜不到
            for (int i = 0; i < ntotal; i++) {
                if (!vis.get(i))
                    search_from_ids.push_back(i), cannot_search++;
            }
#ifdef INTERNAL_CLOCK_TEST
            printf("search from ids = %d; have_next = %d; cannot_search = %d\n", (int)search_from_ids.size(), all_have_next, cannot_search);
            assert(ntotal == search_from_ids.size());
#endif

            for (int i = 0; i < ntotal; i++) // 连graph的时候边id需更新
                rollback_ids[search_from_ids[i]] = i;

            std::vector<float> fastsearch_pool;
            fastsearch_pool.resize(ntotal * d);
#pragma omp parallel for
            for (int i = 0; i < ntotal; i++) {
                int u = search_from_ids[i];
                // printf("u = %d; matrix.size() = %d\n",u, matrix.size());
                memcpy(fastsearch_pool.data() + i * d, x + u * d, d * sizeof(float));
            }
            fastqdis = new SaveneighborDiscomputer(fastsearch_pool.data(), ntotal, d);
        }
        { // 空间局部性优化
            NeighborsContainerType::init_neighbors_pool(d, edges, 16ll * 1024 * 1024 * 1024, build_config.save_neighbor);
            for (int i = 0; i < ntotal; i++) {
                int u = search_from_ids[i];
                auto &pool = edges[u];
                final_graph_neighbors.emplace_back(NeighborsContainerType(d, pool, fastqdis, rollback_ids, build_config.save_neighbor)); // 全部save
            }
        }
        has_built = true;

        threadVt.resize(search_config.num_threads);
        threadRetset.resize(search_config.num_threads);
        neighborDistance.resize(search_config.num_threads);
        threadUsefulset.resize(search_config.num_threads);
        threadFinalset.resize(search_config.num_threads);
        for (int i = 0; i < search_config.num_threads; i++) {
            threadVt[i].init(ntotal);
            threadRetset[i].reserve(std::max(search_config.num_initialize, search_config.search_L));
            neighborDistance[i].reserve(search_config.search_L);
            threadFinalset[i].reserve(search_config.search_L);
            threadUsefulset[i].reserve(build_config.K0 * search_config.beam_size + 1);
        }
    }

    std::vector<int> search_from_ids;


    std::vector<MyVisitedTable> threadVt;                  // threadRetset和之前的retset起到的价值差不多
    std::vector<std::vector<SingleNeighbor>> threadRetset; // threadRetset和之前的retset起到的价值差不多
    std::vector<std::vector<SingleNeighbor>> threadUsefulset;
    std::vector<std::vector<SingleNeighbor>> threadFinalset;
    std::vector<std::vector<float>> neighborDistance;

    void searchSingle(int threadid, int queryid, MyDistanceComputer &realqdis, const SearchConfig &search_config, int max_degree, const int topk, int *indices,
                      float *dists, bool output) {
        const int num_threads = search_config.num_threads;
        const int beam_size = search_config.beam_size;
        const int search_L = search_config.search_L;
        const int num_initialize = search_config.num_initialize;
        const int refine_max = search_config.refine_max;
        assert(0 <= threadid && threadid < num_threads);
#ifdef INTERNAL_CLOCK_TEST
        auto prevtime = std::chrono::high_resolution_clock::now(), nowtime = prevtime;
        float init_time = 0, getneighbor_time = 0, calculate_time = 0, update_time = 0;
        assert(num_initialize >= topk);
        assert(num_initialize >= search_L);
        assert(max_degree * beam_size >= max_degree);
#endif
        FAISS_THROW_IF_NOT_MSG(has_built, "The index is not build yet.");

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
            fastqdis->distances_batch_4(queryid, i + 0, i + 1, i + 2, i + 3, retset[i].distance, retset[i + 1].distance, retset[i + 2].distance,
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
        sort(retset.begin(), retset.end());

        for (size_t i = 0; i < topk; i++) {
            indices[i] = retset[i].id;
            dists[i] = retset[i].distance;
        }

        vt.advance();

#ifdef INTERNAL_CLOCK_TEST
        nowtime = std::chrono::high_resolution_clock::now();
        float recalculate_time = std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
        prevtime = nowtime;
        if (output) {
            printf("bfs: %d; calc=%d\n", bfs_length, calcdis);
            // for (int x = 0; x < 10; ++x)
            //     printf("%f ", finalset[x].distance);
            // for (int x = 0; x < 10; ++x)
            //     for (int y = 0; y < 10; ++y)
            //         printf("%f ", qdis.symmetric_dis(x, y)); puts("<- inside dfs");
            // puts("<- dist");
            printf("time: init_time = %f, getneighbor_time = %f, calculate_time = %f, "
                   "update_time = %f; recalculate_time = %f; alltime = %f\n",
                   init_time, getneighbor_time, calculate_time, update_time, recalculate_time,
                   init_time + getneighbor_time + calculate_time + update_time + recalculate_time);
        }
        {
            // 需要atomic加锁
            std::lock_guard<std::mutex> lock(m_mutex);
            query_count++;
            this->init_time += init_time;
            this->getneighbor_time += getneighbor_time;
            this->update_time += update_time;
            this->calculate_time += calculate_time;
            this->recalculate_time += recalculate_time;
            this->calculate_distance_count += calcdis;
            this->bfs_length += bfs_length;
            this->for_count += for_count;
            this->bfs_items += bfs_items;
            this->calculate_distance_count += calculate_distance_count;
            this->useful_count += useful_count;
        }
#endif
    }
#ifdef INTERNAL_CLOCK_TEST
    std::mutex m_mutex;
#endif

    void reset_time() {
#ifdef INTERNAL_CLOCK_TEST
        if (query_count == 0) {
            return;
        }
        printf("RNN Descent Running Time: mean bfs %f times; %f nodes; init_time = "
               "%f, getneighbor_time = %f, calculate_time = %f, update_time = %f; "
               "recalculate_time = %f; alltime = %f; calculate_dist_count = %f; "
               "for_count = %f; useful_count = %f\n",
               (float)bfs_length / query_count, (float)bfs_items / query_count, init_time / query_count, getneighbor_time / query_count,
               calculate_time / query_count, update_time / query_count, recalculate_time / query_count,
               (init_time + getneighbor_time + calculate_time + update_time + recalculate_time) / query_count, (float)calculate_distance_count / query_count,
               (float)for_count / query_count, (float)useful_count / query_count);
        init_time = 0, getneighbor_time = 0, calculate_time = 0, update_time = 0, recalculate_time = 0;
        query_count = 0, calculate_distance_count = 0, for_count = 0, bfs_length = 0, useful_count = 0, bfs_items = 0;
#endif
    }

    void reset() {
        has_built = false;
        ntotal = 0;
        std::vector<NeighborsContainerType>().swap(final_graph_neighbors);
        // final_graph_neighbors.resize(0);
        // final_graph.resize(0);
        search_from_ids.resize(0);
        // std::vector<MyVisitedTable>().swap(threadVt); // 这个比较大

        if (fastqdis != nullptr) {
            delete fastqdis;
            fastqdis = nullptr;
        }

        reset_time();
    }

    /// Initialize the KNN graph randomly
    void init_graph(KNNGraph &graph, MyDistanceComputer &qdis, const BuildConfig &build_config) {
        lockwarppers.resize(ntotal);
        graph.reserve(ntotal);
        graph.resize(ntotal);
        for (int i = 0; i < ntotal; i++) {
            // graph[i].reserve(initialize_L);
            graph[i].reserve(build_config.R * 2);
        }

#pragma omp parallel
        {
            std::mt19937 rng(build_config.random_seed * 7741 + omp_get_thread_num());
#pragma omp for
            for (int i = 0; i < ntotal; i++) {
                std::vector<int> tmp(build_config.S);

                gen_random(rng, tmp.data(), build_config.S, ntotal);

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

    void update_neighbors(KNNGraph &graph, MyDistanceComputer &qdis, const BuildConfig &build_config) {
        std::vector<std::vector<XNeighbor>> new_pools(build_config.num_threads);
        std::vector<std::vector<XNeighbor>> old_pools(build_config.num_threads);
#pragma omp parallel for schedule(dynamic, 16)
        for (int u = 0; u < ntotal; ++u) {
            auto &nhood = graph[u];
            auto &pool = nhood;
            auto &new_pool = new_pools[omp_get_thread_num()];
            auto &old_pool = old_pools[omp_get_thread_num()];
            new_pool.clear();
            old_pool.clear();
            {
                std::lock_guard<std::mutex> guard(lockwarppers[u]);
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
                std::lock_guard<std::mutex> guard(lockwarppers[u]);
                pool.insert(pool.end(), new_pool.begin(), new_pool.end());
            }
        }
    }
    void add_reverse_edges(KNNGraph &graph, const BuildConfig &build_config) {
        std::vector<std::vector<XNeighbor>> reverse_pools(ntotal);

#pragma omp parallel for
        for (int u = 0; u < ntotal; ++u) {
            for (auto &nn : graph[u]) {
                std::lock_guard<std::mutex> guard(lockwarppers[nn.id()]);
                reverse_pools[nn.id()].emplace_back(XNeighbor(u, nn.distance, nn.flag()));
            }
        }

#pragma omp parallel for
        for (int u = 0; u < ntotal; ++u) {
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

#pragma omp parallel for
        for (int u = 0; u < ntotal; ++u) {
            for (auto &nn : reverse_pools[u]) {
                std::lock_guard<std::mutex> guard(lockwarppers[nn.id()]);
                graph[nn.id()].emplace_back(u, nn.distance, nn.flag()); // 所有edge
            }
        }

#pragma omp parallel for
        for (int u = 0; u < ntotal; ++u) {
            auto &pool = graph[u];
            std::sort(pool.begin(), pool.end()); // 这里sort可能需要考虑度数了
            if (pool.size() > build_config.R) {
                pool.resize(build_config.R);
            }
        }
    }

    void insert_nn(KNNGraph &graph, int id, int nn_id, float distance, bool flag) {
        {
            std::lock_guard<std::mutex> guard(lockwarppers[id]);
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

    bool has_built = false;


    int d;                // dimensions
    int initialize_L = 8; // initial size of memory allocation

    int ntotal = 0;

    std::vector<NeighborsContainerType> final_graph_neighbors;
    MyDistanceComputer *fastqdis = nullptr; // 新的disComputer
    // std::vector<MyVisitedTable> threadVt;
};

} // namespace rnndescent

namespace rnndescent {} // namespace rnndescent