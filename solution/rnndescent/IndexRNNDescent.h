#pragma once

#include <faiss/Index.h>
#include <omp.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <unordered_set>

#include "RNNDescent.h"
#include <cblas.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>

namespace rnndescent {

using idx_t = faiss::idx_t;
using FINTEGER = int;

extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_(const char *transa, const char *transb, FINTEGER *m, FINTEGER *n, FINTEGER *k, const float *alpha, const float *a, FINTEGER *lda, const float *b,
           FINTEGER *ldb, float *beta, float *c, FINTEGER *ldc);
} // namespace

struct IndexRNNDescent {
    bool verbose;
    faiss::Index *index = nullptr;
    MyDistanceComputer *disComputer = nullptr;
    int ntotal = 0, d = 0;
    faiss::MetricType metric_type = faiss::METRIC_L2;
    bool is_trained = false;

    RNNDescent rnndescent;
    RNNDescent::BuildConfig build_config;
    RNNDescent::SearchConfig search_config;

    explicit IndexRNNDescent(int d = 0, faiss::MetricType metric = faiss::METRIC_L2) : verbose(false), d(d), rnndescent(d), metric_type(metric), ntotal(0) {}

    ~IndexRNNDescent() { reset(); }

    faiss::PCAMatrix pca;
    std::vector<float> pcax;      // query pca-matrix; need preserve
    std::vector<float> meand;     // 图1平均值
    std::vector<float> refined_x; // 图1
    std::vector<int> idxmap;      // 图1映射
    void add(idx_t n, const float *x) {
        FAISS_THROW_IF_NOT(is_trained);
        FAISS_THROW_IF_NOT(ntotal == 0); // 暂时不支持增删
        FAISS_THROW_IF_NOT_MSG(x != nullptr, "add data pointer is null");
        FAISS_THROW_IF_NOT_MSG(n > 0, "add requires at least one vector");

        if (ntotal != 0) {
            fprintf(stderr, "WARNING NNDescent doest not support dynamic insertions,"
                            "multiple insertions would lead to re-building the index");
        }
        assert(metric_type == faiss::METRIC_L2); // IP不支持了; disComputer在里面初始化

        build_config = RNNDescent::sanitize_build_config(build_config);
        search_config = RNNDescent::sanitize_search_config(search_config);

        auto prevtime = std::chrono::high_resolution_clock::now(), nowtime = std::chrono::high_resolution_clock::now();
        float process_time;
        ntotal = n;
        // if (false) {
        if (false && (d == 1024 || d == 1536)) {
            puts("train PCA Matrix");
            pca.d_in = d;
            if (d == 512) {
                pca.d_out = 384;
            } else if (d == 1024) {
                pca.d_out = 256;
            } else if (d == 1536) {
                pca.d_out = 384;
            } else {
                throw std::runtime_error("Invalid dimension");
            }
            // throw std::runtime_error("Invalid dimension");
            // pca.verbose = true;
            pca.train(n, x);
            std::vector<float> pcaMatrix;
            pcaMatrix.resize(n * pca.d_out);
            pca.apply_noalloc(n, x, pcaMatrix.data());
            printf("End Calc PCA Matrix\n");
            nowtime = std::chrono::high_resolution_clock::now();
            process_time = std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
            prevtime = nowtime;
            printf("PCA Process done in %f ms\n", process_time);
            rnndescent.d = pca.d_out;
            rnndescent.build(ntotal, verbose, pcaMatrix.data(), build_config, search_config);

            const int maxquery = 10000;
            pcax.reserve(pca.d_out * maxquery);
        } else {
            rnndescent.build(ntotal, verbose, x, build_config, search_config);
        }

        nowtime = std::chrono::high_resolution_clock::now();
        process_time = std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
        prevtime = nowtime;
        printf("RNN Descent build done in %f ms\n", process_time);

        // delete disComputer;
        assert(disComputer == nullptr);
        // disComputer = new CblasDistanceComputerL2(x, ntotal, d, false);
        // disComputer = new SimdDistanceComputerFP16L2(x, ntotal, d);
        // disComputer = new SimdDistanceComputerInt16L2(x, ntotal, d);
        if (idxmap.size() == 0) {
            disComputer = new SimdDistanceComputerFP32L2(x, ntotal, d);
            // disComputer = new CblasDistanceComputerFP32L2(x, ntotal, d);
        } else {
            disComputer = new SimdDistanceComputerFP32L2(refined_x.data(), idxmap.size(), d);
            // disComputer = new CblasDistanceComputerFP32L2(refined_x.data(), idxmap.size(), d);
        }

        // // index = new faiss::IndexScalarQuantizer(d,
        // // faiss::ScalarQuantizer::QT_8bit_uniform); index->train(n, matrix.data());
        // // index->add(n, matrix.data());
        // nowtime = std::chrono::high_resolution_clock::now();
        // auto pca_time = std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
        // prevtime = nowtime;
        // printf("RNN Descent FastIndex build done in %f ms\n", pca_time);
    }

    void train(idx_t n, const float *x) {
        FAISS_THROW_IF_NOT_MSG(x != nullptr, "train data pointer is null");
        FAISS_THROW_IF_NOT_MSG(n > 0, "train requires at least one vector");
        // nndescent structure does not require training
        is_trained = true;
    }

    int left = 1e6, right = 0;
    int count = 0;

    void pca_apply_noalloc(idx_t n, const float *x, float *xt) {
        FAISS_THROW_IF_NOT_MSG(pca.is_trained, "Transformation not trained yet");

        float c_factor;
        if (pca.have_bias) {
            FAISS_THROW_IF_NOT_MSG(pca.b.size() == pca.d_out, "Bias not initialized");
#pragma omp parallel for
            for (int i = 0; i < n; i++)
                memcpy(xt + i * pca.d_out, pca.b.data(), pca.d_out * sizeof(float));
            // for (int j = 0; j < pca.d_out; j++)
            //     xt[i * pca.d_out + j] = pca.b[j];
            c_factor = 1.0;
        } else {
            c_factor = 0.0;
        }

        FAISS_THROW_IF_NOT_MSG(pca.A.size() == pca.d_out * pca.d_in, "Transformation matrix not initialized");

        float one = 1;
        FINTEGER nbiti = pca.d_out, ni = n, di = pca.d_in;
        // int nbiti = pca.d_out, ni = n, di = pca.d_in;
        sgemm_("Transposed", "Not transposed", &nbiti, &ni, &di, &one, pca.A.data(), &di, x, &di, &c_factor, xt, &nbiti);
        // cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, nbiti, ni, di, one, pca.A.data(), di, x, di, c_factor, xt, nbiti);

        //         // 貌似评测机的sgemm已经自带了多线程? 但是我咋这么不信呢; 但是事实就是加上去一点用没有
        // #pragma omp parallel for
        //         for (int i = 0; i < 16; i++) {
        //             int nk = ni / 16;
        //             cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, nbiti, nk, di, one, pca.A.data(), di, x + i * nk * d, di, c_factor, xt + i * nk *
        //             pca.d_out, nbiti);
        //         }
    }

    void search(idx_t n, const float *x, idx_t k, float *distances, int *labels, const faiss::SearchParameters *params = nullptr) {
        FAISS_THROW_IF_NOT_MSG(!params, "search params not supported for this index");
        FAISS_THROW_IF_NOT_MSG(disComputer != nullptr, "index has not been built");
        FAISS_THROW_IF_NOT_MSG(rnndescent.fastqdis != nullptr, "search graph has not been initialized");
        FAISS_THROW_IF_NOT_MSG(x != nullptr, "query pointer is null");
        FAISS_THROW_IF_NOT_MSG(distances != nullptr, "distance output buffer is null");
        FAISS_THROW_IF_NOT_MSG(labels != nullptr, "label output buffer is null");
        FAISS_THROW_IF_NOT_MSG(n > 0, "search requires at least one query");
        FAISS_THROW_IF_NOT_MSG(k > 0, "search requires positive top-k");
        auto searchstarttime = std::chrono::high_resolution_clock::now();

#ifdef INTERNAL_CLOCK_TEST
        auto prevtime = std::chrono::high_resolution_clock::now();
        auto nowtime = prevtime, starttime = prevtime;
        float process_time;
#endif

        disComputer->set_query(x, n);
        if (pca.is_trained) {
            pcax.resize(n * pca.d_out);
            // pca.apply_noalloc(n, x, pcax.data());  // 这句话太慢了
            pca_apply_noalloc(n, x, pcax.data());
            rnndescent.fastqdis->set_query(pcax.data(), n);
        } else {
            rnndescent.fastqdis->set_query(x, n);
        }

#ifdef INTERNAL_CLOCK_TEST
        nowtime = std::chrono::high_resolution_clock::now();
        process_time = std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
        prevtime = nowtime;
        printf("search: PCA Process done in %f ms/item\n", process_time / n);
#endif

#pragma omp parallel for schedule(dynamic, 4)
        for (int queryid = 0; queryid < n; queryid++) {
            int threadid = omp_get_thread_num();
            rnndescent.searchSingle(threadid, queryid, *disComputer, search_config, build_config.K0, k, labels + queryid * k, distances + queryid * k, false);
            if (idxmap.size() != 0) { // 图1
                for (int i = 0; i < k; i++) {
                    // left = std::min(left, labels[i + queryid * k]);
                    // right = std::max(right, labels[i + queryid * k]);
                    labels[i + queryid * k] = idxmap[labels[i + queryid * k]];
                }
            }
            if (metric_type == faiss::METRIC_INNER_PRODUCT) {
                for (int i = 0; i < k; i++) {
                    distances[i + queryid * k] = -distances[i + queryid * k];
                }
            }
        }

#ifdef INTERNAL_CLOCK_TEST
        nowtime = std::chrono::high_resolution_clock::now();
        process_time = std::chrono::duration<float, std::milli>(nowtime - prevtime).count();
        prevtime = nowtime;
        auto all_time = std::chrono::duration<float, std::milli>(nowtime - starttime).count();
        printf("search: RNNDescent Search Process done in %f ms/item; all = %f ms/item; internal point %f\n", process_time / n, all_time / n,
               1000. * n / all_time);
#endif
    }

    void searchSingle(const float *x, idx_t k, float *distances, int *labels) {
        // auto disComputer = index->get_distance_computer();
        // delete disComputer;
        FAISS_THROW_MSG("Not Implemented");
    }

    void reset() {
        rnndescent.reset();
        if (index != nullptr) {
            delete index;
            index = nullptr;
        }
        if (disComputer != nullptr) {
            delete disComputer;
            disComputer = nullptr;
        }
        ntotal = 0;
    }
};

} // namespace rnndescent