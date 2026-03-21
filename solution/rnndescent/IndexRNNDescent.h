#pragma once

#include <faiss/Index.h>
#include <omp.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <memory>
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
    // Current implementation only supports L2 distance.
    struct PCAConfig {
        bool enabled = false;
        int out_dim = 0;
    };

    bool verbose;
    std::unique_ptr<MyDistanceComputer> disComputer;
    int d = 0;
    bool is_trained = false;

    RNNDescent rnndescent;
    RNNDescent::BuildConfig build_config;
    RNNDescent::SearchConfig search_config;
    PCAConfig pca_config;

    explicit IndexRNNDescent(int d = 0) : verbose(false), d(d), rnndescent(d) {}

    ~IndexRNNDescent() { reset(); }

    faiss::PCAMatrix pca;
    std::vector<float> pcax; // query pca-matrix; need preserve
    void add(const RNNDescent::FloatMatrixView &base) {
        FAISS_THROW_IF_NOT(is_trained);
        FAISS_THROW_IF_NOT(disComputer == nullptr); // 暂时不支持增删
        base.validate("add data");
        FAISS_THROW_IF_NOT_MSG(base.dimension() == d, "add data dimension does not match index dimension");
        const idx_t n = base.row_count();
        const float *x = base.data_ptr();

        build_config = RNNDescent::sanitize_build_config(build_config);
        search_config = RNNDescent::sanitize_search_config(search_config);

        auto prevtime = std::chrono::high_resolution_clock::now(), nowtime = std::chrono::high_resolution_clock::now();
        float process_time;
        const bool use_pca = pca_config.enabled && pca_config.out_dim > 0 && pca_config.out_dim < d;
        if (use_pca) {
            puts("train PCA Matrix");
            pca = faiss::PCAMatrix();
            pca.d_in = d;
            pca.d_out = pca_config.out_dim;
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
            rnndescent.build(RNNDescent::FloatMatrixView::from_vector(pcaMatrix, rnndescent.d), verbose, build_config, search_config);

            const int maxquery = 10000;
            pcax.reserve(pca.d_out * maxquery);
        } else {
            pca = faiss::PCAMatrix();
            rnndescent.d = d;
            rnndescent.build(RNNDescent::FloatMatrixView::from_buffer(x, n, d), verbose, build_config, search_config);
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
        disComputer = SelectedDistanceComputerFactory::create_refine_search(x, n, d);
        // disComputer = new CblasDistanceComputerFP32L2(x, ntotal, d);
    }

    void train(const RNNDescent::FloatMatrixView &base) {
        base.validate("train data");
        FAISS_THROW_IF_NOT_MSG(base.dimension() == d, "train data dimension does not match index dimension");
        // nndescent structure does not require training
        is_trained = true;
    }

    void pca_apply_noalloc(const RNNDescent::FloatMatrixView &input, const RNNDescent::MutableFloatMatrixView &output) {
        FAISS_THROW_IF_NOT_MSG(pca.is_trained, "Transformation not trained yet");
        input.validate("pca input");
        output.validate("pca output");
        FAISS_THROW_IF_NOT_MSG(input.row_count() == output.row_count(), "pca input/output row count mismatch");
        FAISS_THROW_IF_NOT_MSG(input.dimension() == pca.d_in, "pca input dimension mismatch");
        FAISS_THROW_IF_NOT_MSG(output.dimension() == pca.d_out, "pca output dimension mismatch");
        const idx_t n = input.row_count();
        const float *x = input.data_ptr();
        float *xt = output.data_ptr();

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

    void search(const RNNDescent::FloatMatrixView &queries, const RNNDescent::SearchResultView &result, const faiss::SearchParameters *params = nullptr) {
        FAISS_THROW_IF_NOT_MSG(!params, "search params not supported for this index");
        FAISS_THROW_IF_NOT_MSG(disComputer != nullptr, "index has not been built");
        FAISS_THROW_IF_NOT_MSG(rnndescent.fastqdis != nullptr, "search graph has not been initialized");
        queries.validate("query batch");
        result.validate();
        FAISS_THROW_IF_NOT_MSG(queries.dimension() == d, "query dimension does not match index dimension");
        const idx_t n = queries.row_count();
        const idx_t k = result.topk();
        const float *x = queries.data_ptr();
        float *distances = result.distances_ptr();
        int *labels = result.indices_ptr();
#ifdef INTERNAL_CLOCK_TEST
        auto prevtime = std::chrono::high_resolution_clock::now();
        auto nowtime = prevtime, starttime = prevtime;
        float process_time;
#endif

        disComputer->set_query(x, n);
        if (pca.is_trained) {
            pcax.resize(n * pca.d_out);
            // pca.apply_noalloc(n, x, pcax.data());  // 这句话太慢了
            pca_apply_noalloc(RNNDescent::FloatMatrixView::from_buffer(x, static_cast<int>(n), d),
                              RNNDescent::MutableFloatMatrixView::from_vector(pcax, pca.d_out));
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

#pragma omp parallel for num_threads(search_config.num_threads) schedule(dynamic, 4)
        for (int queryid = 0; queryid < n; queryid++) {
            int threadid = omp_get_thread_num();
            rnndescent.searchSingle(threadid, queryid, *disComputer, search_config, build_config.K0, result.slice(queryid));
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

    void reset() {
        rnndescent.reset();
        disComputer.reset();
        pca = faiss::PCAMatrix();
        pcax.clear();
    }
};

} // namespace rnndescent