rm -rf build
mkdir -p build
cd build
cmake .. && make -j20

# export OMP_NUM_THREADS=24
export OMP_NUM_THREADS=16
# export DATASET="sift"
export DATASET="siftsmall"
export RESULT_DIR="../benches/results/$DATASET"
mkdir -p $RESULT_DIR
# sh ./../benches/bench_hnsw.sh
# sh ./../benches/bench_nsg.sh
# sh ./../benches/bench_nndescent.sh
# sh ./../benches/bench_rnndescent.sh
./algorithm