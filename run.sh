rm -rf build
mkdir -p build
cd build
cmake .. && make -j20

# export OMP_NUM_THREADS=24
export OMP_NUM_THREADS=16
export DATASET="sift"
# export DATASET="siftsmall"
export RESULT_DIR="../benches/results/$DATASET"
mkdir -p $RESULT_DIR
# ./algorithm
./algorithm --mode dataset --dataset-dir "../benches/datasets/$DATASET"
