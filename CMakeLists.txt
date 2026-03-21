cmake_minimum_required(VERSION 3.23)
project(algorithm)
add_definitions(-std=c++17)
# lscpu查看指令集

# add_compile_options(-Wall -Wextra -O3)
# add_compile_options(-O3)
add_compile_options(-O3 -mcmodel=medium -mcpu=native)
# add_compile_options(-O3 -mcpu=native)
# add_compile_options(-O3 -mfpu=neon)
# add_compile_options(-O3 -mfpu=vfpv3-fp16)
# add_compile_options(-march=armv8.2-a+fp16fml)
# add_compile_options(-march=armv8.4-a+fp16fml)
# add_compile_options(-msse2 -msse -mavx512f -mavx512vl -mbmi2 -mavx512bw -mavx512vnni)
add_compile_options(-msse2 -msse -mavx2 -mbmi2 -msse4.1 -msse4.2)

add_compile_options(-g)
add_compile_options(-DINTERNAL_CLOCK_TEST)

FIND_PACKAGE(OpenMP REQUIRED)

if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# FIND_PACKAGE(OpenBLAS REQUIRED)
# if(OpenBLAS_FOUND)
# message("OpenBLAS FOUND ${OpenBLAS_INCLUDE_DIRS} ${OpenBLAS_LIBRARIES}")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenBLAS_C_FLAGS}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenBLAS_CXX_FLAGS}")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenBLAS_EXE_LINKER_FLAGS}")
# endif()

# # 添加 Faiss 库的路径

include_directories(
/usr/local/include/faiss,
/usr/local/include,
/usr/include/openblas,
/usr/include/mkl,
# /usr/local/include/eigen3
)

# include_directories(${OpenBLAS_INCLUDE_DIRS})

file(GLOB cpp_srcs ${CMAKE_SOURCE_DIR}/main.cpp ${CMAKE_SOURCE_DIR}/solution/*.cpp)

link_directories(
/usr/local/lib64
/usr/lib64
)

add_executable(${PROJECT_NAME} ${cpp_srcs})
target_include_directories(${PROJECT_NAME} PRIVATE /usr/include/openblas)

#target_link_libraries(${PROJECT_NAME} faiss blas pthread m jemalloc)
target_link_libraries(${PROJECT_NAME} faiss ${OpenBLAS_LIBRARIES} pthread openblas m)

#add_executable(bench_hnsw benches/bench_hnsw.cpp)
#target_link_libraries(bench_hnsw faiss openblas)

#add_executable(bench_nsg benches/bench_nsg.cpp)
#target_link_libraries(bench_nsg faiss openblas)

#add_executable(bench_nndescent benches/bench_nndescent.cpp)
#target_link_libraries(bench_nndescent faiss openblas)

#add_executable(bench_rnndescent benches/bench_rnndescent.cpp)
#target_link_libraries(bench_rnndescent faiss openblas)
