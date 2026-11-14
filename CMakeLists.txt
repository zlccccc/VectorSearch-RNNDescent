cmake_minimum_required(VERSION 3.23)
project(algorithm LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(RNNDESCENT_SIMD "native" CACHE STRING "SIMD target: native, avx2, avx512, or neon")
set_property(CACHE RNNDESCENT_SIMD PROPERTY STRINGS native avx2 avx512 neon)

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(faiss CONFIG QUIET)
if(NOT TARGET faiss)
    find_path(FAISS_INCLUDE_DIR NAMES faiss/impl/NNDescent.h REQUIRED)
    find_library(FAISS_LIBRARY NAMES faiss REQUIRED)
    add_library(faiss UNKNOWN IMPORTED)
    set_target_properties(faiss PROPERTIES
        IMPORTED_LOCATION "${FAISS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${FAISS_INCLUDE_DIR}")
endif()

find_package(OpenBLAS CONFIG QUIET)
if(NOT OpenBLAS_FOUND)
    find_path(OpenBLAS_INCLUDE_DIRS NAMES cblas.h PATH_SUFFIXES openblas openblas-pthread REQUIRED)
    find_library(OpenBLAS_LIBRARIES NAMES openblas REQUIRED)
endif()

add_library(rnndescent_build_options INTERFACE)
target_compile_features(rnndescent_build_options INTERFACE cxx_std_17)
target_compile_definitions(rnndescent_build_options INTERFACE INTERNAL_CLOCK_TEST)
target_include_directories(rnndescent_build_options INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}" ${OpenBLAS_INCLUDE_DIRS})
target_link_libraries(rnndescent_build_options INTERFACE
    faiss ${OpenBLAS_LIBRARIES} OpenMP::OpenMP_CXX)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" processor)
if(processor MATCHES "^(x86_64|amd64)$")
    if(RNNDESCENT_SIMD STREQUAL "native")
        target_compile_options(rnndescent_build_options INTERFACE -march=native -mavx2 -mfma)
    elseif(RNNDESCENT_SIMD STREQUAL "avx2")
        target_compile_options(rnndescent_build_options INTERFACE -mavx2 -mfma)
        target_compile_definitions(rnndescent_build_options INTERFACE RNNDESCENT_FORCE_AVX2)
    elseif(RNNDESCENT_SIMD STREQUAL "avx512")
        target_compile_options(rnndescent_build_options INTERFACE -mavx512f -mavx512bw -mavx512vl -mavx512vnni)
        target_compile_definitions(rnndescent_build_options INTERFACE RNNDESCENT_FORCE_AVX512)
    else()
        message(FATAL_ERROR "Set RNNDESCENT_SIMD to native, avx2, or avx512 on x86-64")
    endif()
elseif(processor MATCHES "^(aarch64|arm64)$")
    if(NOT RNNDESCENT_SIMD MATCHES "^(native|neon)$")
        message(FATAL_ERROR "Set RNNDESCENT_SIMD to native or neon on AArch64")
    endif()
    target_compile_options(rnndescent_build_options INTERFACE -march=armv8.2-a+dotprod)
    target_compile_definitions(rnndescent_build_options INTERFACE RNNDESCENT_FORCE_NEON)
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

add_library(rnndescent_solution STATIC solution/solution.cpp)
target_link_libraries(rnndescent_solution PUBLIC rnndescent_build_options)

add_executable(vector_search_app main.cpp)
set_target_properties(vector_search_app PROPERTIES OUTPUT_NAME algorithm)
target_link_libraries(vector_search_app PRIVATE rnndescent_solution)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
