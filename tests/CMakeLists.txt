find_package(GTest 1.14 QUIET CONFIG)
if(NOT GTest_FOUND)
    if(POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)
    endif()
    include(FetchContent)
    set(INSTALL_GTEST OFF)
    FetchContent_Declare(googletest
        URL https://codeload.github.com/google/googletest/tar.gz/refs/tags/v1.17.0
        URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    )
    FetchContent_MakeAvailable(googletest)
elseif(NOT TARGET GTest::gmock)
    message(FATAL_ERROR "Install GoogleMock alongside GoogleTest, or set CMAKE_DISABLE_FIND_PACKAGE_GTest=ON to use the bundled download.")
endif()

include(GoogleTest)
foreach(kind IN ITEMS unit integration)
    if(kind STREQUAL "unit")
        set(sources simd_test.cpp benchmark_test.cpp pca_test.cpp refinement_test.cpp candidate_ids_test.cpp graph_pruning_test.cpp)
    else()
        set(sources solution_test.cpp)
    endif()
    set(target rnndescent_${kind}_tests)
    add_executable(${target} ${sources})
    target_link_libraries(${target} PRIVATE rnndescent_solution GTest::gtest_main GTest::gmock)
    gtest_discover_tests(${target}
        DISCOVERY_MODE PRE_TEST
        PROPERTIES
            LABELS "${kind}"
            TIMEOUT 30
    )
endforeach()

set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/TestEnvironment.cmake")
