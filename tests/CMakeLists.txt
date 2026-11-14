add_executable(rnndescent_tests
    main.cpp
    simd_test.cpp
    benchmark_test.cpp
    solution_test.cpp
)
target_link_libraries(rnndescent_tests PRIVATE rnndescent_solution)

foreach(suite IN ITEMS simd cache benchmark solution recall)
    add_test(NAME rnndescent.${suite} COMMAND rnndescent_tests ${suite})
    set_tests_properties(rnndescent.${suite} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1;OMP_DYNAMIC=FALSE"
        TIMEOUT 30)
endforeach()
set_tests_properties(rnndescent.simd rnndescent.cache rnndescent.benchmark PROPERTIES LABELS unit)
set_tests_properties(rnndescent.solution rnndescent.recall PROPERTIES LABELS integration)
