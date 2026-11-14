# Apply the list after discovery so older CMake releases keep every variable.
foreach(test IN LISTS rnndescent_unit_tests_TESTS rnndescent_integration_tests_TESTS)
    set_tests_properties("${test}" PROPERTIES ENVIRONMENT
        "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1;OMP_DYNAMIC=FALSE")
endforeach()
