#include "Checks.h"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    try {
        tests::require(argc == 2, "expected one test suite name");
        const std::string suite = argv[1];
        if (suite == "simd")
            tests::test_simd();
        else if (suite == "cache")
            tests::test_cache();
        else if (suite == "benchmark")
            tests::test_benchmark();
        else if (suite == "solution")
            tests::test_solution();
        else if (suite == "recall")
            tests::test_recall();
        else
            throw std::runtime_error("unknown test suite: " + suite);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
