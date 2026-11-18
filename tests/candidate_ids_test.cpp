#include "solution/rnndescent/ActiveCandidateIds.h"

#include <gtest/gtest.h>

namespace tests {
namespace {

TEST(ActiveCandidateIdsTest, ErasingFromAWrappedClusterPreservesOtherIds) {
    rnndescent::ActiveCandidateIds ids;
    ids.reset(4);
    // The slots filled in order are 7, 0, 1, 2. IDs 8 and 16 hash to slot 7.
    for (int id : {8, 0, 16, 2})
        ASSERT_TRUE(ids.insert(id));

    ids.erase(8);
    for (int id : {0, 16, 2})
        EXPECT_FALSE(ids.insert(id));
    EXPECT_TRUE(ids.insert(8));

    ids.erase(0);
    for (int id : {8, 16, 2})
        EXPECT_FALSE(ids.insert(id));
    EXPECT_TRUE(ids.insert(0));
}

TEST(ActiveCandidateIdsTest, ResetClearsIdsWhenTheQueryBudgetChanges) {
    rnndescent::ActiveCandidateIds ids;
    for (int budget : {4, 4, 16, 4}) {
        ids.reset(budget);
        for (int id : {8, 0, 16, 2}) {
            EXPECT_TRUE(ids.insert(id));
            EXPECT_FALSE(ids.insert(id));
        }
    }
}

} // namespace
} // namespace tests
