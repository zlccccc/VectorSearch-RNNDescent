#include "solution/rnndescent/CandidateBitmap.h"

#include <gtest/gtest.h>

namespace tests {
namespace {

TEST(CandidateBitmapTest, MarksIdsAcrossWordBoundaries) {
    rnndescent::CandidateBitmap ids;
    ids.initialize(129);
    for (int id : {0, 1, 63, 64, 65, 127, 128})
        ASSERT_TRUE(ids.insert(id)) << id;
    for (int id : {0, 1, 63, 64, 65, 127, 128})
        EXPECT_FALSE(ids.insert(id)) << id;
}

TEST(CandidateBitmapTest, ClearsMarksBetweenQueries) {
    rnndescent::CandidateBitmap ids;
    ids.initialize(70);
    for (int id : {0, 63, 64, 69})
        ASSERT_TRUE(ids.insert(id)) << id;

    ids.clear();
    for (int id : {0, 63, 64, 69}) {
        EXPECT_TRUE(ids.insert(id)) << id;
        EXPECT_FALSE(ids.insert(id)) << id;
    }
}

} // namespace
} // namespace tests
