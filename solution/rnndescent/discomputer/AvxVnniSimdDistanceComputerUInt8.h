#pragma once

#include "Avx2SimdDistanceComputerUInt8.h"

namespace rnndescent {

inline __m256i avx_vnni_accumulate_byte_products(__m256i sum, __m256i q, __m256i x) {
    return _mm256_dpbusd_avx_epi32(sum, q, x);
}

using AvxVnniUInt8DotProduct = AvxUInt8DotProduct<avx_vnni_accumulate_byte_products>;
using AvxVnniSimdDistanceComputerUInt8L2 = QuantizedDistanceComputer<AvxVnniUInt8DotProduct>;

} // namespace rnndescent
