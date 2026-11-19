#pragma once

#include "Avx2UInt8Kernel.h"

namespace rnndescent {

using Avx2UInt8DotProduct = Avx2UInt8Kernel;
using Avx2SimdDistanceComputerUInt8L2 = QuantizedDistanceComputer<Avx2UInt8DotProduct>;

} // namespace rnndescent
