#pragma once

#include "NeonInt8Kernel.h"

namespace rnndescent {

using SimdDistanceComputerInt8L2 = QuantizedDistanceComputer<NeonInt8DotProduct>;
using Int8Neighbors = QuantizedNeighbors<NeonInt8DotProduct>;

} // namespace rnndescent
