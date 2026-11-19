#pragma once

#include "AvxVnniUInt8Kernel.h"

namespace rnndescent {

using AvxVnniUInt8DotProduct = AvxVnniUInt8Kernel;
using AvxVnniSimdDistanceComputerUInt8L2 = QuantizedDistanceComputer<AvxVnniUInt8DotProduct>;

} // namespace rnndescent
