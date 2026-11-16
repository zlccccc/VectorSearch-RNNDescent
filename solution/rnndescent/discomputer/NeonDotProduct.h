#pragma once

#include <array>
#include <cstddef>

namespace rnndescent {

template <class Kernel> struct NeonDotProduct : Kernel {
    using Code = typename Kernel::Code;
    using Dot = typename Kernel::Dot;
    static constexpr bool prefetch_norms = false;

    static Dot dot(const Code *a, const Code *b, size_t d) {
        auto sum = Kernel::zero();
        const size_t full = d - d % Kernel::block_elements;
        for (size_t k = 0; k < full; k += Kernel::block_elements)
            sum = Kernel::multiply_add(sum, Kernel::load(a + k), Kernel::load(b + k));
        Dot result = Kernel::reduce(sum);
        for (size_t k = full; k < d; ++k)
            result += static_cast<Dot>(a[k]) * static_cast<Dot>(b[k]);
        return result;
    }

    static std::array<Dot, 4> dot4(const Code *q, const Code *x0, const Code *x1, const Code *x2, const Code *x3, size_t d) {
        auto sum0 = Kernel::zero(), sum1 = Kernel::zero(), sum2 = Kernel::zero(), sum3 = Kernel::zero();
        const size_t full = d - d % Kernel::block_elements;
        for (size_t k = 0; k < full; k += Kernel::block_elements) {
            const auto q_block = Kernel::load(q + k);
            sum0 = Kernel::multiply_add(sum0, q_block, Kernel::load(x0 + k));
            sum1 = Kernel::multiply_add(sum1, q_block, Kernel::load(x1 + k));
            sum2 = Kernel::multiply_add(sum2, q_block, Kernel::load(x2 + k));
            sum3 = Kernel::multiply_add(sum3, q_block, Kernel::load(x3 + k));
        }
        std::array<Dot, 4> dots{Kernel::reduce(sum0), Kernel::reduce(sum1), Kernel::reduce(sum2), Kernel::reduce(sum3)};
        for (size_t k = full; k < d; ++k) {
            const Dot v = static_cast<Dot>(q[k]);
            dots[0] += v * static_cast<Dot>(x0[k]);
            dots[1] += v * static_cast<Dot>(x1[k]);
            dots[2] += v * static_cast<Dot>(x2[k]);
            dots[3] += v * static_cast<Dot>(x3[k]);
        }
        return dots;
    }
};

} // namespace rnndescent
