#pragma once

#include "Definitions.hpp"
#include "Common.hpp"

namespace Toucan::Utility {

    KOKKOS_INLINE_FUNCTION
    uint32_t    ijk_to_p(const uint32_t i, const uint32_t j, const uint32_t k, const uint32_t xnum, const uint32_t ynum, const uint32_t znum) {
        return i * ynum * znum + j * znum + k;
    }
    KOKKOS_INLINE_FUNCTION
    uint32_t    ijkl_to_p(const uint32_t i, const uint32_t j, const uint32_t k, const uint8_t l, const uint32_t xnum, const uint32_t ynum, const uint32_t znum, const uint32_t tnum) {
        return i * ynum * znum * tnum + j * znum * tnum + k * tnum + l;
    }
    KOKKOS_INLINE_FUNCTION
    uint8_t dxyx_to_n(const int8_t dx, const int8_t dy, const int8_t dz) {
        return 3 * 3 * (dx + 1) + 3 * (dy + 1) + (dz + 1);
    }
    KOKKOS_INLINE_FUNCTION
    uint8_t invert_n(const uint8_t n) {
        return 26 - n;
    }

    // TODO::REMOVE??
    // Needed for better initialization
}