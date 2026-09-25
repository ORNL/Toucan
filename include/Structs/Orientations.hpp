#pragma once

#include "Definitions.hpp"
#include "Structs/GrainID.hpp"

namespace Toucan::Structs {

    template<typename memory_space>
    class Orientations {
    private:
        /////////////////////////////////////////
        //// Set quick access to views types ////
        /////////////////////////////////////////

        using float_view = Kokkos::View<float*, layout, memory_space>;

    public:

        float_view dirInfo_view;
        float_view faceInfo_view;

        KOKKOS_INLINE_FUNCTION
        float& d1x(const uint32_t p) const {
            return dirInfo_view(9 * p);
        }

        KOKKOS_INLINE_FUNCTION
        float& d1y(const uint32_t p) const {
            return dirInfo_view(9 * p + 1);
        }

        KOKKOS_INLINE_FUNCTION
        float& d1z(const uint32_t p) const {
            return dirInfo_view(9 * p + 2);
        }

        KOKKOS_INLINE_FUNCTION
        float& d2x(const uint32_t p) const {
            return dirInfo_view(9 * p + 3);
        }

        KOKKOS_INLINE_FUNCTION
        float& d2y(const uint32_t p) const {
            return dirInfo_view(9 * p + 4);
        }

        KOKKOS_INLINE_FUNCTION
        float& d2z(const uint32_t p) const {
            return dirInfo_view(9 * p + 5);
        }

        KOKKOS_INLINE_FUNCTION
        float& d3x(const uint32_t p) const {
            return dirInfo_view(9 * p + 6);
        }

        KOKKOS_INLINE_FUNCTION
        float& d3y(const uint32_t p) const {
            return dirInfo_view(9 * p + 7);
        }

        KOKKOS_INLINE_FUNCTION
        float& d3z(const uint32_t p) const {
            return dirInfo_view(9 * p + 8);
        }

        KOKKOS_INLINE_FUNCTION
        float& Fx0(const uint32_t p) const {
            return faceInfo_view(12 * p);
        }

        KOKKOS_INLINE_FUNCTION
        float& Fx1(const uint32_t p) const {
            return faceInfo_view(12 * p + 1);
        }

        KOKKOS_INLINE_FUNCTION
        float& Fx2(const uint32_t p) const {
            return faceInfo_view(12 * p + 2);
        }

        KOKKOS_INLINE_FUNCTION
        float& Fx3(const uint32_t p) const {
            return faceInfo_view(12 * p + 3);
        }

        KOKKOS_INLINE_FUNCTION
        float& Fy0(const uint32_t p) const {
            return faceInfo_view(12 * p + 4);
        }

        KOKKOS_INLINE_FUNCTION
        float& Fy1(const uint32_t p) const {
            return faceInfo_view(12 * p + 5);
        }

        KOKKOS_INLINE_FUNCTION
        float& Fy2(const uint32_t p) const {
            return faceInfo_view(12 * p + 6);
        }

        KOKKOS_INLINE_FUNCTION
        float& Fy3(const uint32_t p) const {
            return faceInfo_view(12 * p + 7);
        }

        KOKKOS_INLINE_FUNCTION
        float& Fz0(const uint32_t p) const {
            return faceInfo_view(12 * p + 8);
        }

        KOKKOS_INLINE_FUNCTION
        float& Fz1(const uint32_t p) const {
            return faceInfo_view(12 * p + 9);
        }

        KOKKOS_INLINE_FUNCTION
        float& Fz2(const uint32_t p) const {
            return faceInfo_view(12 * p + 10);
        }

        KOKKOS_INLINE_FUNCTION
        float& Fz3(const uint32_t p) const {
            return faceInfo_view(12 * p + 11);
        }

    };

    class Orientations_Dual {
    public:
        Orientations<host_space> hostOrient;
        Orientations<device_space> deviceOrient;
    };
}
