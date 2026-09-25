#pragma once

#include "Definitions.hpp"

namespace Toucan::Structs{
    // Steering vector to direct computation
    // [ONLY ON DEVICE]
    class SteeringVector{
    public:
        SteeringVector(const uint32_t inputSize){
            size = 0;
            maxSize = inputSize;
            view = uint32_deviceView(Kokkos::ViewAllocateWithoutInitializing("steerView"), maxSize);
            viewSize = uint32_deviceView(Kokkos::ViewAllocateWithoutInitializing("viewSize"), 1);
        };

        KOKKOS_INLINE_FUNCTION
        void push_back(const uint32_t newVal) const {
            view(Kokkos::atomic_fetch_add(&viewSize(0), 1)) = newVal;
        }

        void reset_size() {
            Kokkos::deep_copy(viewSize, 0);
        }

        uint32_t size;
        uint32_t maxSize;
        uint32_deviceView view;
        uint32_deviceView viewSize;
    };
}