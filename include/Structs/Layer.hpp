#pragma once

#include "Definitions.hpp"
#include "Common.hpp"

namespace Toucan::Structs{

    struct SolidifyEvent {
        float x, y, z, tm, tl, cr;
    };

    struct Layer {
        uint32_t numEvents;
        uint32_t extent[3];
        float bounds[6];
        Common::vector<float> eventInfo;

        Layer() = default;
    };
}