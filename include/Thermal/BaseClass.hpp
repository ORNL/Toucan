#pragma once

// Internal includes
#include "Definitions.hpp"

// External includes
#include <Stork_Core.hpp>

namespace Toucan::Thermal {

    // Parent class for thermal data sources.
    template <typename ToucanFloat, typename ToucanSpace>
    class ThermalSource {
    public:
        using DualRDF = Stork::Structs::RDF_Dual<ToucanFloat>;

        // This destructor is virtual so deleting through a ThermalSource pointer still runs the child class destructor correctly.
        virtual ~ThermalSource() = default;

        // Pure virtual function.
        virtual DualRDF GetRDF(const uint32_t layer) = 0;
    };
}
