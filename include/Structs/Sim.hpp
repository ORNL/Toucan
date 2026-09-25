#pragma once

#include "Definitions.hpp"
#include "Common.hpp"

namespace Toucan::Structs{

    struct OutputRegion {
        uint32_t localStart[3] = {0, 0, 0};
        uint32_t count[3] = {0, 0, 0};
        uint32_t stride[3] = {1, 1, 1};
        uint32_t globalStart[3] = {0, 0, 0};
        uint32_t globalStride[3] = {1, 1, 1};
        Common::string sliceSuffix;

        bool empty() const {
            return count[0] == 0 || count[1] == 0 || count[2] == 0;
        }

        size_t size() const {
            return static_cast<size_t>(count[0]) * count[1] * count[2];
        }
    };

    template<typename memory_space>
    struct Sim{
    private:
        /////////////////////////////////////////
        //// Set quick access to views types ////
        /////////////////////////////////////////

        using bool_view = Kokkos::View<bool*, layout, memory_space>;
        using uint8_view = Kokkos::View<uint8_t*, layout, memory_space>;
        using uint16_view = Kokkos::View<uint16_t*, layout, memory_space>;
        using uint32_view = Kokkos::View<uint32_t*, layout, memory_space>;
        using float_view = Kokkos::View<float*, layout, memory_space>;

    public:

        /////////////////////////////////////////
        //// Views and easy access functions ////
        /////////////////////////////////////////

        float_view simFloats_view;
        uint32_view simInts_view;

        // Start Floats HostView
        KOKKOS_INLINE_FUNCTION
        float& xmin() const {
            return simFloats_view(0);
        }

        KOKKOS_INLINE_FUNCTION
        float& xmax() const {
            return simFloats_view(1);
        }

        KOKKOS_INLINE_FUNCTION
        float& ymin() const {
            return simFloats_view(2);
        }

        KOKKOS_INLINE_FUNCTION
        float& ymax() const {
            return simFloats_view(3);
        }

        KOKKOS_INLINE_FUNCTION
        float& zmin() const {
            return simFloats_view(4);
        }

        KOKKOS_INLINE_FUNCTION
        float& zmax() const {
            return simFloats_view(5);
        }

        KOKKOS_INLINE_FUNCTION
        float& res() const {
            return simFloats_view(6);
        }

        KOKKOS_INLINE_FUNCTION
        float& subGrainSize() const {
            return simFloats_view(7);
        }

        KOKKOS_INLINE_FUNCTION
        float& nuclProb() const {
            return simFloats_view(8);
        }

        KOKKOS_INLINE_FUNCTION
        float& meanUnder() const {
            return simFloats_view(9);
        }

        KOKKOS_INLINE_FUNCTION
        float& stdUnder() const {
            return simFloats_view(10);
        }

        KOKKOS_INLINE_FUNCTION
        float& a() const {
            return simFloats_view(11);
        }

        KOKKOS_INLINE_FUNCTION
        float& b() const {
            return simFloats_view(12);
        }

        KOKKOS_INLINE_FUNCTION
        float& b1() const {
            return simFloats_view(13);
        }

        KOKKOS_INLINE_FUNCTION
        float& bn1() const {
            return simFloats_view(14);
        }

        KOKKOS_INLINE_FUNCTION
        float& layerHeight() const {
            return simFloats_view(15);
        }

        KOKKOS_INLINE_FUNCTION
        float& windowHeight() const {
            return simFloats_view(16);
        }

        // Start Ints HostView
        KOKKOS_INLINE_FUNCTION
        uint32_t& xnum() const {
            return simInts_view(0);
        }

        KOKKOS_INLINE_FUNCTION
        uint32_t& ynum() const {
            return simInts_view(1);
        }

        KOKKOS_INLINE_FUNCTION
        uint32_t& znum() const {
            return simInts_view(2);
        }

        KOKKOS_INLINE_FUNCTION
        uint32_t& tnum() const {
            return simInts_view(3);
        }

        KOKKOS_INLINE_FUNCTION
        uint32_t& numOrientations() const {
            return simInts_view(4);
        }

        KOKKOS_INLINE_FUNCTION
        uint32_t& numLayers() const {
            return simInts_view(5);
        }

        KOKKOS_INLINE_FUNCTION
        uint32_t& numUniqueLayers() const {
            return simInts_view(6);
        }

    };

    class Sim_Dual{
    public:

        Sim_Dual(){
            const int floatsSize = 17;
            const int intsSize = 7;
            deviceSim.simFloats_view = float_deviceView(Kokkos::ViewAllocateWithoutInitializing("simFloats_view"), floatsSize);
            deviceSim.simInts_view = uint32_deviceView(Kokkos::ViewAllocateWithoutInitializing("simInts_view"), intsSize);
            hostSim.simFloats_view = Kokkos::create_mirror_view(host_memory(), deviceSim.simFloats_view);
            hostSim.simInts_view = Kokkos::create_mirror_view(host_memory(), deviceSim.simInts_view);
        };

        Sim<host_space> hostSim;
        Sim<device_space> deviceSim;
        uint32_deviceView nucleatedRepeatCounts;
        uint32_deviceView nucleatedRepeatCounts_prev;
        int rank;

        Common::vector<Common::string> thermalFiles;
        Common::string simName;
        Common::string mpiMode;
        Common::string outputMode;
        Common::string outputFormat = "xdmf";
        uint32_t outputEvery[3] = {1, 1, 1};
        uint32_t outputOffset[3] = {0, 0, 0};
        bool outputEverySet[3] = {false, false, false};
        bool outputOffsetSet[3] = {false, false, false};
        bool useGrainIdentifier = false;
        bool outputUnderresolvedGrains = false;
        uint32_t prevGridToSubRepeatIDMax = 0;
        bool hasPrevGridToSubRepeatIDMax = false;
        int seed = 0;
        int orientationSeed = 0;
        int substrateSeed = 0;
        int overwriteFiles = 0;
        uint32_t maxIterations = UINT32_MAX;

        // Copy data to device
        void CopyToDevice() {
            Kokkos::deep_copy(deviceSim.simFloats_view, hostSim.simFloats_view);
            Kokkos::deep_copy(deviceSim.simInts_view, hostSim.simInts_view);
        }

        void InitializeNucleatedRepeatCounts() {
            if (!useGrainIdentifier) {
                return;
            }
            const size_t numOrientations = hostSim.numOrientations();
            nucleatedRepeatCounts = uint32_deviceView("nucleatedRepeatCounts", numOrientations);
            nucleatedRepeatCounts_prev = uint32_deviceView("nucleatedRepeatCounts_prev", numOrientations);
            Kokkos::deep_copy(nucleatedRepeatCounts, 0u);
            Kokkos::deep_copy(nucleatedRepeatCounts_prev, 0u);
        }

    };
}
