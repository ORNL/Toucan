#pragma once

#include "Definitions.hpp"

#include "Structs/GrainID.hpp"

#include "Utility/Util.hpp"

#include "Structs/Rng.hpp"
#include "Structs/Sim.hpp"

namespace Toucan::Structs{

    template<typename memory_space, typename GrainID = BaseGrainIdentifier<memory_space>>
    struct Substrate{
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

        bool_view fromGrid_view;
        uint16_view shuffled_id_list;
        uint16_view dirID_view;
        GrainID grainID;
        uint32_view subNums_view;

        KOKKOS_INLINE_FUNCTION
        uint16_t& dirID(const uint32_t p) const {
            return dirID_view(p);
        }

        KOKKOS_INLINE_FUNCTION
        bool& fromGrid(const uint32_t p) const {
            return fromGrid_view(p);
        }

        // Constants
        KOKKOS_INLINE_FUNCTION
        uint32_t& xnum() const {
                return subNums_view(0);
        }

        KOKKOS_INLINE_FUNCTION
        uint32_t& ynum() const {
                return subNums_view(1);
        }

        KOKKOS_INLINE_FUNCTION
        uint32_t& znum() const {
                return subNums_view(2);
        }

        ///////////////////////////
        //// Usefull Functions ////
        ///////////////////////////
        KOKKOS_INLINE_FUNCTION
        uint32_t    ijk_to_p(const uint32_t i, const uint32_t j, const uint32_t k) const {
            return i + j*xnum() + k*xnum()*ynum();
        }

        KOKKOS_INLINE_FUNCTION
        void    p_to_ijk(uint32_t* ijk, const uint32_t p) const {
            ijk[0] = (p % xnum());
            ijk[1] = (p / xnum()) % ynum();
            ijk[2] = (p / (xnum()*ynum()));
        }

    };

    template<typename memory_space>
    struct Substrate_3D{
    private:
        /////////////////////////////////////////
        //// Set quick access to views types ////
        /////////////////////////////////////////
        using view_layout = Kokkos::LayoutLeft;
        using bool_view = Kokkos::View<bool***, view_layout, memory_space>;
        using uint16_view = Kokkos::View<uint16_t***, view_layout, memory_space>;

    public:
        /////////////////////////////////////////
        //// Views and easy access functions ////
        /////////////////////////////////////////

        bool_view fromGrid_view_3D;
        uint16_view dirID_view_3D;

        void make_fromGrid(bool* ptr, const size_t (&extent)[3]){
            fromGrid_view_3D = bool_view(ptr, extent[0], extent[1], extent[2]);
        }
        void make_dirID(uint16_t* ptr, const size_t (&extent)[3]){
            dirID_view_3D = uint16_view(ptr, extent[0], extent[1], extent[2]);
        }
    };

    template<template<typename> class GrainID = BaseGrainIdentifier>
    class Substrate_Dual{
    public:

        Substrate_Dual() {
            // Make size views
            deviceSub.subNums_view = uint32_deviceView(Kokkos::ViewAllocateWithoutInitializing("deviceSub.subNums_view"), 3);
            hostSub.subNums_view = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, host_memory(), deviceSub.subNums_view);
        }

        // Grids on both spaces
        Substrate<host_space, GrainID<host_space>> hostSub;
        Substrate<device_space, GrainID<device_space>> deviceSub;

        // 3D Grids on both space (just pointers...for slicing, striding, etc.)
        Substrate_3D<host_space> hostSub_3D;
        Substrate_3D<device_space> deviceSub_3D;

        // Spatial information
        uint32_t size;              // Total Size of Substrate
        uint32_t windowSize;        // Size of Window in number of z points
        uint32_t grain_mult;
        uint32_t layerNum = 0;
        uint32_t lastLayerOutput = 0;
        uint32_t windowNum = 1;    // Always start halfway up "substrate"

        // For Header
        Stork::Structs::RegularGrid_Header<float, host_space> header;
        float res, zmin;

        // Persistent scratch for non-contiguous output collection.
        size_t outputScratchCapacity = 0;
        bool_deviceView outputScratch_fromGrid_device;
        bool_hostView outputScratch_fromGrid_host;
        uint16_deviceView outputScratch_dirID_device;
        uint16_hostView outputScratch_dirID_host;
        GrainID<device_space> outputScratch_grainID_device;
        GrainID<host_space> outputScratch_grainID_host;

        // Initialize 3D views (for slicing, striding, etc.)
        void MakeViews_3D(){
            // Get extents
            const size_t extent[3] = {hostSub.xnum(),hostSub.ynum(),hostSub.znum()};

            // Make host views
            hostSub_3D.make_fromGrid(hostSub.fromGrid_view.data(),extent);
            hostSub_3D.make_dirID(hostSub.dirID_view.data(),extent);

            // Make device views
            deviceSub_3D.make_fromGrid(deviceSub.fromGrid_view.data(),extent);
            deviceSub_3D.make_dirID(deviceSub.dirID_view.data(),extent);
        }

        void InitializeOutputScratch(const size_t capacity) {
            outputScratchCapacity = capacity;
            outputScratch_fromGrid_device = bool_deviceView(Kokkos::ViewAllocateWithoutInitializing("outputScratch_fromGrid_device"), capacity);
            outputScratch_fromGrid_host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, host_memory(), outputScratch_fromGrid_device);
            outputScratch_dirID_device = uint16_deviceView(Kokkos::ViewAllocateWithoutInitializing("outputScratch_dirID_device"), capacity);
            outputScratch_dirID_host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, host_memory(), outputScratch_dirID_device);
            outputScratch_grainID_device.Init(capacity, "outputScratch_grainID_device");
            outputScratch_grainID_host.createHostMirrorsFrom(outputScratch_grainID_device);
        }

        template<class View1D>
        static auto Make3DView(const View1D& view, const size_t (&extent)[3]) {
            using value_type = typename View1D::non_const_value_type;
            using memory_space_type = typename View1D::memory_space;
            using view3D = Kokkos::View<value_type***, Kokkos::LayoutLeft, memory_space_type, Kokkos::MemoryUnmanaged>;
            return view3D(view.data(), extent[0], extent[1], extent[2]);
        }

        void CopyGrainIDContiguous(
            const std::pair<size_t, size_t>& iRange,
            const std::pair<size_t, size_t>& jRange,
            const std::pair<size_t, size_t>& kRange)
        {
            const size_t extent[3] = {hostSub.xnum(), hostSub.ynum(), hostSub.znum()};
            deviceSub.grainID.forEachFieldPair(hostSub.grainID, [&](auto deviceField, auto hostField) {
                auto deviceField3D = Make3DView(deviceField, extent);
                auto hostField3D = Make3DView(hostField, extent);
                Kokkos::deep_copy(
                    Kokkos::subview(hostField3D, iRange, jRange, kRange),
                    Kokkos::subview(deviceField3D, iRange, jRange, kRange)
                );
            });
        }

        void CopyFromDevice(const OutputRegion& region) {

            if (region.empty()) {
                return;
            }

            const size_t total = region.size();
            if (total > outputScratchCapacity) {
                throw std::runtime_error("Output scratch capacity is smaller than requested output region.");
            }

            const std::pair<size_t, size_t> iRange = std::make_pair(region.localStart[0], region.localStart[0] + region.count[0]);
            const std::pair<size_t, size_t> jRange = std::make_pair(region.localStart[1], region.localStart[1] + region.count[1]);
            const std::pair<size_t, size_t> kRange = std::make_pair(region.localStart[2], region.localStart[2] + region.count[2]);

            // Create subviews based on the range
            auto hostSub_3D_fromGrid_subview = Kokkos::subview(hostSub_3D.fromGrid_view_3D,iRange,jRange,kRange);
            auto hostSub_3D_dirID_subview = Kokkos::subview(hostSub_3D.dirID_view_3D,iRange,jRange,kRange);

            auto deviceSub_3D_fromGrid_subview = Kokkos::subview(deviceSub_3D.fromGrid_view_3D,iRange,jRange,kRange);
            auto deviceSub_3D_dirID_subview = Kokkos::subview(deviceSub_3D.dirID_view_3D,iRange,jRange,kRange);

            // If the views are contiguous
            const bool unitStride = region.stride[0] == 1 && region.stride[1] == 1 && region.stride[2] == 1;
            const bool subviewContiguous = unitStride && hostSub_3D_fromGrid_subview.span_is_contiguous();
            if (subviewContiguous){
                // Just copy directly
                Kokkos::deep_copy(hostSub_3D_fromGrid_subview,deviceSub_3D_fromGrid_subview);
                Kokkos::deep_copy(hostSub_3D_dirID_subview,deviceSub_3D_dirID_subview);
                CopyGrainIDContiguous(iRange, jRange, kRange);
            }
            else{
                const uint32_t start0 = region.localStart[0];
                const uint32_t start1 = region.localStart[1];
                const uint32_t start2 = region.localStart[2];
                const uint32_t count0 = region.count[0];
                const uint32_t count1 = region.count[1];
                const uint32_t stride0 = region.stride[0];
                const uint32_t stride1 = region.stride[1];
                const uint32_t stride2 = region.stride[2];

                auto deviceFromGrid = deviceSub.fromGrid_view;
                auto deviceDirID = deviceSub.dirID_view;
                auto scratchFromGridDevice = outputScratch_fromGrid_device;
                auto scratchDirIDDevice = outputScratch_dirID_device;
                auto deviceGrainID = deviceSub.grainID;
                auto scratchGrainIDDevice = outputScratch_grainID_device;
                const uint32_t xnumDevice = hostSub.xnum();
                const uint32_t ynumDevice = hostSub.ynum();

                Kokkos::parallel_for(
                    "Substrate Output Gather",
                    Kokkos::RangePolicy<device_space>(0, total),
                    KOKKOS_LAMBDA(const size_t pOut)
                    {
                        const uint32_t di = pOut % count0;
                        const uint32_t dj = (pOut / count0) % count1;
                        const uint32_t dk = pOut / (count0 * count1);
                        const uint32_t i = start0 + di * stride0;
                        const uint32_t j = start1 + dj * stride1;
                        const uint32_t k = start2 + dk * stride2;
                        const uint32_t p = i + j * xnumDevice + k * xnumDevice * ynumDevice;
                        scratchFromGridDevice(pOut) = deviceFromGrid(p);
                        scratchDirIDDevice(pOut) = deviceDirID(p);
                        scratchGrainIDDevice.copyFrom(deviceGrainID, pOut, p);
                    }
                );

                const std::pair<size_t, size_t> scratchRange(0, total);
                auto scratchFromGridDeviceSubview = Kokkos::subview(outputScratch_fromGrid_device, scratchRange);
                auto scratchFromGridHostSubview = Kokkos::subview(outputScratch_fromGrid_host, scratchRange);
                auto scratchDirIDDeviceSubview = Kokkos::subview(outputScratch_dirID_device, scratchRange);
                auto scratchDirIDHostSubview = Kokkos::subview(outputScratch_dirID_host, scratchRange);
                Kokkos::deep_copy(scratchFromGridHostSubview, scratchFromGridDeviceSubview);
                Kokkos::deep_copy(scratchDirIDHostSubview, scratchDirIDDeviceSubview);
                outputScratch_grainID_device.forEachFieldPair(outputScratch_grainID_host, [&](auto deviceField, auto hostField) {
                    Kokkos::deep_copy(
                        Kokkos::subview(hostField, scratchRange),
                        Kokkos::subview(deviceField, scratchRange)
                    );
                });

                auto hostFromGrid = hostSub.fromGrid_view;
                auto hostDirID = hostSub.dirID_view;
                auto scratchFromGridHost = outputScratch_fromGrid_host;
                auto scratchDirIDHost = outputScratch_dirID_host;
                auto hostGrainID = hostSub.grainID;
                auto scratchGrainIDHost = outputScratch_grainID_host;
                const uint32_t xnumHost = hostSub.xnum();
                const uint32_t ynumHost = hostSub.ynum();

                Kokkos::parallel_for(
                    "Substrate Output Scatter",
                    Kokkos::RangePolicy<host_space>(0, total),
                    KOKKOS_LAMBDA(const size_t pOut)
                    {
                        const uint32_t di = pOut % count0;
                        const uint32_t dj = (pOut / count0) % count1;
                        const uint32_t dk = pOut / (count0 * count1);
                        const uint32_t i = start0 + di * stride0;
                        const uint32_t j = start1 + dj * stride1;
                        const uint32_t k = start2 + dk * stride2;
                        const uint32_t p = i + j * xnumHost + k * xnumHost * ynumHost;
                        hostFromGrid(p) = scratchFromGridHost(pOut);
                        hostDirID(p) = scratchDirIDHost(pOut);
                        hostGrainID.copyFrom(scratchGrainIDHost, p, pOut);
                    }
                );
            }
        }

        void CopyFromDevice(std::pair<size_t, size_t> iRange, std::pair<size_t, size_t> jRange, std::pair<size_t, size_t> kRange) {
            OutputRegion region;
            region.localStart[0] = iRange.first;
            region.localStart[1] = jRange.first;
            region.localStart[2] = kRange.first;
            region.count[0] = iRange.second - iRange.first;
            region.count[1] = jRange.second - jRange.first;
            region.count[2] = kRange.second - kRange.first;
            CopyFromDevice(region);
        }
    };
}
