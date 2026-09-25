#pragma once

#include "Definitions.hpp"
#include "Common.hpp"

#include "Utility/Util.hpp"

#include "Structs/Rng.hpp"
#include "Structs/Sim.hpp"
#include "Structs/Layer.hpp"
#include "Structs/Orientations.hpp"
#include "Structs/Substrate.hpp"
#include "Structs/SteeringVector.hpp"

namespace Toucan::Structs {

    template<typename memory_space, typename GrainID = BaseGrainIdentifier<memory_space>>
    class Grid {
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

        // Spatial Info
        uint32_view extent_view;

        KOKKOS_INLINE_FUNCTION
        uint32_t& xnum() const {
            return extent_view(0);
        }

        KOKKOS_INLINE_FUNCTION
        uint32_t& ynum() const {
            return extent_view(1);
        }

        KOKKOS_INLINE_FUNCTION
        uint32_t& znum() const {
            return extent_view(2);
        }

        KOKKOS_INLINE_FUNCTION
        uint32_t& tnum() const {
            return extent_view(3);
        }

        KOKKOS_INLINE_FUNCTION
        uint32_t& xShift() const {
            return extent_view(4);
        }

        KOKKOS_INLINE_FUNCTION
        uint32_t& yShift() const {
            return extent_view(5);
        }

        KOKKOS_INLINE_FUNCTION
        uint32_t& zShift() const {
            return extent_view(6);
        }

        // Grid Location Info
        uint32_view ijk_view;
        uint8_view l_view;

        KOKKOS_INLINE_FUNCTION
        uint32_t& i(const uint32_t p) const {
            return ijk_view(3 * p);
        }

        KOKKOS_INLINE_FUNCTION
        uint32_t& j(const uint32_t p) const {
            return ijk_view(3 * p + 1);
        }

        KOKKOS_INLINE_FUNCTION
        uint32_t& k(const uint32_t p) const {
            return ijk_view(3 * p + 2);
        }

        KOKKOS_INLINE_FUNCTION
        uint8_t& l(const uint32_t p) const {
            return l_view(p);
        }

        // Neighbor info and capture time information
        uint32_view N_view;
        float_view tN_view;
        uint8_view NCap_view;
        float_view tCap_view;

        KOKKOS_INLINE_FUNCTION
        uint32_t& N(const uint32_t p, const uint8_t n) const {
            return N_view(27 * p + n);
        }

        KOKKOS_INLINE_FUNCTION
        float& tN(const uint32_t p, const uint8_t n) const {
            return tN_view(27 * p + n);
        }

        KOKKOS_INLINE_FUNCTION
        uint8_t& NCap(const uint32_t p) const {
            return NCap_view(p);
        }

        KOKKOS_INLINE_FUNCTION
        float& tCap(const uint32_t p) const {
            return tCap_view(p);
        }

        // Solidification Info - Melt time, Solidify time, Cooling rate
        float_view solInfo_view;

        KOKKOS_INLINE_FUNCTION
        float& tm(const uint32_t p) const {
            return solInfo_view(3 * p);
        }

        KOKKOS_INLINE_FUNCTION
        float& tl(const uint32_t p) const {
            return solInfo_view(3 * p + 1);
        }

        KOKKOS_INLINE_FUNCTION
        float& cr(const uint32_t p) const {
            return solInfo_view(3 * p + 2);
        }

        // Nucleation
        float_view tNucl_view;

        KOKKOS_INLINE_FUNCTION
        float& tNucl(const uint32_t p) const {
            return tNucl_view(p);
        }

        // Grain Position Info
        float_view Gstats_view;

        KOKKOS_INLINE_FUNCTION
        float& gx(const uint32_t p) const {
            return Gstats_view(4 * p);
        }

        KOKKOS_INLINE_FUNCTION
        float& gy(const uint32_t p) const {
            return Gstats_view(4 * p + 1);
        }

        KOKKOS_INLINE_FUNCTION
        float& gz(const uint32_t p) const {
            return Gstats_view(4 * p + 2);
        }

        KOKKOS_INLINE_FUNCTION
        float& G0(const uint32_t p) const {
            return Gstats_view(4 * p + 3);
        }

        // Simulation Utility
        bool_view utility_view;
        uint8_view inSteer_view;

        KOKKOS_INLINE_FUNCTION
        bool& edge(const uint32_t p) const {
            return utility_view(4 * p);
        }

        KOKKOS_INLINE_FUNCTION
        bool& MpiWall(const uint32_t p) const {
            return utility_view(4 * p + 1);
        }

        KOKKOS_INLINE_FUNCTION
        bool& cantCap(const uint32_t p) const {
            return utility_view(4 * p + 2);
        }

        KOKKOS_INLINE_FUNCTION
        bool& last(const uint32_t p) const {
            return utility_view(4 * p + 3);
        }

        KOKKOS_INLINE_FUNCTION
        uint8_t& inSteer(const uint32_t p) const {
            return inSteer_view(p);
        }

        // Grain Identity
        uint16_view dirID_view;
        uint16_view dirID_org_view;
        GrainID grainID;
        GrainID grainID_org;

        KOKKOS_INLINE_FUNCTION
        uint16_t& dirID(const uint32_t p) const {
            return dirID_view(p);
        }

        KOKKOS_INLINE_FUNCTION
        uint16_t& dirID_org(const uint32_t p) const {
            return dirID_org_view(p);
        }

        ///////////////////
        //// Functions ////
        ///////////////////

        // Reset All the Arrays Before an DECA Simulation
        void ResetArrays() {

            // Reset relevant neighbor info and capture time information
            Kokkos::deep_copy(tN_view, FLT_MAX);
            Kokkos::deep_copy(NCap_view, UINT8_MAX);
            Kokkos::deep_copy(tCap_view, FLT_MAX);

            // Reset the Nucleation info
            Kokkos::deep_copy(tNucl_view, FLT_MAX);

            // Reset the grain statistics
            Kokkos::deep_copy(Gstats_view, 0.0f);

            // Reset the assigned grain ID
            Kokkos::deep_copy(dirID_view, UINT16_MAX);
            Kokkos::deep_copy(dirID_org_view, UINT16_MAX);

            // Reset if it's in the steering vector
            Kokkos::deep_copy(inSteer_view, UINT8_MAX);

            // Reset grain identifiers
            grainID.Reset();
            grainID_org.Reset();
        }

        // Reset Steering Vector before DECA Iteration
        void ResetSteeringVector(){
            Kokkos::deep_copy(inSteer_view, UINT8_MAX);
        }

        // Init identifiers linking unique grain ID to a crystal orientation
        KOKKOS_INLINE_FUNCTION
        void initializeGrainOrientation(const uint32_t p, const uint16_t dir_id, const uint32_t repeat_id, const int rank) const {
            // Give it an orientation ID
            dirID_org(p) = dir_id;
            dirID(p) = dir_id;
            grainID_org.setUniqueID(p, repeat_id, rank);
            grainID.setUniqueID(p, repeat_id, rank);
        }

        // Reset Grain Directions Based On Rotation Info
        KOKKOS_INLINE_FUNCTION
        void resetGrainID(const uint32_t p) const {
            // Set all grain identifiers to that of the original
            dirID(p) = dirID_org(p);
            grainID.copyFrom(grainID_org, p);
        }

        // Copy original grain directions and ID from another cell
        KOKKOS_INLINE_FUNCTION
        void hardCopyGrainID(const uint32_t p, const uint32_t pN) const {
            // Set original state
            dirID_org(p) = dirID(pN);
            grainID_org.copyFrom(grainID, p, pN);
            // Propagate to spatiotemporal neighbor (future-self)
            dirID(p) = dirID(pN);
            grainID.copyFrom(grainID, p, pN);
        }

        // Copy grain directions and ID from another cell
        KOKKOS_INLINE_FUNCTION
        void softCopyGrainID(const uint32_t p, const uint32_t pN) const {
            dirID(p) = dirID(pN);
            grainID.copyFrom(grainID, p, pN);
        }

        KOKKOS_INLINE_FUNCTION
        void AddToSteer(const uint32_t p, const SteeringVector& steer, uint32_t& numSteer) const {
            steer.push_back(p);
            numSteer++;
        }

        KOKKOS_INLINE_FUNCTION
        void PrepForDualSteer(const uint32_t p, const uint8_t n) const {
            Kokkos::atomic_compare_exchange(&inSteer(p), UINT8_MAX, n);
        }

        KOKKOS_INLINE_FUNCTION
        void SendEvent(const uint32_t p, const float tEvent, const uint8_t n_in) const {
            // Invert index to get into neighbors frame of reference
            const uint8_t n = (27 - 1) - n_in;

            // Set capture time of neighbor
            tN(p, n) = tEvent;

            // if something new is from neighbor which previously captured the cell
            // OR
            // if event is less than the current capture time
            if (n == NCap(p) || tEvent < tCap(p) || NCap(p) == UINT8_MAX) {
                // Make sure only one neighbor can try to add it to the steering vector
                PrepForDualSteer(p, n_in);
            }
        }

        // Find who captured it and when
        KOKKOS_INLINE_FUNCTION
        void updateMinTime(const uint32_t p) const {
            float minTime = FLT_MAX;
            uint8_t minPos = UINT8_MAX;
            for (uint8_t i=0;i<27;i++){
                if (tN(p,i) < minTime){
                    minTime = tN(p,i);
                    minPos = i;
                }
            }
            tCap(p) = minTime;
            NCap(p) = minPos;
        }

        KOKKOS_INLINE_FUNCTION
        void CalculateCapture(const uint32_t p, const Orientations<memory_space>& orient, const SteeringVector& steer, uint32_t& numSteer) const {
            // Usings
            using Common::self;

            //const float tCap_old = tCap(p); //(for safer and safest)

            // find the new minimum time and set indices appropriately
            updateMinTime(p);

            // if self has minimum time, reset to initial directions and add to steering vector
            const uint32_t pN = N(p,NCap(p));

            if (NCap(p) == self) {
                resetGrainID(p);
                AddToSteer(p, steer, numSteer);
            }
            // Otherwise, add to steering vector and calculate capture
            //else if (inSteer(pN) == UINT8_MAX || tCap(p)<tCap_old){ // Safest
            //else if (inSteer(pN) == UINT8_MAX || tCap(p)>tCap_old){ // Safer
            else if (inSteer(pN) == UINT8_MAX){ // least safe

                // Add to steering vector
                AddToSteer(p, steer, numSteer);

                // Get cell which should now have captured this cell
                const float dx = ((NCap(p) / 3) / 3) - 1.0f;
                const float dy = ((NCap(p) / 3) % 3) - 1.0f;
                const float dz = (NCap(p) % 3) - 1.0f;

                // distance from grain center to center of neighbor
                const float x0 = (-dx) - gx(pN); // dx to neighbor is -dx from neighbor
                const float y0 = (-dy) - gy(pN); // dy to neighbor is -dy from neighbor
                const float z0 = (-dz) - gz(pN); // dz to neighbor is -dz from neighbor

                // Associated Cell Orientations ID
                const uint32_t ID = dirID(pN);

                // distance corners need to move to have face capture center
                const float D[4] = {
                    x0 * orient.Fx0(ID) + y0 * orient.Fy0(ID) + z0 * orient.Fz0(ID),
                    x0 * orient.Fx1(ID) + y0 * orient.Fy1(ID) + z0 * orient.Fz1(ID),
                    x0 * orient.Fx2(ID) + y0 * orient.Fy2(ID) + z0 * orient.Fz2(ID),
                    x0 * orient.Fx3(ID) + y0 * orient.Fy3(ID) + z0 * orient.Fz3(ID)
                };

                // distance is equal to the maximum distance from above
                const float Dfabs = max(max(abs(D[0]), abs(D[1])), max(abs(D[2]), abs(D[3])));

                // Set grain orientation to the same
                softCopyGrainID(p, pN);
                grainID.setUnderResolved(p, tl(p), tCap(p));

                // Calculate projections of captured cell center onto diagonals
                const float p1 = (x0 * orient.d1x(ID)) + (y0 * orient.d1y(ID)) + (z0 * orient.d1z(ID));
                const float p2 = (x0 * orient.d2x(ID)) + (y0 * orient.d2y(ID)) + (z0 * orient.d2z(ID));
                const float p3 = (x0 * orient.d3x(ID)) + (y0 * orient.d3y(ID)) + (z0 * orient.d3z(ID));

                // Based on which corner is closest (projection is largest), set new the projection to be the new center (branchless code)
                const bool cond1 = (abs(p1) > abs(p2) && abs(p1) > abs(p3));
                const bool cond2 = (abs(p2) > abs(p1) && abs(p2) > abs(p3));
                const bool cond3 = 1 - cond1 - cond2;
                gx(p) = -(x0 - orient.d1x(ID) * p1 * cond1 - orient.d2x(ID) * p2 * cond2 - orient.d3x(ID) * p3 * cond3);
                gy(p) = -(y0 - orient.d1y(ID) * p1 * cond1 - orient.d2y(ID) * p2 * cond2 - orient.d3y(ID) * p3 * cond3);
                gz(p) = -(z0 - orient.d1z(ID) * p1 * cond1 - orient.d2z(ID) * p2 * cond2 - orient.d3z(ID) * p3 * cond3);
                G0(p) = Dfabs - abs(p1) * cond1 - abs(p2) * cond2 - abs(p3) * cond3;
            }
            else {
                // Reset Capture Time and NCap (so that it will be set to captured again)
                tCap(p) = FLT_MAX;
                NCap(p) = UINT8_MAX;
                // Set inSteer to special value (above 26 and below 255) for MPI Comms
                inSteer(p) = 69;
            }
        }
    };

    template<template<typename> class GrainID = BaseGrainIdentifier>
    class Grid_Dual{
    public:

        // Total size of grid
        uint32_t size;

        // Grid on device. Host-side staging should be explicit and temporary.
        Grid<device_space, GrainID<device_space>> deviceGrid;

        // Allocate memory for all the host and device views
        // [DONE ON HOST]
        void AllocateAndInitializeViews() {

            // Spatial Information
            deviceGrid.extent_view = uint32_deviceView(Kokkos::ViewAllocateWithoutInitializing("deviceGrid.extent_view"), 7);

            // Grid Location Information
            deviceGrid.ijk_view = uint32_deviceView(Kokkos::ViewAllocateWithoutInitializing("deviceGrid.ijk_view"), 3*size);
            deviceGrid.l_view = uint8_deviceView(Kokkos::ViewAllocateWithoutInitializing("deviceGrid.l_view"), size);

            // Neighbor info and capture time information
            deviceGrid.N_view = uint32_deviceView(Kokkos::ViewAllocateWithoutInitializing("deviceGrid.N_view"), 27*size);
            deviceGrid.tN_view = float_deviceView(Kokkos::ViewAllocateWithoutInitializing("deviceGrid.tN_view"), 27*size);
            deviceGrid.NCap_view = uint8_deviceView(Kokkos::ViewAllocateWithoutInitializing("deviceGrid.NCap_view"), size);
            deviceGrid.tCap_view = float_deviceView(Kokkos::ViewAllocateWithoutInitializing("deviceGrid.tCap_view"), size);

            Kokkos::deep_copy(deviceGrid.tN_view, FLT_MAX);
            Kokkos::deep_copy(deviceGrid.NCap_view, UINT8_MAX);
            Kokkos::deep_copy(deviceGrid.tCap_view, FLT_MAX);

            // Solidification Info - Melt time, Solidify time, Cooling rate
            deviceGrid.solInfo_view = float_deviceView(Kokkos::ViewAllocateWithoutInitializing("deviceGrid.solInfo_view"), 3*size);

            // TODO::NECESSARY?
            // [IS THIS NECESSARY AT ALL? OR JUST FOR VISUALIZATION?]
            // Nucleation
            deviceGrid.tNucl_view = float_deviceView(Kokkos::ViewAllocateWithoutInitializing("deviceGrid.tNucl_view"), size);
            // TODO::NECESSARY?
            Kokkos::deep_copy(deviceGrid.tNucl_view, FLT_MAX);

            // Grain Position Info
            deviceGrid.Gstats_view = float_deviceView(Kokkos::ViewAllocateWithoutInitializing("deviceGrid.Gstats_view"), 4*size);
            Kokkos::deep_copy(deviceGrid.Gstats_view, 0.0f);

            // Grain identity
            deviceGrid.dirID_view = uint16_deviceView(Kokkos::ViewAllocateWithoutInitializing("deviceGrid.dirID_view"), size);
            deviceGrid.dirID_org_view = uint16_deviceView(Kokkos::ViewAllocateWithoutInitializing("deviceGrid.dirID_org_view"), size);
            Kokkos::deep_copy(deviceGrid.dirID_view, UINT16_MAX);
            Kokkos::deep_copy(deviceGrid.dirID_org_view, UINT16_MAX);

            deviceGrid.grainID.Init(size, "deviceGrid.grainID");
            deviceGrid.grainID_org.Init(size, "deviceGrid.grainID_org");
            deviceGrid.grainID.Reset();
            deviceGrid.grainID_org.Reset();

            // Simulation Utility
            deviceGrid.utility_view = bool_deviceView(Kokkos::ViewAllocateWithoutInitializing("deviceGrid.utility_view"), 4*size);
            Kokkos::deep_copy(deviceGrid.utility_view, false);

            // Steering Vector
            deviceGrid.inSteer_view = uint8_deviceView(Kokkos::ViewAllocateWithoutInitializing("deviceGrid.inSteer_view"), size);
            Kokkos::deep_copy(deviceGrid.inSteer_view, UINT8_MAX);
        }
    };
}
