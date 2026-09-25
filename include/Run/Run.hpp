#pragma once

#include "Definitions.hpp"
#include "Common.hpp"

#include "Structs/Grid.hpp"
#include "Structs/SteeringVector.hpp"
#include "Structs/Orientations.hpp"
#include "Structs/Sim.hpp"
#include "Comms/Structs.hpp"
#include "Comms/Funcs.hpp"

#include "IO/Out.hpp"

namespace Toucan::Run {

    // Actual DECA Algorithm
    // (Only runs on Device)
    template<template<typename> class GrainID>
    void DECA(Structs::Grid_Dual<GrainID>& DualGrid, Structs::Mpi_Dual<GrainID>& DualMpi, Structs::Orientations_Dual& DualOrient, Structs::Sim_Dual& DualSim, Structs::SteeringVector& steer) {
        // Usings
        using Common::self;

        // Set references to correct memory spaces
        Structs::Grid<device_space, GrainID<device_space>>& grid = DualGrid.deviceGrid;
        Structs::Orientations<device_space>& orient = DualOrient.deviceOrient;
        Structs::Sim<device_space>& sim = DualSim.deviceSim;

        // Make container for second
        Structs::SteeringVector steerDual(steer.maxSize);

        // Iteration Counter
        uint32_t itert = 0;

        // MPI steering sizes
        uint32_t maxLocalSteerSize;
        uint32_t localSteerSize = steer.size;
        MPI_Allreduce(&localSteerSize, &maxLocalSteerSize, 1, MPI_UINT32_T, MPI_MAX, MPI_COMM_WORLD);

        // Synchronize with neighbors to get edges right (not supposed to nucleate recv points)
        Structs::Mpi_Synchronize(DualGrid, DualMpi, steer);
        Kokkos::fence(); // TODO::NECESSARY?

        // Continue as long as the steering vector is active
        while (maxLocalSteerSize) {
            
            // Break if the iteration count has reached the DEBUG maximum number of iterations
            if (itert == DualSim.maxIterations){
                std::cout << "DEBUG::Max Iterations Reached: " << itert << std::endl;
                break;
            }

            // Outputs iteration count
            std::cout << "\tIteration: " << itert << "\tSteerSize: " << steer.size << " / " << DualGrid.size << std::endl;

            // Reset grid points in steering vector so they can be added by neighbors again
            grid.ResetSteeringVector();

            // Run an iteration of DECA
            Kokkos::parallel_for(
                "DECA Iteration (Sends)",
                Kokkos::RangePolicy<device_space>(0, 27*steer.size),
                KOKKOS_LAMBDA(const uint32_t i)
                {
                    // Get cell number
                    const uint32_t p = steer.view(i/27);

                    // Get neighbor to send too
                    const uint8_t n = i%27;

                    // get neighbor cell's reference
                    const uint32_t pN = grid.N(p, n);

                    // if neighbor is edge or OOB, can't "capture"
                    if (pN != UINT32_MAX) {
                        // If capturing future-self and self tm>tcap
                        if (n == self) {
                            // Copy grain ID and put into steering vector
                            grid.hardCopyGrainID(pN, p);
                            grid.PrepForDualSteer(pN, self);
                            // Set capture time to be the same (don't worry, we will only start propagation at undercool=0)
                            grid.tCap(pN)=grid.tCap(p);
                        }
                        // If we can capture and this neighbor didn't capture us (prevent wasteful computation)
                        else if (!grid.cantCap(pN) && (grid.NCap(p)!=n)) {
                            // Associated Cell Orientations ID
                            const uint32_t ID = grid.dirID(p);

                            // get relative neighbor offsets
                            const float dx = ((n / 3) / 3) - 1.0f;
                            const float dy = ((n / 3) % 3) - 1.0f;
                            const float dz = (n % 3) - 1.0f;

                            // distance from grain center to center of neighbor
                            const float x0 = dx - grid.gx(p);
                            const float y0 = dy - grid.gy(p);
                            const float z0 = dz - grid.gz(p);

                            // distance corners need to move to have face capture center
                            const float D[4] = {
                                x0 * orient.Fx0(ID) + y0 * orient.Fy0(ID) + z0 * orient.Fz0(ID),
                                x0 * orient.Fx1(ID) + y0 * orient.Fy1(ID) + z0 * orient.Fz1(ID),
                                x0 * orient.Fx2(ID) + y0 * orient.Fy2(ID) + z0 * orient.Fz2(ID),
                                x0 * orient.Fx3(ID) + y0 * orient.Fy3(ID) + z0 * orient.Fz3(ID)
                            };

                            // distance is equal to the maximum distance from above
                            const float Dfabs = max(max(abs(D[0]), abs(D[1])), max(abs(D[2]), abs(D[3])));

                            // time of capture event of neighbor
                            const float Dist = (Dfabs - grid.G0(p)) * sim.res();
                            const float tUnder = max(grid.tCap(p) - grid.tl(p), 0.0f);

                            const float temp = (abs(Dist) * sim.b1()) / (sim.a() * pow(grid.cr(p), sim.b()));
                            const float dt = pow(temp + pow(tUnder, sim.b1()), sim.bn1());

                            // Capture time is liquid time + dt (if cell needs to grow), otherwise
                            const float t_temp = (Dist < 0) ? tUnder : dt;
                            const float t = grid.tl(p) + t_temp;

                            // Send calculated time to neighbor and add to steering vector if relevant
                            grid.SendEvent(pN, t, n);
                        }
                    }
                }
            );

            Kokkos::fence();

            // Scan will make insterting into steer less contentious while also trimming "temporal selfs" from steer if "self" is in steer
            Kokkos::parallel_scan(
                "DECA Iteration (Scan)",
                Kokkos::RangePolicy<device_space>(0, 27*steer.size),
                KOKKOS_LAMBDA(const uint32_t i, uint32_t& chunk_start, bool isFinal)
                {
                    // Get cell number
                    const uint32_t p = steer.view(i/27);

                    // Get neighbor to send too
                    const uint8_t n = i%27;

                    // get neighbor cell's reference
                    const uint32_t pN = grid.N(p, n);

                    // If neighbor is valid and this was the cell to capture it
                    if (pN != UINT32_MAX && grid.inSteer(pN)==n) {
                        // If the neighbor is the next temporal neighbor (n==self) and I'm in the steerig vector, DONT add
                        if (n!=self || grid.inSteer(p) == UINT8_MAX) {
                            if (isFinal) {
                                steerDual.view(chunk_start)=pN;
                                chunk_start++;
                            }
                            else {
                                chunk_start++;
                            }
                        }
                        else if (isFinal) {
                            grid.inSteer(pN)=69;
                        }
                    }
                }
            , steerDual.size);

            // Reset steer size so it can be reused
            steer.reset_size();

            // Loop over potential new steering points, see if they should be calculated/captured, and then potentially calculate
            Kokkos::parallel_reduce(
                "DECA Iteration (Captures)",
                Kokkos::RangePolicy<device_space>(0, steerDual.size),
                KOKKOS_LAMBDA(const uint32_t i, uint32_t& numSteer)
                {
                    // Get cell number
                    const uint32_t p = steerDual.view(i);

                    // If not an edge, regular capture
                    if (!grid.edge(p)){
                        grid.CalculateCapture(p, orient, steer, numSteer);
                    }
                    // If edge, stick into steer
                    else{
                        // grid.resetGrainID(p); // TODO::REMOVE b/c unnecessary (edges can't change so no need to reset)
                        grid.AddToSteer(p, steer, numSteer);
                    }
                }
            , steer.size);

            // Synchronize with neighbors
            Structs::Mpi_Synchronize(DualGrid, DualMpi, steer);

            // Get global steering vector size
            localSteerSize = steer.size;
            MPI_Allreduce(&localSteerSize, &maxLocalSteerSize, 1, MPI_UINT32_T, MPI_MAX, MPI_COMM_WORLD);

            // Update iteration counter
            itert++;
        }
    }

    // Nucleate grid and transfer from substrate
    // (ONLY ON DEVICE)
    template<template<typename> class GrainID>
    void Nucleate_Grid(Structs::Grid_Dual<GrainID>& DualGrid, Structs::Substrate_Dual<GrainID>& DualSub, Structs::Sim_Dual& DualSim, Structs::RNG_Dual& DualGen, Structs::SteeringVector& steer) {
        // Usings
        using Common::self;

        // Get number of events
        const size_t numEvents = DualGrid.size;

        // Get layer number
        const uint32_t layerNum = DualSub.layerNum;

        // Get rank
        const int rank = DualSim.rank;

        // Reset Steering Vector Size
        steer.reset_size();

        // Get References
        Structs::Grid<device_space, GrainID<device_space>>& grid = DualGrid.deviceGrid;
        Structs::Substrate<device_space, GrainID<device_space>>& sub = DualSub.deviceSub;
        Structs::Sim<device_space>& sim = DualSim.deviceSim;
        uint32_deviceView nucleatedRepeatCounts = DualSim.nucleatedRepeatCounts;
        uint32_deviceView nucleatedRepeatCounts_prev = DualSim.nucleatedRepeatCounts_prev;

        // Reset Grid for next DECA Sim
        grid.ResetArrays();

        // Generate random numbers for nucleation sites
        bool_deviceView  nucleation_site(Kokkos::ViewAllocateWithoutInitializing("nucleation_site"), numEvents);
        Kokkos::parallel_for(
            "RNGInit",
            Kokkos::RangePolicy<device_space>(0, numEvents),
            KOKKOS_LAMBDA(const uint32_t p)
            {
                if (DualGen.ZeroToOne<device_space>() > 1.0f - sim.nuclProb()) {
                    nucleation_site(p) = true;
                }
                else {
                    nucleation_site(p) = false;
                }
            }
        );
        Kokkos::fence();

        // Loop over points and nucleate
        Kokkos::parallel_reduce(
            "Nucleation Routine",
            Kokkos::RangePolicy<device_space>(0, numEvents),
            KOKKOS_LAMBDA(const uint32_t p, uint32_t& numSteer)
            {
                // If previously detected as an edge, set it to be whatever the substrate is
                if (grid.edge(p)) {
                    grid.tNucl(p) = grid.tl(p);
                    grid.tCap(p) = grid.tNucl(p);
                    grid.tN(p, self) = grid.tNucl(p);
                    grid.NCap(p) = self;
                    // If first time, and not an MpiWall (ie. received from another rank), inherit from substrate and add to steering vector
                    if (grid.l(p) == 0 && !grid.MpiWall(p)) {
                        // Get shifts for window, layer, and grid
                        // TODO::IMPROVE make constants of sub
                        const uint32_t windowShift = static_cast<uint32_t>(sim.windowHeight() / sim.res() + 0.5);        // How many points are in a window
                        const uint32_t layerShift = static_cast<uint32_t>(sim.layerHeight() / sim.res() + 0.5);        // How many points are from moving up in layers
                        const uint32_t gridShift = grid.zShift();                // How many points "deep" the grid goes
                        // Shift for point in k is total shift remainder 2*windowShift
                        const uint32_t kShifted = ((grid.k(p) - gridShift) + windowShift + layerNum*layerShift )%(2*windowShift);
                        // Set grid ID to intersecting substrate ID and initialize
                        const uint32_t p_sub = sub.ijk_to_p(grid.i(p), grid.j(p), kShifted);
                        grid.dirID_org(p) = sub.dirID(p_sub); 
                        grid.grainID_org.copyFrom(sub.grainID, p, p_sub);
                        grid.resetGrainID(p);
                        // Add to steering vector
                        grid.AddToSteer(p,steer,numSteer);
                        grid.inSteer(p) = self; // For MPI comms
                    }
                }
                // Otherwise, the event has chance to nucleate a grain unless it is an mpi wall
                else if (nucleation_site(p)) {
                    // Set nucleation time based on gaussian
                    float underCool = DualGen.Normal<device_space>(sim.meanUnder(), sim.stdUnder());
                    underCool = max(underCool, 0.0f);
                    grid.tNucl(p) = grid.tl(p) + underCool / grid.cr(p);
                    grid.tCap(p) = grid.tNucl(p);
                    grid.tN(p,self) = grid.tNucl(p);
                    // Get dirID and repeatID for this grain, init orientation based on MPI rank
                    const uint16_t dir_id_ordered = static_cast<uint16_t>(DualGen.UInt<device_space>(sim.numOrientations()));
                    // Transform grain direction ID into a randomly shuffled one
                    const uint16_t dir_id = DualGen.shuffled_orientations_list(dir_id_ordered);
                    uint32_deviceView repeatCounts = nucleatedRepeatCounts;
                    const int rankLocal = rank;
                    uint32_t repeat_id = 0;
                    if constexpr (GrainID<device_space>::hasUniqueID()) {
                        repeat_id = Kokkos::atomic_fetch_add(&repeatCounts(dir_id), 1u);
                    }
                    grid.initializeGrainOrientation(p, dir_id, repeat_id, rankLocal);
                    
                    // if (grid.l(p) == 0) {
                    //     grid.AddToSteer_Nucleate(p, steer, numSteer, self);
                    // }
                }
            }
        , steer.size);
          
        if constexpr(GrainID<device_space>::hasUniqueID()) {
            // Loop over orientation directions
            const uint32_t numOrientations = DualSim.hostSim.numOrientations();
            uint32_t overflow = 0u;
            Kokkos::parallel_reduce(
                "Nucleated RepeatID Overflow Check",
                Kokkos::RangePolicy<device_space>(0, numOrientations),
                KOKKOS_LAMBDA(const uint32_t p, uint32_t& overflow_th)
                {
                    if (nucleatedRepeatCounts(p) < nucleatedRepeatCounts_prev(p)){
                        overflow_th++;
                    }
                    nucleatedRepeatCounts_prev(p) = nucleatedRepeatCounts(p);
                },
                overflow
            );
            if (overflow != 0u) {
                throw std::runtime_error("RepeatID overflow detected while nucleating grains.");
            }
        }
    }

    // Transfer the Grid ID's back to the Substrate
    // (ONLY ON DEVICE)
    template<template<typename> class GrainID>
    void GridToSub(Structs::Grid_Dual<GrainID>& DualGrid, Structs::Substrate_Dual<GrainID>& DualSub, Structs::Sim_Dual& DualSim) {

        // Get number of events
        const size_t numEvents = DualGrid.size;

        // Get layer number
        const uint32_t layerNum = DualSub.layerNum;

        // Set reference to device grid
        Structs::Grid<device_space, GrainID<device_space>>& grid = DualGrid.deviceGrid;
        Structs::Sim<device_space>& sim = DualSim.deviceSim;
        Structs::Substrate<device_space, GrainID<device_space>>& sub = DualSub.deviceSub;

        // Put grid onto substrate
        Kokkos::parallel_for(
            "Grid2Sub",
            Kokkos::RangePolicy<device_space>(0, numEvents), KOKKOS_LAMBDA (int n)
            {
                //TODO::BAND-AID
                if (grid.last(n) && !grid.MpiWall(n) && grid.dirID(n)!=UINT16_MAX){
                    // Get shifts for window, layer, and grid
                    const uint32_t windowShift = static_cast<uint32_t>(sim.windowHeight() / sim.res() + 0.5);
                    const uint32_t layerShift = static_cast<uint32_t>(sim.layerHeight() / sim.res() + 0.5);
                    const uint32_t gridShift = grid.zShift();
                    // Shift for point in k is total shift remainder 2*windowShift
                    const uint32_t kShifted = ((grid.k(n) - gridShift) + windowShift + layerNum*layerShift )%(2*windowShift);
                    // Get substrate number at that spot
                    const uint32_t p = sub.ijk_to_p(grid.i(n), grid.j(n), kShifted);
                    // Set substrate grain ID to the overlapping cell with the last event number
                    sub.dirID(p) = grid.dirID(n);
                    if constexpr(GrainID<device_space>::hasUniqueID() || GrainID<device_space>::hasUnderResolved()) {
                        sub.grainID.copyFrom(grid.grainID, p, n);
                    }
                    sub.fromGrid(p) = true;

                }
            }
        );
        if constexpr(GrainID<device_space>::hasUniqueID()) {
            uint64_t packedMax = 0;
            Kokkos::parallel_reduce(
                "Grid2Sub RepeatID Max",
                Kokkos::RangePolicy<device_space>(0, numEvents),
                KOKKOS_LAMBDA(const uint32_t n, uint64_t& localMax)
                {
                    if (grid.last(n) && !grid.MpiWall(n) && grid.dirID(n)!=UINT16_MAX) {
                        const uint32_t repeatID = grid.grainID.repeatID_view(n);
                        if (repeatID != UINT32_MAX) {
                            const uint64_t candidate = (uint64_t{1} << 32) | static_cast<uint64_t>(repeatID);
                            if (candidate > localMax) {
                                localMax = candidate;
                            }
                        }
                    }
                },
                Kokkos::Max<uint64_t>(packedMax)
            );
            if ((packedMax >> 32) != 0u) {
                const uint32_t currentMax = static_cast<uint32_t>(packedMax & UINT32_MAX);
                if (DualSim.hasPrevGridToSubRepeatIDMax && currentMax < DualSim.prevGridToSubRepeatIDMax) {
                    throw std::runtime_error("RepeatID wraparound detected during GridToSub.");
                }
                DualSim.prevGridToSubRepeatIDMax = currentMax;
                DualSim.hasPrevGridToSubRepeatIDMax = true;
            }
        }
    }

    // Generate substrate
    // (ONLY ON DEVICE)
    template<template<typename> class GrainID>
    void GenerateSubstrate(Structs::Mpi_Dual<GrainID>& DualMpi, Structs::Substrate_Dual<GrainID>& DualSub, Structs::Sim_Dual& DualSim, Structs::RNG_Dual& DualGen, const uint32_t windowNum){

        // Set references
        const Structs::Mpi<host_space>& hostMpi = DualMpi.hostMpi;
        const Structs::Mpi<device_space>& deviceMpi = DualMpi.deviceMpi;
        const Structs::Substrate<device_space, GrainID<device_space>>& deviceSub = DualSub.deviceSub;
        const Structs::Sim<host_space>& hostSim = DualSim.hostSim;
        const Structs::Sim<device_space>& deviceSim = DualSim.deviceSim;
        const uint16_t randRange = hostSim.numOrientations();

        // Useful numbers
        const uint32_t grainMult = DualSub.grain_mult; // Number of cells before making a new grain
        const uint32_t windowSize = DualSub.windowSize;

        // Number of grains in x and y in global domain
        const uint32_t num_grains_x = std::ceil((hostMpi.imax_global() - hostMpi.imin_global() + 1) / grainMult);
        const uint32_t num_grains_y = std::ceil((hostMpi.jmax_global() - hostMpi.jmin_global() + 1) / grainMult);

        // Calculate k_offset for the grain for the start of the window
        const uint32_t grain_offset_k = windowNum * windowSize;
        const uint32_t sub_offset_k = (windowSize)*(windowNum%2);

        // Set Orientations (on half the substrate at a time) [On Device]
        const uint32_t halfSize = DualSub.size/2;
        Kokkos::parallel_for(
            "Substrate (Generation)",
            Kokkos::RangePolicy<device_space>(0, halfSize),
            KOKKOS_LAMBDA(const uint32_t p)
            {
                // Get i,j,k values
                uint32_t ijk[3];
                deviceSub.p_to_ijk(ijk,p);
                const uint32_t& i = ijk[0];
                const uint32_t& j = ijk[1];
                const uint32_t& k = ijk[2];

                // Get global i,j,k values
                const uint32_t i_global = i + deviceMpi.imin_global();
                const uint32_t j_global = j + deviceMpi.jmin_global();
                const uint32_t k_global = k + grain_offset_k;

                // Get grain indices numbers
                const uint64_t i_grain = i_global / grainMult;
                const uint64_t j_grain = j_global / grainMult;
                const uint64_t k_grain = k_global / grainMult;

                // Get un-transformed grain ID (positive integer, 0-inclusive)
                const uint64_t grain_id_ordered = k_grain * num_grains_x * num_grains_y + j_grain * num_grains_x + i_grain;
                
                // Get grain repeat number and direction ID number
                const uint64_t grain_repeat_id_64 = grain_id_ordered / deviceSim.numOrientations();
                const uint32_t grain_repeat_id = (grain_repeat_id_64 > UINT32_MAX) ? UINT32_MAX : static_cast<uint32_t>(grain_repeat_id_64);
                const uint16_t grain_dir_id_ordered = static_cast<uint16_t>((grain_id_ordered) % deviceSim.numOrientations());
                
                // Transform grain direction ID into a randomly shuffled one
                const uint16_t grain_dir_id = DualGen.shuffled_orientations_list(grain_dir_id_ordered);
                
                // Get substrate index number
                const uint32_t p_sub = deviceSub.ijk_to_p(i,j,k+sub_offset_k);

                // Set substrate ID's
                deviceSub.dirID(p_sub) = grain_dir_id;
                deviceSub.grainID.setUniqueID(p_sub, grain_repeat_id, UINT32_MAX);

                // Reset from grid
                deviceSub.fromGrid(p_sub) = false;
            }
        );
    }

    // Nucleate the initial Substrate
    // (ONLY ON DEVICE)
    template<template<typename> class GrainID>
    void Nucleate_Substrate(Structs::Mpi_Dual<GrainID>& DualMpi, Structs::Substrate_Dual<GrainID>& DualSub, Structs::Sim_Dual& DualSim, Structs::RNG_Dual& DualGen){
        Run::GenerateSubstrate(DualMpi, DualSub, DualSim, DualGen, 0);
        Run::GenerateSubstrate(DualMpi, DualSub, DualSim, DualGen, 1);
    }

    // Make a ready future
    namespace impl{
        // Helper function to create a ready future
        Common::future<void> make_ready_future() {
            Common::promise<void> promise;
            promise.set_value(); // Mark the promise as ready
            return promise.get_future();
        }
    }

    // For outputting results
    namespace impl{
        struct OutputAxisSelection {
            uint32_t localStart = 0;
            uint32_t count = 0;
            uint32_t stride = 1;
            uint32_t globalStart = 0;
            uint32_t globalStride = 1;
        };

        OutputAxisSelection SelectOutputAxis(const uint32_t localGlobalBegin, const uint32_t localGlobalEnd, const uint32_t localIndexBase, const uint32_t every, const uint32_t offset) {
            OutputAxisSelection selection;
            selection.globalStride = every;

            if (localGlobalBegin >= localGlobalEnd) {
                return selection;
            }

            if (every == UINT32_MAX) {
                if (offset >= localGlobalBegin && offset < localGlobalEnd) {
                    selection.localStart = localIndexBase + (offset - localGlobalBegin);
                    selection.count = 1;
                    selection.stride = 1;
                    selection.globalStart = offset;
                }
                return selection;
            }

            uint32_t first = offset;
            if (first < localGlobalBegin) {
                const uint32_t delta = localGlobalBegin - first;
                first += ((delta + every - 1) / every) * every;
            }

            if (first >= localGlobalEnd) {
                return selection;
            }

            selection.localStart = localIndexBase + (first - localGlobalBegin);
            selection.count = ((localGlobalEnd - 1 - first) / every) + 1;
            selection.stride = (selection.count > 1) ? every : 1;
            selection.globalStart = first;
            return selection;
        }

        uint32_t CenterIndex(const uint32_t begin, const uint32_t end) {
            const uint32_t extent = end - begin;
            return begin + ((extent == 0) ? 0 : ((extent - 1) / 2));
        }

        void ValidateEffectiveOutputAxis(const uint32_t every, const uint32_t offset, const bool offsetWasSet) {
            if (every == 0) {
                throw std::runtime_error("Input error: Output Selection Stride values must be greater than zero.");
            }
            if (every != UINT32_MAX && offsetWasSet && offset >= every) {
                throw std::runtime_error("Input error: Output Selection Offset values must be less than Output Selection Stride values.");
            }
        }

        void FillRegionAxis(Structs::OutputRegion& region, const int axis, const OutputAxisSelection& selection) {
            region.localStart[axis] = selection.localStart;
            region.count[axis] = selection.count;
            region.stride[axis] = selection.stride;
            region.globalStart[axis] = selection.globalStart;
            region.globalStride[axis] = selection.globalStride;
        }

        template<template<typename> class GrainID>
        Common::vector<Structs::OutputRegion> BuildOutputRegions(Structs::Substrate_Dual<GrainID>& DualSub, Structs::Mpi_Dual<GrainID>& DualMpi, Structs::Sim_Dual& DualSim, const uint32_t windowNum) {
            Common::vector<Structs::OutputRegion> regions;

            Structs::Sim<host_space>& sim = DualSim.hostSim;

            uint32_t every[3] = {DualSim.outputEvery[0], DualSim.outputEvery[1], DualSim.outputEvery[2]};
            uint32_t offset[3] = {DualSim.outputOffset[0], DualSim.outputOffset[1], DualSim.outputOffset[2]};
            bool offsetWasSet[3] = {DualSim.outputOffsetSet[0], DualSim.outputOffsetSet[1], DualSim.outputOffsetSet[2]};

            const uint32_t localXEnd = DualMpi.header.global_i0() + DualMpi.header.local_inum();
            const uint32_t localYEnd = DualMpi.header.global_j0() + DualMpi.header.local_jnum();
            const uint32_t localXBegin =
                DualMpi.header.global_i0() + ((DualMpi.neighbor_ranks[7] != MPI_PROC_NULL && localXEnd > DualMpi.header.global_i0()) ? 1u : 0u);
            const uint32_t localYBegin =
                DualMpi.header.global_j0() + ((DualMpi.neighbor_ranks[1] != MPI_PROC_NULL && localYEnd > DualMpi.header.global_j0()) ? 1u : 0u);
            const uint32_t localXBase = localXBegin - DualSub.header.global_i0();
            const uint32_t localYBase = localYBegin - DualSub.header.global_j0();

            uint32_t localMin[2] = {localXBegin, localYBegin};
            uint32_t localMax[2] = {localXEnd, localYEnd};
            uint32_t globalMin[2] = {0, 0};
            uint32_t globalMax[2] = {0, 0};
            MPI_Allreduce(localMin, globalMin, 2, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(localMax, globalMax, 2, MPI_UINT32_T, MPI_MAX, MPI_COMM_WORLD);

            const uint32_t layerShift = std::max(static_cast<uint32_t>(sim.layerHeight()/DualSub.res + 0.5), 1u);
            if (DualSim.outputMode == "XY" && !DualSim.outputEverySet[2]) {
                every[2] = layerShift;
            }
            if (DualSim.outputMode == "XZ") {
                if (!DualSim.outputEverySet[1]) {
                    every[1] = UINT32_MAX;
                }
                if (!DualSim.outputOffsetSet[1]) {
                    offset[1] = CenterIndex(globalMin[1], globalMax[1]);
                    offsetWasSet[1] = false;
                }
            }
            if (DualSim.outputMode == "YZ") {
                if (!DualSim.outputEverySet[0]) {
                    every[0] = UINT32_MAX;
                }
                if (!DualSim.outputOffsetSet[0]) {
                    offset[0] = CenterIndex(globalMin[0], globalMax[0]);
                    offsetWasSet[0] = false;
                }
            }

            for (int axis = 0; axis < 3; axis++) {
                ValidateEffectiveOutputAxis(every[axis], offset[axis], offsetWasSet[axis]);
                if (every[axis] != UINT32_MAX && offset[axis] >= every[axis]) {
                    offset[axis] %= every[axis];
                }
            }

            const uint32_t localZBase = (windowNum % 2) ? DualSub.windowSize : 0;

            const OutputAxisSelection xAxis = SelectOutputAxis(localXBegin, localXEnd, localXBase, every[0], offset[0]);
            const OutputAxisSelection yAxis = SelectOutputAxis(localYBegin, localYEnd, localYBase, every[1], offset[1]);
            OutputAxisSelection zAxis = SelectOutputAxis(0, DualSub.windowSize, localZBase, every[2], offset[2]);
            zAxis.globalStart += windowNum * DualSub.windowSize;

            auto makeRegion = [&](const OutputAxisSelection& x, const OutputAxisSelection& y, const OutputAxisSelection& z, const Common::string& suffix) {
                Structs::OutputRegion region;
                FillRegionAxis(region, 0, x);
                FillRegionAxis(region, 1, y);
                FillRegionAxis(region, 2, z);
                region.sliceSuffix = suffix;
                if (!region.empty()) {
                    regions.push_back(region);
                }
            };

            if (DualSim.outputMode == "XYZ") {
                makeRegion(xAxis, yAxis, zAxis, "");
            }
            else if (DualSim.outputMode == "XY") {
                for (uint32_t slice = 0; slice < zAxis.count; slice++) {
                    OutputAxisSelection zSlice = zAxis;
                    zSlice.count = 1;
                    zSlice.stride = 1;
                    zSlice.localStart = zAxis.localStart + slice * zAxis.stride;
                    zSlice.globalStart = zAxis.globalStart + slice * zAxis.globalStride;
                    makeRegion(xAxis, yAxis, zSlice, ".Slice.Z" + std::to_string(zSlice.globalStart));
                }
            }
            else if (DualSim.outputMode == "XZ") {
                for (uint32_t slice = 0; slice < yAxis.count; slice++) {
                    OutputAxisSelection ySlice = yAxis;
                    ySlice.count = 1;
                    ySlice.stride = 1;
                    ySlice.localStart = yAxis.localStart + slice * yAxis.stride;
                    ySlice.globalStart = yAxis.globalStart + slice * yAxis.globalStride;
                    makeRegion(xAxis, ySlice, zAxis, ".Slice.Y" + std::to_string(ySlice.globalStart));
                }
            }
            else if (DualSim.outputMode == "YZ") {
                for (uint32_t slice = 0; slice < xAxis.count; slice++) {
                    OutputAxisSelection xSlice = xAxis;
                    xSlice.count = 1;
                    xSlice.stride = 1;
                    xSlice.localStart = xAxis.localStart + slice * xAxis.stride;
                    xSlice.globalStart = xAxis.globalStart + slice * xAxis.globalStride;
                    makeRegion(xSlice, yAxis, zAxis, ".Slice.X" + std::to_string(xSlice.globalStart));
                }
            }
            else{
                throw std::runtime_error("Input error: Keyword 'FILE::OUTPUT_MODE' has invalid value.\nValid values are [None, XY, XZ, YZ, XYZ].");
            }

            return regions;
        }

        // Just for outputting
        template<template<typename> class GrainID>
        void SyncAndWriteToDisk(Structs::Substrate_Dual<GrainID>& DualSub, Structs::Mpi_Dual<GrainID>& DualMpi, Structs::Sim_Dual& DualSim, Toucan::Common::future<void>& write_done_future, const uint32_t windowNum){

            // Namespaces
            using Toucan::Common::thread;
            using Toucan::Common::promise;
            using Toucan::Common::ref;

            // Wait until previous write is complete
            write_done_future.wait();
            promise<void> write_done;
            write_done_future = write_done.get_future();

            // Only copy data if output is desired
            if (DualSim.outputMode!="None"){
                Common::vector<Structs::OutputRegion> regions = BuildOutputRegions(DualSub, DualMpi, DualSim, windowNum);

                for (const auto& region : regions) {
                    DualSub.CopyFromDevice(region);
                }

                // Output results asynchronously
                const bool shiftZ = (windowNum%2); // If current windowNum is odd, then next output requires a z-shift (so shift it)
                thread writer_thread(IO::SubToOutput_async<GrainID>, ref(DualSub), ref(DualMpi), ref(DualSim), windowNum, shiftZ, regions, std::move(write_done));
                writer_thread.detach();
            }
            else{
                // Signal that the write is done (because output is "None")
                write_done.set_value();
            }
        }
    }

    // Output results
    // (DEVICE->HOST)
    template<template<typename> class GrainID>
    void Output_Results(Structs::Substrate_Dual<GrainID>& DualSub, Structs::Mpi_Dual<GrainID>& DualMpi, Structs::Sim_Dual& DualSim, Structs::RNG_Dual& DualGen, Toucan::Common::future<void>& write_done_future){
        // Usings
        using Common::string;
        using Common::thread;
        using Common::future;
        using Common::promise;
        using Common::ref;
        using Common::move;

        // References
        Structs::Sim<host_space>& sim = DualSim.hostSim;

        // Calculate the windows that the current and next layer be in
        const uint32_t windowShift = DualSub.windowSize;
        const uint32_t layerShift = static_cast<uint32_t>(sim.layerHeight()/DualSub.res + 0.5);
        const uint32_t curWindow = ((layerShift*DualSub.layerNum)%(2*windowShift))/windowShift;
        const uint32_t nextWindow = ((layerShift*(DualSub.layerNum+1))%(2*windowShift))/windowShift;

        // If the next increment will go into the next window, increment the window number and output the relevant substrate data
        if (nextWindow!=curWindow){
            // Copy to host and write data to disk
            impl::SyncAndWriteToDisk(DualSub, DualMpi, DualSim, write_done_future, DualSub.windowNum - 1);
            write_done_future.wait();

            // Generate next substrate
            Run::GenerateSubstrate(DualMpi, DualSub, DualSim, DualGen, DualSub.windowNum + 1);

            // Increment window num and output
            DualSub.windowNum++;

            // Increment layer number
            DualSub.layerNum++;
            DualSub.lastLayerOutput = DualSub.layerNum;
        }
        else{
            // Increment layer number
            DualSub.layerNum++;
        }

    }

    // TODO::Get Working (Work for volume or surface)
    template<template<typename> class GrainID>
    void Output_Final_Results(Structs::Substrate_Dual<GrainID>& DualSub, Structs::Mpi_Dual<GrainID>& DualMpi, Structs::Sim_Dual& DualSim, Toucan::Common::future<void>& write_done_future){

        // Write previous window
        impl::SyncAndWriteToDisk(DualSub, DualMpi, DualSim, write_done_future, DualSub.windowNum - 1);

        // TODO::MAY BE EMPTY IF SURFACE
        // Write current window only if different from last output
        if (DualSub.layerNum != DualSub.lastLayerOutput){
            impl::SyncAndWriteToDisk(DualSub, DualMpi, DualSim, write_done_future, DualSub.windowNum);
        }

    }

}
