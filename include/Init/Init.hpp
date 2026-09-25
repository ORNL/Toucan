#pragma once

#include "Definitions.hpp"
#include "Common.hpp"

#include "Structs/Sim.hpp"
#include "Structs/Layer.hpp"
#include "Structs/Substrate.hpp"
#include "Structs/Grid.hpp"
#include "Comms/Structs.hpp"
#include "Structs/SteeringVector.hpp"
#include "Structs/Rng.hpp"

namespace Toucan::Init{

    template<template<typename> class GrainID>
    void Mpi(Structs::Mpi_Dual<GrainID>& DualMpi, Structs::Sim_Dual& DualSim, Stork::Structs::RegularGrid_Header<float, host_space>& RDF_header){

        // Set references
        Structs::Sim<host_space>& sim = DualSim.hostSim;
        Structs::Mpi<host_space>& mpi = DualMpi.hostMpi;

        // Set base RDF header to be the same (for the check)
        DualMpi.RDF_Input_header = RDF_header;

        // Get x-y bounds for local domain [HOST]
        DualMpi.header.global_x0() = RDF_header.global_x0();    // Same global origin
        DualMpi.header.global_y0() = RDF_header.global_y0();    // Same global origin
        DualMpi.header.global_z0() = RDF_header.global_z0();    // Same global origin
        DualMpi.header.global_k0() = RDF_header.global_k0();    // Z stuff doesn't change
        DualMpi.header.local_knum() = RDF_header.local_knum();    // Z stuff doesn't change
        DualMpi.header.gridResolution() = RDF_header.gridResolution(); // Resolution doesn't change

        // Global Decomposition
        const int I = DualMpi.dims[0];
        const int J = DualMpi.dims[1];

        // Local Decomposition Number
        const int i = DualMpi.coords[0];
        const int j = DualMpi.coords[1];

        // TODO:: -> Logic might be wrong for strong/weak
        // Split up whole file appropriately
        if (DualSim.mpiMode=="Strong"){
            // Resolution
            mpi.res() = RDF_header.gridResolution();

            // x bounds
            mpi.imin_global() = RDF_header.global_i0() + ((RDF_header.local_inum()-1)*i)/I + 1*(i!=0);
            mpi.imax_global() = RDF_header.global_i0() + ((RDF_header.local_inum()-1)*(i+1))/I;

            // y bounds
            mpi.jmin_global() = RDF_header.global_j0() + ((RDF_header.local_jnum()-1)*j)/J + 1*(j!=0);
            mpi.jmax_global() = RDF_header.global_j0() + ((RDF_header.local_jnum()-1)*(j+1))/J;
        }
        // Increment if not an edge so that recv's are correct
        else if (DualSim.mpiMode=="OneToOne"){
            // Resolution
            mpi.res() = DualMpi.header.gridResolution();

            // x bounds (adding and subtracting 1 so the decomposition scheme knows what is owned and what is edge)
            mpi.imin_global() = RDF_header.global_i0() + 1*(i!=0);
            mpi.imax_global() = RDF_header.global_i0() + (RDF_header.local_inum()-1) - 1*(i!=(I-1));

            // y bounds (adding and subtracting 1 so the decomposition scheme knows what is owned and what is edge)
            mpi.jmin_global() = RDF_header.global_j0() + 1*(j!=0);
            mpi.jmax_global() = RDF_header.global_j0() + (RDF_header.local_jnum()-1) - 1*(j!=(J-1));
        }
        else {
            throw std::runtime_error("Input error: Keyword 'MPI::MODE' has invalid value.\nValid values are [Strong, OneToOne].");
        }

        // Set rest of header information
        DualMpi.header.global_i0() = mpi.imin_global();
        DualMpi.header.local_inum() = (1+(mpi.imax_global()-mpi.imin_global()));
        DualMpi.header.global_j0() = mpi.jmin_global();
        DualMpi.header.local_jnum() = (1+(mpi.jmax_global()-mpi.jmin_global()));

        // Copy limits to GPU
        Kokkos::deep_copy(DualMpi.deviceMpi.mpiIndices_view, DualMpi.hostMpi.mpiIndices_view);
        Kokkos::deep_copy(DualMpi.deviceMpi.mpiFloats_view, DualMpi.hostMpi.mpiFloats_view);
    }

    template<template<typename> class GrainID>
    void Substrate(Structs::Substrate_Dual<GrainID>& DualSub, Structs::RNG_Dual& DualGen, Structs::Sim_Dual& DualSim, Structs::Mpi_Dual<GrainID>& DualMpi){

        // Get References
        Structs::Sim<host_space>& hostSim = DualSim.hostSim;
        Structs::Mpi<host_space>& hostMpi = DualMpi.hostMpi;
        auto& hostSub = DualSub.hostSub;
        auto& deviceSub = DualSub.deviceSub;

        // Global Decomposition
        const int I = DualMpi.dims[0];
        const int J = DualMpi.dims[1];

        // Local Decomposition Number
        const int i = DualMpi.coords[0];
        const int j = DualMpi.coords[1];

        // Set Substrate Constants (includes mpi buffers buffer)
        DualSub.header = DualMpi.header;

        // TODO::MAKE THE SAME AS RDF?
        // Make slightly bigger (for buffer...shoudn't be used though)
        const uint32_t iMin = hostMpi.imin_global() - 1*(i!=0);
        const uint32_t iMax = hostMpi.imax_global() + 1*(i!=(I-1));
        const uint32_t jMin = hostMpi.jmin_global() - 1*(j!=0);
        const uint32_t jMax = hostMpi.jmax_global() + 1*(j!=(J-1));
        DualSub.header.global_i0() = iMin;
        DualSub.header.global_j0() = jMin;
        DualSub.header.local_inum() = (1+(iMax-iMin));
        DualSub.header.local_jnum() = (1+(jMax-jMin));

        // Set substrate variables
        DualSub.res = DualSub.header.gridResolution();
        DualSub.zmin = -hostSim.windowHeight();
        DualSub.windowSize = static_cast<uint32_t>((hostSim.windowHeight()/DualSub.header.gridResolution()) + 0.5f);

        // Set Substrate Sizes
        hostSub.xnum() = (1+(iMax-iMin));
        hostSub.ynum() = (1+(jMax-jMin));
        hostSub.znum() = 2*DualSub.windowSize;
        DualSub.grain_mult = static_cast<uint32_t>(DualSim.hostSim.subGrainSize()/DualSub.res);
        DualSub.size = hostSub.xnum() * hostSub.ynum() * hostSub.znum();

        // Make Appropriately Sized Views For Substrate
        deviceSub.dirID_view = uint16_deviceView(Kokkos::ViewAllocateWithoutInitializing("deviceSub.dirID_view"), DualSub.size);
        deviceSub.fromGrid_view = bool_deviceView("deviceSub.fromGrid_view", DualSub.size);    // Initializes as false
        hostSub.dirID_view = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, host_memory(), deviceSub.dirID_view);
        hostSub.fromGrid_view = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, host_memory(), deviceSub.fromGrid_view);
        deviceSub.grainID.Init(DualSub.size, "deviceSub.grainID");
        deviceSub.grainID.Reset();
        hostSub.grainID.createHostMirrorsFrom(deviceSub.grainID);

        // Intialize 3D views (not new data, just pointers and strides)
        DualSub.MakeViews_3D();
        DualSub.InitializeOutputScratch(DualSub.size);

        // Copy substrate index and offset information to device
        Kokkos::deep_copy(deviceSub.subNums_view, hostSub.subNums_view);
    }

    // For outputting orientation information
    namespace impl{
        void writeOrientations(Structs::Sim_Dual& DualSim, Structs::Orientations_Dual& DualOrient){
            // Get references
            Structs::Sim<host_space>& hostSim = DualSim.hostSim;
            Structs::Orientations<host_space>& hostOrient = DualOrient.hostOrient;

            // How many orientaions
            const uint16_t numOrientations = hostSim.numOrientations();

            // Set directory
            const Common::string dir = DualSim.simName + "/";

            // Output CSV
            std::ofstream datafile;
            datafile.exceptions(std::ofstream::failbit | std::ofstream::badbit);
            Common::string out_file = dir + "Orientations.csv";
            try {
                datafile.open(out_file.c_str());
                datafile << "dirID,d1x,d1y,d1z,d2x,d2y,d2z,d3x,d3y,d3z\n";
                // Loop through the data using the precomputed condition - orientation 1 maps to unique grain IDs +/-1, looping back around at the absolute value of the grain ID exceeds numOrientations
                for (uint32_t p = 0; p < numOrientations; p++) {
                    datafile << p+1 << ",";
                    datafile << hostOrient.d1x(p) << "," << hostOrient.d1y(p) << "," << hostOrient.d1z(p) << ",";
                    datafile << hostOrient.d2x(p) << "," << hostOrient.d2y(p) << "," << hostOrient.d2z(p) << ",";
                    datafile << hostOrient.d3x(p) << "," << hostOrient.d3y(p) << "," << hostOrient.d3z(p) << "\n";
                }
            }
            catch (const std::ofstream::failure& e) {
                std::cout << "Exception writing data file, check that Data directory exists\n";
            }
            datafile.close();
        }
    }

    template<template<typename> class GrainID>
    void Orientations(Structs::Orientations_Dual& DualOrient, Structs::Sim_Dual& DualSim, Structs::Mpi_Dual<GrainID>& DualMpi, Structs::RNG_Dual& DualGen){

        // Get orientation references
        Structs::Orientations<host_space>& hostOrient = DualOrient.hostOrient;
        Structs::Orientations<device_space>& deviceOrient = DualOrient.deviceOrient;

        // Set sim reference
        Structs::Sim<host_space>& sim = DualSim.hostSim;

        // Number of orientations
        const size_t numOrientations = sim.numOrientations();

        // Make views apporpriate size
        deviceOrient.dirInfo_view = float_deviceView(Kokkos::ViewAllocateWithoutInitializing("deviceOrient.dirInfo_view"), 9 * numOrientations);
        hostOrient.dirInfo_view = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, host_memory(), deviceOrient.dirInfo_view);

        deviceOrient.faceInfo_view = float_deviceView(Kokkos::ViewAllocateWithoutInitializing("deviceOrient.faceInfo_view"), 12 * numOrientations);
        hostOrient.faceInfo_view = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, host_memory(), deviceOrient.faceInfo_view);

        // Reference host Orient before for loop
        Structs::Orientations<host_space>& orient = hostOrient;

        // Initilaize with numbers
        Kokkos::parallel_for(
                "Orientation Initialization",
                Kokkos::RangePolicy<host_space>(0, numOrientations),
                KOKKOS_LAMBDA(const uint32_t p)
                {
                    // Generate Random Orientation Numbers and Set Directions
                    const float theta = DualGen.ZeroToOne<host_space>()* 2.0f * static_cast<float>(M_PI);
                    const float phi = acosf(1.0f - 2.0f * DualGen.ZeroToOne<host_space>());
                    const float theta2 = DualGen.ZeroToOne<host_space>() * 2.0f * static_cast<float>(M_PI);
                    const float phi2 = acosf(1.0f - 2.0f * DualGen.ZeroToOne<host_space>());
                    // Generate first random direction
                    orient.d1x(p) = cosf(theta) * sinf(phi); orient.d1y(p) = sinf(theta) * sinf(phi); orient.d1z(p) = cosf(phi);
                    // Generate second random direction
                    orient.d2x(p) = cosf(theta2) * sinf(phi2); orient.d2y(p) = sinf(theta2) * sinf(phi2); orient.d2z(p) = cosf(phi2);
                    // Remove similar component (so they become orthogonal)
                    const float p12 = orient.d1x(p) * orient.d2x(p) + orient.d1y(p) * orient.d2y(p) + orient.d1z(p) * orient.d2z(p);
                    orient.d2x(p) -= p12 * orient.d1x(p); orient.d2y(p) -= p12 * orient.d1y(p); orient.d2z(p) -= p12 * orient.d1z(p);
                    const float norm = sqrt(orient.d2x(p) * orient.d2x(p) + orient.d2y(p) * orient.d2y(p) + orient.d2z(p) * orient.d2z(p));
                    orient.d2x(p) /= norm; orient.d2y(p) /= norm; orient.d2z(p) /= norm;
                    // Final direction is cross product of first two
                    orient.d3x(p) = orient.d1y(p) * orient.d2z(p) - orient.d2y(p) * orient.d1z(p);
                    orient.d3y(p) = orient.d1z(p) * orient.d2x(p) - orient.d2z(p) * orient.d1x(p);
                    orient.d3z(p) = orient.d1x(p) * orient.d2y(p) - orient.d2x(p) * orient.d1y(p);

                    // Now set face values
                    orient.Fx0(p) = orient.d1x(p) + orient.d2x(p) + orient.d3x(p);
                    orient.Fx1(p) = orient.d1x(p) + orient.d2x(p) - orient.d3x(p);
                    orient.Fx2(p) = orient.d1x(p) - orient.d2x(p) + orient.d3x(p);
                    orient.Fx3(p) = orient.d1x(p) - orient.d2x(p) - orient.d3x(p);

                    orient.Fy0(p) = orient.d1y(p) + orient.d2y(p) + orient.d3y(p);
                    orient.Fy1(p) = orient.d1y(p) + orient.d2y(p) - orient.d3y(p);
                    orient.Fy2(p) = orient.d1y(p) - orient.d2y(p) + orient.d3y(p);
                    orient.Fy3(p) = orient.d1y(p) - orient.d2y(p) - orient.d3y(p);

                    orient.Fz0(p) = orient.d1z(p) + orient.d2z(p) + orient.d3z(p);
                    orient.Fz1(p) = orient.d1z(p) + orient.d2z(p) - orient.d3z(p);
                    orient.Fz2(p) = orient.d1z(p) - orient.d2z(p) + orient.d3z(p);
                    orient.Fz3(p) = orient.d1z(p) - orient.d2z(p) - orient.d3z(p);
                }
        );

        // Output Orientation Information (rank 0)
        if (!DualMpi.rank){
            impl::writeOrientations(DualSim, DualOrient);
        }

        // Broadcast Orientation Data to other devices
        MPI_Bcast(hostOrient.dirInfo_view.data(), 9 * numOrientations, MPI_FLOAT, 0, DualMpi.comm);
        MPI_Bcast(hostOrient.faceInfo_view.data(), 12 * numOrientations, MPI_FLOAT, 0, DualMpi.comm);

        // Copy Orientation Directional Information to Device
        // [DONE ON HOST->DEVICE]
        Kokkos::deep_copy(deviceOrient.dirInfo_view, hostOrient.dirInfo_view);
        Kokkos::deep_copy(deviceOrient.faceInfo_view, hostOrient.faceInfo_view);
    }

    // For MPI decomposition of the Grid (strong scaling)
    namespace impl{

        // DEBUG Functions for checking sends and recieves
        namespace
        {
            template<template<typename> class GrainID>
            void CheckSendRecvWithNeighbor(Structs::Mpi_Dual<GrainID>& DualMpi){

                // Initialize
                MPI_Request send_requests[16]; // Store send requests
                MPI_Request recv_requests[16]; // Store receive requests
                std::vector<int> received_send_sizes(8);
                std::vector<int> received_recv_sizes(8);
                int send_request_count = 0;
                int recv_request_count_A = 0; // Count for "A" receives
                int recv_request_count_B = 0; // Count for "B" receives


                // Loop over neighbors
                for (int n = 0; n < 8; ++n) {
                    if (DualMpi.neighbor_ranks[n] != MPI_PROC_NULL) {
                        const int neighbor_rank = DualMpi.neighbor_ranks[n];

                        // Initiate non-blocking send of hostMpi.send_size(n)
                        MPI_Isend(&DualMpi.hostMpi.send_size(n), 1, MPI_INT, neighbor_rank, 0, MPI_COMM_WORLD, &send_requests[send_request_count++]);

                        // Initiate non-blocking send of hostMpi.recv_size(n)
                        MPI_Isend(&DualMpi.hostMpi.recv_size(n), 1, MPI_INT, neighbor_rank, 1, MPI_COMM_WORLD, &send_requests[send_request_count++]);

                        // Initiate non-blocking receive of A (neighbor's .send_size)
                        MPI_Irecv(&received_send_sizes[n], 1, MPI_INT, neighbor_rank, 0, MPI_COMM_WORLD, &recv_requests[recv_request_count_A++]);

                        // Initiate non-blocking receive of B (neighbor's .recv_size)
                        MPI_Irecv(&received_recv_sizes[n], 1, MPI_INT, neighbor_rank, 1, MPI_COMM_WORLD, &recv_requests[8 + recv_request_count_B++]);  //Correct, keep the +8
                    }
                }

                // Wait for all receives to complete
                MPI_Waitall(recv_request_count_A, recv_requests, MPI_STATUSES_IGNORE); // Wait for A's
                MPI_Waitall(recv_request_count_B, &recv_requests[8], MPI_STATUSES_IGNORE); //Wait for the B's.  Correct

                // Now that receives are complete, do the checking
                for (int n = 0; n < 8; ++n) {
                    if (DualMpi.neighbor_ranks[n] != MPI_PROC_NULL) {
                        const int neighbor_rank = DualMpi.neighbor_ranks[n];
                        if (received_send_sizes[n] != DualMpi.hostMpi.recv_size(n) || received_recv_sizes[n] != DualMpi.hostMpi.send_size(n)) {
                            std::cout << "Rank " << DualMpi.rank << ": Mismatch with neighbor " << neighbor_rank << "\n"
                                      << "Neighbor's send_size) = " << received_send_sizes[n] << ", expected recv_size = " << DualMpi.hostMpi.recv_size(n) << "\n"
                                      << "Neighbor's recv_size) = " << received_recv_sizes[n] << ", expected send_size = " << DualMpi.hostMpi.send_size(n) << "\n"
                                      << std::endl;
                        }
                    }
                }

                // Wait for all sends to complete
                MPI_Waitall(send_request_count, send_requests, MPI_STATUSES_IGNORE);
            }
        }

        template<template<typename> class GrainID>
        void MPIDecomposition(Structs::Grid_Dual<GrainID>& DualGrid, Structs::Mpi_Dual<GrainID>& DualMpi, Stork::Structs::RDF_Dual<float>& RDF){

            // References to RDF
            Stork::Structs::RegularGrid_Header<float, device_space>& RDF_header = RDF.device_header;
            Stork::Structs::RDF_Data<float, device_space>& RDF_data = RDF.device_data;

            // References to MPI
            Structs::Mpi<host_space>& hostMpi = DualMpi.hostMpi;
            Structs::Mpi<device_space>& deviceMpi = DualMpi.deviceMpi;

            // TODO::Make stuff contiguuous (event numbers)
            // If only one processor, no need to decompose anything
            if (DualMpi.nproc == 1) {
                DualGrid.size = RDF.numEvents;
                return;
            }

            // TODO::MAKE **EVERYTHING IN CODE START INDEXING AT 1 (so 0 would be buffer and is always "safe")**
            // TODO::SHIFT MPI BOUNDS BY 1? (so we don't check -1 if 0 is lower limit)
            hostMpi.imin_global() += 1;
            hostMpi.imax_global() += 1;
            hostMpi.jmin_global() += 1;
            hostMpi.jmax_global() += 1;
            Kokkos::deep_copy(DualMpi.deviceMpi.mpiIndices_view, DualMpi.hostMpi.mpiIndices_view);

            // Make single object for reduction and scan
            Structs::MPISizes sizes;
            const uint32_t numEvents = RDF.numEvents;

            // Loop over all points and see if they are in the grid or should be sent or received
            Kokkos::parallel_reduce(
                "MPI Decompose (reduce)",
                Kokkos::RangePolicy<device_space>(0, numEvents),
                KOKKOS_LAMBDA(const uint32_t n, Structs::MPISizes& chunk)
                {
                    uint32_t ijk_global[3];
                    RDF_header.LOCAL_p_to_GLOBAL_ijk(ijk_global,RDF_data.p(n));

                    // TODO::SHIFT
                    const uint32_t i_global = ijk_global[0] + 1;
                    const uint32_t j_global = ijk_global[1] + 1;

                    // If in total domain (including buffer), then include the point
                    if (deviceMpi.in_total_domain(i_global, j_global)) {
                        chunk.in++;
                        // Check interior
                        if (deviceMpi.in_domain(i_global, j_global)) {
                            // Check sends
                            if (deviceMpi.in_sendDownLeft(i_global, j_global)) {
                                chunk.send[0]++;
                            }
                            if (deviceMpi.in_sendDown(i_global, j_global)) {
                                chunk.send[1]++;
                            }
                            if (deviceMpi.in_sendDownRight(i_global, j_global)) {
                                chunk.send[2]++;
                            }
                            if (deviceMpi.in_sendRight(i_global, j_global)) {
                                chunk.send[3]++;
                            }
                            if (deviceMpi.in_sendUpRight(i_global, j_global)) {
                                chunk.send[4]++;
                            }
                            if (deviceMpi.in_sendUp(i_global, j_global)) {
                                chunk.send[5]++;
                            }
                            if (deviceMpi.in_sendUpLeft(i_global, j_global)) {
                                chunk.send[6]++;
                            }
                            if (deviceMpi.in_sendLeft(i_global, j_global)) {
                                chunk.send[7]++;
                            }
                        }
                        else {
                            // Check buffers
                            if (deviceMpi.in_recvDownLeft(i_global, j_global)) {
                                chunk.recv[0]++;
                            }
                            else if (deviceMpi.in_recvDown(i_global, j_global)) {
                                chunk.recv[1]++;
                            }
                            else if (deviceMpi.in_recvDownRight(i_global, j_global)) {
                                chunk.recv[2]++;
                            }
                            else if (deviceMpi.in_recvRight(i_global, j_global)) {
                                chunk.recv[3]++;
                            }
                            else if (deviceMpi.in_recvUpRight(i_global, j_global)) {
                                chunk.recv[4]++;
                            }
                            else if (deviceMpi.in_recvUp(i_global, j_global)) {
                                chunk.recv[5]++;
                            }
                            else if (deviceMpi.in_recvUpLeft(i_global, j_global)) {
                                chunk.recv[6]++;
                            }
                            else if (deviceMpi.in_recvLeft(i_global, j_global)) {
                                chunk.recv[7]++;
                            }
                        }
                    }
                }
            , sizes);

            // Make sizes and start positions
            DualGrid.size = sizes.in;

            // Set MPI indices
            // TODO::Make single function for cleaner code like: mpi.set_sizes(sizes);
            hostMpi.send_size(0) = sizes.send[0];
            hostMpi.send_size(1) = sizes.send[1];
            hostMpi.send_size(2) = sizes.send[2];
            hostMpi.send_size(3) = sizes.send[3];
            hostMpi.send_size(4) = sizes.send[4];
            hostMpi.send_size(5) = sizes.send[5];
            hostMpi.send_size(6) = sizes.send[6];
            hostMpi.send_size(7) = sizes.send[7];

            hostMpi.recv_size(0) = sizes.recv[0];
            hostMpi.recv_size(1) = sizes.recv[1];
            hostMpi.recv_size(2) = sizes.recv[2];
            hostMpi.recv_size(3) = sizes.recv[3];
            hostMpi.recv_size(4) = sizes.recv[4];
            hostMpi.recv_size(5) = sizes.recv[5];
            hostMpi.recv_size(6) = sizes.recv[6];
            hostMpi.recv_size(7) = sizes.recv[7];

            hostMpi.send_start(0) = 0;
            hostMpi.send_start(1) = hostMpi.send_start(0) + hostMpi.send_size(0);
            hostMpi.send_start(2) = hostMpi.send_start(1) + hostMpi.send_size(1);
            hostMpi.send_start(3) = hostMpi.send_start(2) + hostMpi.send_size(2);
            hostMpi.send_start(4) = hostMpi.send_start(3) + hostMpi.send_size(3);
            hostMpi.send_start(5) = hostMpi.send_start(4) + hostMpi.send_size(4);
            hostMpi.send_start(6) = hostMpi.send_start(5) + hostMpi.send_size(5);
            hostMpi.send_start(7) = hostMpi.send_start(6) + hostMpi.send_size(6);
            DualMpi.send_totalSize = hostMpi.send_start(7) + hostMpi.send_size(7);

            hostMpi.recv_start(0) = 0;
            hostMpi.recv_start(1) = hostMpi.recv_start(0) + hostMpi.recv_size(0);
            hostMpi.recv_start(2) = hostMpi.recv_start(1) + hostMpi.recv_size(1);
            hostMpi.recv_start(3) = hostMpi.recv_start(2) + hostMpi.recv_size(2);
            hostMpi.recv_start(4) = hostMpi.recv_start(3) + hostMpi.recv_size(3);
            hostMpi.recv_start(5) = hostMpi.recv_start(4) + hostMpi.recv_size(4);
            hostMpi.recv_start(6) = hostMpi.recv_start(5) + hostMpi.recv_size(5);
            hostMpi.recv_start(7) = hostMpi.recv_start(6) + hostMpi.recv_size(6);
            DualMpi.recv_totalSize = hostMpi.recv_start(7) + hostMpi.recv_size(7);

            // Makes appropriately sized comms based on sizes defined above
            DualMpi.makeComms();

            // Make other views
            deviceMpi.sendIdx_view = uint32_deviceView(Kokkos::ViewAllocateWithoutInitializing("sendIdx_device"), DualMpi.send_totalSize);
            deviceMpi.recvIdx_view = uint32_deviceView(Kokkos::ViewAllocateWithoutInitializing("recvIdx_device"), DualMpi.recv_totalSize);
            deviceMpi.sendNeighbors_view = uint8_deviceView(Kokkos::ViewAllocateWithoutInitializing("sendNeighbors_device"), DualMpi.send_totalSize);
            deviceMpi.recvNeighbors_view = uint8_deviceView(Kokkos::ViewAllocateWithoutInitializing("recvNeighbors_device"), DualMpi.recv_totalSize);
            Kokkos::deep_copy(deviceMpi.sendInfo_view, hostMpi.sendInfo_view);
            Kokkos::deep_copy(deviceMpi.recvInfo_view, hostMpi.recvInfo_view);

            // Make new header and data
            Stork::Structs::RDF_Dual<float> RDF_trimmed;
            RDF_trimmed.template Make_Data_Views<device_space>(DualGrid.size);
            Stork::Structs::RDF_Data<float, device_space>& RDF_data_trimmed = RDF_trimmed.device_data;

            // Now construct vectors
            Kokkos::parallel_scan(
                "MPI Decompose (scan)",
                Kokkos::RangePolicy<device_space>(0, numEvents),
                KOKKOS_LAMBDA(const uint32_t n, Structs::MPISizes& chunk, bool isFinal)
                {
                    uint32_t ijk_global[3];
                    RDF_header.LOCAL_p_to_GLOBAL_ijk(ijk_global,RDF_data.p(n));

                    // TODO::SHIFT
                    const uint32_t i_global = ijk_global[0] + 1;
                    const uint32_t j_global = ijk_global[1] + 1;

                    // If in total domain (including buffer), then include the point
                    if (deviceMpi.in_total_domain(i_global, j_global)) {
                        // Check interior
                        if (deviceMpi.in_domain(i_global, j_global)) {
                            // Check sends
                            if (deviceMpi.in_sendDownLeft(i_global, j_global)) {
                                if (isFinal) {
                                    const uint32_t i = deviceMpi.send_start(0) + chunk.send[0];
                                    deviceMpi.sendIdx_view(i) = chunk.in;
                                    deviceMpi.sendNeighbors_view(i) = 0;
                                }
                                chunk.send[0]++;
                            }
                            if (deviceMpi.in_sendDown(i_global, j_global)) {
                                if (isFinal) {
                                    const uint32_t i = deviceMpi.send_start(1) + chunk.send[1];
                                    deviceMpi.sendIdx_view(i) = chunk.in;
                                    deviceMpi.sendNeighbors_view(i) = 1;
                                }
                                chunk.send[1]++;
                            }
                            if (deviceMpi.in_sendDownRight(i_global, j_global)) {
                                if (isFinal) {
                                    const uint32_t i = deviceMpi.send_start(2) + chunk.send[2];
                                    deviceMpi.sendIdx_view(i) = chunk.in;
                                    deviceMpi.sendNeighbors_view(i) = 2;
                                }
                                chunk.send[2]++;
                            }
                            if (deviceMpi.in_sendRight(i_global, j_global)) {
                                if (isFinal) {
                                    const uint32_t i = deviceMpi.send_start(3) + chunk.send[3];
                                    deviceMpi.sendIdx_view(i) = chunk.in;
                                    deviceMpi.sendNeighbors_view(i) = 3;
                                }
                                chunk.send[3]++;
                            }
                            if (deviceMpi.in_sendUpRight(i_global, j_global)) {
                                if (isFinal) {
                                    const uint32_t i = deviceMpi.send_start(4) + chunk.send[4];
                                    deviceMpi.sendIdx_view(i) = chunk.in;
                                    deviceMpi.sendNeighbors_view(i) = 4;
                                }
                                chunk.send[4]++;
                            }
                            if (deviceMpi.in_sendUp(i_global, j_global)) {
                                if (isFinal) {
                                    const uint32_t i = deviceMpi.send_start(5) + chunk.send[5];
                                    deviceMpi.sendIdx_view(i) = chunk.in;
                                    deviceMpi.sendNeighbors_view(i) = 5;
                                }
                                chunk.send[5]++;
                            }
                            if (deviceMpi.in_sendUpLeft(i_global, j_global)) {
                                if (isFinal) {
                                    const uint32_t i = deviceMpi.send_start(6) + chunk.send[6];
                                    deviceMpi.sendIdx_view(i) = chunk.in;
                                    deviceMpi.sendNeighbors_view(i) = 6;
                                }
                                chunk.send[6]++;
                            }
                            if (deviceMpi.in_sendLeft(i_global, j_global)) {
                                if (isFinal) {
                                    const uint32_t i = deviceMpi.send_start(7) + chunk.send[7];
                                    deviceMpi.sendIdx_view(i) = chunk.in;
                                    deviceMpi.sendNeighbors_view(i) = 7;
                                }
                                chunk.send[7]++;
                            }
                        }
                        else {
                            // Check buffers
                            if (deviceMpi.in_recvDownLeft(i_global, j_global)) {
                                if (isFinal) {
                                    const uint32_t i = deviceMpi.recv_start(0) + chunk.recv[0];
                                    deviceMpi.recvIdx_view(i) = chunk.in;
                                    deviceMpi.recvNeighbors_view(i) = 0;
                                }
                                chunk.recv[0]++;
                            }
                            else if (deviceMpi.in_recvDown(i_global, j_global)) {
                                if (isFinal) {
                                    const uint32_t i = deviceMpi.recv_start(1) + chunk.recv[1];
                                    deviceMpi.recvIdx_view(i) = chunk.in;
                                    deviceMpi.recvNeighbors_view(i) = 1;
                                }
                                chunk.recv[1]++;
                            }
                            else if (deviceMpi.in_recvDownRight(i_global, j_global)) {
                                if (isFinal) {
                                    const uint32_t i = deviceMpi.recv_start(2) + chunk.recv[2];
                                    deviceMpi.recvIdx_view(i) = chunk.in;
                                    deviceMpi.recvNeighbors_view(i) = 2;
                                }
                                chunk.recv[2]++;
                            }
                            else if (deviceMpi.in_recvRight(i_global, j_global)) {
                                if (isFinal) {
                                    const uint32_t i = deviceMpi.recv_start(3) + chunk.recv[3];
                                    deviceMpi.recvIdx_view(i) = chunk.in;
                                    deviceMpi.recvNeighbors_view(i) = 3;
                                }
                                chunk.recv[3]++;
                            }
                            else if (deviceMpi.in_recvUpRight(i_global, j_global)) {
                                if (isFinal) {
                                    const uint32_t i = deviceMpi.recv_start(4) + chunk.recv[4];
                                    deviceMpi.recvIdx_view(i) = chunk.in;
                                    deviceMpi.recvNeighbors_view(i) = 4;
                                }
                                chunk.recv[4]++;
                            }
                            else if (deviceMpi.in_recvUp(i_global, j_global)) {
                                if (isFinal) {
                                    const uint32_t i = deviceMpi.recv_start(5) + chunk.recv[5];
                                    deviceMpi.recvIdx_view(i) = chunk.in;
                                    deviceMpi.recvNeighbors_view(i) = 5;
                                }
                                chunk.recv[5]++;
                            }
                            else if (deviceMpi.in_recvUpLeft(i_global, j_global)) {
                                if (isFinal) {
                                    const uint32_t i = deviceMpi.recv_start(6) + chunk.recv[6];
                                    deviceMpi.recvIdx_view(i) = chunk.in;
                                    deviceMpi.recvNeighbors_view(i) = 6;
                                }
                                chunk.recv[6]++;
                            }
                            else if (deviceMpi.in_recvLeft(i_global, j_global)) {
                                if (isFinal) {
                                    const uint32_t i = deviceMpi.recv_start(7) + chunk.recv[7];
                                    deviceMpi.recvIdx_view(i) = chunk.in;
                                    deviceMpi.recvNeighbors_view(i) = 7;
                                }
                                chunk.recv[7]++;
                            }
                        }
                        // If final run, populate trimmed RDF too
                        if (isFinal){
                            RDF_data_trimmed.p(chunk.in) = RDF_data.p(n);
                            RDF_data_trimmed.tm(chunk.in) = RDF_data.tm(n);
                            RDF_data_trimmed.tl(chunk.in) = RDF_data.tl(n);
                            RDF_data_trimmed.cr(chunk.in) = RDF_data.cr(n);
                        }
                        chunk.in++;
                    }
                }
            );

            // Update RDF data
            RDF_data = RDF_data_trimmed;

            // TODO::SHIFT MPI BOUNDS BY 1?
            hostMpi.imin_global() -= 1;
            hostMpi.imax_global() -= 1;
            hostMpi.jmin_global() -= 1;
            hostMpi.jmax_global() -= 1;
            Kokkos::deep_copy(DualMpi.deviceMpi.mpiIndices_view, DualMpi.hostMpi.mpiIndices_view);

            // TODO::DEBUG Check bounds
            CheckSendRecvWithNeighbor(DualMpi);
        }
    }

    // Custom Pairs for sorting times when places on temporary grid
    namespace impl{
        // Custom Pairs for sorting times when places on temporary grid
        struct customPair{
            uint32_t p = UINT32_MAX;
            float t = FLT_MAX;
        };

        struct customPairComparator {
            KOKKOS_FUNCTION constexpr bool operator()(const customPair& a, const customPair& b) const {
                return a.t <= b.t; //a precedes b if a is smaller
            }
        };
    }

    template<template<typename> class GrainID>
    void Grid(Structs::Grid_Dual<GrainID>& DualGrid, Structs::Substrate_Dual<GrainID>& DualSub, Structs::Sim_Dual& DualSim,  Structs::Mpi_Dual<GrainID>& DualMpi, Stork::Structs::RDF_Dual<float>& RDF){

        ////////////////////////////////////////////////////////
        // MPI Decomposition and Communication Initialization //
        ////////////////////////////////////////////////////////
        using eventPointer_deviceView = Kokkos::View<impl::customPair**, layout, device_memory>;

        // Usings
        using Common::self;
        using std::max;

        impl::MPIDecomposition(DualGrid, DualMpi, RDF);

        ///////////////////////////////////
        // Allocate and Initialize Views //
        ///////////////////////////////////

        // Grid Size
        const uint32_t numEvents = DualGrid.size;

        // Allocate and Initialize the views
        DualGrid.AllocateAndInitializeViews();

        // References
        auto& deviceGrid = DualGrid.deviceGrid;
        Stork::Structs::RegularGrid_Header<float, device_space>& RDF_header = RDF.device_header;
        Stork::Structs::RDF_Data<float, device_space>& RDF_data = RDF.device_data;
        uint32_hostView hostExtent(Kokkos::ViewAllocateWithoutInitializing("grid_extent_host"), 7);

        // Bounds and extents
        //// Substrate for x-y it may be MPI decomposed in x-y
        //// Layer information for z layer for z since it is not decomposed in z (and substrate extends farther)
        hostExtent(0) = DualSub.hostSub.xnum();
        hostExtent(1) = DualSub.hostSub.ynum();
        hostExtent(2) = RDF.host_header.local_knum();
        hostExtent(3) = 2;    // Minimum is 1 point + buffer

        // Shifts for correct local <i,j> placement after a potential decomposition
        hostExtent(4) = static_cast<uint32_t>(DualSub.header.global_i0()-RDF.host_header.global_i0());
        hostExtent(5) = static_cast<uint32_t>(DualSub.header.global_j0()-RDF.host_header.global_j0());

        // Shift for placing points on substrate correctly
        hostExtent(6) = static_cast<uint32_t>((-RDF.host_header.global_z0() / RDF.host_header.gridResolution()) + 0.5);

        // Adjusted to make buffers (so no bad indexing)
        hostExtent(0) += 2;
        hostExtent(1) += 2;
        hostExtent(2) += 2;

        // Copy to device
        Kokkos::deep_copy(deviceGrid.extent_view, hostExtent);

        // Make layer size in 3D
        const uint32_t layerSize3D = hostExtent(0) * hostExtent(1) * hostExtent(2);

        /////////////////////////
        // Grid Initialization //
        /////////////////////////

        // Make temp view for each number of events per 3D location, set grid.tnum() to be 1 over that (as a buffer)
        uint8_deviceView tNum("tNum", layerSize3D);
        Kokkos::parallel_for(
            "Grid Init (Set Info and Find tNum)",
            Kokkos::RangePolicy<device_space>(0, numEvents),
            KOKKOS_LAMBDA(const uint32_t n)
            {
                const uint32_t& LOCAL_p = RDF_data.p(n);
                uint32_t LOCAL_ijk[3];
                RDF_header.LOCAL_p_to_LOCAL_ijk(LOCAL_ijk, LOCAL_p);
                // Start ijk with +1 for buffer
                deviceGrid.i(n) = (LOCAL_ijk[0]-deviceGrid.xShift()) + 1;
                deviceGrid.j(n) = (LOCAL_ijk[1]-deviceGrid.yShift()) + 1;
                deviceGrid.k(n) = LOCAL_ijk[2] + 1;    // Z is not shifted b/c only ever 2D decomposition
                // Set solidification information
                deviceGrid.tm(n) = RDF_data.tm(n);
                deviceGrid.tl(n) = RDF_data.tl(n);
                deviceGrid.cr(n) = RDF_data.cr(n);
                const uint32_t pt3D = Utility::ijk_to_p(deviceGrid.i(n), deviceGrid.j(n), deviceGrid.k(n), deviceGrid.xnum(), deviceGrid.ynum(), deviceGrid.znum());
                // Atomically add event for that 3D point number
                Kokkos::atomic_add(&tNum(pt3D), 1);
                // Only atomic_max if necessary
                // Note: Quicker than reduce because rarely encountered
                // Note: The "+1" is so we have a buffer of 1
                if (deviceGrid.tnum() < tNum(pt3D) + 1) {
                    //printf("DEBUG 2: %u -> <%u,%u,%u> / <%u,%u,%u> ||| <%u,%u,%u> / <%u,%u,%u> -> %u ||| <%u,%u>\n", LOCAL_p, LOCAL_ijk[0], LOCAL_ijk[1], LOCAL_ijk[2], RDF_header.local_inum(), RDF_header.local_jnum(), RDF_header.local_knum(), deviceGrid.i(n), deviceGrid.j(n), deviceGrid.k(n), deviceGrid.xnum(), deviceGrid.ynum(), deviceGrid.znum(), pt3D, deviceGrid.tnum(), tNum(pt3D));
                    Kokkos::atomic_max(&deviceGrid.tnum(), tNum(pt3D) + 1);
                    //printf("DEBUG 3: %u -> <%u,%u,%u> / <%u,%u,%u> ||| <%u,%u,%u> / <%u,%u,%u> -> %u ||| <%u,%u>\n", LOCAL_p, LOCAL_ijk[0], LOCAL_ijk[1], LOCAL_ijk[2], RDF_header.local_inum(), RDF_header.local_jnum(), RDF_header.local_knum(), deviceGrid.i(n), deviceGrid.j(n), deviceGrid.k(n), deviceGrid.xnum(), deviceGrid.ynum(), deviceGrid.znum(), pt3D, deviceGrid.tnum(), tNum(pt3D));
                }
            }
        );

        // Copy updated tnum to host for the temporary 4D pointer grid allocation.
        Kokkos::deep_copy(hostExtent, deviceGrid.extent_view);

        // Make 4D grid of pointer information
        eventPointer_deviceView ptrGridX("eventPointer", layerSize3D, hostExtent(3));

        // Make 4D grid point
        Kokkos::deep_copy(tNum, 0);
        Kokkos::parallel_for(
            "Grid Init (4D Grid)",
            Kokkos::RangePolicy<device_space>(0, numEvents),
            KOKKOS_LAMBDA(const uint32_t p)
            {
                const uint32_t p3D = Utility::ijk_to_p(deviceGrid.i(p), deviceGrid.j(p), deviceGrid.k(p), deviceGrid.xnum(), deviceGrid.ynum(), deviceGrid.znum());
                const impl::customPair temp = {p, deviceGrid.tm(p)};
                ptrGridX(p3D, Kokkos::atomic_fetch_add(&tNum(p3D),1)) = temp;
            }
        );

        // Sort 4D grid
        Kokkos::parallel_for(
            "Grid Init (Sort 4D Grid)",
            Kokkos::RangePolicy<device_space>(0, layerSize3D),
            KOKKOS_LAMBDA(const uint32_t p3D)
            {
                if (tNum(p3D)>1){
                    for (uint8_t i = 0; i<tNum(p3D);i++){
                        for (uint8_t j = (i+1); j<tNum(p3D);j++){
                            // If greater, swap
                            if (ptrGridX(p3D, i).t > ptrGridX(p3D, j).t) {
                                const uint32_t p_temp = ptrGridX(p3D,j).p;
                                const float t_temp = ptrGridX(p3D,j).t;
                                ptrGridX(p3D,j).p = ptrGridX(p3D,i).p;
                                ptrGridX(p3D,j).t = ptrGridX(p3D,i).t;
                                ptrGridX(p3D,i).p = p_temp;
                                ptrGridX(p3D,i).t = t_temp;
                            }
                        }
                    }
                }
                // Set the event numbers
                for (uint l = 0; l < tNum(p3D); l++) {
                    const uint32_t p = ptrGridX(p3D, l).p;
                    if (p != UINT32_MAX) {
                        deviceGrid.l(p) = l;
                    }
                }
            }
        );

        // Now use temporary grid to find spatiotemporal neighbors and edges
        Kokkos::parallel_for(
            "Grid Init (Spatiotemporal Neighbors & Edges)",
            Kokkos::RangePolicy<device_space>(0, 27*numEvents),
            KOKKOS_LAMBDA(const uint32_t i)
            {
                const uint32_t p = i/27;
                const uint8_t n = i%27;

                if (n == self) {
                    // Set neighbor to next event at location
                    const uint32_t pN_3D = Utility::ijk_to_p(deviceGrid.i(p), deviceGrid.j(p), deviceGrid.k(p), deviceGrid.xnum(), deviceGrid.ynum(), deviceGrid.znum());
                    const uint32_t pN = ptrGridX(pN_3D, deviceGrid.l(p)+1).p;
                    deviceGrid.last(p) = (pN == UINT32_MAX);
                    deviceGrid.N(p,n) = pN;
                }
                else {
                    // Find position of neighbor
                    const int8_t dx = ((n / 3) / 3) - 1;
                    const int8_t dy = ((n / 3) % 3) - 1;
                    const int8_t dz = (n % 3) - 1;
                    const uint32_t pN_3D = Utility::ijk_to_p(deviceGrid.i(p) + dx,deviceGrid.j(p) + dy, deviceGrid.k(p) + dz, deviceGrid.xnum(), deviceGrid.ynum(), deviceGrid.znum());

                    // get neighbor cell's reference
                    uint32_t pN = UINT32_MAX;
                    for (uint8_t l = 0; l < tNum(pN_3D); l++) {
                        const uint32_t pN_temp = ptrGridX(pN_3D, l).p;
                        const float tN_temp = ptrGridX(pN_3D, l).t;
                        if (tN_temp < deviceGrid.tl(p)) {
                            pN = pN_temp;
                        }
                    }
                    deviceGrid.N(p,n) = pN;

                    // Find if edge
                    if (n == 4 || n == 10 || n == 12 || n == 16 || n == 22) {
                        if (pN == UINT32_MAX || deviceGrid.tl(p) < deviceGrid.tm(pN) || deviceGrid.tl(pN) < deviceGrid.tm(p)) {
                            deviceGrid.edge(p) = true;
                            deviceGrid.cantCap(p) = true;
                        }
                    }
                }
            }
        );

        // Do quick initialization routines for fewer checks in main loop
        Kokkos::parallel_for(
            "Grid Init (Safety)",
            Kokkos::RangePolicy<device_space>(0, 27*numEvents),
            KOKKOS_LAMBDA(const uint32_t i)
            {
                const uint32_t p = i/27;
                const uint8_t n = i%27;
                const uint32_t pN = deviceGrid.N(p,n);
                // Make it so can only send event to self edges
                if (pN != UINT32_MAX && n == self && deviceGrid.edge(pN) == false) {
                    deviceGrid.N(p,n) = UINT32_MAX;
                }
                // If neighbor is valid doesn't point back, set to be invalid
                else if (pN != UINT32_MAX && n != self && deviceGrid.N(pN, Utility::invert_n(n)) != p){
                    deviceGrid.N(p,n) = UINT32_MAX;
                }
            }
        );

        // Make MPI walls which can't nucleate or be captured
        Structs::Mpi<device_space>& deviceMpi = DualMpi.deviceMpi;
        Kokkos::parallel_for(
            "Grid Init (MPI Walls)",
            Kokkos::RangePolicy<device_space>(0, DualMpi.recv_totalSize),
            KOKKOS_LAMBDA(const uint32_t i)
            {
                const uint32_t p = deviceMpi.recvIdx_view(i);
                deviceGrid.MpiWall(p) = true;
                deviceGrid.cantCap(p) = true;
                deviceGrid.N(p,self) = UINT32_MAX;
            }
        );

        // TODO::UNNECESSARY IF CODE IS IMPROVED
        // Make everything correct placement (for transferring grainIDs to and from substrate)
        Kokkos::parallel_for(
            "Grid Init (Reset Indexing)",
            Kokkos::RangePolicy<device_space>(0, numEvents),
            KOKKOS_LAMBDA(const uint32_t p)
            {
                deviceGrid.i(p)--;
                deviceGrid.j(p)--;
                deviceGrid.k(p)--;
            }
        );

    }
}
