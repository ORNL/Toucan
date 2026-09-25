#pragma once

#include "Definitions.hpp"
#include "Common.hpp"

#include "Structs/Grid.hpp"
#include "Structs/SteeringVector.hpp"
#include "Comms/Structs.hpp"

namespace Toucan::Structs{

    // To determine if at least one value in an array is true
    namespace impl{
        template<typename T>
        bool anyTrue(const T arr[], int size) {
            for (int i = 0; i < size; ++i) {
                if (arr[i]) {
                    return true; // If any element is true, return false immediately
                }
            }
            return false; // If all elements are false, return true
        }
    }

    // Pack Buffers to Send Info
    template<template<typename> class GrainID>
    void Mpi_PackBuffers(Grid_Dual<GrainID>& DualGrid, Mpi_Dual<GrainID>& DualMpi){

        // Reference to device space
        auto& grid = DualGrid.deviceGrid;
        Mpi<device_space>& deviceMpi = DualMpi.deviceMpi;

        // Find how many to send to each neighbor
        Kokkos::parallel_reduce(
            "MPI (Pack) - Get Sizes",
            Kokkos::RangePolicy<device_space>(0, DualMpi.send_totalSize),
            KOKKOS_LAMBDA(const uint32_t idx, NeighborInts& th_sizes)
            {
                // Get point location
                const uint32_t p = deviceMpi.sendIdx_view(idx);
                // Get neighbor
                const uint8_t n = deviceMpi.sendNeighbors_view(idx);
                // Get offset for this neighbor
                const uint32_t i = idx-deviceMpi.send_start(n);
                // Set memory
                if (grid.inSteer(p)<69){
                    // Reset to zero (for later usage)
                    DualMpi.atomicNumSend(n) = 0;
                    // Increment counter
                    th_sizes.x[n]++;
                }
            }
        , DualMpi.numSend);

        // Update send pointers based on size
        for (uint8_t n=0;n<8;n++){
            DualMpi.updateSendPointers(DualMpi.numSend.x[n], n);
        }

        // Pack send vectors
        Kokkos::parallel_for(
            "MPI (Pack) - Construct",
            Kokkos::RangePolicy<device_space>(0, DualMpi.send_totalSize),
            KOKKOS_LAMBDA(const uint32_t idx)
            {
                // Get point location
                const uint32_t p = deviceMpi.sendIdx_view(idx);
                // Get neighbor
                const uint8_t n = deviceMpi.sendNeighbors_view(idx);
                // Get offset for this neighbor
                const uint32_t i = idx-deviceMpi.send_start(n);
                // Set memory
                if (grid.inSteer(p)<69){
                    const uint32_t j = Kokkos::atomic_fetch_add(&DualMpi.atomicNumSend(n), 1);
                    size_t uint32_offset = Mpi_Dual<GrainID>::mpi_uint32_per_cell * j;
                    size_t float_offset = Mpi_Dual<GrainID>::mpi_float_per_cell * j;
                    size_t uint16_offset = Mpi_Dual<GrainID>::mpi_uint16_per_cell * j;
                    size_t uint8_offset = Mpi_Dual<GrainID>::mpi_uint8_per_cell * j;
                    DualMpi.send_ptrUint32[n][uint32_offset++] = i;
                    DualMpi.send_ptrFloat[n][float_offset++] = grid.gx(p);
                    DualMpi.send_ptrFloat[n][float_offset++] = grid.gy(p);
                    DualMpi.send_ptrFloat[n][float_offset++] = grid.gz(p);
                    DualMpi.send_ptrFloat[n][float_offset++] = grid.G0(p);
                    DualMpi.send_ptrFloat[n][float_offset++] = grid.tCap(p) - grid.tl(p);
                    DualMpi.send_ptrUint16[n][uint16_offset++] = grid.dirID_org(p);
                    DualMpi.send_ptrUint16[n][uint16_offset++] = grid.dirID(p);
                    DualMpi.send_ptrUint8[n][uint8_offset++] = grid.inSteer(p);
                    GrainID<device_space>::pack_mpi(
                        grid.grainID_org,
                        grid.grainID,
                        p,
                        DualMpi.send_ptrUint32[n],
                        uint32_offset,
                        DualMpi.send_ptrFloat[n],
                        float_offset,
                        DualMpi.send_ptrUint16[n],
                        uint16_offset,
                        DualMpi.send_ptrUint8[n],
                        uint8_offset
                    );
                }
            }
        );
    }

    // Upack Buffers and Add to Steering Vector
    template<template<typename> class GrainID>
    void Mpi_UnpackBuffers(Grid_Dual<GrainID>& DualGrid, Mpi_Dual<GrainID>& DualMpi, SteeringVector& steer, const uint8_t n){

        // Reference to device space
        auto& grid = DualGrid.deviceGrid;
        Mpi<device_space>& deviceMpi = DualMpi.deviceMpi;

        // Unpack send vectors
        Kokkos::parallel_for(
            "MPI (UnPack)",
            Kokkos::RangePolicy<device_space>(0, DualMpi.numRecv.x[n]),
            KOKKOS_LAMBDA(const uint32_t j)
            {
                // Get recv point location in send vector
                size_t uint32_offset = Mpi_Dual<GrainID>::mpi_uint32_per_cell * j;
                size_t float_offset = Mpi_Dual<GrainID>::mpi_float_per_cell * j;
                size_t uint16_offset = Mpi_Dual<GrainID>::mpi_uint16_per_cell * j;
                size_t uint8_offset = Mpi_Dual<GrainID>::mpi_uint8_per_cell * j;
                const uint32_t i = DualMpi.recv_ptrUint32[n][uint32_offset++];

                // Get point location
                const uint32_t p = deviceMpi.recvIdx_view(i+deviceMpi.recv_start(n));

                // Set memory
                grid.gx(p) = DualMpi.recv_ptrFloat[n][float_offset++];
                grid.gy(p) = DualMpi.recv_ptrFloat[n][float_offset++];
                grid.gz(p) = DualMpi.recv_ptrFloat[n][float_offset++];
                grid.G0(p) = DualMpi.recv_ptrFloat[n][float_offset++];
                grid.tCap(p) = grid.tl(p) + DualMpi.recv_ptrFloat[n][float_offset++] + 1e-6; //Small delta to prevent looping
                grid.dirID_org(p) = DualMpi.recv_ptrUint16[n][uint16_offset++];
                grid.dirID(p) = DualMpi.recv_ptrUint16[n][uint16_offset++];
                grid.inSteer(p) = DualMpi.recv_ptrUint8[n][uint8_offset++];
                GrainID<device_space>::unpack_mpi(
                    grid.grainID_org,
                    grid.grainID,
                    p,
                    DualMpi.recv_ptrUint32[n],
                    uint32_offset,
                    DualMpi.recv_ptrFloat[n],
                    float_offset,
                    DualMpi.recv_ptrUint16[n],
                    uint16_offset,
                    DualMpi.recv_ptrUint8[n],
                    uint8_offset
                );
                steer.push_back(p);
            }
        );
        steer.size += DualMpi.numRecv.x[n];
    }

    template<template<typename> class GrainID>
    void Mpi_SendRecvAndUnpack(Grid_Dual<GrainID>& DualGrid, Mpi_Dual<GrainID>& DualMpi, SteeringVector& steer){

        // Comm requests
        MPI_Request sendSizeRequests[8] = {
            MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL,
            MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL
        };
        MPI_Request sendRequests[8] = {
            MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL,
            MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL
        };
        MPI_Request recvSizeRequests[8];
        MPI_Request recvRequests[8];

        // Flags for if something should be received
        bool iRecvSizeFlags[8];
        bool iRecvFlags[8];
        bool recvFlags[8];
        for (int n = 0; n < 8; n++){
            iRecvSizeFlags[n] = (DualMpi.neighbor_ranks[n] != MPI_PROC_NULL);
            iRecvFlags[n] = (DualMpi.neighbor_ranks[n] != MPI_PROC_NULL);
            recvFlags[n] = (DualMpi.neighbor_ranks[n] != MPI_PROC_NULL);
        }

        if (DualMpi.deviceComms) {
            // Now do nonblocking sends
            for (int n = 0; n < 8; n++){
                if (DualMpi.neighbor_ranks[n] != MPI_PROC_NULL){
                    // Send the Buffer Size (non-blocking)
                    MPI_Isend(&DualMpi.numSend.x[n], 1, MPI_UINT32_T, DualMpi.neighbor_ranks[n], 0, DualMpi.comm, &sendSizeRequests[n]);
                    // Send the Buffer (non-blocking)
                    MPI_Isend(DualMpi.send_ptrUint32[n], DualMpi.byteMult*DualMpi.numSend.x[n], MPI_CHAR, DualMpi.neighbor_ranks[n], 1, DualMpi.comm, &sendRequests[n]);
                }
            }

            // Loop Until All Messages Received
            while (impl::anyTrue(recvFlags,8)) {
                // Loop over neighbors
                for (int n=0; n<8; n++){
                    if (iRecvSizeFlags[n]){
                        // Set to not do again
                        iRecvSizeFlags[n] = false;
                        // Receive the Buffer Size (non-blocking)
                        MPI_Irecv(&DualMpi.numRecv.x[n], 1, MPI_UINT32_T, DualMpi.neighbor_ranks[n], 0, DualMpi.comm, &recvSizeRequests[n]);
                    }
                    else if (iRecvFlags[n]){
                        int flag;
                        MPI_Test(&recvSizeRequests[n], &flag, MPI_STATUS_IGNORE);
                        if (flag) {
                            // Set to not do again
                            iRecvFlags[n] = false;
                            // Receive the Buffer (non-blocking)
                            MPI_Irecv(DualMpi.recv_ptrUint32[n], DualMpi.byteMult*DualMpi.numRecv.x[n], MPI_CHAR, DualMpi.neighbor_ranks[n], 1, DualMpi.comm, &recvRequests[n]);
                            // Update device recv pointers
                            DualMpi.updateRecvPointers(DualMpi.numRecv.x[n],n);
                        }
                    }
                    else if (recvFlags[n]){
                        int flag;
                        MPI_Test(&recvRequests[n], &flag, MPI_STATUS_IGNORE);
                        if (flag) {
                            // Set to not do again
                            recvFlags[n] = false;
                            // Unpack Buffers
                            Mpi_UnpackBuffers(DualGrid,DualMpi,steer,n);
                        }
                    }
                }
            }
        }
        else {
            // Now do nonblocking sends and receives
            for (int n = 0; n < 8; n++){
                if (DualMpi.neighbor_ranks[n] != MPI_PROC_NULL){
                    // Create subviews
                    char_hostSubView send_subview_host(reinterpret_cast<char*>(DualMpi.send_ptrUint32_host[n]), DualMpi.byteMult*DualMpi.numSend.x[n]);
                    char_deviceSubView send_subview_device(reinterpret_cast<char*>(DualMpi.send_ptrUint32[n]), DualMpi.byteMult*DualMpi.numSend.x[n]);
                    // Deep copy subviews from device to host
                    Kokkos::deep_copy(send_subview_host, send_subview_device);
                    // Send the Buffer Size (non-blocking)
                    MPI_Isend(&DualMpi.numSend.x[n], 1, MPI_UINT32_T, DualMpi.neighbor_ranks[n], 0, DualMpi.comm, &sendSizeRequests[n]);
                    // Send the Buffer (non-blocking)
                    MPI_Isend(DualMpi.send_ptrUint32_host[n], DualMpi.byteMult*DualMpi.numSend.x[n], MPI_CHAR, DualMpi.neighbor_ranks[n], 1, DualMpi.comm, &sendRequests[n]);
                }
            }

            // Loop Until All Messages Received
            while (impl::anyTrue(recvFlags,8)) {
                // Loop over neighbors
                for (int n=0; n<8; n++){
                    if (iRecvSizeFlags[n]){
                        // Set to not do again
                        iRecvSizeFlags[n] = false;
                        // Receive the Buffer Size (non-blocking)
                        MPI_Irecv(&DualMpi.numRecv.x[n], 1, MPI_UINT32_T, DualMpi.neighbor_ranks[n], 0, DualMpi.comm, &recvSizeRequests[n]);
                    }
                    else if (iRecvFlags[n]){
                        int flag;
                        MPI_Test(&recvSizeRequests[n], &flag, MPI_STATUS_IGNORE);
                        if (flag) {
                            // Set to not do again
                            iRecvFlags[n] = false;
                            // Receive the Buffer (non-blocking)
                            MPI_Irecv(DualMpi.recv_ptrUint32_host[n], DualMpi.byteMult*DualMpi.numRecv.x[n], MPI_CHAR, DualMpi.neighbor_ranks[n], 1, DualMpi.comm, &recvRequests[n]);
                            // Update device recv pointers
                            DualMpi.updateRecvPointers(DualMpi.numRecv.x[n],n);
                        }
                    }
                    else if (recvFlags[n]){
                        int flag;
                        MPI_Test(&recvRequests[n], &flag, MPI_STATUS_IGNORE);
                        if (flag) {
                            // Set to not do again
                            recvFlags[n] = false;
                            // Make subviews
                            char_hostSubView recv_subview_host(reinterpret_cast<char*>(DualMpi.recv_ptrUint32_host[n]), DualMpi.byteMult*DualMpi.numRecv.x[n]);
                            char_deviceSubView recv_subview_device(reinterpret_cast<char*>(DualMpi.recv_ptrUint32[n]), DualMpi.byteMult*DualMpi.numRecv.x[n]);
                            // Copy data to device
                            Kokkos::deep_copy(recv_subview_device, recv_subview_host);
                            // Unpack Buffers
                            Mpi_UnpackBuffers(DualGrid,DualMpi,steer,n);
                        }
                    }
                }
            }
        }

        // Fence after all receives
        Kokkos::fence();

        // Ensure all sends are completed before proceeding
        MPI_Waitall(8, sendSizeRequests, MPI_STATUSES_IGNORE);
        MPI_Waitall(8, sendRequests, MPI_STATUSES_IGNORE);
    }

    // Communicate information to neighbors
    template<template<typename> class GrainID>
    void Mpi_Synchronize(Grid_Dual<GrainID>& DualGrid, Mpi_Dual<GrainID>& DualMpi, SteeringVector& steer){

        // Dont do anything if only process
        if (DualMpi.nproc == 1) {
            return;
        }

        // Pack Buffers
        Mpi_PackBuffers(DualGrid,DualMpi);

        // Send, Receive, and Unpack Buffers
        Mpi_SendRecvAndUnpack(DualGrid,DualMpi,steer);
    }
}


// // [BLOCK 1]
// // Now do nonblocking sends and receives
// for (int n = 0; n < 8; n++){
//     if (DualMpi.neighbor_ranks[n] != MPI_PROC_NULL){
//         // Create subviews
//         char_hostSubView send_subview_host(reinterpret_cast<char*>(DualMpi.send_ptrIdx_host[n]), DualMpi.byteMult*DualMpi.numSend.x[n]);
//         char_deviceSubView send_subview_device(reinterpret_cast<char*>(DualMpi.send_ptrIdx[n]), DualMpi.byteMult*DualMpi.numSend.x[n]);
//         // Deep copy subviews from device to host
//         Kokkos::deep_copy(send_subview_host, send_subview_device);
//         // Send the Buffer Size (non-blocking)
//         MPI_Isend(&DualMpi.numSend.x[n], 1, MPI_UINT32_T, DualMpi.neighbor_ranks[n], 0, DualMpi.comm, &sendSizeRequests[n]);
//         // Send the Buffer (non-blocking)
//         MPI_Isend(DualMpi.send_ptrIdx_host[n], DualMpi.byteMult*DualMpi.numSend.x[n], MPI_CHAR, DualMpi.neighbor_ranks[n], 1, DualMpi.comm, &sendRequests[n]);
//     }
// }

// // [BLOCK 2]
// // Loop Until All Messages Received
// while (impl::anyTrue(recvFlags,8)) {
//     // Loop over neighbors
//     for (int n=0; n<8; n++){
//         // [BLOCK 2.1]
//         if (iRecvSizeFlags[n]){
//             // Set to not do again
//             iRecvSizeFlags[n] = false;
//             // Receive the Buffer Size (non-blocking)
//             MPI_Irecv(&DualMpi.numRecv.x[n], 1, MPI_UINT32_T, DualMpi.neighbor_ranks[n], 0, DualMpi.comm, &recvSizeRequests[n]);
//         }
//         // [BLOCK 2.2]
//         else if (iRecvFlags[n]){
//             int flag;
//             MPI_Test(&recvSizeRequests[n], &flag, MPI_STATUS_IGNORE);
//             if (flag) {
//                 // Set to not do again
//                 iRecvFlags[n] = false;
//                 // Receive the Buffer (non-blocking)
//                 MPI_Irecv(DualMpi.recv_ptrIdx_host[n], DualMpi.byteMult*DualMpi.numRecv.x[n], MPI_CHAR, DualMpi.neighbor_ranks[n], 1, DualMpi.comm, &recvRequests[n]);
//                 // Update device recv pointers
//                 DualMpi.updateRecvPointers(DualMpi.numRecv.x[n],n);
//             }
//         }
//         // [BLOCK 2.3]
//         else if (recvFlags[n]){
//             int flag;
//             MPI_Test(&recvRequests[n], &flag, MPI_STATUS_IGNORE);
//             if (flag) {
//                 // Set to not do again
//                 recvFlags[n] = false;
//                 // Make subviews
//                 char_hostSubView recv_subview_host(reinterpret_cast<char*>(DualMpi.recv_ptrIdx_host[n]), DualMpi.byteMult*DualMpi.numRecv.x[n]);
//                 char_deviceSubView recv_subview_device(reinterpret_cast<char*>(DualMpi.recv_ptrIdx[n]), DualMpi.byteMult*DualMpi.numRecv.x[n]);
//                 // Copy data to device
//                 Kokkos::deep_copy(recv_subview_device, recv_subview_host);
//                 // Unpack Buffers
//                 Mpi_UnpackBuffers(DualGrid,DualMpi,steer,n);
//             }
//         }
//     }
// }
