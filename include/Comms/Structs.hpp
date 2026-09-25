#pragma once

#include "Definitions.hpp"
#include "Structs/Sim.hpp"
#include "Structs/Layer.hpp"
#include "Structs/Grid.hpp"
#include "Structs/Orientations.hpp"

/////////////
// Utility //
/////////////

namespace Toucan::Structs {

    // Copy global event info into just the mpi-decomposed local event info
    namespace impl {
        KOKKOS_INLINE_FUNCTION
        void copyEventData_device(const float_deviceView& globalEventInfo, const uint32_t p, const float_deviceView& eventInfo, uint32_t& chunk, const bool isFinal)
        {
            if (isFinal) {
                const uint32_t i = 6*chunk;
                const uint32_t j = 6*p;
                eventInfo(i)=globalEventInfo(j);
                eventInfo(i+1)=globalEventInfo(j+1);
                eventInfo(i+2)=globalEventInfo(j+2);
                eventInfo(i+3)=globalEventInfo(j+3);
                eventInfo(i+4)=globalEventInfo(j+4);
                eventInfo(i+5)=globalEventInfo(j+5);
            }
            chunk++;
        }
    }

    struct NeighborInts {
        uint32_t x[8];
        KOKKOS_FUNCTION
        NeighborInts& operator+=(const NeighborInts& other) {
            for (int i = 0; i < 8; ++i) {
                x[i] += other.x[i];
            }
            return *this;
        }
    };

    struct MPISizes {
        uint32_t in;
        uint32_t send[8];
        uint32_t recv[8];
        KOKKOS_FUNCTION
        MPISizes& operator+=(const MPISizes& other) {
            in += other.in;
            for (int i = 0; i < 8; ++i) {
                send[i] += other.send[i];
                recv[i] += other.recv[i];
            }
            return *this;
        }
    };

    template<typename memory_space>
    class Mpi {
    private:
        /////////////////////////////////////////
        //// Set quick access to views types ////
        /////////////////////////////////////////

        using uint8_view = Kokkos::View<uint8_t*, layout, memory_space>;
        using uint32_view = Kokkos::View<uint32_t*, layout, memory_space>;
        using float_view = Kokkos::View<float*, layout, memory_space>;
    public:

        uint32_view sendIdx_view;
        uint32_view recvIdx_view;

        uint32_view sendInfo_view;  // TODO::COMPILE TIME DIMENSIONS
        uint32_view recvInfo_view;

        uint8_view sendNeighbors_view;
        uint8_view recvNeighbors_view;

        uint32_view mpiIndices_view;    // Size 4
        float_view mpiFloats_view;      // Size 1

        uint32_view numSend;
        uint32_view numRecv;

        //////////////////////
        // Start Pointers  ///
        //////////////////////
        KOKKOS_INLINE_FUNCTION
        uint32_t& send_start(const uint32_t i) const {
            return sendInfo_view(2*i);
        }

        KOKKOS_INLINE_FUNCTION
        uint32_t& send_size(const uint32_t i) const {
            return sendInfo_view(2*i+1);
        }

        KOKKOS_INLINE_FUNCTION
        uint32_t& recv_start(const uint32_t i) const {
            return recvInfo_view(2*i);
        }

        KOKKOS_INLINE_FUNCTION
        uint32_t& recv_size(const uint32_t i) const {
            return recvInfo_view(2*i+1);
        }

        //////////////////////
        // Start MPI Floats //
        //////////////////////

        KOKKOS_INLINE_FUNCTION
        uint32_t& imin_global() const {
            return mpiIndices_view(0);
        }

        KOKKOS_INLINE_FUNCTION
        uint32_t& imax_global() const {
            return mpiIndices_view(1);
        }

        KOKKOS_INLINE_FUNCTION
        uint32_t& jmin_global() const {
            return mpiIndices_view(2);
        }

        KOKKOS_INLINE_FUNCTION
        uint32_t& jmax_global() const {
            return mpiIndices_view(3);
        }

        KOKKOS_INLINE_FUNCTION
        float& res() const {
            return mpiFloats_view(0);
        }

        //////////////////////
        // 2D Region Checks //
        //////////////////////

        // Exclusively in the interior of the domain
        KOKKOS_INLINE_FUNCTION
        bool in_interior_domain(const uint32_t i, const uint32_t j) const {
            return (in_X_interior(i) && in_Y_interior(j));
        }

        // Is the point in the domain, which will be changed by this rank
        KOKKOS_INLINE_FUNCTION
        bool in_domain(const uint32_t i, const uint32_t j) const {
            return (in_X(i) && in_Y(j));
        }

        KOKKOS_INLINE_FUNCTION
        // Is the point in the extended domain (including buffer)
        bool in_total_domain(const uint32_t i, const uint32_t j) const {
            return (in_X_buff(i) && in_Y_buff(j));
        }

        // For sending stuff
        KOKKOS_INLINE_FUNCTION
        bool in_sendUpLeft(const uint32_t i, const uint32_t j) const {
            return (in_left_send_buffer(i) && in_up_send_buffer(j));
        }

        KOKKOS_INLINE_FUNCTION
        bool in_sendUp(const uint32_t i, const uint32_t j) const {
            return (in_X(i) && in_up_send_buffer(j));
        }

        KOKKOS_INLINE_FUNCTION
        bool in_sendUpRight(const uint32_t i, const uint32_t j) const {
            return (in_right_send_buffer(i) && in_up_send_buffer(j));
        }

        KOKKOS_INLINE_FUNCTION
        bool in_sendLeft(const uint32_t i, const uint32_t j) const {
            return (in_left_send_buffer(i) && in_Y(j));
        }

        KOKKOS_INLINE_FUNCTION
        bool in_sendRight(const uint32_t i, const uint32_t j) const {
            return (in_right_send_buffer(i) && in_Y(j));
        }

        KOKKOS_INLINE_FUNCTION
        bool in_sendDownLeft(const uint32_t i, const uint32_t j) const {
            return (in_left_send_buffer(i) && in_down_send_buffer(j));
        }

        KOKKOS_INLINE_FUNCTION
        bool in_sendDown(const uint32_t i, const uint32_t j) const {
            return (in_X(i) && in_down_send_buffer(j));
        }

        KOKKOS_INLINE_FUNCTION
        bool in_sendDownRight(const uint32_t i, const uint32_t j) const {
            return (in_right_send_buffer(i) && in_down_send_buffer(j));
        }

        // For receiving stuff
        KOKKOS_INLINE_FUNCTION
        bool in_recvUpLeft(const uint32_t i, const uint32_t j) const {
            return (in_left_recv_buffer(i) && in_up_recv_buffer(j));
        }

        KOKKOS_INLINE_FUNCTION
        bool in_recvUp(const uint32_t i, const uint32_t j) const {
            return (in_X(i) && in_up_recv_buffer(j));
        }

        KOKKOS_INLINE_FUNCTION
        bool in_recvUpRight(const uint32_t i, const uint32_t j) const {
            return (in_right_recv_buffer(i) && in_up_recv_buffer(j));
        }

        KOKKOS_INLINE_FUNCTION
        bool in_recvLeft(const uint32_t i, const uint32_t j) const {
            return (in_left_recv_buffer(i) && in_Y(j));
        }

        KOKKOS_INLINE_FUNCTION
        bool in_recvRight(const uint32_t i, const uint32_t j) const {
            return (in_right_recv_buffer(i) && in_Y(j));
        }

        KOKKOS_INLINE_FUNCTION
        bool in_recvDownLeft(const uint32_t i, const uint32_t j) const {
            return (in_left_recv_buffer(i) && in_down_recv_buffer(j));
        }

        KOKKOS_INLINE_FUNCTION
        bool in_recvDown(const uint32_t i, const uint32_t j) const {
            return (in_X(i) && in_down_recv_buffer(j));
        }

        KOKKOS_INLINE_FUNCTION
        bool in_recvDownRight(const uint32_t i, const uint32_t j) const {
            return (in_right_recv_buffer(i) && in_down_recv_buffer(j));
        }

        //////////////////////
        // 1D Region Checks //
        //////////////////////

        // In owned domain (i)
        KOKKOS_INLINE_FUNCTION
        bool in_X(const uint32_t i) const {
            return ((i>=imin_global()) && (i<=imax_global()));
        }

        // In owned domain (j)
        KOKKOS_INLINE_FUNCTION
        bool in_Y(const uint32_t j) const {
            return ((j>=jmin_global()) && (j<=jmax_global()));
        }

        // In interior of owned domain (i)
        KOKKOS_INLINE_FUNCTION
        bool in_X_interior(const uint32_t i) const {
            return ((i>=(imin_global()+1)) && (i<=(imax_global()-1)));
        }

        // In interior of owened domain (j)
        KOKKOS_INLINE_FUNCTION
        bool in_Y_interior(const uint32_t j) const {
            return ((j>=(jmin_global()+1)) && (j<=(jmax_global()-1)));
        }

        // In total domain (i)
        KOKKOS_INLINE_FUNCTION
        bool in_X_buff(const uint32_t i) const {
            return ((i>=(imin_global()-1)) && (i<=(imax_global()+1)));
        }

        // In total domain (j)
        KOKKOS_INLINE_FUNCTION
        bool in_Y_buff(const uint32_t j) const {
            return ((j>=(jmin_global()-1)) && (j<=(jmax_global()+1)));
        }

        // One cell into owned domain from left (i)
        KOKKOS_INLINE_FUNCTION
        bool in_left_send_buffer(const uint32_t i) const {
            return (i==imin_global());
        }

        // One cell into owned domain from right (i)
        KOKKOS_INLINE_FUNCTION
        bool in_right_send_buffer(const uint32_t i) const {
            return (i==imax_global());
        }

        // One cell into owned domain from bot (j)
        KOKKOS_INLINE_FUNCTION
        bool in_down_send_buffer(const uint32_t j) const {
            return (j==jmin_global());
        }

        // One cell into owned domain from top (j)
        KOKKOS_INLINE_FUNCTION
        bool in_up_send_buffer(const uint32_t j) const {
            return (j==jmax_global());
        }

        // One cell into owned domain from left (i)
        KOKKOS_INLINE_FUNCTION
        bool in_left_recv_buffer(const uint32_t i) const {
            return (i==(imin_global()-1));
        }

        // One cell into owned domain from right (i)
        KOKKOS_INLINE_FUNCTION
        bool in_right_recv_buffer(const uint32_t i) const {
            return (i==(imax_global()+1));
        }

        // One cell into owned domain from bot (j)
        KOKKOS_INLINE_FUNCTION
        bool in_down_recv_buffer(const uint32_t j) const {
            return (j==(jmin_global()-1));
        }

        // One cell into owned domain from top (j)
        KOKKOS_INLINE_FUNCTION
        bool in_up_recv_buffer(const uint32_t j) const {
            return (j==(jmax_global()+1));
        }

    };

    template<template<typename> class GrainID>
    class Mpi_Dual {
    private:

        // Custom function to get the rank of a neighboring process in a Cartesian communicator
        int getNeighborRank(MPI_Comm comm, int dims[], int periods[], const int coords[], const int displ[]) {
            int rank;
            int shifted_coords[2];

            // Calculate the shifted coordinates
            shifted_coords[0] = coords[0] + displ[0];
            shifted_coords[1] = coords[1] + displ[1];

            // If periodic, take remainder
            if (periods[0]) {
                shifted_coords[0] = shifted_coords[0] % dims[0];
            }
            if (periods[1]) {
                shifted_coords[1] = shifted_coords[1] % dims[1];
            }

            // Check if the shifted coordinates are within the bounds of the Cartesian grid
            if (shifted_coords[0] >= 0 && shifted_coords[0] < dims[0] && shifted_coords[1] >= 0 && shifted_coords[1] < dims[1]) {
                // Get the rank corresponding to the shifted coordinates
                MPI_Cart_rank(comm, shifted_coords, &rank);
            }
            else {
                // Out of bounds, set rank to MPI_PROC_NULL
                rank = MPI_PROC_NULL;
            }

            return rank;
        }

        // Find ranks of 2D neighbors
        void makeNeighbors() {

            // Get MPI Info
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &nproc);

            // Get dims for 2D decomposition
            // TODO::Custom decompositon
            //int dims[2] = {0, 0};
            MPI_Dims_create(nproc, 2, dims);

            // Create coms
            //int periods[2] = {0,0}; // non-periodic boundaries
            MPI_Comm cart_comm;
            MPI_Cart_create(comm, 2, dims, periods, 0, &cart_comm);

            // get the coordinates of the current process
            //int myCoords[2];
            MPI_Cart_coords(cart_comm, rank, 2, coords);

            // Get relative displacements of neighbors
            const int up_left_coords[2] = {-1,1}; const int up_coords[2] = {0,1}; const int up_right_coords[2] = {1,1};
            const int left_coords[2] = {-1,0}; const int right_coords[2] = {1,0};
            const int down_left_coords[2] = {-1,-1}; const int down_coords[2] = {0,-1}; const int down_right_coords[2] = {1,-1};

            // Get ranks of neighbors (start from bot-left and go ctrclockwise)
            neighbor_ranks[0] = getNeighborRank(cart_comm, dims, periods, coords, down_left_coords);
            neighbor_ranks[1] = getNeighborRank(cart_comm, dims, periods, coords, down_coords);
            neighbor_ranks[2] = getNeighborRank(cart_comm, dims, periods, coords, down_right_coords);
            neighbor_ranks[3] = getNeighborRank(cart_comm, dims, periods, coords, right_coords);
            neighbor_ranks[4] = getNeighborRank(cart_comm, dims, periods, coords, up_right_coords);
            neighbor_ranks[5] = getNeighborRank(cart_comm, dims, periods, coords, up_coords);
            neighbor_ranks[6] = getNeighborRank(cart_comm, dims, periods, coords, up_left_coords);
            neighbor_ranks[7] = getNeighborRank(cart_comm, dims, periods, coords, left_coords);
        }

    public:

        Mpi_Dual(MPI_Comm inComm){
            comm = inComm;
            deviceMpi.sendInfo_view = uint32_deviceView(Kokkos::ViewAllocateWithoutInitializing("device.sendInfo_view"), 16);
            deviceMpi.recvInfo_view = uint32_deviceView(Kokkos::ViewAllocateWithoutInitializing("device.recvInfo_view"), 16);
            deviceMpi.mpiIndices_view = uint32_deviceView(Kokkos::ViewAllocateWithoutInitializing("device.recvInfo_view"), 4);
            deviceMpi.mpiFloats_view = float_deviceView(Kokkos::ViewAllocateWithoutInitializing("device.mpiFloats_view"), 1);
            hostMpi.sendInfo_view = Kokkos::create_mirror_view(host_memory(), deviceMpi.sendInfo_view);
            hostMpi.recvInfo_view = Kokkos::create_mirror_view(host_memory(), deviceMpi.recvInfo_view);
            hostMpi.mpiIndices_view = Kokkos::create_mirror_view(host_memory(), deviceMpi.mpiIndices_view);
            hostMpi.mpiFloats_view = Kokkos::create_mirror_view(host_memory(), deviceMpi.mpiFloats_view);
            atomicNumSend = uint32_deviceView(Kokkos::ViewAllocateWithoutInitializing("device.atomicNumSend"), 8);
            atomicNumSend_host = Kokkos::create_mirror_view(host_memory(), atomicNumSend);
            makeNeighbors();
        }

        // Communication Stuff
        MPI_Comm comm;
        bool deviceComms;   // Controls device->device communication

        // MPI Data
        Mpi<host_space> hostMpi;
        Mpi<device_space> deviceMpi;

        // For MPI Initialization
        Stork::Structs::RegularGrid_Header<float, host_space> RDF_Input_header;
        Stork::Structs::RegularGrid_Header<float, host_space> header;

        // For MPI Comms
        char_deviceView sendComms;
        char_deviceView recvComms;

        uint32_deviceView atomicNumSend;
        uint32_hostView atomicNumSend_host;
        NeighborInts numSend;
        NeighborInts numRecv;

        uint32_t* send_ptrUint32[8];
        float* send_ptrFloat[8];
        uint16_t* send_ptrUint16[8];
        uint8_t* send_ptrUint8[8];

        uint32_t* recv_ptrUint32[8];
        float* recv_ptrFloat[8];
        uint16_t* recv_ptrUint16[8];
        uint8_t* recv_ptrUint8[8];

        char_hostView sendComms_host;
        char_hostView recvComms_host;

        uint32_t* send_ptrUint32_host[8];
        float* send_ptrFloat_host[8];
        uint16_t* send_ptrUint16_host[8];
        uint8_t* send_ptrUint8_host[8];

        uint32_t* recv_ptrUint32_host[8];
        float* recv_ptrFloat_host[8];
        uint16_t* recv_ptrUint16_host[8];
        uint8_t* recv_ptrUint8_host[8];

        uint32_t send_totalSize=0;
        uint32_t recv_totalSize=0;
        uint32_t byteMult;

        // MPI info
        int rank;
        int nproc;

        // Dimensionality of decomposition
        int dims[2] = {0,0};    // Dimensionality of decomposition
        int periods[2] = {0,0}; // Periodicity of boundaries
        int coords[2] = {0,0};   // Own coordinates

        // Ranks of Neighbors
        int neighbor_ranks[8];

        static constexpr uint32_t mpi_uint32_base_per_cell = 1; // idx
        static constexpr uint32_t mpi_float_base_per_cell = 5;  // gx, gy, gz, G0, tCap offset
        static constexpr uint32_t mpi_uint16_base_per_cell = 2; // dirID org/current
        static constexpr uint32_t mpi_uint8_base_per_cell = 1;  // inSteer

        static constexpr uint32_t mpi_uint32_per_cell = mpi_uint32_base_per_cell + 2*GrainID<device_space>::mpi_uint32_per_cell();
        static constexpr uint32_t mpi_float_per_cell = mpi_float_base_per_cell + 2*GrainID<device_space>::mpi_float_per_cell();
        static constexpr uint32_t mpi_uint16_per_cell = mpi_uint16_base_per_cell + 2*GrainID<device_space>::mpi_uint16_per_cell();
        static constexpr uint32_t mpi_uint8_per_cell = mpi_uint8_base_per_cell + 2*GrainID<device_space>::mpi_uint8_per_cell();

        void updateSendPointers(const uint32_t numSend, const uint8_t n){
            // Make mults
            uint32_t uint32_mult = mpi_uint32_per_cell * sizeof(uint32_t);
            uint32_t float_mult = mpi_float_per_cell * sizeof(float);
            uint32_t uint16_mult = mpi_uint16_per_cell * sizeof(uint16_t);
            uint32_t uint8_mult = mpi_uint8_per_cell * sizeof(uint8_t);

            // Find offsets
            uint32_t send_offset = 0;
            char* send_basePtr = reinterpret_cast<char*>(send_ptrUint32[n]);

            // Update device pointers
            send_ptrUint32[n] = reinterpret_cast<uint32_t*>(&send_basePtr[send_offset]);
            send_offset += uint32_mult * numSend;

            send_ptrFloat[n] = reinterpret_cast<float*>(&send_basePtr[send_offset]);
            send_offset += float_mult * numSend;

            send_ptrUint16[n] = reinterpret_cast<uint16_t*>(&send_basePtr[send_offset]);
            send_offset += uint16_mult * numSend;

            send_ptrUint8[n] = reinterpret_cast<uint8_t*>(&send_basePtr[send_offset]);
            send_offset += uint8_mult * numSend;
        }

        void updateRecvPointers(const uint32_t numRecv, const uint8_t n){
            // Make mults
            uint32_t uint32_mult = mpi_uint32_per_cell * sizeof(uint32_t);
            uint32_t float_mult = mpi_float_per_cell * sizeof(float);
            uint32_t uint16_mult = mpi_uint16_per_cell * sizeof(uint16_t);
            uint32_t uint8_mult = mpi_uint8_per_cell * sizeof(uint8_t);

            // Find offsets
            uint32_t recv_offset = 0;
            char* recv_basePtr = reinterpret_cast<char*>(recv_ptrUint32[n]);

            // Update device pointers
            recv_ptrUint32[n] = reinterpret_cast<uint32_t*>(&recv_basePtr[recv_offset]);
            recv_offset += uint32_mult * numRecv;

            recv_ptrFloat[n] = reinterpret_cast<float*>(&recv_basePtr[recv_offset]);
            recv_offset += float_mult * numRecv;

            recv_ptrUint16[n] = reinterpret_cast<uint16_t*>(&recv_basePtr[recv_offset]);
            recv_offset += uint16_mult * numRecv;

            recv_ptrUint8[n] = reinterpret_cast<uint8_t*>(&recv_basePtr[recv_offset]);
            recv_offset += uint8_mult * numRecv;
        }

        void makeComms(){
            // Make mults
            uint32_t uint32_mult = mpi_uint32_per_cell * sizeof(uint32_t);
            uint32_t float_mult = mpi_float_per_cell * sizeof(float);
            uint32_t uint16_mult = mpi_uint16_per_cell * sizeof(uint16_t);
            uint32_t uint8_mult = mpi_uint8_per_cell * sizeof(uint8_t);
            byteMult = uint32_mult + float_mult + uint16_mult + uint8_mult;

            // Make offset arrays
            uint32_t send_uint32_offset[8];
            uint32_t recv_uint32_offset[8];
            uint32_t send_float_offset[8];
            uint32_t recv_float_offset[8];
            uint32_t send_uint16_offset[8];
            uint32_t recv_uint16_offset[8];
            uint32_t send_uint8_offset[8];
            uint32_t recv_uint8_offset[8];

            // Find offsets
            uint32_t send_offset = 0;
            uint32_t recv_offset = 0;
            for (int n=0;n<8;n++){
                const uint32_t numSend = hostMpi.send_size(n);
                const uint32_t numRecv = hostMpi.recv_size(n);

                send_uint32_offset[n] = send_offset;
                recv_uint32_offset[n] = recv_offset;
                send_offset += uint32_mult * numSend;
                recv_offset += uint32_mult * numRecv;
                // TODO::All are 64-byte alligned
                // send_offset = ((send_offset + 63) / 64) * 64;
                // recv_offset = ((recv_offset + 63) / 64) * 64;

                send_float_offset[n] = send_offset;
                recv_float_offset[n] = recv_offset;
                send_offset += float_mult * numSend;
                recv_offset += float_mult * numRecv;
                // TODO::All are 64-byte alligned
                // send_offset = ((send_offset + 63) / 64) * 64;
                // recv_offset = ((recv_offset + 63) / 64) * 64;

                send_uint16_offset[n] = send_offset;
                recv_uint16_offset[n] = recv_offset;
                send_offset += uint16_mult * numSend;
                recv_offset += uint16_mult * numRecv;
                // TODO::All are 64-byte alligned
                // send_offset = ((send_offset + 63) / 64) * 64;
                // recv_offset = ((recv_offset + 63) / 64) * 64;

                send_uint8_offset[n] = send_offset;
                recv_uint8_offset[n] = recv_offset;
                send_offset += uint8_mult * numSend;
                recv_offset += uint8_mult * numRecv;
                // TODO::All are 64-byte alligned
                // send_offset = ((send_offset + 63) / 64) * 64;
                // recv_offset = ((recv_offset + 63) / 64) * 64;

                send_offset = ((send_offset + 3) / 4) * 4;
                recv_offset = ((recv_offset + 3) / 4) * 4;
            }

            // Initialize views
            sendComms = char_deviceView(Kokkos::ViewAllocateWithoutInitializing("sendComms"), send_offset);
            recvComms = char_deviceView(Kokkos::ViewAllocateWithoutInitializing("recvComms"), recv_offset);
            // Initialize send and recv pointers
            {
                char* send_basePtr = sendComms.data();
                char* recv_basePtr = recvComms.data();
                for (int n = 0; n < 8; n++) {
                    send_ptrUint32[n] = reinterpret_cast<uint32_t*>(&send_basePtr[send_uint32_offset[n]]);
                    recv_ptrUint32[n] = reinterpret_cast<uint32_t*>(&recv_basePtr[recv_uint32_offset[n]]);
                    send_ptrFloat[n] = reinterpret_cast<float*>(&send_basePtr[send_float_offset[n]]);
                    recv_ptrFloat[n] = reinterpret_cast<float*>(&recv_basePtr[recv_float_offset[n]]);
                    send_ptrUint16[n] = reinterpret_cast<uint16_t*>(&send_basePtr[send_uint16_offset[n]]);
                    recv_ptrUint16[n] = reinterpret_cast<uint16_t*>(&recv_basePtr[recv_uint16_offset[n]]);
                    send_ptrUint8[n] = reinterpret_cast<uint8_t*>(&send_basePtr[send_uint8_offset[n]]);
                    recv_ptrUint8[n] = reinterpret_cast<uint8_t*>(&recv_basePtr[recv_uint8_offset[n]]);
                }
            }
            if (!deviceComms) {
                sendComms_host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, host_memory(), sendComms);
                recvComms_host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, host_memory(), recvComms);
                // Initialize send and recv  pointers
                {
                    char* send_basePtr = sendComms_host.data();
                    char* recv_basePtr = recvComms_host.data();
                    for (int n = 0; n < 8; n++) {
                        send_ptrUint32_host[n] = reinterpret_cast<uint32_t*>(&send_basePtr[send_uint32_offset[n]]);
                        recv_ptrUint32_host[n] = reinterpret_cast<uint32_t*>(&recv_basePtr[recv_uint32_offset[n]]);
                        send_ptrFloat_host[n] = reinterpret_cast<float*>(&send_basePtr[send_float_offset[n]]);
                        recv_ptrFloat_host[n] = reinterpret_cast<float*>(&recv_basePtr[recv_float_offset[n]]);
                        send_ptrUint16_host[n] = reinterpret_cast<uint16_t*>(&send_basePtr[send_uint16_offset[n]]);
                        recv_ptrUint16_host[n] = reinterpret_cast<uint16_t*>(&recv_basePtr[recv_uint16_offset[n]]);
                        send_ptrUint8_host[n] = reinterpret_cast<uint8_t*>(&send_basePtr[send_uint8_offset[n]]);
                        recv_ptrUint8_host[n] = reinterpret_cast<uint8_t*>(&recv_basePtr[recv_uint8_offset[n]]);
                    }
                }
            }
        }
    };
}
