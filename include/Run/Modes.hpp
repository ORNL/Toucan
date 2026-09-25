#pragma once

#include "Definitions.hpp"
#include "Common.hpp"

#include <nlohmann/json.hpp>

#include "Run.hpp"

#include "Init/Init.hpp"

#include "Structs/Grid.hpp"
#include "Structs/Layer.hpp"
#include "Comms/Structs.hpp"
#include "Structs/Sim.hpp"
#include "Structs/Rng.hpp"
#include "Structs/Orientations.hpp"

// MODES: Grids from files, remain in memory
//      : Grids from files, read one by one
//      : Grids from pointers (coupled), remain in memory
//      : Grids from pointers (coupled), read one by one

namespace Toucan::Run {

    // Bounds check between layers
    namespace impl {
        template<template<typename> class GrainID>
        void BoundsCheck(Structs::Sim_Dual& DualSim, Structs::Mpi_Dual<GrainID>& DualMpi, Stork::Structs::RegularGrid_Header<float, host_space>& RDF_header) {
            // Get references for headers
            Stork::Structs::RegularGrid_Header<float, host_space>& oldHeader = DualMpi.RDF_Input_header;
            Stork::Structs::RegularGrid_Header<float, host_space>& newHeader = RDF_header;

            // Make sure relevant header information is the same (for <x,y> and res)
            if (oldHeader.global_i0() != newHeader.global_i0()) {
                throw std::runtime_error("Mpi Error: global_i0 does not match between iterations.");
            }
            if (oldHeader.global_j0() != newHeader.global_j0()) {
                throw std::runtime_error("Mpi Error: global_j0 does not match between iterations.");
            }
            if (oldHeader.global_x0() != newHeader.global_x0()) {
                throw std::runtime_error("Mpi Error: global_x0 does not match between iterations.");
            }
            if (oldHeader.global_y0() != newHeader.global_y0()) {
                throw std::runtime_error("Mpi Error: global_y0 does not match between iterations.");
            }
            if (oldHeader.local_inum() != newHeader.local_inum()) {
                throw std::runtime_error("Mpi Error: local_inum does not match between iterations.");
            }
            if (oldHeader.local_jnum() != newHeader.local_jnum()) {
                throw std::runtime_error("Mpi Error: local_jnum does not match between iterations.");
            }
            if (oldHeader.gridResolution() != newHeader.gridResolution()) {
                throw std::runtime_error("Mpi Error: gridResolution does not match between iterations.");
            }

            // Check substrate size vs depth
            if (newHeader.global_z0() < (-DualSim.hostSim.windowHeight())) {
                throw std::runtime_error("Mpi Error: Thermal information goes too deep. Increase 'MULTILAYER::WINDOW_HEIGHT'.");
            }
        }
    }

    enum class CoupledSignal {
        READ,
        SIMULATE,
        CLEAR,
        DONE
    };
    // Initialization with JSON
    template<template<typename> class GrainID>
    void Initialize(const Toucan::IO::FileReader& inputData, Structs::Mpi_Dual<GrainID>& DualMpi, Structs::Substrate_Dual<GrainID>& DualSub, Structs::Sim_Dual& DualSim, Structs::Orientations_Dual& DualOrient, Structs::RNG_Dual& DualGen, Stork::Structs::RDF_Dual<float>& RDF){
        // Read in sim params from file
        Toucan::IO::ReadSimFile(inputData, DualSim, DualMpi, RDF.host_header);
        // Set orientations and shuffled grain ID list from their own seeds.
        Structs::Sim<host_space>& sim = DualSim.hostSim;
        DualGen = Toucan::Structs::RNG_Dual(sim.numOrientations(), DualSim.orientationSeed, DualSim.substrateSeed);
        // Initialize orientations
        Toucan::Init::Orientations(DualOrient, DualSim, DualMpi, DualGen);
        // Initialize Mpi domain info from first file
        Init::Mpi(DualMpi, DualSim, RDF.host_header);
        // Set up substrate and parameters
        Init::Substrate(DualSub, DualGen, DualSim, DualMpi);
        // Populate Substrate
        Run::Nucleate_Substrate(DualMpi, DualSub, DualSim, DualGen);
    }

    template<template<typename> class GrainID>
    void Initialize(nlohmann::json &inputData, Structs::Mpi_Dual<GrainID>& DualMpi, Structs::Substrate_Dual<GrainID>& DualSub, Structs::Sim_Dual& DualSim, Structs::Orientations_Dual& DualOrient, Structs::RNG_Dual& DualGen, Stork::Structs::RDF_Dual<float>& RDF){
        Toucan::IO::FileReader reader;
        reader.Initialize(inputData);
        Initialize(reader, DualMpi, DualSub, DualSim, DualOrient, DualGen, RDF);
    }

    template<template<typename> class GrainID>
    void FromLayer(Structs::Mpi_Dual<GrainID>& DualMpi, Structs::Substrate_Dual<GrainID>& DualSub, Structs::Sim_Dual& DualSim, Structs::Orientations_Dual& DualOrient, Structs::RNG_Dual& DualGen, const CoupledSignal signal, Stork::Structs::RDF_Dual<float>& RDF){

        // Vector of grids in memory
        static Common::vector<Structs::Grid_Dual<GrainID>> DualGrids;

        // For writing asynchronously
        static Toucan::Common::future<void> write_done_future = impl::make_ready_future();

        // If we are reading, initialize grid from data
        if (signal == CoupledSignal::READ){
            // Make grid
            Structs::Grid_Dual<GrainID> DualGrid;
            // Make sure bounds are consistent
            impl::BoundsCheck(DualSim, DualMpi, RDF.host_header);
            // Initialize CA Grid (delete global layer)
            Init::Grid(DualGrid, DualSub, DualSim, DualMpi, RDF);
            // Add it to the vector of Grids
            DualGrids.push_back(DualGrid);

            // TODO::DEBUG (test sends and recvs)
            // for (int rank=0;rank<2;rank++){
            //     if (DualMpi.rank==rank){
            //         std::cout << "Rank: " << DualMpi.rank << "\n";
            //         std::cout << "<x0,y0,z0>\t" << RDF.host_header.global_x0() << " | " << RDF.host_header.global_y0() << " | " << RDF.host_header.global_z0() << "\n";
            //         std::cout << "<i0,j0,k0>\t" << RDF.host_header.global_i0() << " | " << RDF.host_header.global_j0() << " | " << RDF.host_header.global_k0() << "\n";
            //         std::cout << "<inum,jnum,knum>\t" << RDF.host_header.local_inum() << " | " << RDF.host_header.local_jnum() << " | " << RDF.host_header.local_knum() << "\n";
            //         for (int n=0;n<8;n++){
            //             std::cout << "\tNeighbor: " << n << "\n";
            //             std::cout << "\t\tSends: " << DualMpi.hostMpi.send_size(n) << "\n";
            //             std::cout << "\t\tRecvs: " << DualMpi.hostMpi.recv_size(n) << "\n";
            //         }
            //         std::cout << std::endl;
            //     }
            //     MPI_Barrier(MPI_COMM_WORLD);
            // }
        }
        // If we are simulating, run a layer of the grid
        else if (signal == CoupledSignal::SIMULATE){
            // Set grid references (repeats like a->b->c->...->a->b, etc)
            const int layerRemainder = DualSub.layerNum%DualGrids.size();
            Structs::Grid_Dual<GrainID>& DualGrid = DualGrids[layerRemainder];

            // Initialize Steering Vector [DEVICE ONLY]
            Structs::SteeringVector steer(DualGrid.size);

            // Transfer dirID's from Substrate to Grid AND Nucleate Grid [DEVICE ONLY]
            Run::Nucleate_Grid(DualGrid, DualSub, DualSim, DualGen, steer);

            // Run DECA For Grid [DEVICE ONLY]
            Run::DECA(DualGrid, DualMpi, DualOrient, DualSim, steer);

            // Transfer dirID's from Grid to Substrate [DEVICE ONLY]
            Run::GridToSub(DualGrid, DualSub, DualSim);

            // Output Results (Async) [DEVICE->HOST]
            Run::Output_Results(DualSub, DualMpi, DualSim, DualGen, write_done_future);
        }
        // If we are clearing, clear the grids
        else if (signal == CoupledSignal::CLEAR){
            DualGrids.clear();
        }
        // If the code is all done, clear the grids and output final results
        else if (signal == CoupledSignal::DONE){
            DualGrids.clear();

            // TODO::OUTPUT EVERYTHING NOT YET OUTPUT
            // Output Final Results (Async) [DEVICE->HOST]
            Run::Output_Final_Results(DualSub, DualMpi, DualSim, write_done_future);

            // Make sure last write completes before finishing
            write_done_future.wait();
        }
    }

    template<template<typename> class GrainID>
    void FromLayer(Structs::Mpi_Dual<GrainID>& DualMpi, Structs::Substrate_Dual<GrainID>& DualSub, Structs::Sim_Dual& DualSim, Structs::Orientations_Dual& DualOrient, Structs::RNG_Dual& DualGen, const CoupledSignal signal){
        // Make blank layer
        Stork::Structs::RDF_Dual<float> blankLayer;   // Blank layer for calls when layer info isn't needed

        // Run Mode
        FromLayer(DualMpi, DualSub, DualSim, DualOrient, DualGen, signal, blankLayer);
    }
}
