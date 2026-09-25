#include <Toucan_Core.hpp>

namespace {

    template<template<typename> class GrainID>
    int RunToucan(const Toucan::IO::FileReader& toucanInputData) {
        using ToucanFloat = float;
        using ToucanDevice = Toucan::device_space;

        Toucan::Structs::Mpi_Dual<GrainID> DualMpi(MPI_COMM_WORLD);
        Toucan::Structs::Sim_Dual DualSim;
        Toucan::Structs::RNG_Dual DualGen;
        Toucan::Structs::Orientations_Dual DualOrient;
        Toucan::Structs::Substrate_Dual<GrainID> DualSub;

        // Set DualSim rank
        DualSim.rank = DualMpi.rank;

        // Get thermal generator
        std::unique_ptr<Toucan::Thermal::ThermalSource<ToucanFloat, ToucanDevice>> thermalSource = Toucan::Thermal::Routing<ToucanFloat, ToucanDevice, GrainID>(toucanInputData.thermal_json, DualMpi);

        uint32_t numLayers = 1;
        bool firstLayer = true;
        for (uint32_t layer=0; layer<numLayers; layer++){
            if (DualMpi.rank == 0) {
                std::cout << "Layer: " << layer << std::endl;
            }

            if (DualMpi.rank == 0) {
                std::cout << "\tThermal" << std::endl;
            }
            Stork::Structs::RDF_Dual<ToucanFloat> RDF = thermalSource->GetRDF(layer);

            if (firstLayer) {
                Toucan::Run::Initialize(toucanInputData, DualMpi, DualSub, DualSim, DualOrient, DualGen, RDF);
                numLayers = DualSim.hostSim.numLayers();
                firstLayer = false;
            }

            if (DualMpi.rank == 0) {
                std::cout << "\tMicrostructure" << std::endl;
            }
            Toucan::Run::FromLayer(DualMpi, DualSub, DualSim, DualOrient, DualGen, Toucan::Run::CoupledSignal::READ, RDF);
            Toucan::Run::FromLayer(DualMpi, DualSub, DualSim, DualOrient, DualGen, Toucan::Run::CoupledSignal::SIMULATE);
            Toucan::Run::FromLayer(DualMpi, DualSub, DualSim, DualOrient, DualGen, Toucan::Run::CoupledSignal::CLEAR);
        }

        Toucan::Run::FromLayer(DualMpi, DualSub, DualSim, DualOrient, DualGen, Toucan::Run::CoupledSignal::DONE);
        return 0;
    }

}

int main(int argc, char* argv[]) {

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int result = 0;

    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {
        // Toucan File from command line
        if (argc < 2) {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0) {
                std::cerr << "Usage: Toucan <ToucanSettings.json>\n";
            }
            result = 1;
        }
        else {
            const std::string inputToucanFile = argv[1];

            // Initialize inputs
            Toucan::IO::FileReader toucanInputData;
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0) {
                toucanInputData.Initialize(inputToucanFile);
            }
            toucanInputData.Broadcast(0, MPI_COMM_WORLD);

            const auto grainsIt = toucanInputData.output_json.find("Grains");
            const bool hasGrains = grainsIt != toucanInputData.output_json.end() && grainsIt->is_object();
            const bool useGrainIdentifier = hasGrains ? grainsIt->value("UniqueIdentifiers", false) : false;
            const bool outputUnderresolvedGrains = hasGrains ? grainsIt->value("UnderResolved", false) : false;

            if (useGrainIdentifier && outputUnderresolvedGrains) {
                result = RunToucan<Toucan::Structs::UniqueUnderresolvedGrainIdentifier>(toucanInputData);
            }
            else if (useGrainIdentifier) {
                result = RunToucan<Toucan::Structs::UniqueOnlyGrainIdentifier>(toucanInputData);
            }
            else if (outputUnderresolvedGrains) {
                result = RunToucan<Toucan::Structs::UnderresolvedOnlyGrainIdentifier>(toucanInputData);
            }
            else {
                result = RunToucan<Toucan::Structs::BaseGrainIdentifier>(toucanInputData);
            }
        }
    }

    // Finalize Kokkos
    Kokkos::finalize();

    // Finalize MPI
    MPI_Finalize();
    
    return result;
}
