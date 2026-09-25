#pragma once

#include "Definitions.hpp"
#include "Common.hpp"

#include <nlohmann/json.hpp>

#include "Structs/Layer.hpp"
#include "Structs/Sim.hpp"
#include "Comms/Structs.hpp"

#include <limits>

namespace Toucan::IO {

    // Usings
    using json = nlohmann::json;

    // Severity levels (enum)
    namespace impl {
        enum class Severity {
            WARNING,
            ERROR
        };
    }

    // If file exists
    namespace impl {
        inline bool fileExists(const Common::string& name) {
            std::ifstream f(name.c_str());
            return f.good();
        }
    }

    // For directory creation
    namespace impl {
        void createDirectoryIfNotExists(const std::string& path) {
            std::filesystem::path dirPath(path);
            if (!std::filesystem::exists(dirPath)) {
                if (std::filesystem::create_directories(dirPath)) {
                    std::cout << "Directory created: " << path << "\n";
                } else {
                    std::cerr << "Failed to create directory: " << path << "\n";
                }
            } else {
                std::cout << "Directory already exists: " << path << "\n";
            }
        }
    }

    // TODO::Should be in Stork
    // For finding all files (wildcards and such)
    namespace impl {
        template<template<typename> class GrainID>
        void FindFiles(Structs::Sim_Dual& Sim, Structs::Mpi_Dual<GrainID>& Mpi, Common::string& baseFile, const int numFiles) {
            // If not reading files
            if (baseFile == "") {
                return;
            }

            // Usings
            using Common::string;

            // Generate file names by replacing * with rank and # with layer number
            Common::string baseLayerFile = baseFile;

            // If doing "OneToOne" mode (where each rank has its own file), replace * with rank
            if (Sim.mpiMode == "OneToOne") {

                // Find how many characters to replace and make string
                const int numDigits = std::to_string(Mpi.nproc).length(); // Get the number of digits in the maximum rank
                std::ostringstream oss;
                oss << std::setw(numDigits) << std::setfill('0') << Mpi.rank; // Format the rank with leading zeros

                // Find the rank wildcard
                size_t posRank = baseLayerFile.find('*');

                // If the position exists, replace it
                if (posRank != std::string::npos) {
                    baseLayerFile.replace(posRank, 1, oss.str());
                }
                // Otherwise, throw an error
                else {
                    throw std::runtime_error("Fatal error: Mpi::Mode is 'OneToOne' but rank wildcard '*' is missing.");
                }
            }

            // If multiple files, replace # with numbers
            if (numFiles > 1) {
                for (int fileNum = 0; fileNum < numFiles; fileNum++) {
                    string fileName = baseLayerFile;

                    // Find how many characters to replace and make string
                    const int numDigits = std::to_string(numFiles).length(); // Get the number of digits in the maximum rank
                    std::ostringstream oss;
                    oss << std::setw(numDigits) << std::setfill('0') << fileNum; // Format the rank with leading zeros

                    // Find the layer wildcard
                    size_t posLayer = fileName.find('#');

                    // If the position exists, replace it
                    if (posLayer != std::string::npos) {
                        fileName.replace(posLayer, 1, oss.str());
                        Sim.thermalFiles.push_back(fileName);
                    }
                    // Otherwise, throw an error
                    else {
                        throw std::runtime_error("Fatal error: 'FILE::NUMBER_OF_FILES_PER_RANK' exceeds one but layer wildcard '#' is missing.");
                    }
                }
            }
            // Otherwise it is still just the base file
            else {
                string fileName = baseLayerFile;
                // Add fileName to List
                Sim.thermalFiles.push_back(fileName);
            }

            // Now make sure each file exists
            for (const string& file: Sim.thermalFiles) {
                // Full Filenames
                const string txtFile = file + ".txt";
                const string csvFile = file + ".csv";
                const string binFile = file + ".deca";
                // If binary file doesn't exist, parse text file (SLOW) and save to binary file (FAST)
                if (!impl::fileExists(txtFile) && !impl::fileExists(csvFile) && !impl::fileExists(binFile)) {
                    throw std::runtime_error("Fatal error: file '" + file + "' cannot be found.");
                }
            }

        }
    }

    namespace impl {
        inline json ReadJsonFile(const Common::string& fileName) {
            std::ifstream file(fileName);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open input file: " + fileName);
            }
            return json::parse(file);
        }

        inline const json& GetNestedJson(const json& root, const Common::string& key, const bool isCritical) {
            static const json emptyObject = json::object();

            if (root.contains(key) && root[key].is_object()) {
                return root[key];
            }

            if (isCritical) {
                throw std::runtime_error("Critical input error: missing required object block '" + key + "'.");
            }

            return emptyObject;
        }

        template <typename T>
        inline T ReadValue(const json& root, const Common::string& key, const T& defaultValue, const bool isCritical) {
            if (root.contains(key) && !root[key].is_null()) {
                return root[key].get<T>();
            }

            if (isCritical) {
                throw std::runtime_error("Critical input error: missing required field '" + key + "'.");
            }

            return defaultValue;
        }

        inline json ReadFileOrJson(const json& root, const Common::string& key, const bool isCritical) {
            if (!root.contains(key) || root[key].is_null()) {
                if (isCritical) {
                    throw std::runtime_error("Input '" + key + "' is required.");
                }
                return json::object();
            }

            if (root[key].is_string()) {
                return ReadJsonFile(root[key].get<Common::string>());
            }
            if (root[key].is_object()) {
                return root[key];
            }

            throw std::runtime_error("Input '" + key + "' must be a filename string or JSON object.");
        }

        inline Common::string MaterializeInlineJsonFile(const json& node, const Common::string& stem, const size_t index) {
            const auto ticks = Common::high_resolution_clock::now().time_since_epoch().count();
            const std::filesystem::path filePath =
                std::filesystem::temp_directory_path() /
                ("toucan_" + stem + "_" + std::to_string(ticks) + "_" + std::to_string(index) + ".json");

            std::ofstream out(filePath);
            if (!out.is_open()) {
                throw std::runtime_error("Could not write inline JSON input to: " + filePath.string());
            }
            out << node.dump(4);
            return filePath.string();
        }
    }

    class FileReader {
    public:
        json irf_json = json::object();
        json rng_json = json::object();
        json domain_json = json::object();
        json mpi_json = json::object();
        json nucleation_json = json::object();
        json substrate_json = json::object();
        json temperature_data_json = json::object();
        json thermal_json = json::object();
        json debug_json = json::object();
        json output_json = json::object();

        Common::vector<Common::string> temperature_files;

        void Initialize(const Common::string& inputFile) {
            Initialize(impl::ReadJsonFile(inputFile));
        }

        void Initialize(const json& root) {
            irf_json = impl::ReadFileOrJson(root, "IRF", true);

            if (root.contains("RNG") && root["RNG"].is_object()) {
                rng_json = root["RNG"];
            }
            else {
                rng_json = impl::GetNestedJson(root, "RNG", true);
            }

            domain_json = impl::GetNestedJson(root, "Domain", true);

            if (root.contains("MPI") && root["MPI"].is_object()) {
                mpi_json = root["MPI"];
            }
            else {
                mpi_json = impl::GetNestedJson(root, "MPI", true);
            }

            nucleation_json = impl::GetNestedJson(root, "Nucleation", true);
            substrate_json = impl::GetNestedJson(root, "Substrate", false);
            if (root.contains("Thermal") && root["Thermal"].is_object()) {
                thermal_json = root["Thermal"];
            }
            else {
                thermal_json = json::object();
            }

            if (root.contains("TemperatureData") && root["TemperatureData"].is_object()) {
                temperature_data_json = root["TemperatureData"];
            }
            else {
                temperature_data_json = json::object();
            }

            debug_json = root.value("Debug", json::object());
            output_json = impl::GetNestedJson(root, "Output", true);

            if (!temperature_data_json.empty()) {
                ReadTemperatureFiles();
            }
            else {
                temperature_files.clear();
            }
            Validate();
        }

        json Pack() const {
            return json{
                {"irf_json", irf_json},
                {"rng_json", rng_json},
                {"domain_json", domain_json},
                {"mpi_json", mpi_json},
                {"nucleation_json", nucleation_json},
                {"substrate_json", substrate_json},
                {"temperature_data_json", temperature_data_json},
                {"thermal_json", thermal_json},
                {"debug_json", debug_json},
                {"output_json", output_json},
                {"temperature_files", temperature_files}
            };
        }

        void Unpack(const json& packed) {
            irf_json = packed.at("irf_json");
            rng_json = packed.at("rng_json");
            domain_json = packed.at("domain_json");
            mpi_json = packed.at("mpi_json");
            nucleation_json = packed.at("nucleation_json");
            substrate_json = packed.value("substrate_json", json::object());
            temperature_data_json = packed.at("temperature_data_json");
            thermal_json = packed.value("thermal_json", json::object());
            debug_json = packed.value("debug_json", json::object());
            output_json = packed.at("output_json");
            temperature_files = packed.at("temperature_files").get<Common::vector<Common::string>>();
            Validate();
        }

        void Broadcast(const int root, MPI_Comm comm) {
            int rank = 0;
            MPI_Comm_rank(comm, &rank);

            Common::string payload;
            int payloadSize = 0;

            if (rank == root) {
                payload = Pack().dump();
                if (payload.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
                    throw std::runtime_error("Toucan input broadcast payload is too large for MPI_Bcast.");
                }
                payloadSize = static_cast<int>(payload.size());
            }

            MPI_Bcast(&payloadSize, 1, MPI_INT, root, comm);
            if (rank != root) {
                payload.resize(static_cast<size_t>(payloadSize));
            }

            MPI_Bcast(payload.data(), payloadSize, MPI_CHAR, root, comm);
            if (rank != root) {
                Unpack(json::parse(payload));
            }
        }

        uint8_t FineFactor(const uint8_t defaultValue = 1) const {
            return impl::ReadValue<uint8_t>(temperature_data_json, "FineFactor", defaultValue, false);
        }

    private:
        void ReadTemperatureFiles() {
            temperature_files.clear();

            if (!temperature_data_json.contains("TemperatureFiles") || !temperature_data_json["TemperatureFiles"].is_array()) {
                throw std::runtime_error("Input [TemperatureData][TemperatureFiles] must be an array.");
            }

            const json& filesNode = temperature_data_json["TemperatureFiles"];
            for (size_t index = 0; index < filesNode.size(); index++) {
                const json& fileNode = filesNode[index];
                if (fileNode.is_string()) {
                    temperature_files.push_back(fileNode.get<Common::string>());
                }
                else if (fileNode.is_object()) {
                    temperature_files.push_back(impl::MaterializeInlineJsonFile(fileNode, "temperature", index));
                }
                else {
                    throw std::runtime_error("Each [TemperatureData][TemperatureFiles] entry must be a filename string or JSON object.");
                }
            }
        }

        void Validate() const {
            if (!irf_json.is_object() || irf_json.empty()) {
                throw std::runtime_error("Toucan input validation error: IRF must be a non-empty JSON object.");
            }
            if (!domain_json.is_object() || domain_json.empty()) {
                throw std::runtime_error("Toucan input validation error: Domain must be a non-empty JSON object.");
            }
            if (!mpi_json.is_object() || mpi_json.empty()) {
                throw std::runtime_error("Toucan input validation error: MPI must be a non-empty JSON object.");
            }
            if (!rng_json.is_object() || rng_json.empty()) {
                throw std::runtime_error("Toucan input validation error: RNG must be a non-empty JSON object.");
            }
            if (!nucleation_json.is_object() || nucleation_json.empty()) {
                throw std::runtime_error("Toucan input validation error: Nucleation must be a non-empty JSON object.");
            }
            if (!thermal_json.empty() && !thermal_json.is_object()) {
                throw std::runtime_error("Toucan input validation error: Thermal must be a JSON object.");
            }
            if (!debug_json.is_object()) {
                throw std::runtime_error("Toucan input validation error: Debug must be a JSON object.");
            }
            if (thermal_json.empty()) {
                if (!temperature_data_json.is_object() || temperature_data_json.empty()) {
                    throw std::runtime_error("Toucan input validation error: TemperatureData must be a non-empty JSON object when Thermal is not provided.");
                }
                if (temperature_files.empty()) {
                    throw std::runtime_error("Toucan input validation error: TemperatureFiles must contain at least one entry.");
                }
            }
            if (!output_json.is_object() || output_json.empty()) {
                throw std::runtime_error("Toucan input validation error: Output must be a non-empty JSON object.");
            }
        }
    };

    namespace impl {
        inline void InitializeIRF(const json& irfNode, Structs::Sim_Dual& DualSim) {
            const Structs::Sim<host_space>& hostSim = DualSim.hostSim;

            hostSim.a() = static_cast<float>(irfNode["coefficients"]["A"]);
            hostSim.b() = static_cast<float>(irfNode["coefficients"]["B"]);
            if (irfNode["function"] != "power") {
                throw std::runtime_error("Error: Toucan requires an IRF in the power law form V = a * Undercooling ^ b");
            }

            hostSim.b1() = (1.0f + hostSim.b());
            hostSim.bn1() = (1.0f / hostSim.b1());
        }

        inline void InitializeRNG(const json& rngNode, Structs::Sim_Dual& DualSim) {
            const int defaultSeed = ReadValue<int>(rngNode, "Seed", 0, false);
            const int orientationSeed = ReadValue<int>(rngNode, "Orientation", defaultSeed, false);
            const int substrateSeed = ReadValue<int>(rngNode, "Substrate", orientationSeed, false);

            DualSim.seed = orientationSeed;
            DualSim.orientationSeed = orientationSeed;
            DualSim.substrateSeed = substrateSeed;
        }

        inline void InitializeDomain(
            const json& domainNode,
            Structs::Sim_Dual& DualSim,
            Stork::Structs::RegularGrid_Header<float, host_space>& RDF_header
        ) {
            const Structs::Sim<host_space>& hostSim = DualSim.hostSim;

            hostSim.res() = RDF_header.gridResolution();
            hostSim.numLayers() = ReadValue<uint32_t>(domainNode, "NumberOfLayers", 1, true);
            hostSim.layerHeight() = ReadValue<float>(domainNode, "LayerOffset", 0.0f, true);
            hostSim.windowHeight() = ReadValue<float>(domainNode, "WindowHeight", 0.0f, true);
        }

        template<template<typename> class GrainID>
        inline void InitializeMPI(const json& mpiNode, Structs::Sim_Dual& DualSim, Structs::Mpi_Dual<GrainID>& DualMpi) {
            DualSim.mpiMode = ReadValue<Common::string>(mpiNode, "Mode", Common::string("Strong"), true);
            const Common::string mpiComms = ReadValue<Common::string>(mpiNode, "Comm", Common::string("Host"), true);

            if (mpiComms == "Host") {
                DualMpi.deviceComms = false;
            }
            else if (mpiComms == "Device") {
                DualMpi.deviceComms = true;
            }
            else {
                throw std::runtime_error("Input error: Keyword [MPI][Comm] has invalid value.\nValid values are [Host, Device].");
            }
        }

        inline void InitializeNucleation(const json& nucleationNode, Structs::Sim_Dual& DualSim) {
            const Structs::Sim<host_space>& hostSim = DualSim.hostSim;

            const float nuclDensity = ReadValue<float>(nucleationNode, "Density", 0.0f, true);
            hostSim.nuclProb() = nuclDensity * hostSim.res() * hostSim.res() * hostSim.res();
            hostSim.meanUnder() = ReadValue<float>(nucleationNode, "MeanUndercooling", 0.0f, true);

            hostSim.stdUnder() = ReadValue<float>(nucleationNode, "StdUndercooling", 0.0f, true);
        }

        inline void InitializeSubstrate(const json& substrateNode, Structs::Sim_Dual& DualSim) {
            const Structs::Sim<host_space>& hostSim = DualSim.hostSim;

            hostSim.subGrainSize() = ReadValue<float>(substrateNode, "MeanBaseplateGrainSize", 10.0e-6f, false);
            hostSim.numOrientations() = ReadValue<uint32_t>(substrateNode, "NumGrainOrientations", UINT16_MAX - 1, false);
        }

        inline uint32_t ReadOutputUint(const json& node, const Common::string& label) {
            if (!node.is_number_unsigned() && !node.is_number_integer()) {
                throw std::runtime_error("Input error: Keyword [Output][" + label + "] must be an unsigned integer.");
            }
            const int64_t value = node.get<int64_t>();
            if (value < 0 || static_cast<uint64_t>(value) > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("Input error: Keyword [Output][" + label + "] is outside the valid uint32 range.");
            }
            return static_cast<uint32_t>(value);
        }

        inline int OutputAxisIndex(const Common::string& key) {
            if (key == "X") {
                return 0;
            }
            if (key == "Y") {
                return 1;
            }
            if (key == "Z") {
                return 2;
            }
            throw std::runtime_error("Input error: Output axis keys must be X, Y, or Z (case-sensitive).");
        }

        inline void ReadOutputAxisSetting(const json& settingsNode, const Common::string& blockLabel, const Common::string& key, uint32_t (&values)[3], bool (&wasSet)[3], const bool requirePositive) {
            if (!settingsNode.contains(key) || settingsNode[key].is_null()) {
                return;
            }

            const json& node = settingsNode[key];
            if (node.is_number_unsigned() || node.is_number_integer()) {
                const uint32_t value = ReadOutputUint(node, blockLabel + "][" + key);
                if (requirePositive && value == 0) {
                    throw std::runtime_error("Input error: Keyword [Output][" + blockLabel + "][" + key + "] must be greater than zero.");
                }
                for (int axis = 0; axis < 3; axis++) {
                    values[axis] = value;
                    wasSet[axis] = true;
                }
                return;
            }

            if (node.is_object()) {
                for (auto it = node.begin(); it != node.end(); ++it) {
                    const int axis = OutputAxisIndex(it.key());
                    const uint32_t value = ReadOutputUint(it.value(), blockLabel + "][" + key + "][" + it.key());
                    if (requirePositive && value == 0) {
                        throw std::runtime_error("Input error: Keyword [Output][" + blockLabel + "][" + key + "][" + it.key() + "] must be greater than zero.");
                    }
                    values[axis] = value;
                    wasSet[axis] = true;
                }
                return;
            }

            throw std::runtime_error("Input error: Keyword [Output][" + blockLabel + "][" + key + "] must be an integer or an object keyed by X, Y, and Z (case-sensitive).");
        }

        inline void InitializeOutput(const json& outputNode, Structs::Sim_Dual& DualSim) {
            const json& selectionNode = GetNestedJson(outputNode, "Selection", true);
            const json& grainsNode = GetNestedJson(outputNode, "Grains", true);

            DualSim.simName = ReadValue<Common::string>(outputNode, "Directory", Common::string("ToucanOutput"), true);
            DualSim.outputFormat = ReadValue<Common::string>(outputNode, "Format", Common::string("xdmf"), false);
            DualSim.outputMode = ReadValue<Common::string>(selectionNode, "Dims", Common::string("XYZ"), true);
            DualSim.useGrainIdentifier = ReadValue<bool>(grainsNode, "UniqueIdentifiers", false, false);
            DualSim.outputUnderresolvedGrains = ReadValue<bool>(grainsNode, "UnderResolved", false, false);
            ReadOutputAxisSetting(selectionNode, "Selection", "Stride", DualSim.outputEvery, DualSim.outputEverySet, true);
            ReadOutputAxisSetting(selectionNode, "Selection", "Offset", DualSim.outputOffset, DualSim.outputOffsetSet, false);

            if (selectionNode.contains("Every")) {
                throw std::runtime_error("Input error: Keyword [Output][Selection][Every] is no longer supported. Use [Output][Selection][Stride].");
            }

            if (DualSim.outputMode != "None" &&
                DualSim.outputMode != "XYZ" &&
                DualSim.outputMode != "XY" &&
                DualSim.outputMode != "XZ" &&
                DualSim.outputMode != "YZ") {
                throw std::runtime_error("Input error: Keyword [Output][Selection][Dims] has invalid value.\nValid values are [None, XYZ, XY, XZ, YZ].");
            }

            if (DualSim.outputFormat != "xdmf" && DualSim.outputFormat != "csv") {
                throw std::runtime_error("Input error: Keyword [Output][Format] has invalid value.\nValid values are [xdmf, csv].");
            }

            for (int axis = 0; axis < 3; axis++) {
                if (DualSim.outputEverySet[axis] && DualSim.outputOffsetSet[axis] && DualSim.outputEvery[axis] != std::numeric_limits<uint32_t>::max() && DualSim.outputOffset[axis] >= DualSim.outputEvery[axis]) {
                    throw std::runtime_error("Input error: Keyword [Output][Selection][Offset] must be less than [Output][Selection][Stride] for each specified axis.");
                }
            }

            createDirectoryIfNotExists(DualSim.simName);
        }
    }

    // JSON alternative to SetValues and ReadSimFile - all inputs for spatial dimensions assumed to be in meters
    template<template<typename> class GrainID>
    Structs::Sim_Dual ReadSimFile(const FileReader& inputData, Structs::Sim_Dual& DualSim, Structs::Mpi_Dual<GrainID>& DualMpi, Stork::Structs::RegularGrid_Header<float, host_space>& RDF_header) {

        impl::InitializeIRF(inputData.irf_json, DualSim);
        impl::InitializeRNG(inputData.rng_json, DualSim);
        impl::InitializeDomain(inputData.domain_json, DualSim, RDF_header);
        impl::InitializeMPI(inputData.mpi_json, DualSim, DualMpi);
        impl::InitializeNucleation(inputData.nucleation_json, DualSim);
        impl::InitializeSubstrate(inputData.substrate_json, DualSim);
        impl::InitializeOutput(inputData.output_json, DualSim);
        DualSim.maxIterations = impl::ReadValue<uint32_t>(inputData.debug_json, "MaxIterations", UINT32_MAX, false);

        // Copy values to device
        DualSim.CopyToDevice();
        DualSim.InitializeNucleatedRepeatCounts();

        return DualSim;
    }

    template<template<typename> class GrainID>
    Structs::Sim_Dual ReadSimFile(json& inputData, Structs::Sim_Dual& DualSim, Structs::Mpi_Dual<GrainID>& DualMpi, Stork::Structs::RegularGrid_Header<float, host_space>& RDF_header) {
        FileReader reader;
        reader.Initialize(inputData);
        return ReadSimFile(reader, DualSim, DualMpi, RDF_header);
    }

    void ParseJson_IRF(const json& root, Structs::Sim_Dual& DualSim){
        impl::InitializeIRF(root, DualSim);
    }

    void ParseJson_Domain(const json& root, Structs::Sim_Dual& DualSim, Stork::Structs::RegularGrid_Header<float, host_space>& RDF_header){
        impl::InitializeDomain(root, DualSim, RDF_header);
    }
}
