#pragma once

// Internal includes
#include "Common.hpp"
#include "Definitions.hpp"
#include "IO/In.hpp"
#include "Thermal/BaseClass.hpp"

// External includes
#include <Condor_Core.hpp>
#include <Stork_Core.hpp>

#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Toucan::Thermal {

    using json = nlohmann::json;

    // Runs Condor for the requested layer, then returns Toucan-ready RDF.
    template <typename ToucanFloat = float, typename ToucanSpace = device_space, template<typename> class GrainID = Structs::BaseGrainIdentifier>
    class CondorSource : public ThermalSource<ToucanFloat, ToucanSpace> {
    private:
        using string = std::string; 
        template <typename T> 
        using vector = std::vector<T>;

        using ThermalSpace = device_space;
        using DualRDF = Stork::Structs::RDF_Dual<ToucanFloat>;

        template <typename ThermalFloat>
        using DualSRDF = Stork::Structs::SRDF_Dual<ThermalFloat>;

        enum class RunMode { NONE, RDF, SRDF };
        enum class Precision { NONE, FLOAT, DOUBLE };

        RunMode mode = RunMode::NONE;
        Precision prec = Precision::NONE;
        vector<string> thermalInputFiles;
        const Structs::Mpi_Dual<GrainID>& dualMpi;

        struct Mode_RDF {
            bool reuse = false;
            bool cache = false;

            void Init(const json& settings) {
                reuse = IO::impl::ReadValue<bool>(settings, "Reuse", false, false);
                cache = IO::impl::ReadValue<bool>(settings, "Cache", false, false);
            }
        } rdf_settings;

        struct Mode_SRDF {
            bool reuse = false;
            bool cache = false;
            uint8_t fineFactor = 0;
            bool normTimes = false;
            double timeScale = 1.0e-2;

            void Init(const json& settings) {
                reuse = IO::impl::ReadValue<bool>(settings, "Reuse", false, false);
                cache = IO::impl::ReadValue<bool>(settings, "Cache", false, false);
                fineFactor = static_cast<uint8_t>(IO::impl::ReadValue<int>(settings, "FineFactor", 0, true));
                normTimes = IO::impl::ReadValue<bool>(settings, "NormalizeTimes", true, false);
                timeScale = IO::impl::ReadValue<double>(settings, "TimeScale", 1.0e-2, false);
            }
        } srdf_settings;

        // Containers for reuse
        vector<DualRDF> RDFs;
        vector<DualSRDF<float>> SRDFs_float;
        vector<DualSRDF<double>> SRDFs_double;

        std::filesystem::path CacheDirectory() const {
            return std::filesystem::current_path() / "cache";
        }

        uint64_t HashString(const string& value) const {
            uint64_t hash = 14695981039346656037ull;

            for (const char c : value) {
                hash ^= static_cast<unsigned char>(c);
                hash *= 1099511628211ull;
            }

            return hash;
        }

        string MpiCacheHash(const string& inputFile) const {
            std::ostringstream key;
            key << "input=" << std::filesystem::path(inputFile).lexically_normal().string()
                << ";nproc=" << dualMpi.nproc
                << ";dims=" << dualMpi.dims[0] << "x" << dualMpi.dims[1]
                << ";coords=" << dualMpi.coords[0] << "x" << dualMpi.coords[1]
                << ";rank=" << dualMpi.rank;

            std::ostringstream hash;
            hash << std::hex << std::setw(16) << std::setfill('0') << HashString(key.str());
            return hash.str();
        }

        string CacheStem(const string& inputFile) const {
            std::filesystem::path path(inputFile);
            return (CacheDirectory() / (path.stem().string() + ".condor-" + MpiCacheHash(inputFile))).string();
        }

        string RDFCacheStem(const string& inputFile) const {
            return CacheStem(inputFile) + ".rdf";
        }

        string SRDFCacheStem(const string& inputFile) const {
            return CacheStem(inputFile) + ".srdf";
        }

        bool CacheExists(const string& storkStem) const {
            return std::filesystem::exists(storkStem + ".stork");
        }

        void EnsureCacheDirectoryExists() const {
            std::filesystem::create_directories(CacheDirectory());
        }

        void InitMode(const json& settings) {
            if (settings.contains("RDF") && !settings.contains("SRDF")) {
                mode = RunMode::RDF;
                rdf_settings.Init(settings["RDF"]);
                return;
            }
            else if (settings.contains("SRDF") && !settings.contains("RDF")) {
                mode = RunMode::SRDF;
                srdf_settings.Init(settings["SRDF"]);
                return;
            }
            throw std::runtime_error("Thermal::Condor -> Invalid Format. Valid values [RDF, SRDF].");
        }

        void InitPrecision(const json& settings) {
            const string read_prec = IO::impl::ReadValue<string>(settings, "Precision", string("double"), false);

            if (read_prec == "float") {
                prec = Precision::FLOAT;
                return;
            }

            if (read_prec == "double") {
                prec = Precision::DOUBLE;
                return;
            }

            throw std::runtime_error("Thermal::Condor -> Invalid Precision. Valid values [float, double].");
        }

        DualRDF RunRDF(const uint32_t thermalIndex, const string& paramInputFile) {
            if (rdf_settings.reuse) {
                DualRDF& RDF = RDFs[thermalIndex];
                if (RDF.numEvents == 0) {
                    LoadOrRunRDF(RDF, paramInputFile);
                }

                return RDF;
            }

            DualRDF RDF;
            LoadOrRunRDF(RDF, paramInputFile);
            return RDF;
        }

        void LoadOrRunRDF(DualRDF& RDF, const string& paramInputFile) {
            const string cacheStem = RDFCacheStem(paramInputFile);

            if (rdf_settings.cache && CacheExists(cacheStem)) {
                Stork::IO::Input_RDF_binary<ToucanFloat>(RDF, cacheStem);
                RDF.template Make_Data_Mirrors<host_space, ToucanSpace>();
                RDF.template Copy_All<host_space, ToucanSpace>();
                return;
            }

            Condor::Run::Coupled<ToucanFloat>(RDF, paramInputFile);
            RDF.template Copy_Header<ToucanSpace, host_space>();

            if (rdf_settings.cache) {
                RDF.template Make_Data_Mirrors<ToucanSpace, host_space>();
                RDF.template Copy_All<ToucanSpace, host_space>();
                EnsureCacheDirectoryExists();
                Stork::IO::Output_RDF_binary<ToucanFloat>(RDF, cacheStem);
            }
        }

        template <typename ThermalFloat>
        void FillSRDF(DualSRDF<ThermalFloat>& SRDF, const string& paramInputFile) {
            Condor::Run::Coupled<ThermalFloat>(SRDF, paramInputFile);

            if (srdf_settings.normTimes) {
                Stork::Run::Normalize_SRDF_Times<ThermalFloat, ThermalSpace>(
                    SRDF,
                    static_cast<ThermalFloat>(srdf_settings.timeScale)
                );
            }

            SRDF.template Copy_Header<ThermalSpace, host_space>();
        }

        template <typename ThermalFloat>
        DualRDF ConvertSRDFToRDF(DualSRDF<ThermalFloat>& SRDF) {
            const int I = dualMpi.dims[0];
            const int J = dualMpi.dims[1];
            const int i = dualMpi.coords[0];
            const int j = dualMpi.coords[1];

            const uint8_t fineFactor = srdf_settings.fineFactor;
            const uint32_t RDF_global_iMin =
                SRDF.host_header.global_i0() * fineFactor + (fineFactor / 2) * (i != 0);
            const uint32_t RDF_global_iMax =
                (SRDF.host_header.global_i0() + SRDF.host_header.local_inum() - 1) * fineFactor -
                ((fineFactor - 1) / 2) * (i != (I - 1));
            const uint32_t RDF_global_jMin =
                SRDF.host_header.global_j0() * fineFactor + (fineFactor / 2) * (j != 0);
            const uint32_t RDF_global_jMax =
                (SRDF.host_header.global_j0() + SRDF.host_header.local_jnum() - 1) * fineFactor -
                ((fineFactor - 1) / 2) * (j != (J - 1));

            DualRDF RDF =
                Stork::Run::Interpolate_And_Trim_SRDF_to_RDF<ThermalFloat, ThermalSpace, ToucanFloat, ToucanSpace>(
                    SRDF,
                    fineFactor,
                    RDF_global_iMin,
                    RDF_global_iMax,
                    RDF_global_jMin,
                    RDF_global_jMax
                );

            Stork::Run::Trim_RDF_in_Z<ToucanFloat, ToucanSpace>(RDF);
            RDF.template Copy_Header<ToucanSpace, host_space>();
            return RDF;
        }

        template <typename ThermalFloat>
        DualRDF RunSRDFTyped(
            const uint32_t thermalIndex,
            const string& paramInputFile,
            vector<DualSRDF<ThermalFloat>>& SRDFs
        ) {
            if (srdf_settings.reuse) {
                DualSRDF<ThermalFloat>& SRDF = SRDFs[thermalIndex];
                if (SRDF.numSnaps == 0) {
                    LoadOrRunSRDF<ThermalFloat>(SRDF, paramInputFile);
                }

                return ConvertSRDFToRDF<ThermalFloat>(SRDF);
            }

            DualSRDF<ThermalFloat> SRDF;
            LoadOrRunSRDF<ThermalFloat>(SRDF, paramInputFile);

            return ConvertSRDFToRDF<ThermalFloat>(SRDF);
        }

        template <typename ThermalFloat>
        void LoadOrRunSRDF(DualSRDF<ThermalFloat>& SRDF, const string& paramInputFile) {
            const string cacheStem = SRDFCacheStem(paramInputFile);

            if (srdf_settings.cache && CacheExists(cacheStem)) {
                Stork::IO::Input_SRDF_binary<ThermalFloat>(SRDF, cacheStem);
                SRDF.template Make_Data_Mirrors<host_space, ThermalSpace>();
                SRDF.template Copy_All<host_space, ThermalSpace>();
                return;
            }

            FillSRDF<ThermalFloat>(SRDF, paramInputFile);

            if (srdf_settings.cache) {
                SRDF.template Make_Data_Mirrors<ThermalSpace, host_space>();
                SRDF.template Copy_All<ThermalSpace, host_space>();
                EnsureCacheDirectoryExists();
                Stork::IO::Output_SRDF_binary<ThermalFloat>(SRDF, cacheStem);
            }
        }

        DualRDF RunSRDF(const uint32_t thermalIndex, const string& paramInputFile) {
            if (prec == Precision::FLOAT) {
                return RunSRDFTyped<float>(thermalIndex, paramInputFile, SRDFs_float);
            }

            if (prec == Precision::DOUBLE) {
                return RunSRDFTyped<double>(thermalIndex, paramInputFile, SRDFs_double);
            }

            throw std::runtime_error("Thermal::Condor -> Invalid Precision. Valid values [float, double].");
        }

    public:
        CondorSource(const json& settings, const Structs::Mpi_Dual<GrainID>& DualMpi)
            : dualMpi(DualMpi)
        {
            if (settings.contains("Files") && settings["Files"].is_array()) {
                thermalInputFiles = settings["Files"].get<vector<string>>();
            }
            else {
                throw std::runtime_error("Thermal::Condor -> Input files either not specified or invalid.");
            }

            InitPrecision(settings);
            InitMode(settings);

            RDFs.resize(thermalInputFiles.size());
            SRDFs_float.resize(thermalInputFiles.size());
            SRDFs_double.resize(thermalInputFiles.size());
        }

        DualRDF GetRDF(const uint32_t layer) override {
            if (thermalInputFiles.empty()) {
                throw std::runtime_error("Thermal::Condor -> No thermal input files are available.");
            }

            const uint32_t thermalIndex = static_cast<uint32_t>(layer % thermalInputFiles.size());
            const string& inputFile = thermalInputFiles[thermalIndex];

            if (mode == RunMode::RDF) {
                return RunRDF(thermalIndex, inputFile);
            }

            if (mode == RunMode::SRDF) {
                return RunSRDF(thermalIndex, inputFile);
            }

            throw std::runtime_error("Thermal::Condor -> Invalid Format. Valid values [RDF, SRDF].");
        }
    };
}
