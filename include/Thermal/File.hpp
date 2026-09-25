#pragma once

// Internal includes
#include "Common.hpp"
#include "Definitions.hpp"
#include "IO/In.hpp"
#include "Thermal/BaseClass.hpp"

// External includes
#include <Stork_Core.hpp>

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Toucan::Thermal {

    using json = nlohmann::json;

    // Reads pre-existing Stork thermal files and returns Toucan-ready RDF.
    template <typename ToucanFloat = float, typename ToucanSpace = device_space, template<typename> class GrainID = Structs::BaseGrainIdentifier>
    class FileSource : public ThermalSource<ToucanFloat, ToucanSpace> {
    private:
        using string = std::string;
        template <typename T>
        using vector = std::vector<T>;

        using ThermalSpace = device_space;
        using DualRDF = Stork::Structs::RDF_Dual<ToucanFloat>;

        template <typename ThermalFloat>
        using DualSRDF = Stork::Structs::SRDF_Dual<ThermalFloat>;

        enum class Format { NONE, RDF, SRDF };
        enum class Precision { NONE, FLOAT, DOUBLE };

        Format format = Format::NONE;
        Precision prec = Precision::NONE;
        bool reuse = false;
        uint8_t fineFactor = 1;
        bool normTimes = false;
        double timeScale = 1.0e-2;

        vector<string> filePatterns;
        vector<uint32_t> fileLayers;
        vector<uint32_t> fileRanks;
        const Structs::Mpi_Dual<GrainID>& dualMpi;

        struct IndexedFile {
            uint32_t layer = 0;
            uint32_t rank = 0;
            string stem;
            string source;
        };

        struct TokenState {
            bool hasRank = false;
            bool hasLayer = false;
            uint32_t rank = 0;
            uint32_t layer = 0;
        };

        vector<IndexedFile> indexedFiles;

        // Containers for reuse on device memory.
        vector<DualRDF> RDFs;
        vector<DualSRDF<float>> SRDFs_float;
        vector<DualSRDF<double>> SRDFs_double;

        static string StripStorkExtension(const string& fileName) {
            std::filesystem::path path(fileName);
            if (path.extension() == ".stork") {
                return path.replace_extension("").string();
            }
            return fileName;
        }

        uint32_t LayerNumber(const uint32_t layer) const {
            if (fileLayers.empty()) {
                return layer;
            }
            return fileLayers[layer % fileLayers.size()];
        }

        uint32_t RankNumber() const {
            if (fileRanks.empty()) {
                return static_cast<uint32_t>(dualMpi.rank);
            }

            if (static_cast<size_t>(dualMpi.rank) >= fileRanks.size()) {
                throw std::runtime_error(
                    "Thermal::File -> MPI rank " + std::to_string(dualMpi.rank) +
                    " has no corresponding file rank in Ranks."
                );
            }

            return fileRanks[dualMpi.rank];
        }

        static bool IsDigits(const string& value) {
            if (value.empty()) {
                return false;
            }

            for (const char c : value) {
                if (!std::isdigit(static_cast<unsigned char>(c))) {
                    return false;
                }
            }

            return true;
        }

        static uint32_t ParseUInt32(const string& value, const string& label) {
            if (!IsDigits(value)) {
                throw std::runtime_error("Thermal::File -> Parsed " + label + " is not numeric.");
            }

            const unsigned long parsed = std::stoul(value);
            if (parsed > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("Thermal::File -> Parsed " + label + " exceeds uint32_t.");
            }
            return static_cast<uint32_t>(parsed);
        }

        static size_t CountToken(const string& value, const string& token) {
            size_t count = 0;
            size_t pos = 0;
            while ((pos = value.find(token, pos)) != string::npos) {
                ++count;
                pos += token.size();
            }
            return count;
        }

        static size_t FindNextToken(const string& pattern, const size_t start, string& token) {
            const size_t rankPos = pattern.find("RANK", start);
            const size_t layerPos = pattern.find("LAYER", start);

            if (rankPos == string::npos && layerPos == string::npos) {
                token.clear();
                return string::npos;
            }

            if (layerPos == string::npos || (rankPos != string::npos && rankPos < layerPos)) {
                token = "RANK";
                return rankPos;
            }

            token = "LAYER";
            return layerPos;
        }

        static bool HasToken(const string& value) {
            return value.find("RANK") != string::npos || value.find("LAYER") != string::npos;
        }

        static void ValidatePatternComponent(const string& component) {
            if (CountToken(component, "RANK") > 1 || CountToken(component, "LAYER") > 1) {
                throw std::runtime_error("Thermal::File -> Each path component may only contain one RANK token and one LAYER token.");
            }

            if (component.find("RANKLAYER") != string::npos || component.find("LAYERRANK") != string::npos) {
                throw std::runtime_error("Thermal::File -> RANK and LAYER tokens must be separated by literal text.");
            }
        }

        void ValidatePattern(const string& pattern) const {
            const std::filesystem::path patternPath(pattern);
            const std::filesystem::path relativePattern =
                patternPath.is_absolute() ? patternPath.relative_path() : patternPath;

            for (const auto& component : relativePattern) {
                ValidatePatternComponent(component.string());
            }
        }

        static bool MatchLiteral(const string& fileName, const size_t pos, const string& literal) {
            if (pos > fileName.size() || literal.size() > fileName.size() - pos) {
                return false;
            }
            return fileName.compare(pos, literal.size(), literal) == 0;
        }

        static bool ReadTokenValue(
            const string& fileName,
            size_t& filePos,
            const string& nextLiteral,
            string& tokenValue
        ) {
            if (nextLiteral.empty()) {
                const size_t start = filePos;
                while (filePos < fileName.size() && std::isdigit(static_cast<unsigned char>(fileName[filePos]))) {
                    ++filePos;
                }
                tokenValue = fileName.substr(start, filePos - start);
                return IsDigits(tokenValue);
            }

            size_t literalPos = fileName.find(nextLiteral, filePos);
            while (literalPos != string::npos) {
                tokenValue = fileName.substr(filePos, literalPos - filePos);
                if (IsDigits(tokenValue)) {
                    filePos = literalPos;
                    return true;
                }
                literalPos = fileName.find(nextLiteral, literalPos + 1);
            }

            return false;
        }

        bool ParsePathComponent(
            const string& patternComponent,
            const string& actualComponent,
            TokenState& state
        ) const {
            string rankValue;
            string layerValue;
            size_t patternPos = 0;
            size_t filePos = 0;

            while (patternPos < patternComponent.size()) {
                string token;
                const size_t tokenPos = FindNextToken(patternComponent, patternPos, token);

                if (tokenPos == string::npos) {
                    const string literal = patternComponent.substr(patternPos);
                    if (!MatchLiteral(actualComponent, filePos, literal)) {
                        return false;
                    }
                    filePos += literal.size();
                    patternPos = patternComponent.size();
                    continue;
                }

                const string literal = patternComponent.substr(patternPos, tokenPos - patternPos);
                if (!MatchLiteral(actualComponent, filePos, literal)) {
                    return false;
                }
                filePos += literal.size();

                const size_t afterToken = tokenPos + token.size();
                string nextToken;
                const size_t nextTokenPos = FindNextToken(patternComponent, afterToken, nextToken);
                const string nextLiteral = patternComponent.substr(
                    afterToken,
                    (nextTokenPos == string::npos) ? string::npos : nextTokenPos - afterToken
                );

                string tokenValue;
                if (!ReadTokenValue(actualComponent, filePos, nextLiteral, tokenValue)) {
                    return false;
                }

                if (token == "RANK") {
                    rankValue = tokenValue;
                }
                else {
                    layerValue = tokenValue;
                }

                patternPos = afterToken;
            }

            if (filePos != actualComponent.size()) {
                return false;
            }

            if (!rankValue.empty()) {
                const uint32_t rank = ParseUInt32(rankValue, "RANK");
                if (state.hasRank && state.rank != rank) {
                    return false;
                }
                state.hasRank = true;
                state.rank = rank;
            }

            if (!layerValue.empty()) {
                const uint32_t layer = ParseUInt32(layerValue, "LAYER");
                if (state.hasLayer && state.layer != layer) {
                    return false;
                }
                state.hasLayer = true;
                state.layer = layer;
            }

            return true;
        }

        void AddIndexedFile(const IndexedFile& indexedFile, std::unordered_map<uint32_t, string>& layersSeen) {
            const auto it = layersSeen.find(indexedFile.layer);
            if (it != layersSeen.end()) {
                if (it->second == indexedFile.source) {
                    return;
                }

                throw std::runtime_error(
                    "Thermal::File -> Multiple files matched rank " + std::to_string(indexedFile.rank) +
                    " and layer " + std::to_string(indexedFile.layer) + ":\n  " + it->second +
                    "\n  " + indexedFile.source
                );
            }

            layersSeen[indexedFile.layer] = indexedFile.source;
            indexedFiles.push_back(indexedFile);
        }

        vector<string> PatternComponents(const string& pattern) const {
            const std::filesystem::path patternPath(pattern);
            const std::filesystem::path relativePattern =
                patternPath.is_absolute() ? patternPath.relative_path() : patternPath;

            vector<string> components;
            for (const auto& component : relativePattern) {
                components.push_back(component.string());
            }

            if (components.empty()) {
                throw std::runtime_error("Thermal::File -> File pattern is empty.");
            }

            if (std::filesystem::path(components.back()).extension() != ".stork") {
                components.back() += ".stork";
            }

            return components;
        }

        std::filesystem::path PatternRoot(const string& pattern) const {
            const std::filesystem::path patternPath(pattern);
            if (patternPath.is_absolute()) {
                return patternPath.root_path();
            }
            return std::filesystem::current_path();
        }

        void IndexMatchedFile(
            const std::filesystem::path& filePath,
            const TokenState& state,
            std::unordered_map<uint32_t, string>& layersSeen
        ) {
            const uint32_t rank = state.hasRank ? state.rank : RankNumber();
            const uint32_t layer = state.hasLayer ? state.layer : 0u;

            if (rank != RankNumber()) {
                return;
            }

            AddIndexedFile(
                IndexedFile{
                    layer,
                    rank,
                    StripStorkExtension(filePath.string()),
                    filePath.string()
                },
                layersSeen
            );
        }

        void WalkPatternComponents(
            const vector<string>& components,
            const size_t componentIndex,
            const std::filesystem::path& currentPath,
            const TokenState& state,
            std::unordered_map<uint32_t, string>& layersSeen
        ) {
            const bool isFinalComponent = (componentIndex + 1 == components.size());
            const string& componentPattern = components[componentIndex];

            if (!HasToken(componentPattern)) {
                TokenState nextState = state;
                const std::filesystem::path nextPath = currentPath / componentPattern;

                if (isFinalComponent) {
                    if (std::filesystem::is_regular_file(nextPath) && nextPath.extension() == ".stork") {
                        IndexMatchedFile(nextPath, nextState, layersSeen);
                    }
                    return;
                }

                if (std::filesystem::is_directory(nextPath)) {
                    WalkPatternComponents(components, componentIndex + 1, nextPath, nextState, layersSeen);
                }
                return;
            }

            if (!std::filesystem::is_directory(currentPath)) {
                return;
            }

            for (const auto& entry : std::filesystem::directory_iterator(currentPath)) {
                if (isFinalComponent) {
                    if (!entry.is_regular_file() || entry.path().extension() != ".stork") {
                        continue;
                    }
                }
                else if (!entry.is_directory()) {
                    continue;
                }

                TokenState nextState = state;
                if (!ParsePathComponent(componentPattern, entry.path().filename().string(), nextState)) {
                    continue;
                }

                if (isFinalComponent) {
                    IndexMatchedFile(entry.path(), nextState, layersSeen);
                }
                else {
                    WalkPatternComponents(components, componentIndex + 1, entry.path(), nextState, layersSeen);
                }
            }
        }

        void IndexPatternFiles(const string& pattern, std::unordered_map<uint32_t, string>& layersSeen) {
            ValidatePattern(pattern);

            const vector<string> components = PatternComponents(pattern);
            const std::filesystem::path root = PatternRoot(pattern);

            if (!std::filesystem::exists(root) || !std::filesystem::is_directory(root)) {
                throw std::runtime_error("Thermal::File -> Directory does not exist: " + root.string());
            }

            TokenState state;
            WalkPatternComponents(components, 0, root, state, layersSeen);
        }

        void ValidateAllRequiredFiles(const std::unordered_map<uint32_t, string>& layersSeen) const {
            vector<uint32_t> missingLayers;
            for (const uint32_t layer : fileLayers) {
                if (layersSeen.find(layer) == layersSeen.end()) {
                    missingLayers.push_back(layer);
                }
            }

            if (missingLayers.empty()) {
                return;
            }

            std::ostringstream message;
            message << "Thermal::File -> Missing required thermal files for file rank " << RankNumber() << ":";
            for (const uint32_t layer : missingLayers) {
                message << "\n  layer " << layer;
            }
            throw std::runtime_error(message.str());
        }

        void BuildFileIndex() {
            if (filePatterns.empty()) {
                throw std::runtime_error("Thermal::File -> No file strings are available.");
            }

            std::unordered_map<uint32_t, string> layersSeen;
            for (const string& pattern : filePatterns) {
                IndexPatternFiles(pattern, layersSeen);
            }
            ValidateAllRequiredFiles(layersSeen);
        }

        string ResolveFileName(const uint32_t layer) const {
            const uint32_t fileLayer = LayerNumber(layer);
            for (const IndexedFile& indexedFile : indexedFiles) {
                if (indexedFile.layer == fileLayer) {
                    return indexedFile.stem;
                }
            }

            std::ostringstream message;
            message << "Thermal::File -> Could not find thermal file for layer " << layer
                    << " (file layer " << fileLayer << ", file rank " << RankNumber() << ").";
            throw std::runtime_error(message.str());
        }

        void InitFormat(const json& settings) {
            if (settings.contains("RDF") && !settings.contains("SRDF")) {
                format = Format::RDF;
                return;
            }

            if (settings.contains("SRDF") && !settings.contains("RDF")) {
                format = Format::SRDF;
                return;
            }

            const string readFormat = IO::impl::ReadValue<string>(settings, "Format", string(""), false);
            if (readFormat == "RDF") {
                format = Format::RDF;
                return;
            }
            if (readFormat == "SRDF") {
                format = Format::SRDF;
                return;
            }

            throw std::runtime_error("Thermal::File -> Invalid Format. Valid values [RDF, SRDF].");
        }

        void InitPrecision(const json& settings) {
            const string readPrec = IO::impl::ReadValue<string>(settings, "Precision", string("double"), false);

            if (readPrec == "float") {
                prec = Precision::FLOAT;
                return;
            }

            if (readPrec == "double") {
                prec = Precision::DOUBLE;
                return;
            }

            throw std::runtime_error("Thermal::File -> Invalid Precision. Valid values [float, double].");
        }

        void InitFiles(const json& settings) {
            if (settings.contains("String")) {
                if (settings["String"].is_string()) {
                    filePatterns.push_back(settings["String"].get<string>());
                    return;
                }
                if (settings["String"].is_array()) {
                    filePatterns = settings["String"].get<vector<string>>();
                    return;
                }
            }

            if (settings.contains("Files") && settings["Files"].is_array()) {
                filePatterns = settings["Files"].get<vector<string>>();
                return;
            }

            throw std::runtime_error("Thermal::File -> File strings either not specified or invalid.");
        }

        vector<uint32_t> MakeZeroBasedRange(const int count, const string& key) const {
            if (count <= 0) {
                throw std::runtime_error("Thermal::File -> " + key + " must be greater than zero.");
            }

            vector<uint32_t> values;
            values.reserve(static_cast<size_t>(count));
            for (int value = 0; value < count; ++value) {
                values.push_back(static_cast<uint32_t>(value));
            }
            return values;
        }

        vector<uint32_t> MakeInclusiveRange(const int start, const int stop, const string& key) const {
            if (start < 0 || stop < 0) {
                throw std::runtime_error("Thermal::File -> " + key + " values must be non-negative.");
            }
            if (stop < start) {
                throw std::runtime_error("Thermal::File -> " + key + " stop must be greater than or equal to start.");
            }

            vector<uint32_t> values;
            values.reserve(static_cast<size_t>(stop - start + 1));
            for (int value = start; value <= stop; ++value) {
                values.push_back(static_cast<uint32_t>(value));
            }
            return values;
        }

        vector<uint32_t> ReadNumberRange(const json& settings, const string& key, const int defaultCount) const {
            if (!settings.contains(key) || settings[key].is_null()) {
                return MakeZeroBasedRange(defaultCount, key);
            }

            if (settings[key].is_number_integer()) {
                return MakeZeroBasedRange(settings[key].get<int>(), key);
            }

            if (settings[key].is_array()) {
                if (settings[key].size() != 2) {
                    throw std::runtime_error("Thermal::File -> " + key + " array must be [start, stop].");
                }
                return MakeInclusiveRange(settings[key][0].get<int>(), settings[key][1].get<int>(), key);
            }

            throw std::runtime_error("Thermal::File -> " + key + " must be an integer count or [start, stop].");
        }

        void InitLayerNumbers(const json& settings) {
            fileLayers = ReadNumberRange(settings, "Layers", 1);
        }

        void InitRankNumbers(const json& settings) {
            fileRanks = ReadNumberRange(settings, "Ranks", dualMpi.nproc);
        }

        void InitSettings(const json& settings) {
            const json* modeSettings = nullptr;
            if (format == Format::RDF && settings.contains("RDF")) {
                modeSettings = &settings["RDF"];
            }
            if (format == Format::SRDF && settings.contains("SRDF")) {
                modeSettings = &settings["SRDF"];
            }

            const json& source = (modeSettings == nullptr) ? settings : *modeSettings;
            reuse = IO::impl::ReadValue<bool>(source, "Reuse", IO::impl::ReadValue<bool>(settings, "Reuse", false, false), false);
            normTimes = IO::impl::ReadValue<bool>(source, "NormalizeTimes", IO::impl::ReadValue<bool>(settings, "NormalizeTimes", false, false), false);
            timeScale = IO::impl::ReadValue<double>(source, "TimeScale", IO::impl::ReadValue<double>(settings, "TimeScale", 1.0e-2, false), false);

            if (format == Format::SRDF) {
                const int readFineFactor = IO::impl::ReadValue<int>(source, "FineFactor", IO::impl::ReadValue<int>(settings, "FineFactor", 0, false), true);
                if (readFineFactor <= 0) {
                    throw std::runtime_error("Thermal::File -> FineFactor must be greater than zero.");
                }
                fineFactor = static_cast<uint8_t>(readFineFactor);
            }
        }

        template <typename ThermalFloat>
        DualRDF ConvertRDFToToucan(Stork::Structs::RDF_Dual<ThermalFloat>& source) const {
            DualRDF RDF;
            RDF.numEvents = source.numEvents;

            Kokkos::deep_copy(RDF.host_header.index_view, source.host_header.index_view);
            for (size_t n = 0; n < source.host_header.floatType_view.size(); ++n) {
                RDF.host_header.floatType_view(n) = static_cast<ToucanFloat>(source.host_header.floatType_view(n));
            }

            RDF.template Make_Data_Views<host_space>(source.numEvents);
            Kokkos::deep_copy(RDF.host_data.cellNum_view, source.host_data.cellNum_view);
            for (size_t n = 0; n < source.host_data.solInfo_view.size(); ++n) {
                RDF.host_data.solInfo_view(n) = static_cast<ToucanFloat>(source.host_data.solInfo_view(n));
            }

            RDF.template Make_Data_Mirrors<host_space, ToucanSpace>();
            RDF.template Copy_All<host_space, ToucanSpace>();
            return RDF;
        }

        template <typename ThermalFloat>
        DualRDF LoadRDFTyped(const string& fileStem) const {
            Stork::Structs::RDF_Dual<ThermalFloat> RDF;
            Stork::IO::Input_RDF_binary<ThermalFloat>(RDF, fileStem);

            if (normTimes) {
                Stork::Run::Normalize_RDF_Times<ThermalFloat, host_space>(
                    RDF,
                    static_cast<ThermalFloat>(timeScale)
                );
            }

            if constexpr (std::is_same_v<ThermalFloat, ToucanFloat>) {
                RDF.template Make_Data_Mirrors<host_space, ToucanSpace>();
                RDF.template Copy_All<host_space, ToucanSpace>();
                return RDF;
            }
            else {
                return ConvertRDFToToucan<ThermalFloat>(RDF);
            }
        }

        DualRDF LoadRDF(const string& fileStem) const {
            if (prec == Precision::FLOAT) {
                return LoadRDFTyped<float>(fileStem);
            }

            if (prec == Precision::DOUBLE) {
                return LoadRDFTyped<double>(fileStem);
            }

            throw std::runtime_error("Thermal::File -> Invalid Precision. Valid values [float, double].");
        }

        template <typename ThermalFloat>
        void LoadSRDF(DualSRDF<ThermalFloat>& SRDF, const string& fileStem) const {
            Stork::IO::Input_SRDF_binary<ThermalFloat>(SRDF, fileStem);

            if (normTimes) {
                Stork::Run::Normalize_SRDF_Times<ThermalFloat, host_space>(
                    SRDF,
                    static_cast<ThermalFloat>(timeScale)
                );
            }

            SRDF.template Make_Data_Mirrors<host_space, ThermalSpace>();
            SRDF.template Copy_All<host_space, ThermalSpace>();
        }

        template <typename ThermalFloat>
        DualRDF ConvertSRDFToRDF(DualSRDF<ThermalFloat>& SRDF) const {
            const int I = dualMpi.dims[0];
            const int J = dualMpi.dims[1];
            const int i = dualMpi.coords[0];
            const int j = dualMpi.coords[1];

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

        DualRDF RunRDF(const uint32_t fileIndex, const string& fileStem) {
            if (reuse) {
                DualRDF& RDF = RDFs[fileIndex];
                if (RDF.numEvents == 0) {
                    RDF = LoadRDF(fileStem);
                }

                return RDF;
            }

            return LoadRDF(fileStem);
        }

        template <typename ThermalFloat>
        DualRDF RunSRDFTyped(
            const uint32_t fileIndex,
            const string& fileStem,
            vector<DualSRDF<ThermalFloat>>& SRDFs
        ) {
            if (reuse) {
                DualSRDF<ThermalFloat>& SRDF = SRDFs[fileIndex];
                if (SRDF.numSnaps == 0) {
                    LoadSRDF<ThermalFloat>(SRDF, fileStem);
                }

                return ConvertSRDFToRDF<ThermalFloat>(SRDF);
            }

            DualSRDF<ThermalFloat> SRDF;
            LoadSRDF<ThermalFloat>(SRDF, fileStem);
            return ConvertSRDFToRDF<ThermalFloat>(SRDF);
        }

        DualRDF RunSRDF(const uint32_t fileIndex, const string& fileStem) {
            if (prec == Precision::FLOAT) {
                return RunSRDFTyped<float>(fileIndex, fileStem, SRDFs_float);
            }

            if (prec == Precision::DOUBLE) {
                return RunSRDFTyped<double>(fileIndex, fileStem, SRDFs_double);
            }

            throw std::runtime_error("Thermal::File -> Invalid Precision. Valid values [float, double].");
        }

    public:
        FileSource(const json& settings, const Structs::Mpi_Dual<GrainID>& DualMpi)
            : dualMpi(DualMpi)
        {
            InitFiles(settings);
            InitLayerNumbers(settings);
            InitRankNumbers(settings);
            InitFormat(settings);
            InitPrecision(settings);
            InitSettings(settings);
            BuildFileIndex();

            RDFs.resize(fileLayers.size());
            SRDFs_float.resize(fileLayers.size());
            SRDFs_double.resize(fileLayers.size());
        }

        DualRDF GetRDF(const uint32_t layer) override {
            const uint32_t fileIndex = static_cast<uint32_t>(layer % fileLayers.size());
            const string fileStem = ResolveFileName(layer);

            if (format == Format::RDF) {
                return RunRDF(fileIndex, fileStem);
            }

            if (format == Format::SRDF) {
                return RunSRDF(fileIndex, fileStem);
            }

            throw std::runtime_error("Thermal::File -> Invalid Format. Valid values [RDF, SRDF].");
        }
    };
}
