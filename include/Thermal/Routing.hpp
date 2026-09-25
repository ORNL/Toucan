#pragma once

// Internal includes
#include "Common.hpp"
#include "Definitions.hpp"
#include "Comms/Structs.hpp"
#include "Thermal/File.hpp"
#if TOUCAN_ENABLE_CONDOR
#include "Thermal/Condor.hpp"
#endif

namespace Toucan::Thermal{
    // Using json
    using json = nlohmann::json;

    // Valid types
    const std::vector<std::string> ValidModes = {
        "File",
#if TOUCAN_ENABLE_CONDOR
        "Condor",
#endif
    };
    std::string ValidModeMessage(){
        std::string message = "[";
        for (size_t i=0;i<ValidModes.size();i++){
            if (i > 0){
                message += ", ";
            }
            message += ValidModes[i];
        }
        message += "]";
        return message;
    }

    // The master function
    template<typename ToucanFloat, typename ToucanSpace, template<typename> class GrainID = Structs::BaseGrainIdentifier>
    std::unique_ptr<ThermalSource<ToucanFloat, ToucanSpace>> Routing(const json& root, const Structs::Mpi_Dual<GrainID>& DualMpi){
#if !TOUCAN_ENABLE_CONDOR
        if (root.contains("Condor")) {
            throw std::runtime_error(
                "Input error: Thermal mode \"Condor\" was requested, but this "
                "Toucan build was compiled without Condor support. Reconfigure "
                "Toucan with -DTOUCAN_ENABLE_CONDOR=ON and ensure Condor is "
                "installed, or select the \"File\" thermal mode."
            );
        }
#endif

        // Ensure that we aren't putting too many modes in
        std::string selectedMode = "";
        uint32_t numModesFound = 0;
        // Count number of valid modes
        for (const std::string& mode : ValidModes) {
            if (root.contains(mode)) {
                selectedMode = mode;
                numModesFound++;
            }
        }
        // Error for no modes
        if (numModesFound == 0) {
            throw std::runtime_error(
                "Input error: Thermal routing requires one mode.\nValid values are " +
                ValidModeMessage() + "."
            );
        }
        // Error for multiple modes
        if (numModesFound > 1) {
            throw std::runtime_error(
                "Input error: Thermal routing received more than one mode. "
                "Choose exactly one from " + ValidModeMessage() + "."
            );
        }
        

        // Select child class which returns thermal data
        if (root.contains("File")){
            return std::make_unique<FileSource<ToucanFloat, ToucanSpace, GrainID>>(root["File"],DualMpi);
        }
#if TOUCAN_ENABLE_CONDOR
        if (root.contains("Condor")){
            return std::make_unique<CondorSource<ToucanFloat, ToucanSpace, GrainID>>(root["Condor"], DualMpi);    
        }
#endif
        throw std::runtime_error("Input error: Thermal routing reached an invalid mode selection.");
    }
}  
