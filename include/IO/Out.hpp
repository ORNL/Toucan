#pragma once

#include "Definitions.hpp"
#include "Common.hpp"

#include "Structs/Orientations.hpp"
#include "Structs/Substrate.hpp"
#include "Structs/Layer.hpp"
#include "Structs/Grid.hpp"

namespace Toucan::IO {

    // For ZeroPadNumber
    namespace impl {
        // Common function
        inline Common::string ZeroPadNumber(const int num, const int width)
        {
            std::ostringstream ss;
            ss << std::setw(width) << std::setfill('0') << num;
            return ss.str();
        }
    }

    template<template<typename> class GrainID>
    void SubToCSV(Structs::Substrate_Dual<GrainID>& DualSub, Structs::Mpi_Dual<GrainID>& DualMpi, Structs::Sim_Dual& DualSim, const uint32_t windowNum, const Structs::OutputRegion& region) {

        // Set References
        auto& sub = DualSub.hostSub;
        Stork::Structs::RegularGrid_Header<float, host_space>& header = DualMpi.header;

        // Set directory
        const Common::string dir = DualSim.simName + "/";

        // Gets starts and extents
        const uint32_t start[3] = {region.localStart[0], region.localStart[1], region.localStart[2]};
        const uint32_t extent[3] = {region.count[0], region.count[1], region.count[2]};
        const uint32_t stride[3] = {region.stride[0], region.stride[1], region.stride[2]};
        const float resolution = header.gridResolution();

        // Output CSV
        std::ofstream datafile;
        datafile.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        Common::string out_file = dir + "Rank." + impl::ZeroPadNumber(DualMpi.rank, std::max(static_cast<int>(log10(DualMpi.nproc)+0.5),1)) + ".Output." + impl::ZeroPadNumber(windowNum,1+int(log10(DualSim.hostSim.numLayers()))) + region.sliceSuffix + ".csv";
        try {
            datafile.open(out_file.c_str());
            datafile << "x,y,z,dirID";
            sub.grainID.csv_header(datafile);
            datafile << "\n";
            // Now iterate ONLY over the copied subview (i->j->k for contiguous access)
            for (uint32_t dk = 0; dk < extent[2]; dk++) {
                for (uint32_t dj = 0; dj < extent[1]; dj++) {
                    for (uint32_t di = 0; di < extent[0]; di++) {
                        // Get indices
                        const uint32_t i = start[0] + di * stride[0];
                        const uint32_t j = start[1] + dj * stride[1];
                        const uint32_t k = start[2] + dk * stride[2];
                            const uint32_t p = sub.ijk_to_p(i, j, k);
                            // If from the grid, output it
                            if (sub.fromGrid(p)) {
                                const float x = header.global_x0() + (region.globalStart[0] + di * region.globalStride[0]) * resolution;
                                const float y = header.global_y0() + (region.globalStart[1] + dj * region.globalStride[1]) * resolution;
                                const float z = k * resolution + DualSub.zmin;
                                const uint32_t dirID_loc = static_cast<uint32_t>(sub.dirID(p));
                                datafile << x << "," << y << "," << z << "," << dirID_loc;
                                sub.grainID.csv_data(datafile, p);
                                datafile << "\n";
                            }
                    }
                }
            }
        }
        catch (const std::ofstream::failure& e) {
            std::cout << "Exception writing data file, check that Data directory exists\n";
        }
        datafile.close();
    }

    // Helper functions for writing in binary (xdmf format)
    namespace {
        template <typename out_data_type>
        void writeRawData(const std::string& filename, const std::vector<out_data_type>& data) {
            std::ofstream bin_file(filename.c_str(), std::ios::binary);
            bin_file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(out_data_type));
            bin_file.close();
        }

        void writeXDMF(const std::string& xmf_filename, const std::vector<std::pair<std::string, std::string>>& attributes, const uint32_t extent[3], const float origin[3], const float spacing[3]) {

            // Unpack extent into named variables for clarity.
            const uint32_t nx = extent[0]; // i-dimension
            const uint32_t ny = extent[1]; // j-dimension
            const uint32_t nz = extent[2]; // k-dimension (slowest changing)

            // Unpack origin and spacing.
            const float ox = origin[0], oy = origin[1], oz = origin[2];
            const float dx = spacing[0], dy = spacing[1], dz = spacing[2];

            std::ofstream xmf_file(xmf_filename);
            if (!xmf_file) {
                std::cerr << "Error: Could not open XMF file for writing: " << xmf_filename << std::endl;
                return;
            }

            // Use a raw string literal (C++11) or standard string streaming for the header.
            xmf_file << "<?xml version=\"1.0\" ?>\n";
            xmf_file << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
            // --- CHANGE 1: Use modern Xdmf Version 3.0 ---
            xmf_file << "<Xdmf Version=\"3.0\">\n";
            xmf_file << "  <Domain>\n";
            xmf_file << "    <Grid Name=\"Microstructure\" GridType=\"Uniform\">\n";
            // Defines the grid structure (slowest-to-fastest order)
            xmf_file << "      <Topology TopologyType=\"3DCORECTMesh\" Dimensions=\"" << nz << " " << ny << " " << nx << "\"/>\n\n";
            // Defines the physical coordinates (order should match topology Z, Y, X to match the Topology)
            xmf_file << "      <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n";
            // --- Write origin in Z, Y, X order ---
            xmf_file << "        <DataItem Name=\"Origin\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n";
            xmf_file << "          " << oz << " " << oy << " " << ox << "\n";
            xmf_file << "        </DataItem>\n";
            // --- Write spacing in Z, Y, X order ---
            xmf_file << "        <DataItem Name=\"Spacing\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n";
            xmf_file << "          " << dz << " " << dy << " " << dx << "\n";
            xmf_file << "        </DataItem>\n";
            // End geometry section
            xmf_file << "      </Geometry>\n\n";
            for (const auto& attribute : attributes) {
                xmf_file << "      <Attribute Name=\"" << attribute.first << "\" AttributeType=\"Scalar\" Center=\"Node\">\n";
                xmf_file << "        <DataItem Dimensions=\"" << nz << " " << ny << " " << nx << "\" NumberType=\"Int\" Format=\"Binary\" >\n";
                xmf_file << "          " << attribute.second << "\n";
                xmf_file << "        </DataItem>\n";
                xmf_file << "      </Attribute>\n\n";
            }
            xmf_file << "    </Grid>\n";
            xmf_file << "  </Domain>\n";
            xmf_file << "</Xdmf>\n";
            xmf_file.close();
        }
    }


    template<template<typename> class GrainID>
    void SubToXDMF(Structs::Substrate_Dual<GrainID>& DualSub, Structs::Mpi_Dual<GrainID>& DualMpi, Structs::Sim_Dual& DualSim, const uint32_t windowNum, const Structs::OutputRegion& region) {

        // Set References
        auto& sub = DualSub.hostSub;
        Stork::Structs::RegularGrid_Header<float, host_space>& header = DualMpi.header;

        // Set directory
        const Common::string dir = DualSim.simName + "/";

        // Gets starts and extents
        const uint32_t start[3] = {region.localStart[0], region.localStart[1], region.localStart[2]};
        const uint32_t extent[3] = {region.count[0], region.count[1], region.count[2]};
        const uint32_t stride[3] = {region.stride[0], region.stride[1], region.stride[2]};

        // Get name
        std::ofstream datafile;
        datafile.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        const Common::string base_fileName = "Rank." + impl::ZeroPadNumber(DualMpi.rank, std::max(static_cast<int>(log10(DualMpi.nproc)+0.5),1)) + ".Output." + impl::ZeroPadNumber(windowNum,1+int(log10(DualSim.hostSim.numLayers()))) + region.sliceSuffix;
        const Common::string xmf_fileName = dir + base_fileName + ".xmf";
        const Common::string dirID_partialName = base_fileName + ".dirID.bin";
        const Common::string dirID_fileName = dir + dirID_partialName;

        // Construct output fields
        std::vector<int32_t> dir_id(extent[0]*extent[1]*extent[2], -1);
        std::vector<int32_t> repeat_id;
        std::vector<int32_t> nucleated_rank_id;
        std::vector<int32_t> under_resolved;
        if constexpr(GrainID<host_space>::hasUniqueID()) {
            repeat_id.resize(extent[0]*extent[1]*extent[2], -1);
            nucleated_rank_id.resize(extent[0]*extent[1]*extent[2], -1);
        }
        if constexpr(GrainID<host_space>::hasUnderResolved()) {
            under_resolved.resize(extent[0]*extent[1]*extent[2], -1);
        }
        for (uint32_t dk = 0; dk < extent[2]; dk++) {
            for (uint32_t dj = 0; dj < extent[1]; dj++) {
                for (uint32_t di = 0; di < extent[0]; di++) {
                    // Get indices
                    const uint32_t i = start[0] + di * stride[0];
                    const uint32_t j = start[1] + dj * stride[1];
                    const uint32_t k = start[2] + dk * stride[2];
                    const uint32_t p = sub.ijk_to_p(i, j, k);
                    // If from the grid, output it
                    const uint32_t p_out = dk*extent[0]*extent[1] + dj*extent[0] + di;
                    if (sub.fromGrid(p)) {
                        dir_id[p_out] = static_cast<int32_t>(sub.dirID(p));
                        if constexpr(GrainID<host_space>::hasUniqueID()) {
                            nucleated_rank_id[p_out] = -1;
                        }
                        sub.grainID.xdmf_data(repeat_id, nucleated_rank_id, under_resolved, p, p_out);
                    }
                }
            }
        }

        // Write out binary data
        writeRawData(dirID_fileName, dir_id);
        std::vector<std::pair<std::string, std::string>> attributes = {{"dirID", dirID_partialName}};
        sub.grainID.xdmf_header(attributes, base_fileName);
        if constexpr(GrainID<host_space>::hasUniqueID()) {
            writeRawData(dir + base_fileName + ".repeatID.bin", repeat_id);
            writeRawData(dir + base_fileName + ".nucleatedRankID.bin", nucleated_rank_id);
        }
        if constexpr(GrainID<host_space>::hasUnderResolved()) {
            writeRawData(dir + base_fileName + ".underResolved.bin", under_resolved);
        }

        // Write out xmf data
        float GLOBAL_origin_xyz[3];
        GLOBAL_origin_xyz[0] = header.global_x0() + region.globalStart[0] * header.gridResolution();
        GLOBAL_origin_xyz[1] = header.global_y0() + region.globalStart[1] * header.gridResolution();
        GLOBAL_origin_xyz[2] = start[2] * header.gridResolution() + DualSub.zmin;
        const float spacing[3] = {
            header.gridResolution() * (extent[0] > 1 ? region.stride[0] : 1),
            header.gridResolution() * (extent[1] > 1 ? region.stride[1] : 1),
            header.gridResolution() * (extent[2] > 1 ? region.stride[2] : 1)
        };
        writeXDMF(xmf_fileName, attributes, extent, GLOBAL_origin_xyz, spacing);
    }

    template<template<typename> class GrainID>
    void SubToOutput_async(Structs::Substrate_Dual<GrainID>& DualSub, Structs::Mpi_Dual<GrainID>& DualMpi, Structs::Sim_Dual& DualSim, const uint32_t windowNum, const bool shiftZ, const Common::vector<Structs::OutputRegion> regions, Common::promise<void> write_done) {

        // Output
        for (const auto& region : regions) {
            if (DualSim.outputFormat == "xdmf") {
                SubToXDMF(DualSub, DualMpi, DualSim, windowNum, region);
            }
            else if (DualSim.outputFormat == "csv") {
                SubToCSV(DualSub, DualMpi, DualSim, windowNum, region);
            }
            else {
                throw std::runtime_error("Input error: Keyword [Output][Format] has invalid value.\nValid values are [xdmf, csv].");
            }
        }

        // Increment zmin once per shifted output window.
        if (shiftZ) {
            DualSub.zmin += (2*DualSub.windowSize)*DualSub.res;
        }

        // Signal that the write is done
        write_done.set_value();
    }
}
