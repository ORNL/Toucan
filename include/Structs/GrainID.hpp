#pragma once

#include "Definitions.hpp"

#include <cstdint>
#include <climits>
#include <cstddef>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace Toucan::Structs {

    template<typename memory_space, bool uniqueID = false, bool underResolved = false>
    class GrainIdentifier {
    private:

        using bool_view = Kokkos::View<bool*, layout, memory_space>;
        using uint32_view = Kokkos::View<uint32_t*, layout, memory_space>;
        using floating_view = Kokkos::View<float*, layout, memory_space>;

    public:
        KOKKOS_INLINE_FUNCTION static constexpr bool hasUniqueID() { return uniqueID; }
        KOKKOS_INLINE_FUNCTION static constexpr bool hasUnderResolved() { return underResolved; }

        // For unique ID
        uint32_view repeatID_view;
        uint32_view nucleatedRankID_view;

        // For underresolved
        bool_view underResolved_view;

        // Initialization functions
        void Init(const size_t size, const char* label = "GrainIdentifier") {
            if constexpr(uniqueID) {
                repeatID_view = uint32_view(Kokkos::ViewAllocateWithoutInitializing(label), size);
                nucleatedRankID_view = uint32_view(Kokkos::ViewAllocateWithoutInitializing(label), size);
            }
            if constexpr(underResolved) {
                underResolved_view = bool_view(Kokkos::ViewAllocateWithoutInitializing(label), size);
            }

        }

        template<typename source_memory_space>
        void createHostMirrorsFrom(const GrainIdentifier<source_memory_space, uniqueID, underResolved>& source) {
            if constexpr(uniqueID) {
                repeatID_view = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, host_memory(), source.repeatID_view);
                nucleatedRankID_view = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, host_memory(), source.nucleatedRankID_view);
            }
            if constexpr(underResolved) {
                underResolved_view = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, host_memory(), source.underResolved_view);
            }
        }

        template<typename other_memory_space, class Func>
        void forEachFieldPair(GrainIdentifier<other_memory_space, uniqueID, underResolved>& other, Func&& func) const {
            if constexpr(uniqueID) {
                func(repeatID_view, other.repeatID_view);
                func(nucleatedRankID_view, other.nucleatedRankID_view);
            }
            if constexpr(underResolved) {
                func(underResolved_view, other.underResolved_view);
            }
        }

        void Reset() {
            if constexpr(uniqueID) {
                Kokkos::deep_copy(repeatID_view, UINT32_MAX);
                Kokkos::deep_copy(nucleatedRankID_view, UINT32_MAX);
            }
            if constexpr(underResolved){
                Kokkos::deep_copy(underResolved_view, false);
            }
        }

        // Information propagation functions
        KOKKOS_INLINE_FUNCTION
        void copyFrom(const GrainIdentifier& src, const uint32_t dest, const uint32_t source) const {
            if constexpr(uniqueID) {
                repeatID_view(dest) = src.repeatID_view(source);
                nucleatedRankID_view(dest) = src.nucleatedRankID_view(source);
            }
            if constexpr(underResolved) {
                underResolved_view(dest) = src.underResolved_view(source);
            }
        }

        KOKKOS_INLINE_FUNCTION
        void copyFrom(const GrainIdentifier& src, const uint32_t p) const {
            copyFrom(src, p, p);
        }

        KOKKOS_INLINE_FUNCTION
        void propagate(const int dest, const int source) const {
            copyFrom(*this, dest, source);
        }

        // IO Functions
        void csv_header(std::ostream& datafile) const {
            if constexpr(uniqueID) {
                datafile << ",repeatID,nucleatedRankID";
            }
            if constexpr(underResolved) {
                datafile << ",underResolved";
            }
        }
        void csv_data(std::ostream& datafile, const uint32_t p) const {
            if constexpr(uniqueID) {
                const uint32_t repeatID_loc = repeatID_view(p);
                const uint32_t nucRankID_loc = nucleatedRankID_view(p);

                datafile << "," << repeatID_loc << ",";
                if (nucRankID_loc == UINT32_MAX) {
                    datafile << -1;
                }
                else {
                    datafile << nucRankID_loc;
                }
            }
            if constexpr(underResolved) {
                datafile << "," << static_cast<int>(underResolved_view(p));
            }
        }
        void xdmf_header(std::vector<std::pair<std::string, std::string>>& attributes, const std::string& base_fileName) const {
            if constexpr(uniqueID) {
                attributes.push_back({"repeatID", base_fileName + ".repeatID.bin"});
                attributes.push_back({"nucleatedRankID", base_fileName + ".nucleatedRankID.bin"});
            }
            if constexpr(underResolved) {
                attributes.push_back({"underResolved", base_fileName + ".underResolved.bin"});
            }
        }
        void xdmf_data(std::vector<int32_t>& repeat_id, std::vector<int32_t>& nucleated_rank_id, std::vector<int32_t>& under_resolved, const uint32_t p, const uint32_t p_out) const {
            if constexpr(uniqueID) {
                repeat_id[p_out] = static_cast<int32_t>(repeatID_view(p));

                const uint32_t nucRankID_loc = nucleatedRankID_view(p);
                if (nucRankID_loc != UINT32_MAX) {
                    nucleated_rank_id[p_out] = static_cast<int32_t>(nucRankID_loc);
                }
            }
            if constexpr(underResolved) {
                under_resolved[p_out] = underResolved_view(p) ? 1 : 0;
            }
        }

        // MPI Functions
        static constexpr uint32_t mpi_uint32_per_cell(){
            uint32_t num = 0;
            if constexpr(uniqueID){
                num+=2;
            }
            return num;
        }
        static constexpr uint32_t mpi_float_per_cell(){
            uint32_t num = 0;
            return num;
        }
        static constexpr uint32_t mpi_uint16_per_cell(){
            uint32_t num = 0;
            return num;
        }
        static constexpr uint32_t mpi_uint8_per_cell(){
            uint32_t num = 0;
            if constexpr(underResolved){
                num+=1;
            }
            return num;
        }

        KOKKOS_INLINE_FUNCTION
        static void pack_mpi(
            const GrainIdentifier& org,
            const GrainIdentifier& current,
            const uint32_t p,
            uint32_t* uint32_buf,
            size_t& uint32_offset,
            float* float_buf,
            size_t& float_offset,
            uint16_t* uint16_buf,
            size_t& uint16_offset,
            uint8_t* uint8_buf,
            size_t& uint8_offset)
        {
            (void)float_buf;
            (void)float_offset;
            (void)uint16_buf;
            (void)uint16_offset;

            if constexpr(hasUniqueID()) {
                uint32_buf[uint32_offset++] = org.repeatID_view(p);
                uint32_buf[uint32_offset++] = current.repeatID_view(p);
                uint32_buf[uint32_offset++] = org.nucleatedRankID_view(p);
                uint32_buf[uint32_offset++] = current.nucleatedRankID_view(p);
            }

            if constexpr(hasUnderResolved()) {
                uint8_buf[uint8_offset++] = static_cast<uint8_t>(org.underResolved_view(p));
                uint8_buf[uint8_offset++] = static_cast<uint8_t>(current.underResolved_view(p));
            }
        }

        KOKKOS_INLINE_FUNCTION
        static void unpack_mpi(
            const GrainIdentifier& org,
            const GrainIdentifier& current,
            const uint32_t p,
            const uint32_t* uint32_buf,
            size_t& uint32_offset,
            const float* float_buf,
            size_t& float_offset,
            const uint16_t* uint16_buf,
            size_t& uint16_offset,
            const uint8_t* uint8_buf,
            size_t& uint8_offset)
        {
            (void)float_buf;
            (void)float_offset;
            (void)uint16_buf;
            (void)uint16_offset;

            if constexpr(hasUniqueID()) {
                org.repeatID_view(p) = uint32_buf[uint32_offset++];
                current.repeatID_view(p) = uint32_buf[uint32_offset++];
                org.nucleatedRankID_view(p) = uint32_buf[uint32_offset++];
                current.nucleatedRankID_view(p) = uint32_buf[uint32_offset++];
            }

            if constexpr(hasUnderResolved()) {
                org.underResolved_view(p) = static_cast<bool>(uint8_buf[uint8_offset++]);
                current.underResolved_view(p) = static_cast<bool>(uint8_buf[uint8_offset++]);
            }
        }

        // Unique ID functions
        KOKKOS_INLINE_FUNCTION
        void setUniqueID(const uint32_t p, const uint32_t repeat_id, const uint32_t rank) const {
            if constexpr(uniqueID) {
                repeatID_view(p) = repeat_id;
                nucleatedRankID_view(p) = rank;
            }
        }

        // Underresolved functions
        KOKKOS_INLINE_FUNCTION
        void setUnderResolved(const uint32_t p, const float tliq, const float tCap) const {
            if constexpr(underResolved) {
                underResolved_view(p) = (tCap < tliq);
            }
        }
    };

    template<typename memory_space>
    using BaseGrainIdentifier = GrainIdentifier<memory_space, false, false>;

    template<typename memory_space>
    using UniqueOnlyGrainIdentifier = GrainIdentifier<memory_space, true, false>;

    template<typename memory_space>
    using UnderresolvedOnlyGrainIdentifier = GrainIdentifier<memory_space, false, true>;

    template<typename memory_space>
    using UniqueUnderresolvedGrainIdentifier = GrainIdentifier<memory_space, true, true>;

}
