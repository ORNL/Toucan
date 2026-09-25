#pragma once

#define _USE_MATH_DEFINES

#include "Definitions.hpp"
#include <random>

namespace Toucan::Structs {

    class RNG_Dual {
    private:
        Kokkos::Random_XorShift64_Pool<host_space> random_pool_host;
        Kokkos::Random_XorShift64_Pool<device_space> random_pool_device;
        using uint16_view = Kokkos::View<uint16_t*, layout, device_space>;
        using uint16_view_host = Kokkos::View<uint16_t*, layout, host_space>;

        template<typename ExecutionSpace, typename std::enable_if_t<std::is_same_v<ExecutionSpace, host_space>, int> = 0>
        KOKKOS_INLINE_FUNCTION
        uint32_t rand(const uint32_t range) const {
            auto generator = random_pool_host.get_state();
            const uint32_t value = generator.urand(range);
            random_pool_host.free_state(generator);
            return value;
        }

        template<typename ExecutionSpace, typename std::enable_if_t<std::conjunction_v<std::negation<std::is_same<ExecutionSpace, host_space>>, std::is_same<ExecutionSpace, device_space>>, int> = 0>
        KOKKOS_INLINE_FUNCTION
        uint32_t rand(const uint32_t range) const {
            auto generator = random_pool_device.get_state();
            const uint32_t value = generator.urand(range);
            random_pool_device.free_state(generator);
            return value;
        }

    public:
        RNG_Dual() = default;

        RNG_Dual(const uint16_t num_orientations, const int seed)
            : RNG_Dual(num_orientations, seed, seed) {}

        RNG_Dual(const uint16_t num_orientations, const int random_seed, const int shuffle_seed) {
            random_pool_host = Kokkos::Random_XorShift64_Pool<host_space>(random_seed);
            // If the two spaces are different, generate one for device too
            if (!std::is_same<host_space, device_space>::value) {
                random_pool_device = Kokkos::Random_XorShift64_Pool<device_space>(random_seed);
            }

            // Random permutation of integers for shuffling list of grain id values
            std::vector<int> shuffled_orientations_vector(num_orientations);
            for (int i = 0; i < num_orientations; i++) {
                shuffled_orientations_vector[i] = i;
            }
            std::mt19937 generator(shuffle_seed);
            std::shuffle(shuffled_orientations_vector.begin(), shuffled_orientations_vector.end(), generator);
            uint16_view_host shuffled_orientations_list_host(Kokkos::ViewAllocateWithoutInitializing("shuffled_orientation_list"), num_orientations);
            for (int i = 0; i < num_orientations; i++) {
                shuffled_orientations_list_host[i] = shuffled_orientations_vector[i];
            }
            shuffled_orientations_list = Kokkos::create_mirror_view_and_copy(device_space(), shuffled_orientations_list_host);
        };

        uint16_view shuffled_orientations_list;

        // Randomly generate number from 0 to 1 (float)
        template<typename exe_space>
        KOKKOS_INLINE_FUNCTION
        float ZeroToOne() const {
            return rand<exe_space>(UINT32_MAX) / static_cast<float>(UINT32_MAX);
        }

        //  Randomly generate number from 0 to RANGE
        template<typename exe_space>
        KOKKOS_INLINE_FUNCTION
        uint32_t UInt(const uint32_t range) const {
            // Generate a random float between 0 and 1 using th_rand()
            return (rand<exe_space>(range));
        }

        //  Randomly generate a number within a normal distribution
        template<typename exe_space>
        KOKKOS_INLINE_FUNCTION
        float Normal(const float mean, const float std) const {
            // Get two random numbers
            const float r1 = ZeroToOne<exe_space>();
            const float r2 = ZeroToOne<exe_space>();
            // Box-Muller transform
            const float Z0 = sqrt(-2.0f * log(r1)) * cos(2.0f * static_cast<float>(M_PI) * r2);
            const float Z1 = sqrt(-2.0f * log(r1)) * sin(2.0f * static_cast<float>(M_PI) * r2);
            return mean + Z0 * std;
        }
    };
}
