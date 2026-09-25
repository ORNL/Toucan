#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <Kokkos_Random.hpp>

namespace Toucan {
    // Space definitions
    using host_space = Kokkos::DefaultHostExecutionSpace;
    using device_space = Kokkos::DefaultExecutionSpace;

    // Kokkos host memory and execution spaces
    using host_memory = Kokkos::DefaultHostExecutionSpace::memory_space;
    using host_exe = Kokkos::DefaultHostExecutionSpace::execution_space;

    // Kokkos device memory and exectution spaces
    using device_memory = Kokkos::DefaultExecutionSpace::memory_space;
    using device_exe = Kokkos::DefaultExecutionSpace::execution_space;

    // Set memory access layout
    typedef typename device_exe::array_layout layout;

    // Quick access to policy types
    using staticSchedule = Kokkos::Schedule<Kokkos::Static>;
    using dynamicSchedule = Kokkos::Schedule<Kokkos::Dynamic>;

    // Set quick access to views
    typedef Kokkos::View<char*, layout, device_memory> char_deviceView;
    typedef Kokkos::View<char*, layout, host_memory> char_hostView;

    typedef Kokkos::View<char*, layout, device_memory, Kokkos::MemoryUnmanaged> char_deviceSubView;
    typedef Kokkos::View<char*, layout, host_memory, Kokkos::MemoryUnmanaged> char_hostSubView;

    typedef Kokkos::View<bool*, layout, device_memory> bool_deviceView;
    typedef Kokkos::View<bool*, layout, host_memory> bool_hostView;

    typedef Kokkos::View<uint8_t*, layout, device_memory> uint8_deviceView;
    typedef Kokkos::View<uint8_t*, layout, host_memory> uint8_hostView;

    typedef Kokkos::View<uint16_t*, layout, device_memory> uint16_deviceView;
    typedef Kokkos::View<uint16_t*, layout, host_memory> uint16_hostView;

    typedef Kokkos::View<uint32_t*, layout, device_memory> uint32_deviceView;
    typedef Kokkos::View<uint32_t*, layout, host_memory> uint32_hostView;

    typedef Kokkos::View<float*, layout, device_memory> float_deviceView;
    typedef Kokkos::View<float*, layout, host_memory> float_hostView;
}