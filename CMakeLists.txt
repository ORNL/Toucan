cmake_minimum_required(VERSION 3.16)
project(TOUCAN LANGUAGES CXX)
set(PROJECT_VERSION "0.1.0-dev")

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24")
    cmake_policy(SET CMP0135 NEW)
endif()

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to enable or disable Condor thermal generation
option(TOUCAN_ENABLE_CONDOR "Enable Condor thermal generation" ON)

# Required dependencies
find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)
find_package(Stork REQUIRED)
find_package(nlohmann_json REQUIRED)

# Optional dependencies
if(TOUCAN_ENABLE_CONDOR)
    find_package(Condor QUIET)
    if(NOT Condor_FOUND)
        message(FATAL_ERROR
            "Toucan Condor support is enabled, but the Condor package was not found.\n"
            "Install Condor and add it to CMAKE_PREFIX_PATH, or configure with "
            "-DTOUCAN_ENABLE_CONDOR=OFF."
        )
    endif()
endif()

# Core library target with headers only (for package support)
add_library(Toucan_Core INTERFACE)

# Specify include directories for the core library
target_include_directories(Toucan_Core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  # For build time
    $<INSTALL_INTERFACE:include>  # For installation
)

# Link dependencies to Toucan_Core
target_link_libraries(Toucan_Core
    INTERFACE
    Kokkos::kokkos
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    Stork::Stork
    nlohmann_json::nlohmann_json
)

# Optional Condor support
if(TOUCAN_ENABLE_CONDOR)
    target_link_libraries(Toucan_Core INTERFACE Condor::Condor)
    target_compile_definitions(Toucan_Core INTERFACE TOUCAN_ENABLE_CONDOR=1)
else()
    target_compile_definitions(Toucan_Core INTERFACE TOUCAN_ENABLE_CONDOR=0)
endif()

include(GNUInstallDirs)

# Set the output directory for the executables during build
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/apps)

# Add executables for each file in apps/ and then install
file(GLOB APP_SOURCES "apps/*.cpp")  # Find all .cpp files in apps/
foreach(app_source ${APP_SOURCES})
    # Output directory during build process
    get_filename_component(app_name ${app_source} NAME_WE)  # Get the file name without extension
    add_executable(${app_name} ${app_source})  # Create an executable for each app
    target_link_libraries(${app_name} PRIVATE Toucan_Core)  # Link each app with Toucan_Core
    
    # Install executable
    install(TARGETS ${app_name} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endforeach()

# Installation rules for the Toucan_Core library
install(TARGETS Toucan_Core
    EXPORT ToucanTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install public headers
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Install the export configuration for package support
install(EXPORT ToucanTargets
    DESTINATION lib/cmake/ToucanCore
)

# Package configuration for find_package() support
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/ToucanConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ToucanCoreConfig.cmake"
    INSTALL_DESTINATION lib/cmake/ToucanCore
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ToucanCoreConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ToucanCoreConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ToucanCoreConfigVersion.cmake"
    DESTINATION lib/cmake/ToucanCore
)
