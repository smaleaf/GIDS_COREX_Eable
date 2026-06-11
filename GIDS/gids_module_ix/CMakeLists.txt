cmake_minimum_required(VERSION 3.12)
project(GIDS_IX_Module LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------
# Iluvatar IXC compiler configuration
# ------------------------------------------------------------------
# Prefer ixc if available; fall back to g++ (for sandbox/dev testing)
find_program(IXC_COMPILER ixc)
if(IXC_COMPILER)
    message(STATUS "Using Iluvatar compiler: ${IXC_COMPILER}")
    set(CMAKE_CUDA_COMPILER ixc)
    set(CMAKE_CUDA_FLAGS "--compiler-options -fPIC")
    set(CMAKE_CUDA_ARCHITECTURES "bi")
    add_definitions(-DUSE_IXC)
    set(SOURCE_FILE_EXT ".cu")
else()
    message(STATUS "ixc not found, using g++ (sandbox mode - no GPU kernel compilation)")
    set(SOURCE_FILE_EXT ".cpp")
    add_definitions(-DIX_SANDBOX_MODE)
endif()

# If using mapping_cudart.h compatibility mode:
# include_directories(${IXDRIVER_INCLUDE_DIR})
# add_definitions(-DILUVATAR_COMPAT_MODE)

# ------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------
# Python / pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Try to find pybind11 from Python's site-packages first
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(PYBIND11_CMAKE_DIR)
    set(pybind11_DIR "${PYBIND11_CMAKE_DIR}")
endif()
find_package(pybind11 REQUIRED)

# Iluvatar CUFile API
set(IXDRIVER_ROOT "$ENV{IXDRIVER_ROOT}" CACHE PATH "ixdriver SDK root")
if(IXDRIVER_ROOT)
    set(IXDRIVER_INCLUDE_DIR "${IXDRIVER_ROOT}/include")
    set(IXDRIVER_LIB_DIR     "${IXDRIVER_ROOT}/lib")
else()
    set(IXDRIVER_INCLUDE_DIR "/home/corex/sw_home_1/sw_home/sdk/ixdriver/include")
    set(IXDRIVER_LIB_DIR     "/home/corex/sw_home_1/sw_home/sdk/ixdriver/lib")
endif()

message(STATUS "ixdriver include: ${IXDRIVER_INCLUDE_DIR}")
message(STATUS "ixdriver lib:     ${IXDRIVER_LIB_DIR}")

# ------------------------------------------------------------------
# Build target
# ------------------------------------------------------------------
if(IXC_COMPILER)
    pybind11_add_module(IXFeatureStore ix_feature_store.cu)
else()
    # Sandbox mode: rename .cu to .cpp and build with standard g++
    add_library(IXFeatureStore SHARED ix_feature_store.cpp)
    target_include_directories(IXFeatureStore PRIVATE ${pybind11_INCLUDE_DIRS})
    target_link_libraries(IXFeatureStore PRIVATE ${Python3_LIBRARIES})
    set_target_properties(IXFeatureStore PROPERTIES PREFIX "" SUFFIX ".cpython-${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}-x86_64-linux-gnu.so")
endif()

target_include_directories(IXFeatureStore PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${IXDRIVER_INCLUDE_DIR}
    ${IXDRIVER_INCLUDE_DIR}/IX
)

# Set library search paths
set(COREX_LIB64 "/home/corex/sw_home_1/sw_home/local/corex/lib64")
target_link_directories(IXFeatureStore PRIVATE
    ${COREX_LIB64}
    /usr/local/lib64
)

# Link against cuFile + cudart libraries
# cuFile: GPU Direct Storage (standard NVIDIA API)
# cudart: IX runtime (ixMalloc, ixHostAlloc, ixStreamCreate, etc.)
# --no-as-needed: force link even if no direct references found
if(IXC_COMPILER)
    target_link_libraries(IXFeatureStore PRIVATE
        ixrts
        -Wl,--no-as-needed cufile cudart -Wl,--as-needed
    )
else()
    target_link_libraries(IXFeatureStore PRIVATE
        -Wl,--no-as-needed cufile cudart -Wl,--as-needed
    )
endif()

# Position independent code for shared library
set_target_properties(IXFeatureStore PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
)
if(NOT IXC_COMPILER)
    set_target_properties(IXFeatureStore PROPERTIES
        SUFFIX ".cpython-${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}-x86_64-linux-gnu.so"
    )
endif()

# ------------------------------------------------------------------
# Install
# ------------------------------------------------------------------
install(TARGETS IXFeatureStore DESTINATION ${CMAKE_SOURCE_DIR}/../GIDS_Setup/GIDS)

message(STATUS "GIDS IX Module configured successfully")
message(STATUS "Build with: mkdir build_ix && cd build_ix && cmake .. && make -j")