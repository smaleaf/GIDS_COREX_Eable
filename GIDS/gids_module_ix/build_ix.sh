#!/bin/bash
# ==============================================================
# GIDS Iluvatar Port - Build Script
# ==============================================================
# Prerequisites:
#   1. source /home/corex/sw_home_1/sw_home/enable
#   2. Python3 + pybind11 available
#   3. ixc or clang++ compiler available
# ==============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${SCRIPT_DIR}/build_ix"

# Activate SDK environment
if [ -f "/home/corex/sw_home_1/sw_home/enable" ]; then
    source /home/corex/sw_home_1/sw_home/enable
    echo "[build] SDK environment activated"
fi

# ixdriver SDK paths
IXDRIVER_ROOT="${IXDRIVER_ROOT:-/home/corex/sw_home_1/sw_home/sdk/ixdriver}"
IXDRIVER_INC="${IXDRIVER_ROOT}/include"
IXDRIVER_LIB="${IXDRIVER_ROOT}/lib"

# Detect compiler
if command -v ixc &>/dev/null; then
    CUDA_COMPILER="ixc"
    echo "[build] Compiler: ixc (Iluvatar GPU mode)"
elif command -v clang++ &>/dev/null; then
    CUDA_COMPILER="clang++"
    echo "[build] Compiler: clang++ (sandbox/dev mode)"
else
    CUDA_COMPILER=""
    echo "[build] WARNING: No ixc or clang++ found, using system g++"
fi

echo "============================================="
echo " GIDS Iluvatar Port - Build"
echo "============================================="
echo " Source dir:   ${SCRIPT_DIR}"
echo " Build dir:    ${BUILD_DIR}"
echo " Compiler:     ${CUDA_COMPILER:-g++}"
echo " ixdriver inc: ${IXDRIVER_INC}"
echo " ixdriver lib: ${IXDRIVER_LIB}"
echo "============================================="

# Check prerequisites
if [ ! -d "${IXDRIVER_INC}" ]; then
    echo "ERROR: ixdriver SDK not found at ${IXDRIVER_ROOT}"
    exit 1
fi

# In sandbox mode (no ixc), cmake expects ix_feature_store.cpp
# Create a copy from .cu since the code uses only host-side APIs
if ! command -v ixc &>/dev/null; then
    cp "${SCRIPT_DIR}/ix_feature_store.cu" "${SCRIPT_DIR}/ix_feature_store.cpp"
    echo "[build] Created ix_feature_store.cpp for sandbox compilation"
fi

# Clean and create build directory
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

cd "${BUILD_DIR}"

# Configure
echo ""
echo "-- Running cmake --"
cmake "${SCRIPT_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DIXDRIVER_ROOT="${IXDRIVER_ROOT}" \
    -DPYTHON_EXECUTABLE="$(which python3)"

# Build
echo ""
echo "-- Building --"
make -j$(nproc)

# Copy to GIDS Setup directory
OUTPUT_DIR="${PROJECT_ROOT}/GIDS_Setup/GIDS"
mkdir -p "${OUTPUT_DIR}"

# Copy the .so file
SO_FILE=$(find "${BUILD_DIR}" -name "IXFeatureStore*.so" 2>/dev/null | head -1)
if [ -n "${SO_FILE}" ]; then
    cp "${SO_FILE}" "${OUTPUT_DIR}/"
    echo ""
    echo "============================================="
    echo " Build SUCCESS"
    echo " Output: ${OUTPUT_DIR}/$(basename ${SO_FILE})"
    echo "============================================="
else
    echo ""
    echo "============================================="
    echo " Build FAILED - .so file not found"
    echo "============================================="
    exit 1
fi