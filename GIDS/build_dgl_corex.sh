#!/bin/bash
set -e

COREX_DIR=/home/corex/sw_home_1/sw_home/local/corex
DGL_VER=1.1.3
BUILD_DIR=/tmp/dgl_build_corex

echo "=== 1/4 克隆 DGL v${DGL_VER} ==="
cd /tmp
rm -rf dgl_build_corex dgl_corex
git clone --depth 1 --branch v${DGL_VER} https://github.com/dmlc/dgl.git dgl_corex
cd dgl_corex
git submodule update --init --recursive

echo "=== 1.5/4 禁用 gpu_cache (Corex %laneid 编译器问题) ==="
# Disable HugeCTR gpu_cache compilation (requires %laneid PTX register, not supported by Corex)
sed -i '/^# Compile gpu_cache/,/^endif(USE_CUDA)/{
  s/^if(USE_CUDA)/if(USE_CUDA AND NOT USE_CUDA)  # DISABLED for Corex/
}' CMakeLists.txt

# Exclude the DGL gpu_cache wrapper from CUDA sources
sed -i '/dgl_config_cuda(DGL_CUDA_SRC)/a\
  # Disable gpu_cache wrapper: requires HugeCTR gpu_cache which is blocked by Corex %laneid issue\
  list(FILTER DGL_CUDA_SRC EXCLUDE REGEX "gpu_cache\\\\.cu$")' CMakeLists.txt

# Make GPUCache import conditional (Python fallback)
sed -i 's/^from .gpu_cache import GPUCache$/try:\n    from .gpu_cache import GPUCache\nexcept ImportError:\n    GPUCache = None/' python/dgl/cuda/__init__.py

# Add CAPI check in gpu_cache.py to raise ImportError if gpu_cache is disabled
sed -i '/^_init_api("dgl.cuda", __name__)$/a\
\
# Check if CAPI was actually registered (will be missing if gpu_cache is disabled at compile time)\
try:\
    _CAPI_DGLGpuCacheCreate\
except NameError:\
    raise ImportError(\
        "dgl.cuda.gpu_cache CAPI not available. "\
        "The HugeCTR gpu_cache module was disabled at compile time "\
        "(Corex compiler does not support %laneid PTX register)."\
    )' python/dgl/cuda/gpu_cache.py

echo "=== 1.6/4 修复 fp16.cuh Corex 兼容性 ==="
# Corex provides __half operators natively; skip DGL's redefinition
sed -i 's/#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 530)$/#if defined(__CUDA_ARCH__) \&\& (__CUDA_ARCH__ < 530) \&\& !defined(__IXCC__)/' src/array/cuda/fp16.cuh

echo "=== 1.7/4 修复 -Xcompiler 逗号分隔格式 (Corex clang++ 不兼容) ==="
patch -p1 < /root/GIDS_cufile/patches/dgl_cuda_cmake_fix.patch

echo "=== 1.8/4 禁用 CCCL cub/thrust/libcudacxx (Corex 不支持 variadic) ==="
# CCCL's libcudacxx uses variadic functions in C++20 concept fallback,
# which Corex device code does not support. Use Corex's own cub/thrust instead.
sed -i 's|^  cuda_include_directories(BEFORE "${CMAKE_SOURCE_DIR}/third_party/cccl/thrust")|  # DISABLED for Corex: \0|' CMakeLists.txt
sed -i 's|^  cuda_include_directories(BEFORE "${CMAKE_SOURCE_DIR}/third_party/cccl/cub")|  # DISABLED for Corex: \0|' CMakeLists.txt
sed -i 's|^  cuda_include_directories(BEFORE "${CMAKE_SOURCE_DIR}/third_party/cccl/libcudacxx/include")|  # DISABLED for Corex: \0|' CMakeLists.txt

echo "=== 1.9/4 修复 array_iterator.h CUB_INLINE (Corex __CUDA_ARCH__ 未定义) ==="
# CUB_INLINE is guarded by __CUDA_ARCH__ but Corex defines __IXCC__ instead.
# Without the fix, device code can't call operator+/operator++ etc.
sed -i 's/#ifdef __CUDA_ARCH__/#if defined(__CUDA_ARCH__) || defined(__IXCC__)/' include/dgl/array_iterator.h

echo "=== 2/4 配置 CMake (Corex CUDA) ==="
mkdir -p build && cd build

cmake .. \
    -DUSE_CUDA=ON \
    -DUSE_OPENMP=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=${COREX_DIR} \
    -DCMAKE_CUDA_COMPILER=${COREX_DIR}/bin/ixc \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
    -DCMAKE_INSTALL_PREFIX=${BUILD_DIR} \
    -DBUILD_CPP_TEST=OFF

echo "=== 3/4 编译 (使用所有 CPU 核心) ==="
make -j$(nproc)

echo "=== 4/4 安装 Python 包 ==="
cd ../python
python3 setup.py install --user

echo "=== 验证 ==="
python3 -c "import dgl; print('DGL', dgl.__version__, 'CUDA OK')"
