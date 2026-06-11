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
