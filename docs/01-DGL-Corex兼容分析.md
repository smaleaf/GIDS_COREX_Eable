# DGL Corex 兼容分析

> **源码位置：** `/root/GIDS_cufile/dgl/` (v1.1.3)
> **上游仓库：** https://github.com/dmlc/dgl.git
> **许可证：** Apache 2.0

---

## 1. 为什么需要 DGL CUDA 版

GIDS 的 `GIDS_DGLDataLoader` 使用 DGL 的以下 CUDA 特性：

| 特性 | 用途 | CPU 版支持 |
|------|------|-----------|
| `graph.pin_memory_()` | 将图数据 pin 到 CPU 内存，供 GPU 零拷贝访问 | ❌ |
| `graph.to('cuda:0')` | 将图移至 GPU 显存 | ❌ |
| UVA Sampling | 统一虚拟地址采样（CPU 端索引 + GPU 端采样） | ❌ |
| `sample_neighbors()` GPU | 邻居采样在 GPU 上执行 | ❌ |

`pip install dgl` 安装的是 CPU 版，必须源码编译 CUDA 版。

---

## 2. 兼容问题清单

### 2.1 `-Xcompiler` 格式不兼容 ✅ 已修复

**现象：**
```
clang++: error: unknown argument: '-fopenmp,-O2,-Wall,...'
```

**根因：** nvcc 用逗号分隔 host compiler flags，ixc (clang++ 封装) 把整个逗号分隔字符串当单个参数。

**修复位置：** `cmake/modules/CUDA.cmake` L239-243

```cmake
# 原代码（nvcc 风格）
string(REGEX REPLACE "[ \\t\\n\\r]" "," CXX_HOST_FLAGS "${CMAKE_CXX_FLAGS}")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "${CXX_HOST_FLAGS}")

# 修复后（ixc 风格）
string(REGEX REPLACE "[ \\t\\n\\r]" " " CXX_HOST_FLAGS "${CMAKE_CXX_FLAGS}")
separate_arguments(CXX_HOST_FLAGS_LIST UNIX_COMMAND "${CXX_HOST_FLAGS}")
list(APPEND CUDA_NVCC_FLAGS ${CXX_HOST_FLAGS_LIST})
```

**状态：** ✅ 已修复，`/root/GIDS_cufile/patches/dgl_cuda_cmake.patch`

---

### 2.2 `meta_group_rank()` 缺失 ✅ 已修复

**现象：**
```
error: no member named 'meta_group_rank' in 'cooperative_groups::thread_block_tile<32>'
```

**根因：** Corex 旧版 `cooperative_groups.h` 只实现了 CUDA 10.2 API，缺少 CUDA 11.0+ 的 `meta_group_rank()` 和 `meta_group_size()`。

**修复位置：** Corex SDK 源码树 `include/IX/ixrt/cooperative_groups.h` (SWPM-918-gids 分支)

```cpp
// 在 thread_block_tile<Size> 类中新增
__device__ unsigned int meta_group_rank() const {
    return threadIdx.x / warp_size;  // 简化实现
}
__device__ unsigned int meta_group_size() const {
    return blockDim.x / warp_size;   // 简化实现
}
```

**验证：**
```bash
/home/corex/sw_home_1/sw_home/local/corex/bin/clang++ -O3 -x ivcore \
    -I/home/corex/sw_home_1/sw_home/sdk/ixdriver/include \
    --cuda-gpu-arch=ivcore11 --cuda-gpu-arch=ivcore20 \
    -fPIC -c apps/cudasamples/cooperativeGroupsMetaGroupTest.cu
```

**状态：** ✅ 已修复，SDK 头文件已部署到已安装路径

---

### 2.3 `%laneid` PTX 寄存器不支持 ❌ 编译器后端阻塞

**现象：**
```
<inline asm>:1:14: error: unknown token in expression
    mov.u32 v4, %laneid;
clang++: error: llc command failed with exit code 1
```

**根因：** `cooperative_groups::tiled_partition` 等 warp 级原语在编译器后端（LLVM llc）被翻译为 PTX 指令，其中引用了 `%laneid` 寄存器。Corex 的 LLVM fork 不支持此寄存器。

**影响范围：** HugeCTR gpu_cache 模块（`static_hash_table.cu`、`nv_gpu_cache.cu`、`uvm_table.cu` 等）

**涉及 API：**
```cpp
#include <cooperative_groups.h>
auto block = cooperative_groups::this_thread_block();
auto tile = cooperative_groups::tiled_partition<tile_size>(block);
auto warp_tile = cooperative_groups::tiled_partition<WARP_SIZE>(block);
auto grid = cooperative_groups::this_grid();
```

**需要的修复：** Corex LLVM fork 中添加 `%laneid` PTX 寄存器的代码生成映射。

**状态：** ❌ 待 Corex 编译器团队支持

---

## 3. 编译流程

### 前置条件

```bash
# 1. 确保 cooperative_groups.h 含 meta_group_rank
grep "meta_group_rank" /home/corex/sw_home_1/sw_home/local/corex/include/cooperative_groups.h

# 2. 激活 Corex 环境
source /home/corex/sw_home_1/sw_home/enable
```

### 编译命令

```bash
cd /root/GIDS_cufile/dgl
git submodule update --init --recursive

# 修复 -Xcompiler 格式
sed -i 's/string(REGEX REPLACE "\[ \\t\\n\\r\]" "," CXX_HOST_FLAGS "${CMAKE_CXX_FLAGS}")/string(REGEX REPLACE "[ \\t\\n\\r]" " " CXX_HOST_FLAGS "${CMAKE_CXX_FLAGS}")\n  separate_arguments(CXX_HOST_FLAGS_LIST UNIX_COMMAND "${CXX_HOST_FLAGS}")\n  list(APPEND CUDA_NVCC_FLAGS ${CXX_HOST_FLAGS_LIST})/' cmake/modules/CUDA.cmake
sed -i '/list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "${CXX_HOST_FLAGS}")/d' cmake/modules/CUDA.cmake

# 禁掉 gpu_cache（%laneid 编译器问题）
# 如果 cooperative_groups.h 不含 meta_group_rank，也需禁用
sed -i '/add_subdirectory.*HugeCTR.*gpu_cache/s/^/#/' third_party/HugeCTR/CMakeLists.txt

mkdir build && cd build
cmake .. \
    -DUSE_CUDA=ON \
    -DUSE_OPENMP=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=/home/corex/sw_home_1/sw_home/local/corex \
    -DCUDA_NVCC_EXECUTABLE=/home/corex/sw_home_1/sw_home/local/corex/bin/ixc \
    -DBUILD_CPP_TEST=OFF

make -j$(nproc)

cd ../python
python3 setup.py install --user
```

### 验证

```bash
python3 -c "
import dgl
g = dgl.rand_graph(10, 20)
g = g.to('cuda:0')
print('DGL', dgl.__version__, 'CUDA OK')
"
```

---

## 4. 依赖的子模块

| 子模块 | 上游 | 用途 | 状态 |
|--------|------|------|------|
| dmlc-core | github.com/dmlc/dmlc-core | DGL 基础库 | ✅ 无问题 |
| dlpack | github.com/dmlc/dlpack | 张量数据结构 | ✅ 头文件，无编译 |
| METIS | github.com/KarypisLab/METIS | 图分区 | ✅ C 代码，gcc 编译 |
| phmap | github.com/greg7mdp/parallel-hashmap | 并行哈希表 | ✅ 头文件，无编译 |
| nanoflann | github.com/jlblancoc/nanoflann | 最近邻搜索 | ✅ 头文件，无编译 |
| libxsmm | github.com/libxsmm/libxsmm | 矩阵乘法 | ✅ C 代码，gcc 编译 |
| **HugeCTR gpu_cache** | (DGL 内置) | GPU 嵌入缓存 | ❌ `%laneid` 编译器问题 |

---

## 5. 相关文档

- [HugeCTR gpu_cache Corex 兼容分析](./02-HugeCTR-gpu_cache-Corex兼容分析.md)
- [GIDS IXFeatureStore 依赖清单](../GIDS-IX-依赖清单.md)
- [GIDS HugeCTR GPU Cache 与两级缓存优化](../GIDS-HugeCTR-GPU-Cache与两级缓存优化.md)