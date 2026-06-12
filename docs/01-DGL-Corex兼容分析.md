# DGL Corex 兼容分析

> **源码位置：** `/root/GIDS_cufile/dgl/` (v1.1.3)
> **上游仓库：** https://github.com/dmlc/dgl.git
> **许可证：** Apache 2.0
> **最后更新：** 2026-06-12（DGL Corex 版编译通过）

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

**根因：** DGL 的 `CUDA.cmake` 把 host compiler flags 用逗号串联成一个 `-Xcompiler` 参数。nvcc 会自动拆分逗号，但 Corex 的 `ixc`（clang++ 封装）把整个字符串当作一个参数拒绝。

**修复位置：** `cmake/modules/CUDA.cmake` L237-250

```cmake
# 原代码（nvcc 风格）
string(REGEX REPLACE "[ \\t\\n\\r]" "," CXX_HOST_FLAGS "${CMAKE_CXX_FLAGS}")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "${CXX_HOST_FLAGS}")

# 修复后（逐个 -Xcompiler 参数）
string(STRIP "${CMAKE_CXX_FLAGS}" CXX_HOST_FLAGS_STRIPPED)
string(REGEX REPLACE "[ \\t\\n\\r]+" ";" CXX_HOST_FLAGS_LIST "${CXX_HOST_FLAGS_STRIPPED}")
foreach(flag ${CXX_HOST_FLAGS_LIST})
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "${flag}")
endforeach()
```

**状态：** ✅ 已修复，补丁 `/root/GIDS_cufile/patches/dgl_cuda_cmake_fix.patch`

---

### 2.2 `meta_group_rank()` 缺失 ✅ 已修复

**现象：**
```
error: no member named 'meta_group_rank' in 'cooperative_groups::thread_block_tile<32>'
```

**根因：** Corex 旧版 `cooperative_groups.h` 只实现了 CUDA 10.2 API，缺少 CUDA 11.0+ 的 `meta_group_rank()` 和 `meta_group_size()`。

**修复位置：** Corex SDK 源码树 `include/IX/ixrt/cooperative_groups.h` (SWPM-918-gids 分支)

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

**重要说明：** `cooperative_groups.h` SWPM-918-gids 补丁通过 `thread_rank()/numThreads` 实现了 `meta_group_rank()`，但这只解决了 API 调用层的问题。`tiled_partition` 的内部实现仍会在编译器后端触发 `%laneid` 生成。**这是编译器后端问题，不是源码级问题**。

**影响范围：** HugeCTR gpu_cache 模块（`static_hash_table.cu`、`nv_gpu_cache.cu`、`uvm_table.cu` 等）

**决策：** 短期禁用 gpu_cache 编译，不影响 GIDS 核心 `IXFeatureStore` 功能。

**状态：** ❌ 待 Corex 编译器团队支持

---

### 2.4 `fp16.cuh` `__half` 运算符重定义 ✅ 已修复

**现象：**
```
error: redefinition of 'operator+'
error: redefinition of 'operator-'
... (20 errors)
```

**根因：** Corex 编译器已原生提供 `__half` 的算术运算符，DGL 的 `fp16.cuh` 在 `__CUDA_ARCH__ < 530` 条件下又定义了一遍。

**修复位置：** `src/array/cuda/fp16.cuh` L48

```cpp
// 原代码
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 530)

// 修复后（Corex 跳过）
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 530) && !defined(__IXCC__)
```

**状态：** ✅ 已修复

---

### 2.5 CCCL cub/thrust/libcudacxx 变参函数不兼容 ✅ 已修复

**现象：**
```
error: CUDA device code does not support variadic functions
    _LIBCUDACXX_CONCEPT_FRAGMENT(...)
```

**根因：** DGL v1.1.3 使用 CCCL（NVIDIA 新版 cub/thrust/libcudacxx），其 `libcudacxx` 的 C++20 concept fallback 使用了 C 风格变参函数 `(...)`，Corex 编译器不支持 device code 中的变参函数。

**修复位置：** `CMakeLists.txt` L129-L131

```cmake
# 原代码（强制使用 CCCL 自带版本）
cuda_include_directories(BEFORE "${CMAKE_SOURCE_DIR}/third_party/cccl/thrust")
cuda_include_directories(BEFORE "${CMAKE_SOURCE_DIR}/third_party/cccl/cub")
cuda_include_directories(BEFORE "${CMAKE_SOURCE_DIR}/third_party/cccl/libcudacxx/include")

# 修复后（注释掉，使用 Corex SDK 自带 cub/thrust）
# DISABLED for Corex: Corex provides its own cub/thrust ...
```

**状态：** ✅ 已修复

---

### 2.6 `omp.h` 找不到 ✅ 已修复

**现象：**
```
fatal error: 'omp.h' file not found with <angled> include
```

**根因：** Corex 的 `clang++` 未自带 `omp.h`，而 DGL 的 host 代码（`parallel_for.h`）需要 OpenMP 头文件。

**修复位置：** `cmake/modules/CUDA.cmake` L247-L254

```cmake
# 自动检测 GCC include 路径并传入 -Xcompiler
execute_process(COMMAND gcc -print-file-name=include
  OUTPUT_VARIABLE GCC_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(GCC_INCLUDE_DIR)
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-I${GCC_INCLUDE_DIR}")
endif()
```

**状态：** ✅ 已修复

---

### 2.7 gpu_cache 编译禁用 ✅ 已处理

**根因：** 如上 2.3 所述，`%laneid` 编译器后端问题。

**修复位置：**

1. `CMakeLists.txt`：gpu_cache 编译块改为 `if(USE_CUDA AND NOT USE_CUDA)`（始终为 false）；排除 `gpu_cache.cu` 源文件
2. `python/dgl/cuda/__init__.py`：`from .gpu_cache import GPUCache` 包裹为 try/except
3. `python/dgl/cuda/gpu_cache.py`：`_init_api` 后检查 CAPI 是否存在，不存在抛出 `ImportError`

**状态：** ✅ 已处理，GPUCache 不可用时自动降级，不影响训练

---

### 2.8 `array_iterator.h` `CUB_INLINE` 未标记 `__device__` ✅ 已修复

**现象：**
```
error: call to __host__ function from __device__ function
    CUB_INLINE PairIterator operator+(const std::ptrdiff_t& movement) const {
```

**根因：** `CUB_INLINE` 宏只在 `__CUDA_ARCH__` 下展开为 `__host__ __device__ __forceinline__`，Corex 编译器定义 `__IXCC__` 而非 `__CUDA_ARCH__`，导致头文件中的迭代器运算符被标记为 host-only，device code 调用时报错。

**修复位置：** `include/dgl/array_iterator.h` L9

```cpp
// 原代码
#ifdef __CUDA_ARCH__
#define CUB_INLINE __host__ __device__ __forceinline__

// 修复后
#if defined(__CUDA_ARCH__) || defined(__IXCC__)
#define CUB_INLINE __host__ __device__ __forceinline__
```

**状态：** ✅ 已修复

---

## 3. 编译流程

### 前置条件

```bash
# 1. 确保 cooperative_groups.h 含 meta_group_rank
grep "meta_group_rank" /home/corex/sw_home_1/sw_home/local/corex/include/cooperative_groups.h

# 2. 激活 Corex 环境
source /home/corex/sw_home_1/sw_home/enable
```

### 一键编译（推荐）

```bash
bash /root/GIDS_cufile/GIDS/build_dgl_corex.sh
```

### 手动编译

```bash
cd /root/GIDS_cufile/dgl
git submodule update --init --recursive

# 应用所有兼容性修复（详见 build_dgl_corex.sh 中步骤 1.5-1.8）
# 1. 禁用 gpu_cache
# 2. 修复 fp16.cuh
# 3. 修复 -Xcompiler 格式
# 4. 禁用 CCCL cub/thrust/libcudacxx

mkdir build && cd build
cmake .. \
    -DUSE_CUDA=ON \
    -DUSE_OPENMP=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=/home/corex/sw_home_1/sw_home/local/corex \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
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
| CCCL (cub/thrust) | github.com/NVIDIA/cccl | GPU 并行算法 | ❌ 已禁用，使用 Corex SDK 自带版本 |
| **HugeCTR gpu_cache** | (DGL 内置) | GPU 嵌入缓存 | ❌ `%laneid` 编译器后端问题，已禁用 |

---

## 5. 相关文档

- [HugeCTR gpu_cache Corex 兼容分析](./02-HugeCTR-gpu_cache-Corex兼容分析.md)
- [GIDS-cuFile适配Corex方案总结](../GIDS-cuFile适配Corex方案总结.md)
- [GIDS IXFeatureStore 依赖清单](../GIDS-IX-依赖清单.md)