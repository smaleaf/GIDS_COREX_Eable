# HugeCTR gpu_cache Corex 兼容分析

> **源码位置：**
> - DGL 内置：`/root/GIDS_cufile/dgl/third_party/HugeCTR/gpu_cache/`
> - 独立仓库：`/root/GIDS_cufile/HugeCTR/gpu_cache/`
> **上游仓库：** https://github.com/NVIDIA-Merlin/HugeCTR
> **许可证：** Apache 2.0
> **最后更新：** 2026-06-12（确认禁用，%laneid 编译器后端问题）

---

## 1. 模块概述

HugeCTR GPU Embedding Cache 是 NVIDIA 为推荐系统 embedding 查表场景设计的 GPU 原生缓存。在 DGL 中以独立子目录集成，编译为 `libgpu_cache.a` 并链接进 `libdgl.so`。

### 源文件

| 文件 | 功能 |
|------|------|
| `nv_gpu_cache.cu` | GPU 嵌入缓存核心实现 |
| `static_hash_table.cu` | 静态哈希表（组相联结构） |
| `static_table.cu` | 静态嵌入表 |
| `uvm_table.cu` | 统一虚拟内存嵌入表 |

### 核心数据结构

```
Set-Associative Cache (组相联缓存)
┌─────────────────────────────────┐
│ Set 0 │ Way 0 │ Way 1 │ ...    │  ← 每个 Set 包含 N 个 Way
│ Set 1 │ Way 0 │ Way 1 │ ...    │
│  ...   │  ...  │  ...  │ ...    │
│ Set N │ Way 0 │ Way 1 │ ...    │
└─────────────────────────────────┘
```

- **Set Index**: `hash(key) % num_sets` 确定 key 属于哪个 Set
- **Way**: 每个 Set 内有 W 个槽位，类 LRU 淘汰策略
- **Warp 级并行**: 一个 warp (32 线程) 协作处理一次查表

---

## 2. 编译器兼容状态

### 2.1 API 层 ✅ 已修复

| API | 归属 | 状态 |
|-----|------|------|
| `cooperative_groups::this_thread_block()` | CG 核心 | ✅ |
| `cooperative_groups::tiled_partition<Size>()` | CG 核心 | ✅ → 但后端仍生成 `%laneid` |
| `cooperative_groups::this_grid()` | CG 核心 | ✅ |
| `meta_group_rank()` | CG CUDA 11.0+ | ✅ SWPM-918-gids 补丁 |
| `meta_group_size()` | CG CUDA 11.0+ | ✅ SWPM-918-gids 补丁 |

### 2.2 编译器后端 ❌ 阻塞

| PTX 寄存器 | 错误现象 | 状态 |
|------------|----------|------|
| `%laneid` | `error: unknown token in expression: mov.u32 v4, %laneid` | ❌ |
| `%warpid` | （待确认） | ⬜ |
| 其他 warp 寄存器 | （待确认） | ⬜ |

**根因分析：**

gpu_cache 大量使用 `cooperative_groups::tiled_partition<>`，其底层依赖 warp 级操作。编译器后端（Corex LLVM fork 的 llc）需要将这些操作翻译为 GPU 指令，翻译过程中引用了 PTX 寄存器 `%laneid`（标识当前线程在其 warp 中的位置，0-31）。

Corex 的 LLVM fork 未实现 `%laneid` 寄存器的代码生成映射。

**关键澄清（2026-06-12）：** SWPM-918-gids 补丁通过 `thread_rank()/numThreads` 实现了 `meta_group_rank()` 的 API 层，但这只解决了 `meta_group_rank()` 和 `meta_group_size()` 函数调用的问题。`tiled_partition` 内部的 warp 级操作在编译器后端翻译时，仍会生成 `%laneid` PTX 寄存器引用。**这是编译器后端问题，无法通过源码级修复绕过。**

### 2.3 源码分析

所有涉及 cooperative_groups 的代码都使用标准 C++ API，**没有任何行内 PTX 汇编**：

```cpp
// static_hash_table.cu:131-132
auto block = cooperative_groups::this_thread_block();
auto tile = cooperative_groups::tiled_partition<tile_size>(block);

// static_hash_table.cu:266-269
auto grid = cooperative_groups::this_grid();
auto block = cooperative_groups::this_thread_block();
auto tile = cooperative_groups::tiled_partition<tile_size>(block);
auto warp_tile = cooperative_groups::tiled_partition<WARP_SIZE>(block);

// nv_gpu_cache.cu:17,21
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
```

编译错误 `%laneid` 完全由编译器后端自动生成，与源码无关。这是编译器移植层面的问题，不是应用层代码问题。

---

## 3. 编译器修复建议（给 Corex 团队）

### 需要支持的 PTX 寄存器

| PTX 寄存器 | 含义 | 来源 API |
|------------|------|----------|
| `%laneid` | Warp 内线程索引 (0-31) | `tiled_partition` 内的线程标识 |
| `%warpid` | Block 内 warp 索引 | `meta_group_rank()` 的底层实现 |
| `%nwarpid` | Grid 内全局 warp 索引 | 多 block 协调 |

### 可能的修复路径

1. **LLVM 后端添加寄存器映射**：在 Corex LLVM Target 描述中添加 `%laneid` 寄存器定义和代码生成逻辑
2. **CG 实现降级**：如果 Corex GPU 硬件有等价的 warp lane index 寄存器，映射到该寄存器
3. **软实现**：用 `threadIdx.x % warpSize` 计算替代物理寄存器（有性能损失）

---

## 4. 当前处理方案：禁用 gpu_cache

### 4.1 CMake 层面

**DGL `CMakeLists.txt`：**

```cmake
# gpu_cache 编译块改为始终为 false
if(USE_CUDA AND NOT USE_CUDA)  # DISABLED for Corex: %laneid PTX register
  ...
  cuda_add_library(gpu_cache STATIC ${gpu_cache_src})
  ...
endif()

# 排除 gpu_cache.cu 源文件
list(FILTER DGL_CUDA_SRC EXCLUDE REGEX "gpu_cache\\.cu$")
```

### 4.2 Python 层面

**`python/dgl/cuda/__init__.py`：**
```python
try:
    from .gpu_cache import GPUCache
except ImportError:
    GPUCache = None
```

**`python/dgl/cuda/gpu_cache.py`：**
```python
def _init_api():
    ...
    try:
        _CAPI_DGLHeteroCacheCreate = _LIB.CAPI_DGLHeteroCacheCreate
    except AttributeError:
        raise ImportError("GPUCache is not available in this build of DGL")
```

### 4.3 对 GIDS 的影响：无

GIDS 使用自己的 `IXFeatureStore`（cuFile → NVMe SSD），不依赖 HugeCTR gpu_cache。禁掉 gpu_cache 编译不影响 GIDS 的核心功能。

---

## 5. 对 GIDS 的影响评估

### 当前影响：无

GIDS 使用自己的 `IXFeatureStore`（cuFile → NVMe SSD），不依赖 HugeCTR gpu_cache。禁掉 gpu_cache 编译不会影响 GIDS 的核心功能。

### 未来影响：两级缓存优化

`%laneid` 支持后，可启用 GPU HBM + NVMe 两级缓存：

| 方案 | 延迟 | 容量 | 适用数据 |
|------|------|------|---------|
| L1: gpu_cache (GPU HBM) | ~100ns | 100-200MB | 热门节点 embedding |
| L2: IXFeatureStore (NVMe SSD) | ~10μs | TB 级 | 全量特征 |

---

## 6. 相关工作

| 项目 | 内容 |
|------|------|
| Corex SDK 分支 | SWPM-918-gids（meta_group_rank 补丁） |
| DGL 编译 | `/root/GIDS_cufile/dgl/build/` |
| DGL 编译脚本 | `/root/GIDS_cufile/GIDS/build_dgl_corex.sh` |
| 独立 HugeCTR | `/root/GIDS_cufile/HugeCTR/` |

---

## 7. 参考

- [HugeCTR GitHub](https://github.com/NVIDIA-Merlin/HugeCTR)
- [CUDA cooperative_groups 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)
- [DGL GPU Cache PR](https://github.com/dmlc/dgl/pull/6694)
- [NVIDIA Merlin 文档](https://developer.nvidia.com/nvidia-merlin/hugectr)
- [DGL Corex 兼容分析](./01-DGL-Corex兼容分析.md)
- **PTX 指令级分析：** [03-gpu_cache-PTX指令清单与Corex兼容状态.md](./03-gpu_cache-PTX指令清单与Corex兼容状态.md)