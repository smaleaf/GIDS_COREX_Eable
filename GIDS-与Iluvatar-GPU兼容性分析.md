# GIDS 与 Iluvatar GPU 兼容性分析

## 1. 核心结论（已验证）

**GIDS API 层可直接映射到 Iluvatar GPU。** 经验证，Iluvatar ixdriver SDK 提供了完整的 CUDA API 兼容层（[mapping_cudart.h](file:///home/corex/sw_home_1/sw_home/sdk/ixdriver/include/IX/mapping_cudart.h)），GIDS 所需的**全部 CUDA Runtime API 均有 1:1 对应的 ix* 接口**。主要工作量在 BaM 框架的 NVMe 用户态驱动移植。

> **验证来源：** `/home/corex/sw_home_1/sw_home/sdk/ixdriver/include/IX/mapping_cudart.h`
> 该文件包含 3000+ 行 CUDA→IX API 宏映射，覆盖所有 CUDA Runtime/Driver API。

---

## 2. CUDA 依赖清单

### 2.1 编译期依赖（CMakeLists.txt）

```
CMAKE_MINIMUM_REQUIRED(VERSION 3.3.0)
PROJECT(cmake-cpp-pybind11 CXX CUDA)          # ← CUDA 语言支持

# CUDA 编译选项
target_compile_options(BAM_Feature_Store PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:               # ← CUDA 编译器
        -std=c++11;
        -O3;
        --default-stream per-thread;
    >)

# GPU 架构代码生成
-gencode arch=compute_70,code=sm_70           # ← NVIDIA SM 70 (Volta)
-gencode arch=compute_80,code=sm_80           # ← NVIDIA SM 80 (Ampere)
```

**移植需求：** 需要将 CMake 项目改为 Iluvatar 的 `ixc` 编译器，并修改 GPU 架构目标代码。

---

### 2.2 CUDA Runtime API 调用

以下是从 [gids_nvme.cu](file:///home/corex/sw_home_1/GIDS_enable/GIDS/gids_module/gids_nvme.cu) 和 [gids_kernel.cu](file:///home/corex/sw_home_1/GIDS_enable/GIDS/gids_module/gids_kernel.cu) 中提取的所有 CUDA API 调用：

| CUDA API | 出现位置 | 功能 | Iluvatar 替代方案 | 验证状态 |
|----------|---------|------|-------------------|----------|
| `cudaHostAlloc(..., cudaHostAllocMapped)` | cpu_backing_buffer() | 分配 Unified Memory 可映射的主机内存 | `ixHostAlloc()` + `ixHostAllocMapped` | ✅ 已验证 |
| `cudaHostGetDevicePointer()` | cpu_backing_buffer() | 获取映射后内存的设备端虚拟地址 | `ixHostGetDevicePointer()` | ✅ 已验证 |
| `cudaMalloc()` | init_controllers() | GPU 显存分配 | `ixMalloc()` | ✅ 已验证 |
| `cudaMemset()` | init_controllers() | GPU 显存初始化 | `ixMemset()` | ✅ 已验证 |
| `cudaMemcpy()` (Host→Device, Device→Host) | 多处 | 数据拷贝 | `ixMemcpy()` | ✅ 已验证 |
| `cudaDeviceSynchronize()` | 各处 kernel launch 后 | 同步 GPU 设备 | `ixDeviceSynchronize()` | ✅ 已验证 |
| `cudaStreamCreate()` | read_feature_hetero() | 创建异步流 | `ixStreamCreate()` | ✅ 已验证 |
| `cudaStreamDestroy()` | read_feature_hetero() | 销毁异步流 | `ixStreamDestroy()` | ✅ 已验证 |
| `cudaStreamSynchronize()` | read_feature_hetero() | 同步流 | `ixStreamSynchronize()` | ✅ 已验证 |
| `cudaDeviceCanAccessPeer()` | P2P 检查 | 检查 GPU 间 P2P 能力 | `ixDeviceCanAccessPeer()` | ✅ 已验证 |
| `cudaDeviceEnablePeerAccess()` | P2P 启用 | 启用 GPU 间 P2P 访问 | `ixDeviceEnablePeerAccess()` | ✅ 已验证 |
| `cudaDeviceGetP2PAttribute()` | P2P 属性查询 | 查询 P2P 属性 | `ixDeviceGetP2PAttribute()` | ✅ 已验证 |
| `cudaLaunchKernel()` | - | Kernel 启动（替代<<<>>>语法） | `ixLaunchKernel()` | ✅ 已验证 |
| `cudaGetDeviceCount()` | - | 获取 GPU 数量 | `ixGetDeviceCount()` | ✅ 已验证 |
| `cudaSetDevice()` | - | 设置当前设备 | `ixSetDevice()` | ✅ 已验证 |
| `cudaGetLastError()` | - | 错误检查 | `ixGetLastError()` | ✅ 已验证 |
| `cuda_err_chk()` | 各处 | CUDA 错误检查宏 | 替换为 `ix_err_chk()` | ✅ 已验证 |

> **验证来源：** [mapping_cudart.h](file:///home/corex/sw_home_1/sw_home/sdk/ixdriver/include/IX/mapping_cudart.h) 中每个 `cuda*` 函数都有对应的 `#define cudaXxx ixXxx` 宏映射。

---

### 2.3 CUDA Kernel Launch 语法

```cuda
// NVIDIA CUDA 语法
read_feature_kernel<TYPE><<<g_size, b_size>>>(args...);
read_feature_kernel<TYPE><<<g_size, b_size, 0, streams[i]>>>(args...);
```

**移植需求：** 需要改为 Iluvatar 的 kernel launch 语法：
```c
// Iluvatar 语法 (示例)
ixLaunchKernel(read_feature_kernel, g_size, b_size, 0, streams[i], args...);
```

---

### 2.4 CUDA 内建变量

| CUDA 内建变量 | 用途 | Iluvatar 替代 |
|--------------|------|--------------|
| `blockIdx.x` | Block 索引 | 同（兼容） |
| `threadIdx.x` | 线程索引 | 同（兼容） |
| `blockDim.x` | Block 维度 | 同（兼容） |
| `atomicAdd()` | 原子加 | 同（兼容） |
| `__global__` | Kernel 函数声明 | 同（兼容） |
| `__device__` | 设备函数声明 | 同（兼容） |
| `<<<>>>` 语法 | Kernel launch | **不兼容！** 需要改为 API 调用 |

---

### 2.5 依赖的第三方 CUDA 库

```cmake
# BaM 框架：libnvm.so
# BaM 本身也是基于 CUDA 的，包含：
#   - cudaMalloc/cudaFree
#   - CUDA kernel launches
#   - GPU 端 NVMe 命令提交
target_link_libraries(BAM_Feature_Store PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../bam/build/lib/libnvm.so)
```

**移植需求：** BaM 框架也需要完整移植到 Iluvatar 平台。

---

## 3. Python 层依赖

### 3.1 PyTorch

```python
import torch
# 使用的 PyTorch CUDA API：
tensor.to('cuda:0')       # Iluvatar Corex PyTorch 使用 cuda:0（不是 ix:0）
tensor.data_ptr()          # GPU 指针，直接传给 kernel
tensor.is_cuda             # Corex PyTorch 兼容
torch.cuda.current_device()# Corex PyTorch 兼容
torch.zeros([...], device='cuda:0')  # device='cuda:0'
torch.load()               # 不变
torch.nonzero()            # 不变
```

**移植需求：** 需要安装 Iluvatar 版本的 PyTorch（`torch_iluvatar`），API 基本兼容。

### 3.2 DGL (Deep Graph Library)

```python
import dgl
# DGL 使用了 DGLHeteroGraph, MultiLayerNeighborSampler 等
# 这些是框架无关的图操作，理论上不需要改
# 但 DGL 的 CUDA 后端需要切换到 Iluvatar 后端
```

### 3.3 nvtx (NVIDIA Tools Extension)

```python
import nvtx            # NVIDIA NVTX 性能标记
import torch.cuda.nvtx as t_nvtx
```

**移植需求：** 需要替换为 Iluvatar 的性能分析工具或直接移除。

---

## 4. 需要修改的关键文件

### 4.1 编译配置

| 文件 | 修改内容 |
|------|---------|
| [CMakeLists.txt](file:///home/corex/sw_home_1/GIDS_enable/GIDS/gids_module/CMakeLists.txt) | 项目语言 CUDA→IXC，GPU 架构 target→iluvatar |
| BaM 的 CMakeLists.txt | 同上 |

### 4.2 kernel 文件

| 文件 | 修改内容 |
|------|---------|
| [gids_kernel.cu](file:///home/corex/sw_home_1/GIDS_enable/GIDS/gids_module/gids_kernel.cu) | kernel launch 语法、CUDA API 替换 |
| [gids_nvme.cu](file:///home/corex/sw_home_1/GIDS_enable/GIDS/gids_module/gids_nvme.cu) | 流管理 API、内存管理 API、错误检查 |
| [bam_nvme.h](file:///home/corex/sw_home_1/GIDS_enable/GIDS/gids_module/include/bam_nvme.h) | CUDA 头文件引用 |

### 4.3 Python 文件

| 文件 | 修改内容 |
|------|---------|
| [GIDS.py](file:///home/corex/sw_home_1/GIDS_enable/GIDS/GIDS_Setup/GIDS/GIDS.py) | cuda→ix 设备字符串 |
| 所有训练脚本 | cuda→ix 设备字符串 |
| [models.py](file:///home/corex/sw_home_1/GIDS_enable/GIDS/evaluation/models.py) | 不变（DGL 模型） |
| [dataloader.py](file:///home/corex/sw_home_1/GIDS_enable/GIDS/evaluation/dataloader.py) | 基本不变 |

---

## 5. 最大难点：BaM 框架移植

GIDS 的核心能力来自 BaM 框架的以下特性：

1. **GPU 端 NVMe 命令提交：** GPU kernel 直接写 NVMe 的 Submission Queue
2. **GPU 端页缓存管理：** page_cache_t 在 GPU 显存中维护 LRU 缓存
3. **透明页故障处理：** bam_ptr 智能指针在 GPU kernel 中处理缺页
4. **PCIe P2P DMA：** GPU 直接通过 PCIe 向 NVMe SSD 发起 DMA 传输

这些特性很多依赖 NVIDIA GPU 特有的硬件能力：

| 能力 | NVIDIA 依赖 | Iluvatar 可行性 |
|------|-----------|----------------|
| GPU→NVMe P2P DMA | nvidia-peermem 内核模块 + GPU BAR 空间映射 | ✅ ixdriver 支持 cuFile API ([ixnvcufile.h](file:///home/corex/sw_home_1/sw_home/sdk/ixdriver/cufileapi/ixnvcufile.h)), 含 `ixdrvFileRead/Write` + `ixdrvFileBufRegister` + RDMA |
| GPU 端 MMIO 写 NVMe 寄存器 | GPU 能访问 PCIe 设备的 MMIO 空间 | ✅ ixdriver 支持 P2P (`ixDeviceEnablePeerAccess`, `ixDeviceGetP2PAttribute`) |
| CUDA Unified Memory | cudaHostAllocMapped | ✅ `ixHostAllocMapped` 已支持 |
| GPU 端内存映射文件 | 通过 BaM 框架实现 | ⚠️ BaM 使用裸 NVMe 命令而非 cuFile，需适配 |

> **关键发现：** Iluvatar 已支持 **cuFile (GPU Direct Storage)** API，含 `ixdrvFileRead`、`ixdrvFileWrite`、`ixdrvFileBufRegister`、`ixdrvfileRDMAInfo` 等。理论上可以直接用 cuFile 替代 BaM 的裸 NVMe 访问方式，大幅简化移植。

---

## 6. 移植路线图

### 阶段 1：可行性验证（1-2 周）

```
1. 确认 Iluvatar GPU 是否支持 PCIe P2P (GPU↔NVMe DMA)
   → 检查 ixsmi 输出
   → 测试 GPU 能否访问 NVMe BAR 空间

2. 确认 Iluvatar 的 Unified Memory 支持
   → 是否有类似 cudaHostAllocMapped 的 API

3. 评估 BaM 移植工作量
   → BaM 代码量
   → CUDA API 数量
```

### 阶段 2：基础移植（3-4 周）

```
1. 替换所有 CUDA API 为 Iluvatar API
   - cudaMalloc → ixMalloc
   - cudaMemcpy → ixMemcpy
   - cudaStream → ixStream
   - cudaDeviceSynchronize → ixDeviceSynchronize

2. 修改 kernel launch 语法
   - <<<>>> → ixLaunchKernel

3. 修改 CMake 构建系统
   - CUDA → IXC
   - SM 目标 → Iluvatar 架构

4. 安装 Iluvatar 版本的 PyTorch 和 DGL
```

### 阶段 3：BaM 框架移植（4-6 周）

```
1. 移植 NVMe 用户态驱动层
   - 可能可以复用 BaM 的大部分代码
   - 关键是 GPU MMIO 访问路径

2. 移植 page_cache 和 array/range 抽象
   - 内存管理 API 替换
   - GPU kernel 修改

3. 移植 bam_ptr 智能指针
   - page fault 处理逻辑

4. 验证 GPU→NVMe P2P 通信
```

### 阶段 4：集成测试（2-3 周）

```
1. 运行单元测试
2. IGB 小规模数据集端到端测试
3. 性能基准测试（vs NVIDIA 平台）
4. 大规模数据集稳定性测试
```

---

## 7. 替代方案

### 方案 A：完全移植 GIDS
- 工作量：**约 12-16 周**（1 名熟练工程师）
- 风险：Iluvatar GPU P2P DMA 能力不确定

### 方案 B：仅移植 GIDS 的优化策略到 Iluvatar 数据加载器
- 将 Window Buffering、CPU Buffer、Accumulator 等算法思想移植
- 底层数据加载使用 Iluvatar 原生的 IO 路径
- 工作量：**约 4-6 周**

### 方案 C：保持 NVIDIA GPU 用于 GIDS 场景
- 如果环境中有 NVIDIA GPU，直接使用
- Iluvatar GPU 用于其他计算任务
- 工作量：**0 周**

---

## 8. 关键 API 映射速查表（已验证）

| NVIDIA CUDA Runtime API | Iluvatar IX API | 状态 | 验证来源 |
|-------------------------|-----------------|------|----------|
| `cudaMalloc` | `ixMalloc` | ✅ 已验证 | mapping_cudart.h:L1784 |
| `cudaMallocHost` | `ixMallocHost` | ✅ 已验证 | mapping_cudart.h:L1787 |
| `cudaMallocManaged` | `ixMallocManaged` | ✅ 已验证 | mapping_cudart.h:L1788 |
| `cudaHostAlloc` | `ixHostAlloc` | ✅ 已验证 | mapping_cudart.h:L2033 |
| `cudaHostAllocMapped` | `ixHostAllocMapped` | ✅ 已验证 | mapping_cudart.h:L2713 |
| `cudaHostAllocWriteCombined` | `ixHostAllocWriteCombined` | ✅ 已验证 | mapping_cudart.h:L2715 |
| `cudaHostGetDevicePointer` | `ixHostGetDevicePointer` | ✅ 已验证 | mapping_cudart.h:L1777 |
| `cudaHostRegister` | `ixHostRegister` | ✅ 已验证 | mapping_cudart.h:L2038 |
| `cudaFree` | `ixFree` | ✅ 已验证 | mapping_cudart.h:L1915 |
| `cudaFreeHost` | `ixFreeHost` | ✅ 已验证 | mapping_cudart.h:L1918 |
| `cudaMemcpy` | `ixMemcpy` | ✅ 已验证 | mapping_cudart.h:L2110 |
| `cudaMemcpyAsync` | `ixMemcpyAsync` | ✅ 已验证 | mapping_cudart.h:L2125 |
| `cudaMemcpyHostToDevice` | `ixMemcpyHostToDevice` | ✅ 已验证 | mapping_cudart.h:L1796 |
| `cudaMemcpyDeviceToHost` | `ixMemcpyDeviceToHost` | ✅ 已验证 | mapping_cudart.h:L1793 |
| `cudaMemset` | `ixMemset` | ✅ 已验证 | mapping_cudart.h:L2084 |
| `cudaStreamCreate` | `ixStreamCreate` | ✅ 已验证 | mapping_cudart.h:L2178 |
| `cudaStreamCreateWithFlags` | `ixStreamCreateWithFlags` | ✅ 已验证 | mapping_cudart.h:L1673 |
| `cudaStreamDestroy` | `ixStreamDestroy` | ✅ 已验证 | mapping_cudart.h:L1674 |
| `cudaStreamSynchronize` | `ixStreamSynchronize` | ✅ 已验证 | ix_runtime_api.h:L70 |
| `cudaDeviceSynchronize` | `ixDeviceSynchronize` | ✅ 已验证 | mapping_cudart.h:L1893 |
| `cudaSetDevice` | `ixSetDevice` | ✅ 已验证 | mapping_cudart.h:L1888 |
| `cudaGetDeviceCount` | `ixGetDeviceCount` | ✅ 已验证 | mapping_cudart.h:L1641 |
| `cudaLaunchKernel` | `ixLaunchKernel` | ✅ 已验证 | mapping_cudart.h:L2057 |
| `cudaLaunchKernel_ptsz` | `ixLaunchKernel_ptsz` | ✅ 已验证 | mapping_cudart.h:L2060 |
| `cudaDeviceEnablePeerAccess` | `ixDeviceEnablePeerAccess` | ✅ 已验证 | mapping_cudart.h:L1871 |
| `cudaDeviceCanAccessPeer` | `ixDeviceCanAccessPeer` | ✅ 已验证 | mapping_cudart.h:L1869 |
| `cudaDeviceGetP2PAttribute` | `ixDeviceGetP2PAttribute` | ✅ 已验证 | mapping_cudart.h:L1880 |
| `cudaGetLastError` | `ixGetLastError` | ✅ 已验证 | mapping_cudart.h:L1644 |
| `cudaGetErrorString` | `ixGetErrorString` | ✅ 已验证 | mapping_cudart.h:L1643 |
| `cudaEventCreate` | `ixEventCreate` | ✅ 已验证 | mapping_cudart.h:L1760 |
| `cudaEventRecord` | `ixEventRecord` | ✅ 已验证 | mapping_cudart.h:L1631 |
| `cudaFuncSetCacheConfig` | `ixFuncSetCacheConfig` | ✅ 已验证 | mapping_cudart.h:L1768 |

### GPU Direct Storage (cuFile) 映射

| NVIDIA cuFile API | Iluvatar cuFile API | 状态 |
|-------------------|---------------------|------|
| `cuFileDriverOpen` | `ixdrvFileDriverOpen` | ✅ 已验证 | cuFile 标准 API |
| `cuFileDriverClose` | `ixdrvFileDriverClose` | ✅ 已验证 | cuFile 标准 API |
| `cuFileHandleRegister` | `ixdrvFileHandleRegister` | ✅ 已验证 | cuFile 标准 API |
| `cuFileHandleDeregister` | `ixdrvFileHandleDeregister` | ✅ 已验证 | cuFile 标准 API |
| `cuFileBufRegister` | `ixdrvFileBufRegister` | ✅ 已验证 | cuFile 标准 API |
| `cuFileBufDeregister` | `ixdrvFileBufDeregister` | ✅ 已验证 | cuFile 标准 API |
| `cuFileRead` | `ixdrvFileRead` | ✅ 已验证 | cuFile 标准 API |
| `cuFileWrite` | `ixdrvFileWrite` | ✅ 已验证 | cuFile 标准 API |
| `cufileRDMAInfo` | `ixdrvfileRDMAInfo` | ✅ 已验证 | cuFile 标准 API |

> GIDS-IX 代码中使用标准 cuFile API 名称（`cuFileRead` 等），通过 Iluvatar Corex SDK 的 `libcufile.so` 提供底层实现。

### GDR (GPU Direct RDMA) 映射

| NVIDIA GDR API | Iluvatar IX API | 状态 |
|---------------|-----------------|------|
| `gdrapi.h` | `gdrapi.h` (经过 `mapping_cudart.h` 包装) | ✅ 已验证 |

### 编译工具链映射

| NVIDIA | Iluvatar | 状态 |
|--------|----------|------|
| `nvcc` | `ixc` | ✅ (ixdriver SDK 包含) |
| `__CUDACC__` | `__IXCC__` | ✅ mapping_cudart.h:L48-50 |
| `__CUDA_ARCH__` | `__IX_ARCH__` | ✅ mapping_cudart.h:L36-38 |
| `<<<grid, block>>>` | `<<<grid, block>>>` 或 `ixLaunchKernel()` | ✅ 兼容两种方式 |
| CUDA Runtime API v10.2 | IXRT v10.02 | ✅ ix_runtime_api.h:#define IXRT_VERSION 10020 |