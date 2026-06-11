# GIDS 架构总览

## 1. 项目概述

**GIDS (GPU-Initiated Direct Storage Accesses) Dataloader** 是一个用于加速大规模图神经网络（GNN）训练的开源数据加载系统。核心思想是让 GPU 直接从 NVMe SSD 读取图节点特征数据，绕过 CPU，消除传统数据流水线中的 CPU 瓶颈。

### 关键特性
- GPU Direct Storage：GPU kernel 直接发起 NVMe 读请求
- Window Buffering：预取窗口缓冲机制，隐藏 SSD 延迟
- Storage Access Accumulator：批量合并多次 SSD 访问请求
- CPU Feature Buffer：热数据缓存到 CPU 内存（Zero-copy 映射到 GPU）
- 同构图 + 异构图支持
- 多 SSD 条带化（Page-level striping）

---

## 2. 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                    Python 应用层                          │
│  ┌──────────────┐  ┌─────────────────────────────────┐   │
│  │ GIDS.py       │  │ evaluation/*.py (训练脚本)       │   │
│  │ ┌───────────┐ │  │ homogenous_train.py             │   │
│  │ │ GIDS 类   │ │  │ heterogeneous_train.py          │   │
│  │ │ - 初始化   │ │  │ GIDS_unit_test.py              │   │
│  │ │ - 窗口缓冲 │ │  └─────────────────────────────────┘   │
│  │ │ - 特征拉取 │ │                                        │
│  │ │ - 设置缓存 │ │  DGL Framework                       │
│  │ └───────────┘ │  ┌──────────────────────────────────┐  │
│  │ ┌───────────┐ │  │ GIDS_DGLDataLoader               │  │
│  │ │ DataLoader│ │  │ (自定义 DGL DataLoader)          │  │
│  │ └───────────┘ │  │ - 图采样委托给 DGL               │  │
│  └──────────────┘  │ - 特征加载交给 GIDS               │  │
│                     └──────────────────────────────────┘  │
├──────────────────────────────────────────────────────────┤
│                 pybind11 绑定层                            │
│         BAM_Feature_Store.so (C++/CUDA → Python)          │
├──────────────────────────────────────────────────────────┤
│                   CUDA/C++ 核心层                          │
│  ┌──────────────────┐  ┌───────────────────────────────┐ │
│  │ gids_kernel.cu    │  │ gids_nvme.cu                  │ │
│  │ (GPU Kernel)      │  │ (Host 端管理逻辑)             │ │
│  │ ┌───────────────┐ │  │ ┌───────────────────────────┐ │ │
│  │ │read_feature   │ │  │ │BAM_Feature_Store 类       │ │ │
│  │ │  _kernel      │ │  │ │- init_controllers()       │ │ │
│  │ │               │ │  │ │- read_feature()            │ │ │
│  │ │cpu_buffer     │ │  │ │- read_feature_hetero()    │ │ │
│  │ │  _kernel      │ │  │ │- read_feature_merged()    │ │ │
│  │ │               │ │  │ │- cpu_backing_buffer()     │ │ │
│  │ │window_buffer  │ │  │ │- set_cpu_buffer()         │ │ │
│  │ │  _kernel      │ │  │ │- store_tensor()           │ │ │
│  │ │               │ │  │ │- window_buffering()       │ │ │
│  │ │write_kernel   │ │  │ └───────────────────────────┘ │ │
│  │ └───────────────┘ │  │ ┌───────────────────────────┐ │ │
│  └──────────────────┘  │ │GIDS_Controllers 类         │ │ │
│                         │ │- 管理 NVMe Controller     │ │ │
│                         │ └───────────────────────────┘ │ │
│                         └───────────────────────────────┘ │
├──────────────────────────────────────────────────────────┤
│                    BaM 框架层                              │
│  ┌──────────────────────────────────────────────────┐    │
│  │ libnvm.so                                        │    │
│  │ - Controller (NVMe 控制器抽象)                   │    │
│  │ - page_cache_t (GPU 端页缓存管理)                │    │
│  │ - range_t / array_t (地址映射)                   │    │
│  │ - bam_ptr<T> (GPU 端智能指针，支持透明页故障)    │    │
│  │ - nvm_parallel_queue (NVMe 命令队列)             │    │
│  └──────────────────────────────────────────────────┘    │
├──────────────────────────────────────────────────────────┤
│                    硬件层                                  │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │ NVIDIA   │  │ NVMe SSD     │  │ PCIe 总线         │   │
│  │ GPU      │◄─┤ /dev/libnvm0 │  │ (GPU ↔ SSD P2P)  │   │
│  │ (CUDA)   │  │ /dev/libnvm1 │  │                   │   │
│  └──────────┘  └──────────────┘  └───────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

---

## 3. 数据流

### 3.1 训练循环数据流

```
1. DGL Sampler 采样
   ┌──────────┐    邻居采样     ┌──────────────┐
   │ 大图 (G) │ ──────────────► │ 子图 (blocks) │
   └──────────┘                 │ + node IDs    │
                                └──────────────┘

2. GIDS 特征读取 (GPU Direct)
   ┌──────────────┐   GPU Kernel    ┌──────────────────┐
   │ node IDs     │ ──────────────► │ BAM_Feature_Store │
   │ (GPU memory) │                 │ (GPU 端)          │
   └──────────────┘                 └───────┬──────────┘
                                            │
                          bam_ptr<T> 智能指针
                          (透明页故障处理)
                                            │
                                   ┌────────▼──────────┐
                                   │ page_cache_t      │
                                   │ (GPU 显存页缓存)   │
                                   └────────┬──────────┘
                                            │ Cache Miss
                                   ┌────────▼──────────┐
                                   │ NVMe Controller   │
                                   │ /dev/libnvmX      │
                                   └────────┬──────────┘
                                            │ PCIe P2P
                                   ┌────────▼──────────┐
                                   │ NVMe SSD          │
                                   │ (节点特征数据)     │
                                   └───────────────────┘

3. GNN 模型训练
   ┌──────────────┐                ┌──────────────┐
   │ 特征 Tensor   │ ─────────────►│ GNN Model    │
   │ (GPU memory)  │               │ (SAGE/GCN...) │
   └──────────────┘                └──────────────┘
```

### 3.2 Window Buffering 机制

```
Timeline (CUDA Streams):
─────────────────────────────────────────────────────────────────►
Stream[0]: [GNN Train] [GNN Train] [GNN Train] [GNN Train] ...
Stream[1]: [WB Prefetch] [WB Prefetch] [WB Prefetch] ...
Stream[2]: [Feature Read] [Feature Read] [Feature Read] ...

WB = Window Buffering: 提前通知 BaM 缓存即将访问的页面
     (set_window_buffer_counter → 增加页面的预取优先级)
```

---

## 4. 核心组件详解

### 4.1 BaM (GPU-Initiated NVMe Access Framework)

BaM 是 GIDS 的底层依赖，来自 [ZaidQureshi/bam](https://github.com/ZaidQureshi/bam.git)（window_buffer 分支）。

**核心抽象：**

| 组件 | 作用 |
|------|------|
| `Controller` | 管理一个 NVMe 设备，提供命令提交队列 |
| `page_cache_t` | GPU 端 DRAM 页缓存，管理页的分配/淘汰/刷新 |
| `range_t<T>` | 定义一段数据的地址映射（逻辑地址→物理页） |
| `array_t<T>` | 可包含多个 range，支持条带化 |
| `bam_ptr<T>` | GPU 端智能指针，operator[] 透明处理 page fault |

**关键设计：**
- 数据分布模式：`STRIPE`（条带化）或 `REPLICATE`（复制）
- Page Cache 管理：LRU-like 淘汰策略
- NVMe 命令提交：GPU kernel 直接写入 NVMe SQ（Submission Queue）

### 4.2 BAM_Feature_Store（C++/CUDA 核心类）

```cpp
template <typename TYPE>
struct BAM_Feature_Store {
    // BaM 核心对象
    page_cache_t *h_pc;     // Host 端页缓存句柄
    range_t<TYPE> *h_range; // 数据范围映射
    array_t<TYPE> *a;       // GPU 端数组（bam_ptr 操作对象）
    
    // CPU 缓冲优化
    GIDS_CPU_buffer<TYPE> CPU_buffer;
    bool cpu_buffer_flag;   // 是否启用 CPU 缓冲
    bool seq_flag;          // 顺序/哈希模式
    
    // 控制参数
    uint32_t pageSize;      // 页大小 (默认 4096B)
    uint64_t numElems;      // 数据集总元素数
    uint32_t n_ctrls;       // SSD 数量
    
    // 核心方法
    void init_controllers(...);  // 初始化 NVMe 控制器 + 页缓存
    void read_feature(...);      // 同构图特征读取（单 stream）
    void read_feature_hetero(...); // 异构图特征读取（多 stream 并发）
    void read_feature_merged(...); // 批量合并读取
    void cpu_backing_buffer(...);  // 分配 CPU 缓冲 (cudaHostAlloc)
    void set_cpu_buffer(...);      // 预设热节点到 CPU 缓冲
    void set_window_buffering(...); // 标记窗口预取页面
    void store_tensor(...);        // 将 tensor 写入 SSD
};
```

### 4.3 GIDS Python 类

```python
class GIDS:
    def __init__(self, page_size, off, cache_dim, num_ele, 
                 num_ssd, ssd_list, cache_size, ctrl_idx,
                 window_buffer, wb_size, accumulator_flag, 
                 long_type, heterograph, heterograph_map):
        # 创建 BAM_Feature_Store 实例
        # 初始化 GIDS_Controllers
        # 配置窗口缓冲、累加器等优化
        
    def fetch_feature(self, dim, it, device):
        # 核心数据加载循环：
        # 1. 从迭代器获取下一批节点 ID
        # 2. (可选) 窗口缓冲预取
        # 3. (可选) Storage Access Accumulator 合并
        # 4. 调用 BAM_FS.read_feature() GPU Direct 读取
        # 5. 返回特征 tensor
        
    def cpu_backing_buffer(self, dim, length):
        # 分配 CPU pinned memory 作为热数据缓存
        
    def set_cpu_buffer(self, ten, N):
        # 将 PageRank Top-N 节点预设入 CPU 缓冲
        
    def window_buffering(self, batch):
        # 提前通知 page cache 即将访问的页面
```

### 4.4 GIDS_DGLDataLoader

自定义 DGL DataLoader，图采样部分委托 DGL，特征加载交给 GIDS：

```python
class GIDS_DGLDataLoader(torch.utils.data.DataLoader):
    def __iter__(self):
        # 包装 DGL 原生迭代器
        return _PrefetchingIter(self, super().__iter__(), 
                                GIDS_Loader=self.GIDS_Loader)

class _PrefetchingIter:
    def __next__(self):
        # 1. 获取 DGL 采样的批次
        # 2. 通过 GIDS_Loader.fetch_feature() 获取特征
        # 3. 返回 (input_nodes, seeds, blocks, features)
```

---

## 5. 三大优化策略

### 5.1 Window Buffering（窗口缓冲）

**问题：** GPU kernel 访问 SSD 时，page fault 导致高延迟
**方案：** 在处理当前 batch 时，提前通知 BaM 缓存下一页面的范围

```python
# 在 fetch_feature 中
if self.window_buffering_flag:
    self.window_buffering(next_batch)  # 预取下一批
batch = self.window_buffer.pop(0)      # 使用已缓存的当前批
```

GPU kernel 层面：
```cuda
__global__ void set_window_buffering_kernel(array_d_t<T>* dr, 
    uint64_t *index_ptr, uint64_t page_size, int hash_off) {
    bam_ptr<T> ptr(dr);
    // 增加指定页面的预取计数器
    ptr.set_window_buffer_counter(page_idx * page_size/sizeof(T), 1);
}
```

### 5.2 Storage Access Accumulator（存储访问累加器）

**问题：** 每次 SSD 访问都有固定延迟开销，小批量访问效率低
**方案：** 累积多个 batch 的访问请求后合并提交

```python
# 根据 SSD 带宽/延迟计算最优合并数量
accesses = (p * bw * 1024 / page_size * (l_ssd + l_system) * num_ssd) / (1-p)

# 累积足够访问量后，合并到一次 read_feature_merged 调用
self.BAM_FS.read_feature_merged(num_iter, return_torch_list, 
                                 index_ptr_list, ...)
```

### 5.3 CPU Feature Buffer（CPU 特征缓冲）

**问题：** 部分节点被频繁访问（热数据），每次都从 SSD 读取浪费带宽
**方案：** 基于 PageRank 识别热节点，预加载到 CPU pinned memory

```cpp
// 使用 cudaHostAllocMapped 实现 zero-copy
cudaHostAlloc(&cpu_buffer_ptr, sizeof(TYPE) * dim * len, cudaHostAllocMapped);
cudaHostGetDevicePointer(&d_cpu_buffer_ptr, cpu_buffer_ptr, 0);
```

GPU kernel 分流逻辑：
```cuda
if (row_index < CPU_buffer.cpu_buffer_len) {
    // 从 CPU buffer 读取 (通过 PCIe)
    temp = CPU_buffer.device_cpu_buffer[row_index * cache_dim + tid];
} else {
    // 从 SSD 读取 (GPU Direct)
    temp = ptr.read(row_index * cache_dim + tid);
}
```

---

## 6. 目录结构

```
GIDS/
├── .gitmodules                    # BaM 子模块配置
├── .gitignore
├── README.md
├── head.html
│
├── gids_module/                   # C++/CUDA 核心模块
│   ├── CMakeLists.txt             # CMake 构建配置
│   ├── gids_kernel.cu             # GPU Kernel 实现
│   ├── gids_nvme.cu               # Host 端管理 + pybind11 绑定
│   ├── include/
│   │   ├── bam_nvme.h             # BAM_Feature_Store + GIDS_Controllers 声明
│   │   ├── page_cache_backup.h    # 页缓存备份声明
│   │   └── example.h
│   ├── BAM_Feature_Store/
│   │   ├── __init__.py
│   │   └── setup.py
│   └── example/
│       ├── __init__.py
│       └── setup.py
│
├── GIDS_Setup/                    # Python 包安装
│   ├── setup.py
│   ├── GIDS/
│   │   ├── __init__.py            # 导出 GIDS 和 GIDS_DGLDataLoader
│   │   ├── GIDS.py                # 核心 Python 实现
│   │   └── test.py
│   └── dist/
│
└── evaluation/                    # 评估与示例
    ├── homogenous_train.py        # 同构图 GIDS/BaM 训练
    ├── homogenous_train_baseline.py # 同构图 mmap 基线
    ├── heterogeneous_train.py     # 异构图 GIDS/BaM 训练
    ├── heterogeneous_train_baseline.py # 异构图 mmap 基线
    ├── homogenous_train_ClusterGCN.py
    ├── models.py                  # GNN 模型定义 (SAGE/GCN/GAT/RGCN/RSAGE)
    ├── dataloader.py              # IGB/OGB 数据集加载
    ├── tensor_write.py            # Tensor → SSD 写入工具
    ├── page_rank_node_list_gen.py # PageRank 热节点列表生成
    ├── ladies_sampler.py          # LADIES 采样器
    ├── mlperf_model.py
    ├── lock_mem.cpp
    ├── GIDS_unit_test.py          # 单元测试
    ├── gids_unit_test.sh
    ├── run_GIDS_IGBH.sh           # 运行脚本：GIDS 模式
    ├── run_BaM_IGBH.sh            # 运行脚本：BaM 模式
    ├── run_base_IGBH.sh           # 运行脚本：mmap 基线模式
    ├── write_data.sh
    └── write_data_full.sh
```

---

## 7. 依赖关系

```
GIDS
 ├── BaM (git submodule, window_buffer branch)
 │    └── libnvm.so (NVMe 用户态驱动)
 ├── DGL (Deep Graph Library)
 ├── PyTorch (深度学习框架)
 ├── pybind11 (C++/Python 绑定)
 ├── CUDA Toolkit (>= 10.0, compute capability >= 7.0)
 └── NVMe SSD (至少 1 块，支持多 SSD 条带化)
```

---

## 8. 硬件要求

| 组件 | 要求 |
|------|------|
| GPU | NVIDIA GPU，计算能力 ≥ 7.0 (Volta/Turing/Ampere) |
| 存储 | NVMe SSD，通过 `/dev/libnvmX` 用户态驱动访问 |
| 内存 | 足以容纳图结构数据 + CPU 缓冲 |
| PCIe | 支持 GPU↔SSD P2P (Peer-to-Peer DMA) |
| 数据集 | IGB、OGB、MAG 等大规模图数据集，特征存储在 SSD |