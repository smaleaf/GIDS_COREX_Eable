# HugeCTR GPU Embedding Cache 技术详解与 GIDS 优化方案

> **最后更新：** 2026-06-11 | 基于 HugeCTR gpu_cache 源码分析

---

## 目录

1. [HugeCTR GPU Cache 技术原理](#1-hugectr-gpu-cache-技术原理)
2. [DGL 集成方式](#2-dgl-集成方式)
3. [与 GIDS IXFeatureStore 的对比](#3-与-gids-ixfeaturestore-的对比)
4. [GIDS 两级缓存优化方案](#4-gids-两级缓存优化方案)
5. [实现路线图](#5-实现路线图)

---

## 1. HugeCTR GPU Cache 技术原理

### 1.1 核心思想

HugeCTR GPU Cache 是 NVIDIA 为推荐系统 embedding 查表场景设计的 GPU 原生缓存。它将高频访问的 embedding 向量缓存在 GPU 显存中，避免重复从 CPU/SSD 读取。

### 1.2 数据结构

```
┌─────────────────────────────────────────────────┐
│                  GPU Cache                       │
│                                                 │
│  ┌──────────┐  ┌──────────┐      ┌──────────┐  │
│  │  Set 0   │  │  Set 1   │ ...  │  Set N-1 │  │
│  │ ┌──────┐ │  │ ┌──────┐ │      │ ┌──────┐ │  │
│  │ │Way 0 │ │  │ │Way 0 │ │      │ │Way 0 │ │  │
│  │ │32keys│ │  │ │32keys│ │      │ │32keys│ │  │
│  │ ├──────┤ │  │ ├──────┤ │      │ ├──────┤ │  │
│  │ │Way 1 │ │  │ │Way 1 │ │      │ │Way 1 │ │  │
│  │ │32keys│ │  │ │32keys│ │      │ │32keys│ │  │
│  │ └──────┘ │  │ └──────┘ │      │ └──────┘ │  │
│  └──────────┘  └──────────┘      └──────────┘  │
│                                                 │
│  Set-Associativity: 2-way (可配置)               │
│  Slab Size: 32 keys (一个 warp)                  │
│  容量: capacity_in_set × num_sets × associativity │
└─────────────────────────────────────────────────┘
```

### 1.3 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `set_associativity` | 2 | 组相联度，每组 2 路 |
| `warp_size` | 32 | 每个 slab 存 32 个 key（一个 warp 大小） |
| `SLAB_SIZE` | 32 | 同上 |
| `capacity_in_set` | 用户配置 | 每组（set）的容量 |
| `TASK_PER_WARP_TILE` | 1 | 每个 warp 处理的 key 数 |

### 1.4 缓存策略：全局计数器 + 槽位计数器

HugeCTR 使用 **全局逻辑时钟 + 槽位时间戳** 实现类 LRU 淘汰：

```
全局计数器 (global_counter): 每次 Query 递增，作为逻辑时钟
槽位计数器 (slot_counter):  记录每个槽位最后被访问的时间戳

淘汰策略:
  遍历 set 的所有 way，找到 slot_counter 最小的槽位（最久未访问）
  → 替换该槽位
```

这是一个 **近似 LRU**，不需要维护链表，线程安全且 GPU 友好。

### 1.5 核心 API

```cpp
template <typename key_type, ...>
class gpu_cache {
public:
    // 查询：命中则返回 value，未命中则记录到 missing 列表
    void Query(const key_type* d_keys, size_t len,
               float* d_values,
               uint64_t* d_missing_index,   // 未命中的位置
               key_type* d_missing_keys,     // 未命中的 key
               size_t* d_missing_len,        // 未命中数量
               cudaStream_t stream);

    // 替换：将 missing keys 对应的 values 写入缓存（淘汰旧数据）
    void Replace(const key_type* d_keys, size_t len,
                 const float* d_values, cudaStream_t stream);

    // 原地更新：更新已存在缓存中的 embedding
    void Update(const key_type* d_keys, size_t len,
                const float* d_values, cudaStream_t stream);
};
```

### 1.6 Warp 级并行

每个 warp（32 线程）处理一个 set 的查询，利用 CUDA cooperative groups 实现 warp 内协作：

```cuda
cg::thread_block_tile<32> warp_tile = cg::tiled_partition<32>(cg::this_thread_block());
size_t lane_idx = warp_tile.thread_rank();           // warp 内线程号 0~31
size_t global_warp_id = blockIdx.x * warps_per_block + warp_tile.meta_group_rank();
```

每个 warp 独立负责一个 key 的查表，32 个 lane 并行比较 slab 中的 32 个 key。

### 1.7 哈希函数

```cpp
// Set 级别哈希：key → set index
set_hasher(key) % num_sets  →  确定 key 属于哪个 set

// Slab 级别哈希（可选）
slab_hasher(key) % capacity_in_set  →  在 set 内定位 slab
```

默认使用 MurmurHash3_32，同时支持自定义哈希函数。

---

## 2. DGL 集成方式

DGL 将 HugeCTR gpu_cache 编译进 `libdgl.so`，用于 DGL 原生 embedding 场景：

```
DGL Embedding Lookup
       │
       ▼
  ┌─────────┐    命中    ┌──────────┐
  │gpu_cache│ ────────▶ │ 返回 value│
  │  Query  │           └──────────┘
  └────┬────┘
       │ 未命中
       ▼
  ┌─────────┐
  │ 从 CPU  │
  │ 读取并  │
  │ Replace │
  └─────────┘
```

DGL 的 `dgl.nn.Embedding` 或自定义 embedding 层可以通过 gpu_cache 加速高频 key 的查表。

---

## 3. 与 GIDS IXFeatureStore 的对比

| 维度 | gpu_cache (HugeCTR) | IXFeatureStore (GIDS) |
|------|---------------------|----------------------|
| **存储介质** | GPU 显存 (HBM) | NVMe SSD |
| **访问延迟** | ~100 ns | ~10 μs (NVMe) / ~100 μs (HDD) |
| **容量** | GB 级 (受 GPU 显存限制) | TB 级 |
| **缓存策略** | 类 LRU (全局计数器) | 页缓存 (page_cache_t) |
| **并行粒度** | Warp 级 (32 线程) | 线程级 |
| **数据来源** | CPU 内存 | NVMe SSD 文件 |
| **适用场景** | 高频热数据 | 全量冷数据 + 温数据 |

### 互补关系

```
           ┌──────────────────────┐
           │    访问频率           │
           │                      │
     高    │  ██ gpu_cache ██     │  ← GPU HBM (~100ns)
           │  (热数据, 高频节点)    │
           │                      │
     中    │  ░░ IXFeatureStore ░░ │  ← NVMe SSD (~10μs)
           │  (温数据, 中频节点)    │
           │                      │
     低    │  ░░ IXFeatureStore ░░ │  ← NVMe SSD
           │  (冷数据, 低频节点)    │
           └──────────────────────┘
```

**两者不是替代关系，而是互补关系。** GIDS 的 IXFeatureStore 解决"数据放不下内存"的问题，gpu_cache 解决"热点数据加速"的问题。

---

## 4. GIDS 两级缓存优化方案

### 4.1 方案概述

在 GIDS 现有架构上增加一层 GPU 嵌入缓存，形成两级缓存：

```
                    ┌─────────────────────┐
                    │   GIDS_DGLDataLoader │
                    │   (Neighbor Sampler) │
                    └──────────┬──────────┘
                               │ 采样得到 input_nodes
                               ▼
                    ┌─────────────────────┐
                    │   L1: gpu_cache     │  ← GPU HBM 热缓存
                    │   Query(keys)       │
                    └──────┬──────┬───────┘
                      命中  │      │  未命中
                           ▼      ▼
                    ┌─────────┐ ┌──────────────────┐
                    │ 返回    │ │ L2: IXFeatureStore│  ← NVMe SSD 冷存储
                    │ value   │ │ Read from SSD     │
                    └─────────┘ └────────┬─────────┘
                                         │
                                         ▼
                                  ┌─────────────┐
                                  │ gpu_cache   │
                                  │ Replace/    │  ← 回填热缓存
                                  │ Update      │
                                  └─────────────┘
```

### 4.2 具体实现方案

#### 4.2.1 在 GIDS 类中集成 gpu_cache

```python
class GIDS:
    def __init__(self, ..., use_gpu_cache=True, gpu_cache_capacity=1000000):
        # 现有 IXFeatureStore（L2 冷存储）
        self.IX_FS = IXFeatureStore.IXFeatureStore_float()
        self.IX_FS.init_controllers(...)

        # 新增 GPU 缓存（L1 热存储）
        if use_gpu_cache:
            self.gpu_cache = gpu_cache_api(
                capacity_in_set=gpu_cache_capacity // num_sets,
                embedding_vec_size=cache_dim,
                set_associativity=2,
                warp_size=32
            )
        else:
            self.gpu_cache = None

    def fetch_feature(self, dim, it, device):
        """改造后的特征读取：先查 GPU 缓存，未命中再读 SSD"""
        batch = next(it)
        input_nodes = batch[0]  # 采样的输入节点 ID

        values = torch.empty(len(input_nodes), dim, device=device)

        if self.gpu_cache is not None:
            # L1: 查询 GPU 缓存
            missing_idx, missing_keys, missing_len = self.gpu_cache.Query(
                input_nodes, values
            )

            if missing_len > 0:
                # L2: 未命中 → 从 SSD 读取
                ssd_values = self.IX_FS.read_feature(
                    missing_keys, missing_len, dim
                )
                values[missing_idx] = ssd_values

                # 回填 GPU 缓存
                self.gpu_cache.Replace(missing_keys, missing_len, ssd_values)
        else:
            # 无 GPU 缓存 → 直接走 SSD
            self.IX_FS.read_feature(input_nodes, values, dim)

        return batch, values
```

#### 4.2.2 热点感知的预取策略

图采样天然产生热点——**高度数节点被采样频率远高于低度数节点**。利用这一特性：

```python
class GIDS:
    def __init__(self, ...):
        self.access_counter = {}      # 节点访问计数
        self.hot_threshold = 100      # 热节点阈值
        self.prefetch_queue = []      # 预取队列

    def update_access_stats(self, input_nodes):
        """更新节点访问统计，识别热节点"""
        for nid in input_nodes:
            self.access_counter[nid] = self.access_counter.get(nid, 0) + 1

    def prefetch_hot_nodes(self):
        """异步预取热节点到 GPU 缓存"""
        hot_nodes = [
            nid for nid, count in self.access_counter.items()
            if count >= self.hot_threshold
        ]
        if hot_nodes:
            values = self.IX_FS.read_feature(hot_nodes, len(hot_nodes), self.cache_dim)
            self.gpu_cache.Replace(hot_nodes, len(hot_nodes), values)
```

#### 4.2.3 窗口缓冲 + GPU 缓存的协同

GIDS 现有的 `window_buffering` 机制预取后续 batch 的节点 ID，可以与 GPU 缓存协同：

```python
def window_buffering_with_cache(self, batch):
    """窗口缓冲 + GPU 缓存预取"""
    input_tensor = batch[0].to(self.gids_device)

    # 1. 通知 IXFeatureStore 预取（SSD 异步读取）
    self.IX_FS.set_window_buffering(input_tensor.data_ptr(), len(input_tensor), 0)

    # 2. 同时查询 GPU 缓存（命中的直接从显存返回）
    if self.gpu_cache is not None:
        self.gpu_cache.Query(input_tensor, self._gpu_cache_buffer)
```

### 4.3 缓存容量规划

| 数据集 | 节点数 | 特征维度 | 全量大小 | GPU 缓存 (10%) | GPU 缓存 (20%) |
|--------|--------|---------|---------|---------------|---------------|
| ogbn-products | 2.4M | 100 | 960 MB | 96 MB | 192 MB |
| ogbn-papers100M | 111M | 128 | 57 GB | 5.7 GB | 11.4 GB |

对于 ogbn-products，分配 100-200MB GPU 缓存即可覆盖 10-20% 的热节点，预期命中率可达 60-80%（服从幂律分布）。

### 4.4 性能预期

| 场景 | 无缓存 | 仅 GPU 缓存 | 两级缓存 |
|------|--------|------------|---------|
| 热节点访问延迟 | ~10 μs (SSD) | ~100 ns (HBM) | ~100 ns |
| 冷节点访问延迟 | ~10 μs (SSD) | ~10 μs (SSD) | ~10 μs |
| 缓存命中率 (预期) | 0% | 60-80% | 60-80% |
| 额外 GPU 显存 | 0 | 100-200 MB | 100-200 MB |
| 吞吐提升 (预期) | 1x | 3-5x (热数据) | 2-3x (整体) |

---

## 5. 实现路线图

### 阶段 1：编译集成 (当前)

- [x] 确认 gpu_cache 在 Corex 上的编译兼容性
- [x] 修复 `meta_group_rank()` 兼容问题（SDK 头文件补丁，SWPM-918-gids）
- [ ] ~~完成 DGL + gpu_cache 的 Corex 编译~~ **阻塞**：编译器后端 `%laneid` 不支持
- [ ] 等待 Corex 编译器团队支持 `%laneid` PTX 寄存器
- **详细分析**：`/root/GIDS_cufile/docs/02-HugeCTR-gpu_cache-Corex兼容分析.md`

### 阶段 2：基础集成

- [ ] 在 GIDS 类中增加 `gpu_cache` 可选组件
- [ ] 实现 `fetch_feature` 的两级查表流程（Query → SSD → Replace）
- [ ] 添加 `--use_gpu_cache` 和 `--gpu_cache_capacity` 训练参数
- [ ] 基准测试：对比有/无 GPU 缓存的吞吐量

### 阶段 3：热点优化

- [ ] 实现节点访问频率统计
- [ ] 实现异步热节点预取
- [ ] 窗口缓冲 + GPU 缓存协同
- [ ] 针对图幂律分布特性优化缓存容量分配

### 阶段 4：高级特性

- [ ] 自适应缓存容量（根据命中率动态调整）
- [ ] 多 GPU 分布式缓存
- [ ] 缓存一致性协议（训练中 embedding 更新时同步）

---

## 参考资料

- HugeCTR GPU Cache 源码: `third_party/HugeCTR/gpu_cache/`
- NVIDIA HugeCTR: https://github.com/NVIDIA-Merlin/HugeCTR
- DGL GPU Cache 集成 PR: https://github.com/dmlc/dgl/pull/5500
- GIDS BaM 论文: "BaM: A Case for Enabling Fine-grain High Throughput GPU-orchestrated Access to Storage"
- **Corex 兼容分析：** `/root/GIDS_cufile/docs/01-DGL-Corex兼容分析.md`
- **HugeCTR gpu_cache 编译器兼容：** `/root/GIDS_cufile/docs/02-HugeCTR-gpu_cache-Corex兼容分析.md`
- **兼容库总览：** `/root/GIDS_cufile/docs/00-Corex兼容库总览.md`