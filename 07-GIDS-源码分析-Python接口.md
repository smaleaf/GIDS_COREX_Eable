# GIDS 源码分析：Python 接口与 DGL DataLoader 集成

## 1. 文件概览

| 文件 | 角色 |
|------|------|
| [GIDS.py](file:///home/corex/sw_home_1/GIDS_enable/GIDS/GIDS_Setup/GIDS/GIDS.py) | 核心 Python 实现：GIDS 类 + GIDS_DGLDataLoader |
| [__init__.py](file:///home/corex/sw_home_1/GIDS_enable/GIDS/GIDS_Setup/GIDS/__init__.py) | 导出 `GIDS` 和 `GIDS_DGLDataLoader` |
| [setup.py](file:///home/corex/sw_home_1/GIDS_enable/GIDS/GIDS_Setup/setup.py) | pip 安装配置 |
| [test.py](file:///home/corex/sw_home_1/GIDS_enable/GIDS/GIDS_Setup/GIDS/test.py) | 简单功能测试 |

---

## 2. GIDS 类详解

### 2.1 构造函数

```python
class GIDS():
    def __init__(self, 
        page_size=4096,        # 页大小（字节）
        off=0,                 # 数据起始偏移
        cache_dim=1024,        # 缓存维度
        num_ele=300*1000*1000*1024,  # 数据集总元素数
        num_ssd=1,             # SSD 数量
        ssd_list=None,         # SSD 设备列表 [0,1,2,...]
        cache_size=10,         # GPU 页缓存大小（页数）
        ctrl_idx=0,            # CUDA 设备 ID
        window_buffer=False,   # 是否启用窗口缓冲
        wb_size=8,             # 窗口缓冲大小
        accumulator_flag=False,# 是否启用访问累加器
        long_type=False,       # 是否使用 int64 类型
        heterograph=False,     # 是否为异构图
        heterograph_map=None   # 异构图节点类型→偏移映射
    ):
```

**初始化流程：**

```
1. 选择数据类型
   long_type? → BAM_Feature_Store.BAM_Feature_Store_long()
              → BAM_Feature_Store.BAM_Feature_Store_float()

2. 创建 GIDS_Controllers
   → init_GIDS_controllers(num_ssd, queueDepth=1024, numQueues=128, ssd_list)

3. 初始化 BAM_Feature_Store
   → init_controllers(controllers, page_size, off, cache_size, num_ele, num_ssd)

4. 配置优化参数
   → window_buffering_flag, accumulator_flag, heterograph, ...
```

**关键计算：**
```python
# 偏移按页大小对齐
self.off = math.ceil(math.ceil(off / page_size) / num_ssd)
```

---

### 2.2 fetch_feature — 核心数据加载

```python
def fetch_feature(self, dim, it, device):
    """
    dim: 特征维度
    it: DGL DataLoader 迭代器
    device: CUDA 设备
    """
    GIDS_time_start = time.time()

    # === 阶段1: 初始化窗口缓冲 ===
    if self.window_buffering_flag:
        if self.wb_init == False:
            self.fill_wb(it, self.wb_size)  # 预填充 wb_size 个 batch
            self.wb_init = True

    # === 阶段2: 获取下一批样本 ===
    next_batch = next(it)  # DGL 图采样
    self.window_buffer.append(next_batch)

    # 对下一批执行窗口缓冲预取
    if self.window_buffering_flag:
        self.window_buffering(next_batch)

    # === 阶段3: 根据 accumulator 模式分流 ===
    if self.accumulator_flag:
        return self._fetch_with_accumulator(dim, it)
    else:
        return self._fetch_direct(dim)
```

**完整数据流：**

```
                 ┌──────────────────────┐
                 │  DGL Sampler (CPU)   │
                 │  图采样 → node IDs   │
                 └──────────┬───────────┘
                            │ next(it)
                 ┌──────────▼───────────┐
                 │  window_buffer 队列  │
                 │  (FIFO, wb_size)     │
                 └──────────┬───────────┘
                            │ pop(0)
                 ┌──────────▼───────────┐
                 │ 模式判断             │
                 │ ┌────────┐┌────────┐│
                 │ │direct  ││accumul.││
                 │ └───┬────┘└───┬────┘│
                 └─────┼─────────┼──────┘
                       │         │
          ┌────────────▼─┐  ┌───▼──────────────┐
          │单 batch 读取  │  │多 batch 合并读取  │
          │read_feature() │  │read_feature_     │
          │               │  │  merged()        │
          └───────────────┘  └──────────────────┘
                       │         │
                 ┌─────▼─────────▼─────┐
                 │ BAM_Feature_Store   │
                 │ GPU Direct SSD Read │
                 └─────────────────────┘
```

---

### 2.3 _fetch_direct — 直接模式

```python
# 同构图直接模式
def _fetch_direct_homo(self, dim):
    batch = self.window_buffer.pop(0)
    index = batch[0].to(self.gids_device)
    index_size = len(index)
    index_ptr = index.data_ptr()

    return_torch = torch.zeros(
        [index_size, dim], dtype=torch.float, 
        device=self.gids_device
    ).contiguous()

    # GPU Direct read
    self.BAM_FS.read_feature(
        return_torch.data_ptr(),  # 输出 tensor
        index_ptr,                # 节点 ID
        index_size,               # 数量
        dim,                      # 特征维度
        self.cache_dim,           # 缓存维度
        0                         # key 偏移
    )

    # 拼接结果: batch + features
    batch.append(return_torch)
    return batch

# 返回值: (input_nodes, seeds, blocks, features)
```

**异构图模式：**
```python
# 异构图：每种节点类型单独读取
for key, v in batch[0].items():
    if len(v) == 0:
        ret_ten[key] = torch.empty((0, dim)).to(self.gids_device)
    else:
        key_off = self.heterograph_map.get(key, 0)
        g_index = v.to(self.gids_device)
        return_torch = torch.zeros([len(g_index), dim], ...)
        # 收集各类型参数
        return_torch_list.append(return_torch.data_ptr())
        index_ptr_list.append(g_index.data_ptr())
        index_size_list.append(len(g_index))
        key_list.append(key_off)
        ret_ten[key] = return_torch

# 多流并发读取
self.BAM_FS.read_feature_hetero(
    num_keys, return_torch_list, index_ptr_list, 
    index_size_list, dim, self.cache_dim, key_list
)
```

---

### 2.4 _fetch_with_accumulator — 累加器模式

**核心逻辑：** 累积足够多的访问请求后，合并为一次批量 SSD 读取

```python
def _fetch_with_accumulator(self, dim, it):
    # 如果之前已累积合并的 batch，直接返回
    if len(self.return_torch_buffer) != 0:
        return_ten = self.return_torch_buffer.pop(0)
        return_batch = self.window_buffer.pop(0)
        return_batch.append(return_ten)
        return return_batch

    buffer_size = len(self.window_buffer)
    current_access = 0
    num_iter = 0

    # 累积直到访问量超过阈值
    while True:
        if num_iter >= buffer_size:
            batch = next(it)
            # 统计访问量
            for k, v in batch[0].items():
                current_access += len(v)
            self.window_buffer.append(batch)
        else:
            batch = self.window_buffer[num_iter]
            for k, v in batch[0].items():
                current_access += len(v)

        num_iter += 1
        required_accesses += self.prev_cpu_access
        if current_access > required_accesses:
            break  # 达到阈值，停止累积

    # 合并读取
    self.BAM_FS.read_feature_merged(
        num_iter, return_torch_list, index_ptr_list, 
        index_size_list, dim, self.cache_dim
    )

    # 更新 CPU 缓冲命中率（用于下次阈值计算）
    cpu_access_count = self.BAM_FS.get_cpu_access_count()
    self.prev_cpu_access = int(cpu_access_count / num_iter)
```

**阈值计算公式：**
```python
def set_required_storage_access(self, bw, l_ssd, l_system, num_ssd, p):
    # bw: SSD 峰值带宽 (GB/s)
    # l_ssd: SSD 延迟 (μs)
    # l_system: 系统延迟 (μs)
    # p: 峰值带宽利用率 (0.95)
    accesses = (p * bw * 1024 / self.page_size * 
                (l_ssd + l_system) * num_ssd) / (1 - p)
    self.required_accesses = accesses
```

---

### 2.5 window_buffering — 窗口预取

```python
def window_buffering(self, batch):
    """对下一批节点提前设置窗口缓冲标记"""
    if self.heterograph:
        for key, value in batch[0].items():
            if len(value) == 0:
                continue
            key_off = self.heterograph_map.get(key, 0)
            input_tensor = value.to(self.gids_device)
            num_pages = len(input_tensor)
            self.BAM_FS.set_window_buffering(
                input_tensor.data_ptr(), num_pages, key_off)
    else:
        input_tensor = batch[0].to(self.gids_device)
        num_pages = len(input_tensor)
        self.BAM_FS.set_window_buffering(
            input_tensor.data_ptr(), num_pages, 0)
```

---

### 2.6 CPU 缓冲管理

```python
def cpu_backing_buffer(self, dim, length):
    """分配 CPU 端的 pinned memory 作为热数据缓冲"""
    self.BAM_FS.cpu_backing_buffer(dim, length)

def set_cpu_buffer(self, ten, N):
    """将 PageRank Top-N 节点加载到 CPU 缓冲"""
    topk_ten = ten[:N]  # 取 Top-N 热节点
    topk_len = len(topk_ten)
    d_ten = topk_ten.to(self.gids_device)
    self.BAM_FS.set_cpu_buffer(d_ten.data_ptr(), topk_len)
```

---

## 3. GIDS_DGLDataLoader 详解

### 3.1 类结构

```python
class GIDS_DGLDataLoader(torch.utils.data.DataLoader):
    """
    自定义 DGL DataLoader，继承 PyTorch DataLoader
    
    图采样 → DGL 原生的 MultiLayerNeighborSampler
    特征加载 → GIDS GPU Direct Storage
    """
    def __init__(self, graph, indices, graph_sampler, 
                 batch_size, dim, GIDS, device=None, **kwargs):
        # 保存 DGL 图相关属性
        self.graph = graph
        self.indices = indices
        self.graph_sampler = graph_sampler
        self.GIDS_Loader = GIDS
        self.dim = dim

        # 图处理优化
        if isinstance(self.graph, DGLHeteroGraph):
            self.graph.create_formats_()   # 创建 CSR/CSC 格式
            if not self.graph._graph.is_pinned():
                self.graph._graph.pin_memory_()  # 锁定到 GPU

        # 使用 CollateWrapper 包装采样函数
        super().__init__(
            self.dataset,
            collate_fn=CollateWrapper(
                self.graph_sampler.sample, graph, self.device),
            batch_size=None,  # batch 已由 CollateWrapper 处理
            pin_memory=False,
            **kwargs
        )
```

### 3.2 CollateWrapper — 批采样包装

```python
class CollateWrapper(object):
    """包装 DGL 的采样函数，处理 batch collation"""
    def __init__(self, sample_func, g, device):
        self.sample_func = sample_func  # DGL sampler.sample
        self.g = g
        self.device = device

    def __call__(self, items):
        # items → GPU device
        items = recursive_apply(items, lambda x: x.to(self.device))
        # 执行图采样
        batch = self.sample_func(self.g, items)
        # 清理 parent storage 引用
        return recursive_apply(batch, remove_parent_storage_columns, self.g)
```

### 3.3 _PrefetchingIter — 预取迭代器

```python
class _PrefetchingIter(object):
    """将 DataLoader 迭代器与 GIDS 特征加载衔接"""
    def __init__(self, dataloader, dataloader_it, GIDS_Loader=None):
        self.dataloader_it = dataloader_it     # DGL 原生迭代器
        self.dataloader = dataloader
        self.GIDS_Loader = GIDS_Loader

    def __next__(self):
        # 委托给 GIDS_Loader.fetch_feature
        batch = self.GIDS_Loader.fetch_feature(
            self.dataloader.dim, 
            self.dataloader_it, 
            self.GIDS_Loader.gids_device
        )
        return batch   # (input_nodes, seeds, blocks, features)
```

---

## 4. 训练流程集成

### 4.1 同构图训练示例

```python
from GIDS import GIDS_DGLDataLoader
import GIDS

# 1. 初始化 GIDS
GIDS_Loader = GIDS.GIDS(
    page_size=args.page_size,
    off=args.offset,
    num_ele=args.num_ele,
    num_ssd=args.num_ssd,
    cache_size=args.cache_size,
    window_buffer=args.window_buffer,
    wb_size=args.wb_size,
    accumulator_flag=args.accumulator,
)

# 2. (可选) 配置 CPU 缓冲
if args.cpu_buffer:
    num_nodes = g.number_of_nodes()
    num_pinned = int(num_nodes * args.cpu_buffer_percent)
    GIDS_Loader.cpu_backing_buffer(dim, num_pinned)
    pr_ten = torch.load(args.pin_file)  # PageRank 结果
    GIDS_Loader.set_cpu_buffer(pr_ten, num_pinned)

# 3. (可选) 配置累加器
if args.accumulator:
    GIDS_Loader.set_required_storage_access(
        args.bw, args.l_ssd, args.l_system, 
        args.num_ssd, args.peak_percent
    )

# 4. 创建 GIDS DataLoader
sampler = dgl.dataloading.MultiLayerNeighborSampler(
    [int(fanout) for fanout in args.fan_out.split(',')]
)

train_dataloader = GIDS_DGLDataLoader(
    g, train_nid, sampler, args.batch_size, 
    dim, GIDS_Loader,
    shuffle=True, drop_last=False, num_workers=args.num_workers
)

# 5. 训练循环
for step, (input_nodes, seeds, blocks, ret) in enumerate(train_dataloader):
    batch_inputs = ret          # GIDS 从 SSD 加载的特征
    batch_labels = blocks[-1].dstdata['labels']

    blocks = [block.to(device) for block in blocks]
    batch_labels = batch_labels.to(device)

    batch_pred = model(blocks, batch_inputs)
    loss = loss_fcn(batch_pred, batch_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 4.2 异构图训练关键差异

```python
# 异构图需要提供节点类型到偏移的映射
GIDS_Loader = GIDS.GIDS(
    heterograph=True,
    heterograph_map={
        'paper': 0,           # paper 节点从偏移 0 开始
        'author': 269346174,  # author 节点从偏移 269346174 开始
        'fos': 546567057,
        'institute': 547280017
    }
)
```

### 4.3 mmap 基线 vs GIDS

| 对比维度 | mmap 基线 | GIDS |
|---------|----------|------|
| 数据传输路径 | SSD → CPU (mmap) → GPU (cudaMemcpy) | SSD → GPU (P2P DMA) |
| CPU 参与度 | 高（CPU 管理页故障） | 低（GPU 直接控制） |
| 带宽瓶颈 | PCIe 双向 + CPU 内存带宽 | PCIe P2P 单向 |
| DataLoader | DGL 原生 DataLoader | GIDS_DGLDataLoader |

---

## 5. 数据集支持

### IGB 数据集
- 节点特征: `node_feat.npy` (float32, shape=[N, 1024])
- 图结构: DGL CSC 格式
- 支持 experimental/small/medium/large/full 五种规模

### OGB 数据集
- 使用 DGL 的 OGB 封装
- 节点数可达 111M+

### 数据写入 SSD
```python
# tensor_write.py: 将 NumPy tensor 写入 SSD
GIDS_Loader = GIDS.GIDS(page_size=4096, off=args.offset, 
                         num_ele=args.num_ele, ...)

emb = np.memmap(path, dtype='float32', mode='r', 
                shape=(num_nodes, emb_size))
GIDS_Loader.store_mmap_tensor(emb, offset)
GIDS_Loader.flush_cache()
```

---

## 6. 模型支持 ([models.py](file:///home/corex/sw_home_1/GIDS_enable/GIDS/evaluation/models.py))

| 模型 | 同构图 | 异构图 | 层类型 |
|------|--------|--------|--------|
| SAGE | ✓ | - | SAGEConv (mean aggregator) |
| GCN | ✓ | - | GraphConv |
| GAT | ✓ | - | GATConv (multi-head) |
| RGCN | - | ✓ | HeteroGraphConv + GraphConv |
| RSAGE | - | ✓ | HeteroGraphConv + SAGEConv |