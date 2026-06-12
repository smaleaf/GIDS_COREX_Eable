# BaM `bam_ptr` GPU 内存缓存技术原理与接口详解

> 文档版本：v1.0  
> 日期：2026-06-12  
> 来源分析：`/root/GIDS_cufile/bam/include/page_cache.h`、`nvm_types.h`、`nvm_parallel_queue.h`、`buffer.h`

---

## 一、核心思路：GPU DRAM 作为 NVMe 的软件管理页缓存

BaM（Block-Accelerator Memory）的根本思想是：**把 GPU 显存（HBM/GDDR）当作 NVMe SSD 的分页缓存**。

```
┌─────────────────────────────────────────────────────────────────┐
│ GPU 核                                                          │
│  array[i]  →  bam_ptr::operator[]  →  缓存命中？               │
│                     命中 ↓              未命中 ↓               │
│              直接返回 GPU DRAM 地址   ←  发起 NVMe 读命令      │
│                                         填充 GPU DRAM 缓存槽   │
│                                         再返回地址              │
└────────────────┬────────────────────────────────────────────────┘
                 │ GPU 直接发 NVMe 命令（无 CPU 介入）
                 │  ① 写 Submission Queue Entry（GPU DRAM→SQ）
                 │  ② 写 NVMe BAR Doorbell（PCIe Store）
                 │  ③ 等 Completion Queue Entry（轮询 CQ）
                 ▼
┌───────────────────────────────┐
│ NVMe SSD                      │
│  DMA 读→ GPU DRAM 缓存槽      │
└───────────────────────────────┘
```

> **关键**：整条路径不经过 CPU，不经过系统内存，延迟完全由 GPU-NVMe PCIe 路径决定。

---

## 二、核心数据结构

### 2.1 整体层次关系

```
page_cache_t  (Host C++ 对象，管理 GPU DRAM 缓存池)
  └── page_cache_d_t  (GPU 侧描述符，存放在 GPU DRAM 中)
        ├── cache_pages[]    ← 每个缓存槽的锁 + 反向地址翻译
        ├── prp1[]/prp2[]    ← 预计算好的 NVMe DMA 物理地址（GPU pages→NVMe）
        ├── d_ctrls[]        ← NVMe 控制器句柄数组
        ├── ranges[]         ← 指向各 range 的 data_page_t[] 数组
        └── base_addr        ← GPU DRAM 缓存区起始地址

range_t<T>  (Host 对象，描述 NVMe 数据范围→cache 的映射)
  └── range_d_t<T>  (GPU 侧描述符)
        ├── pages[]          ← 每个逻辑页的状态（INVALID/VALID/DIRTY/…）
        ├── index_start/count ← 负责的虚拟元素下标区间
        ├── page_start/count  ← 对应的 NVMe LBA 起始页
        └── cache            ← page_cache_d_t 的内联副本（快速访问）

array_d_t<T>  (GPU 侧数组句柄，包含多个 range)
  └── d_ranges[]            ← range_d_t<T>[] 数组

bam_ptr<T>              (GPU 侧智能指针，无 TLB 版本)
bam_ptr_tlb<T, n=32>   (GPU 侧智能指针，带 32 项 TLB)
```

---

### 2.2 `data_page_t`——逻辑页状态描述符

```cpp
typedef struct __align__(32) {
    simt::atomic<uint32_t, simt::thread_scope_device>  state;  // 页状态 + 引用计数
    uint32_t offset;           // 对应的 cache slot 下标（缓存槽号）
} data_page_t;
```

`state` 字段的位域含义：

```
bit 31   : VALID  (0x80000000) — 页数据已从 NVMe 加载到 GPU DRAM
bit 30   : BUSY   (0x40000000) — 正在加载（I/O in-flight）
bit 29   : DIRTY  (0x20000000) — 页被 GPU 写过，需要回写 NVMe
bit 28-0 : CNT    (0x1fffffff) — 活跃引用计数（有多少 thread 持有此页）
```

状态机：

```
INVALID(0) ─── 首次访问 ──→  BUSY (加载中)
                                  │  NVMe read_data() 完成
                                  ▼
                             VALID (可读写)
                                  │  写操作
                                  ▼
                          VALID | DIRTY (脏页)
                                  │  被驱逐时
                                  ▼
                         写回 NVMe (write_data)
                                  │
                                  ▼
                             INVALID (槽可重用)
```

---

### 2.3 `cache_page_t`——缓存槽管理描述符

```cpp
typedef struct __align__(32) {
    simt::atomic<uint32_t> page_take_lock;  // FREE(2) / LOCKED(1) / UNLOCKED(0)
    uint64_t page_translation;              // 编码了 (page_offset << n_ranges_bits | range_id)
} cache_page_t;
```

物理 cache slot 通过 `page_translation` 字段反向查找该槽当前缓存的是哪个逻辑页，用于驱逐时找到对应的 `data_page_t` 并做回写。

---

### 2.4 `nvm_queue_t`——NVMe I/O 队列对

```cpp
typedef struct {
    simt::atomic<uint32_t, thread_scope_device>  head_lock;  // SQ 头部锁
    simt::atomic<uint32_t, thread_scope_device>  tail_lock;  // SQ 尾部锁
    simt::atomic<uint32_t, thread_scope_device>  head;       // SQ 头指针
    simt::atomic<uint32_t, thread_scope_device>  tail;       // SQ 尾指针
    simt::atomic<uint32_t, thread_scope_system>  tail_copy;  // 系统可见尾副本（用于门铃）
    simt::atomic<uint32_t, thread_scope_system>  head_copy;  // CQ 头部（CPU/设备可见）

    volatile uint32_t*  db;      // ← NVMe BAR 门铃寄存器指针（write-only！）
    volatile void*      vaddr;   // SQ/CQ 在 GPU DRAM 中的虚拟地址
    uint64_t            ioaddr;  // SQ/CQ 的 DMA 物理地址（NVMe 控制器视角）
    uint32_t            qs;      // 队列深度（条目数）
    uint16_t            no;      // 队列编号
    ...
} nvm_queue_t;
```

> **关键成员 `db`**：指向 NVMe BAR0 中门铃寄存器的指针，由 `cudaHostRegister(IoMemory)` 映射。GPU kernel 执行 `*db = new_tail` 即完成门铃写，通知 NVMe 控制器 SQ 有新命令。

---

## 三、页缓存工作原理

### 3.1 缓存初始化（Host 侧）

`page_cache_t` 构造函数完成以下操作：

```
1. cudaMalloc(cache_size = page_size × n_pages)
   → 分配 GPU DRAM 缓存池（base_addr）

2. nv_p2p_get_pages() / itr_p2p_get_dev_pages()（Corex 替换）
   → 将 GPU DRAM 物理页注册为 PCIe DMA 目标
   → 获取 GPU 页的 bus address（NVMe 控制器视角）

3. 预计算 PRP 表（prp1[], prp2[]）：
   if page_size ≤ NVMe_ctrl_page_size:
       prp1[i] = bus_addr(cache_page[i])   // 单 PRP
   elif page_size ≤ 2 × NVMe_ctrl_page_size:
       prp1[i], prp2[i] = 两段 PRP         // 双 PRP
   else:
       prp1[i] = 第一页, prp2[i] → PRP 链表 // PRP list

4. cache_pages[] 初始化为 FREE（所有槽均空闲）

5. cudaMemcpy(d_pc_ptr, &pdt, HostToDevice)
   → 将 page_cache_d_t 描述符上传 GPU DRAM
```

**PRP（Physical Region Page）** 是 NVMe 规范中描述 DMA 内存地址的方式。BaM 预先把 GPU DRAM 的物理地址打包成 NVMe 规范格式，发命令时直接填入，零 CPU 开销。

---

### 3.2 缓存访问（GPU 侧）：命中路径

当 `bam_ptr<T>::operator[](i)` 调用时：

```
1. 检查 i 是否在 [start, end) 范围内（当前持有的 cache page 区间）

2. 命中（i ∈ [start, end)）：
   → 直接返回 addr[i - start]
   → 无任何 atomic 操作，零额外开销
```

在 `array_d_t::seq_read()` / `coalesce_page()` 路径中，命中判断更精细：

```
1. range_d_t::acquire_page(page, count, write=false, ctrl, queue)
2. 读取 pages[page].state
   case V_NB (VALID 且 not BUSY)：
     → hit_cnt++
     → 返回 pages[page].offset（cache slot 号）
     → base_addr + offset × page_size = GPU DRAM 缓存地址
```

---

### 3.3 缓存访问（GPU 侧）：缺失路径

```
case NV_NB (INVALID 且 not BUSY)：
  1. CAS: pages[page].state.fetch_or(BUSY) — 独占加载权
  2. find_slot() — 分配一个空闲/可驱逐的 cache slot（见 3.4）
  3. read_data(pc, qp, lba, n_blocks, page_trans)
       ↳ 构造 NVMe Read 命令（prp1 已指向 GPU DRAM 目标）
       ↳ sq_enqueue() — 写入 Submission Queue
       ↳ *qp->sq.db = new_tail — GPU 直接写 NVMe BAR 门铃 ←核心操作
       ↳ cq_poll() — GPU 自旋等待 Completion Queue 条目
       ↳ cq_dequeue() — 消费完成条目
  4. pages[page].offset = page_trans
  5. state.fetch_xor(BUSY→清除 BUSY, VALID→置位)
  6. 返回 cache_page_addr(page_trans)

case NV_B / V_B (BUSY 中)：
  → __nanosleep(8ns → 256ns 指数退避) 自旋等待
```

整条 miss 路径在 GPU kernel 内以 lockstep 方式完成，**不唤醒 CPU，不切换上下文**。

---

### 3.4 缓存驱逐（`find_slot()`）

使用**轮询式竞争替换**策略（非严格 LRU）：

```
1. page = page_ticket.fetch_add(1) % n_pages   // 原子计数器轮询

2. 对候选 slot 尝试加锁（CAS: UNLOCKED→LOCKED）

3. 检查该 slot 对应的旧逻辑页状态：
   if (refcount == 0 && not BUSY):
     if (DIRTY):
       write_data(...)   // 先回写 NVMe
     pages[old_page].state &= CNT_MASK  // 清除 VALID/DIRTY
     cache_pages[slot].page_translation = new_global_address
     fail = false        // 成功占用
   else:
     解锁，尝试下一个 slot  // 被持有，跳过

4. 解锁 slot（page_take_lock = UNLOCKED）
```

脏页驱逐时，GPU 直接发起 NVMe Write 命令将数据写回 SSD，完全不经 CPU。

---

### 3.5 Warp 合并优化（`coalesce_page()`）

这是 BaM 最关键的性能优化，避免同一 warp 内多个 thread 重复加载同一页：

```cpp
// 找出活跃 warp 中访问同一 page 的所有 lanes
uint32_t eq_mask = __match_any_sync(mask, gaddr);
eq_mask &= __match_any_sync(mask, (uint64_t)this);

// 选 master lane（最低 active bit）
int master = __ffs(eq_mask) - 1;
uint32_t count = __popc(eq_mask);  // 有几个 lane 需要这个页

// 只有 master 执行 acquire_page，引用计数一次加 count
if (master == lane)
    base = r_->acquire_page(page, count, dirty, ctrl, queue);

// 广播 cache 地址给所有 lane
base_master = __shfl_sync(eq_mask, base_master, master);
```

效果：32 个 thread 访问同一页 → 仅 1 次 `acquire_page` + 1 次 NVMe I/O（若缺失）。

---

### 3.6 `flush_cache()`——批量脏页回写

```cpp
void page_cache_t::flush_cache() {
    __flush<<<n_blocks, 64>>>(d_pc_ptr);  // GPU kernel 并行扫描所有 cache slot
}

__global__ void __flush(page_cache_d_t* pc) {
    // 每个 thread 负责一个 cache slot
    uint32_t state = pc->ranges[range][addr].state.load();
    if (state & DIRTY)
        write_data(pc, qp, lba, n_blocks, page);  // NVMe 写回
        pc->ranges[range][addr].state.fetch_and(~DIRTY);
}
```

调用场景：程序退出前、checkpoint 前显式刷新缓存。

---

## 四、`bam_ptr<T>` 智能指针——完整接口说明

### 4.1 数据成员

```cpp
template<typename T>
struct bam_ptr {
    data_page_t*   page     = nullptr;  // 当前持有页的状态描述符
    array_d_t<T>*  array    = nullptr;  // 指向所属 array
    size_t         start    = 0;        // 当前页覆盖的元素下标起始
    size_t         end      = 0;        // 当前页覆盖的元素下标结束（不含）
    int64_t        range_id = -1;       // 当前页所属 range ID
    T*             addr     = nullptr;  // GPU DRAM 缓存地址（直接访问）
};
```

### 4.2 方法接口

| 方法 | 签名 | 说明 |
|------|------|------|
| 构造 | `bam_ptr(array_d_t<T>* a)` | 绑定到 array，不触发 I/O |
| 析构 | `~bam_ptr()` | 调用 `fini()` 释放当前持有页 |
| 只读访问 | `T operator[](size_t i) const` | 若 i 不在当前页，触发 `update_page(i)` |
| 读写访问 | `T& operator[](size_t i)` | 同上，并设置页的 `DIRTY` 标志 |
| 原始指针 | `T* memref(size_t i)` | 返回 i 所在页的 GPU DRAM 起始指针（不计偏移）|
| 显式更新 | `T* update_page(size_t i)` | 先 `fini()` 旧页，再 `acquire_page()` 新页 |
| 显式释放 | `void fini()` | 引用计数 -1，释放对当前页的持有 |

**生命周期示意**（GPU kernel 内）：

```cuda
__global__ void kernel(array_d_t<float>* arr) {
    bam_ptr<float> ptr(arr);          // 构造，start=end=0，不加载
    float v = ptr[1000];              // 首次访问，触发缺失 → NVMe I/O → 命中返回
    ptr[1001] = 3.14f;               // 同页，直接写 GPU DRAM；设 DIRTY
    float v2 = ptr[65536];           // 跨页，fini() 旧页，加载新页
    // 析构时自动 release_page()
}
```

---

## 五、`bam_ptr_tlb<T, n=32>` TLB 加速版本

在 `bam_ptr` 基础上增加了 **线程/Block 级软件 TLB**，大幅减少全局页表查找开销。

### 5.1 TLB 结构

```cpp
template<typename T, size_t n = 32, simt::thread_scope _scope = thread_scope_device>
struct tlb {
    tlb_entry<_scope> entries[n];   // n 个 TLB 条目，默认 32
    array_d_t<T>* array;
};

struct tlb_entry {
    uint64_t global_id;                      // 已缓存的全局页 ID (gid)
    simt::atomic<uint32_t, _scope> state;    // 状态 + 引用计数
    data_page_t* page;                       // 指向对应的 data_page_t
};
```

TLB 查找：**直接映射**（Direct-Mapped），`entry = gid % n`。

### 5.2 TLB 查找流程

```cpp
T* tlb::acquire(i, gid, start, end, range, page_):
  1. ent = gid % n                          // 直接映射
  2. entry = entries[ent]
  3. 使用 __match_any_sync 找到同 warp 内访问相同 gid 的所有 lanes
  4. master lane 执行：
     a. 先取 entry 的 VALID_ 锁（spin acquire）
     b. if (entry->global_id == gid):        // TLB 命中
          state += count; 返回缓存地址
     c. elif (entry->page == null || refcount == 0):  // TLB 缺失
          释放旧 entry; array->acquire_page_() 加载新页
          entry->global_id = gid; entry->page = new_page
     d. else: 自旋等待（__nanosleep 退避）
  5. __shfl_sync 广播地址给其他 lanes
```

### 5.3 TLB 使用接口

```cpp
template<typename T, size_t n = 32, ...>
struct bam_ptr_tlb {
    tlb<T,n>*      tlb_   = nullptr;
    array_d_t<T>*  array  = nullptr;
    range_d_t<T>*  range;
    size_t         page, start, end, gid;
    T*             addr   = nullptr;

    // 构造（在 kernel 内）
    bam_ptr_tlb(array_d_t<T>* a, tlb<T,n>* t);

    // 析构：释放 TLB 引用计数
    ~bam_ptr_tlb();

    // 只读访问
    T operator[](size_t i) const;

    // 读写访问（自动标脏）
    T& operator[](size_t i);

    // 显式更新到新页
    void update_page(size_t i);

    // 释放当前页的 TLB 引用
    void fini();
};
```

**TLB 作用域（`_scope`）**：

| 作用域 | 含义 | 使用场景 |
|--------|------|----------|
| `thread_scope_device` (默认) | TLB 在所有 thread 间共享 | block 内多 thread 共享缓存 |
| `thread_scope_block` | TLB 在 block 内共享 | block 独立缓存窗口 |

---

## 六、`array_d_t<T>` 数组操作接口（GPU 侧）

`bam_ptr` 底层调用的是 `array_d_t<T>`，它支持更底层的操作：

| 方法 | 语义 | 特点 |
|------|------|------|
| `seq_read(i)` | 读元素 i | warp 合并，自动 release |
| `seq_write(i, val)` | 写元素 i，标脏 | warp 合并 |
| `AtomicAdd(i, val)` | 原子加到元素 i | 使用 `atomicAdd` 在 GPU DRAM 中操作 |
| `get_raw(i)` | 返回 `returned_cache_page_t<T>` | 包含指针+大小+偏移，需手动 release |
| `release_raw(i)` | 释放 `get_raw()` 获取的页 | 引用计数 -1 |
| `memcpy(i, count, dest)` | 将元素 i 所在页整体拷贝到 `dest` | 使用 `warp_memcpy<ulonglong4>` 32B/warp |
| `find_range(i)` | 查找元素 i 属于哪个 range | 线性搜索所有 range（range 数量一般很小）|
| `operator[](i)` | 等同 `seq_read(i)` | — |
| `operator()(i, val)` | 等同 `seq_write(i, val)` | — |

`returned_cache_page_t<T>` 结构：

```cpp
template<typename T>
struct returned_cache_page_t {
    T*       addr;    // 页的 GPU DRAM 起始地址
    uint32_t size;    // 页中元素个数 (page_size / sizeof(T))
    uint32_t offset;  // 元素 i 在页内的偏移

    T  operator[](size_t i) const;  // 只读索引
    T& operator[](size_t i);        // 读写索引
};
```

---

## 七、Host 侧 API（构造与配置）

### 7.1 `page_cache_t`

```cpp
page_cache_t(
    uint64_t page_size,          // 缓存页大小（须是 NVMe block size 的整数倍，默认 64KB）
    uint64_t n_pages,            // 缓存页总数（= GPU DRAM 缓存容量 / page_size）
    uint32_t cudaDevice,         // GPU 设备号
    const Controller& ctrl,      // NVMe 控制器（用于确定 block size 等）
    uint64_t max_range,          // 最大 range 数量（必须是 2 的幂）
    const std::vector<Controller*>& ctrls  // 所有 NVMe 控制器列表
);

// 方法
void add_range(range_t<T>* range);    // 注册一个数据范围（自动调用，通常不需手动调用）
void flush_cache();                    // 显式回写所有脏页
void print_reset_stats();              // 打印缓存统计并清零
```

### 7.2 `range_t<T>`

```cpp
range_t<T>(
    uint64_t index_start,    // 虚拟数组元素下标起始（例如 0）
    uint64_t count,          // 元素总数（例如 1 billion）
    uint64_t page_start,     // NVMe 上起始页号
    uint64_t page_count,     // NVMe 页数
    uint64_t page_start_offset, // 第一个元素在首页内的字节偏移
    uint64_t page_size,      // （冗余，取自 page_cache_t）
    page_cache_t* cache,     // 所属 page cache
    uint32_t cudaDevice,
    data_dist_t dist = REPLICATE  // REPLICATE 或 STRIPE
);
```

### 7.3 `array_t<T>`

```cpp
array_t<T>(
    uint64_t num_elems,                        // 总元素数
    uint64_t disk_start_offset,                // NVMe 起始字节偏移
    const std::vector<range_t<T>*>& ranges,    // 所有 range（可多段）
    uint32_t cudaDevice
);

// GPU 侧指针（传入 kernel）
array_d_t<T>* d_array_ptr;
```

### 7.4 典型使用模式

```cpp
/* Host 初始化 */
// 创建 NVMe 控制器和队列对（省略具体步骤）
Controller ctrl(fd, n_qps, cudaDevice);

// 创建 64KB 页、1024 页缓存（= 64MB GPU DRAM 缓存）
page_cache_t cache(64*1024, 1024, cudaDevice, ctrl, 2, {&ctrl});

// 描述 NVMe 上的数据布局：元素 0..N-1 → NVMe page 0..M-1
range_t<float> range(0, N, 0, M, 0, 64*1024, &cache, cudaDevice, STRIPE);

// 创建数组句柄
array_t<float> array(N, 0, {&range}, cudaDevice);

/* 传入 GPU kernel */
my_kernel<<<grid, block>>>(array.d_array_ptr);

/* GPU kernel 内使用 */
__global__ void my_kernel(array_d_t<float>* arr) {
    bam_ptr<float> ptr(arr);
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = ptr[i];            // 透明分页访问
    ptr[i] = v * 2.0f;          // 写操作，自动标脏
}

/* Host 收尾 */
cache.flush_cache();    // 回写所有脏页
cudaDeviceSynchronize();
```

---

## 八、数据分布策略

`data_dist_t` 控制多 NVMe 控制器的数据布局：

| 策略 | 值 | 说明 | 适用场景 |
|------|-----|------|---------|
| `REPLICATE` | 0 | 每个控制器存一份完整数据 | 读密集，高带宽需求 |
| `STRIPE` | 1 | 页按 round-robin 分散到各控制器 | 大数据集，存储均衡 |

分布决策函数（GPU 侧）：

```cpp
// 计算逻辑页 page_offset 对应哪个 NVMe 控制器
uint64_t ctrl_id = get_backing_ctrl_(page_offset, n_ctrls, dist);
// STRIPE:    ctrl_id = page_offset % n_ctrls
// REPLICATE: ctrl_id = ALL_CTRLS (0xffffffff...，全部写/选一个读)

// 计算逻辑页对应的 NVMe LBA
uint64_t lba = get_backing_page_(page_start, page_offset, n_ctrls, dist);
// STRIPE:    lba = page_start + page_offset / n_ctrls
// REPLICATE: lba = page_start + page_offset
```

---

## 九、统计与调优接口

每个 `range_d_t<T>` 内置原子计数器：

| 计数器 | 类型 | 说明 |
|--------|------|------|
| `access_cnt` | `simt::atomic<uint64_t, device>` | 总访问次数 |
| `hit_cnt` | 同上 | 缓存命中次数 |
| `miss_cnt` | 同上 | 缓存缺失次数（触发 NVMe I/O）|
| `read_io_cnt` | 同上 | 实际 NVMe 读命令数 |

读取方式：

```cpp
// 通过 array_t 的统计接口
array.print_reset_stats();
// 输出: #READ IOs, #Accesses, #Misses, Miss Rate, #Hits, Hit Rate, CLSize
```

调优建议：
- **Miss Rate 高** → 增大 `n_pages`（更大 GPU DRAM 缓存）
- **read_io_cnt ≫ miss_cnt** → 页抖动（thrashing），考虑 STRIPE 策略
- **Hit Rate 低于 80%** → 数据访问局部性差，考虑数据预排序

---

## 十、CoreX 适配要点（结合本项目）

| 原 CUDA/NVIDIA 接口 | CoreX 替代 | 影响层 |
|---------------------|-----------|--------|
| `simt::atomic<T, scope>` | `cuda::atomic<T, scope>`（头文件 `cuda/std/atomic`）| 全文件 |
| `nv-p2p.h`: `nvidia_p2p_get_pages()` | `rdma_itr_p2p_get_pages()` / `itr_p2p_get_dev_pages()` | `libnvm.ko` 内核模块 |
| `cudaMalloc` + DMA 注册 | `ixMalloc`（通过 `#define cudaMalloc ixMalloc` 自动映射）| `buffer.h` |
| `cudaHostRegister(IoMemory)` | `ixHostRegister(IoMemory)`（已验证 ✅）| `ctrl.h`（门铃注册）|
| `cudaHostGetDevicePointer` | `ixHostGetDevicePointer`（已验证 ✅）| `ctrl.h` |
| `__threadfence_system()` | `__threadfence()`（ivcore11 backend 限制）| `page_cache.h` SQ doorbell 写后 |
| `__nanosleep(ns)` | 验证中（ivcore 是否支持）| 多处自旋等待 |

---

## 十一、总结：缓存技术全景

```
┌──────────────────────── BaM GPU 页缓存全景 ────────────────────────────┐
│                                                                         │
│  GPU kernel 层:  bam_ptr[i]  →  bam_ptr_tlb[i]  →  array_d_t[i]      │
│                       ↓               ↓                  ↓             │
│  TLB 层:        无 TLB           32-entry              同 array_d_t    │
│                                  直接映射 TLB                           │
│                                       ↓                                │
│  页缓存层:           range_d_t::acquire_page()                         │
│                     ┌───命中────→ 直接返回 GPU DRAM 地址               │
│                     └───缺失───→ find_slot() → read_data()             │
│                                       ↓               ↓                │
│  替换策略:          round-robin 轮询         page_ticket 原子计数器     │
│  脏页回写:          write_data() (驱逐时) + flush_cache() (显式)       │
│                                       ↓                                │
│  NVMe I/O 层:    sq_enqueue → *db = tail → cq_poll → cq_dequeue       │
│                  （GPU 直接操作 NVMe SQ/CQ，门铃写通过 IoMemory 映射） │
│                                       ↓                                │
│  DMA 层:         prp1[]/prp2[] = GPU DRAM bus_addr                     │
│                  → NVMe 控制器直接 DMA 到 GPU DRAM                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

BaM 的创新在于**将传统 OS 页缓存的三层架构（页表、替换算法、I/O 调度）全部搬入 GPU kernel**，用 GPU 的海量并行核心和片上原子操作实现高并发的 GPU-NVMe 直通访问，消除了 CPU、操作系统、PCIe 来回的所有软件开销。
EOF
echo "文档已生成"