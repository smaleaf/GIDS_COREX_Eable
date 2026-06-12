# GIDS 移植到 Iluvatar：GDS vs cuFile 方案深度分析

## 1. 背景

GIDS 当前的数据存储访问基于 **BaM 框架**的裸 NVMe 方式：
- GPU kernel 直接通过 MMIO 写入 NVMe 控制器的 Submission Queue
- 需要 `/dev/libnvmX` 用户态 NVMe 驱动
- 使用 BaM 提供的 `bam_ptr<T>` 智能指针和 `page_cache_t` 页缓存

移植到 Iluvatar 有两种技术路线：
- **方案 A（cuFile）：** 用标准 cuFile API (`cuFileRead/Write`) 替代 BaM 的裸 NVMe
- **方案 B（GDS-like）：** 保留 BaM 的架构思想，但移植其 NVMe 用户态驱动层

> **最终选择：方案 A (cuFile)**。代码使用标准 cuFile API 名称，通过 Iluvatar Corex SDK 的 `libcufile.so` 提供底层实现。

---

## 2. 两种方案详细对比

### 2.1 架构对比

```
方案 A: cuFile 方案                      方案 B: GDS-like / BaM 移植
═══════════════════                     ═══════════════════════════

GIDS Python 层                           GIDS Python 层
     │                                        │
     ▼                                        ▼
ixFeatureStore (新)                      BAM_Feature_Store (移植)
     │                                        │
     ▼                                        ▼
ixdrvFileRead/Write                     BaM bam_ptr<T> (移植)
(NVIDIA cuFile 等价)                    BaM page_cache_t (移植)
     │                                        │
     ▼                                        ▼
内核 NVMe 驱动 + 文件系统               移植的 NVMe 用户态驱动
(kernel VFS → nvme.ko)                  (/dev/libnvmX)
     │                                        │
     ▼                                        ▼
NVMe SSD                                NVMe SSD

方案 A GDS 加速路径：
  libcufile.so → ioctl(ITRFS_READ/WRITE) → itrfs.ko → nvme.ko → NVMe SSD
```

### 2.2 核心差异

| 维度 | 方案 A: cuFile | 方案 B: BaM 移植 |
|------|---------------|------------------|
| **存储访问方式** | 文件 I/O（通过 kernel VFS） | 裸 NVMe 命令（用户态直接驱动） |
| **设备路径** | `/mnt/nvme0/features.bin`（普通文件） | `/dev/libnvm0`（字符设备） |
| **GPU 端 API** | `ixdrvFileRead(fh, devPtr, size, offset)` | `bam_ptr[idx]`（智能指针透明访问） |
| **页缓存** | 依赖 kernel page cache | BaM 自定义 GPU 端 page_cache_t |
| **数据粒度** | 字节流（任意 offset/size） | 页粒度（page_size 对齐） |
| **条带化** | 应用层手动实现 | BaM range_t STRIPE 模式内置 |
| **移植工作量** | ~300 行新 C++ 代码 + 适配 | ~5000+ 行移植（BaM 全部代码） |
| **性能** | 稍高延迟（过 kernel），吞吐相当 | 更低延迟（绕过 kernel） |
| **稳定性** | 依赖成熟的 kernel NVMe 驱动 | 用户态驱动需要大量测试 |
| **调试难度** | 低（标准文件 I/O） | 高（NVMe 协议级调试） |
| **Iluvatar 兼容** | ✅ 已验证：`libcufile.so` + `itrfs.ko`（GDS 内核驱动）+ `/dev/itrfs` | ⚠️ 需确认 /dev/libnvmX 设备支持 |

### 2.3 API 对照

```
┌─────────────────────────────────────────────────────────────────┐
│ 当前 GIDS (NVIDIA + BaM)                                        │
├─────────────────────────────────────────────────────────────────┤
│ // 初始化                                                        │
│ Controller ctrl("/dev/libnvm0", ns, gpuid, qd, nq);            │
│ page_cache_t pc(page_size, n_pages, gpuid, ...);               │
│ range_t<float> rng(0, nElem, off, nPages, ..., &pc, STRIPE);   │
│ array_t<float> arr(nElem, 0, {&rng}, gpuid);                   │
│                                                                  │
│ // GPU kernel 中读取                                             │
│ bam_ptr<float> ptr(arr.d_array_ptr);                            │
│ float val = ptr[row * cache_dim + col];  // 透明 page fault    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 方案 A: cuFile                                                  │
├─────────────────────────────────────────────────────────────────┤
│ // 初始化                                                        │
│ cuFileHandle_t fh;                                              │
│ CUfileDescr_t descr = {.type=CU_FILE_HANDLE_TYPE_OPAQUE_FD,     │
│     .handle.fd = open("/mnt/nvme/...")};                        │
│ cuFileHandleRegister(&fh, &descr);                              │
│ cuFileBufRegister(gpu_buffer, buffer_size, 0);                  │
│                                                                  │
│ // 读取（CPU 发起，GPU DMA）                                     │
│ cuFileRead(fh, gpu_dev_ptr, nbytes, file_offset, 0);            │
│                                                                  │
│ // 需要在 CPU 端显式调用，无法在 GPU kernel 中透明访问           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 方案 A 详细设计：cuFile 替代方案

### 3.1 核心思路

**关键洞察：** BaM 的 `bam_ptr` 透明访问在 GIDS 中的实际使用方式并非真正"GPU kernel 内按需触发"，而是 `read_feature_kernel` 中显式调用 `ptr.read(idx)`，本质上等价于一次批量读取。因此可以用 cuFile 的批量 DMA 读取替代。

### 3.2 新架构设计

```
┌──────────────────────────────────────────────────────┐
│                 ixFeatureStore (新类)                 │
│  替代 BAM_Feature_Store + BaM framework              │
├──────────────────────────────────────────────────────┤
│                                                      │
│  数据存储：普通文件                                   │
│    /mnt/nvme0/node_feat.bin（单 SSD）                │
│    /mnt/nvme{0..N}/node_feat_part_{i}.bin（多 SSD）  │
│                                                      │
│  CPU 端接口（pybind11 暴露）：                        │
│    init(num_ssd, file_paths, page_size, cache_size)  │
│    read_feature(tensor_ptr, index_ptr, num_idx,      │
│                 dim, cache_dim, key_off)              │
│    cpu_backing_buffer(dim, len)                      │
│    set_cpu_buffer(idx_ptr, num)                      │
│    set_window_buffering(idx_ptr, num_pages)          │
│    store_tensor(tensor_ptr, num, offset)             │
│    flush_cache()  ← cuFile 不需要，直接返回               │
│                                                      │
│  GPU Kernel：                                        │
│    copy_from_cpu_buffer_kernel (同原版)               │
│    ← read_feature 改为 CPU 发起 cuFile DMA           │
│                                                      │
│  优化策略保留：                                       │
│    ✅ Window Buffering：提前 cuFileBufRegister       │
│    ✅ CPU Buffer：保持 cudaHostAllocMapped 方式      │
│    ✅ Accumulator：合并多次 read 为一次大 read        │
│    ⚠️ 条带化：应用层按 page_size 分发到多个文件       │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 3.3 关键变化

| GIDS 原版 | 移植版 |
|-----------|--------|
| `bam_ptr.read(idx)` in GPU kernel | CPU 端 `cuFileRead()` 批量 DMA |
| `page_cache_t` GPU 端缓存管理 | kernel page cache（OS 管理）或自建 GPU 缓存 |
| `range_t::STRIPE` 硬件条带化 | 应用层 page 分发 |
| `h_pc->flush_cache()` 同步写回 | `fdatasync()` + `ixDeviceSynchronize()` |

### 3.4 数据布局

```
原 BaM 单 SSD:
  /dev/libnvm0  ──  直接裸写入，无文件系统

cuFile 单 SSD:
  /mnt/nvme0/node_feat.bin  ──  ext4/xfs 文件系统上的普通文件
  写入: cuFileWrite(fh, src_ptr, size, offset)
  读取: cuFileRead(fh, dst_ptr, size, offset)

cuFile 多 SSD 条带化:
  /mnt/nvme0/node_feat_part_0.bin  (page 0, 2, 4, ...)
  /mnt/nvme1/node_feat_part_1.bin  (page 1, 3, 5, ...)
  读取时按 page_idx % num_ssd 选择文件
```

### 3.5 性能预期

| 操作 | BaM 裸 NVMe | cuFile | 差异 |
|------|------------|--------|------|
| 单次 4KB 读延迟 | ~11μs | ~15-20μs | cuFile 多 kernel 往返 |
| 大块顺序读带宽 | ~5.8 GB/s | ~5.5 GB/s | 接近（DMA 路径相同） |
| GPU kernel 内访问 | 透明 page fault | 不支持（需 CPU 发起） | 架构差异 |
| 实现复杂度 | 极高 | 低 | cuFile 大幅简化 |

---

## 4. 方案 B 详细设计：BaM 移植

### 4.1 需要移植的组件

```
BaM 框架 (bam git submodule) 需要移植：
├── include/
│   ├── nvm_admin.h      ← NVMe Admin 命令定义
│   ├── nvm_cmd.h         ← NVMe I/O 命令定义
│   ├── nvm_ctrl.h        ← NVMe 控制器抽象
│   ├── nvm_queue.h       ← SQ/CQ 管理
│   ├── nvm_parallel_queue.h
│   ├── nvm_types.h       ← 数据结构
│   ├── nvm_util.h        ← 工具函数
│   ├── buffer.h          ← DMA buffer 管理
│   ├── ctrl.h            ← Controller 类
│   ├── page_cache.h      ← GPU 端页缓存
│   ├── queue.h           ← 队列抽象
│   └── event.h           ← 事件/同步
├── src/
│   └── *.cpp, *.cu       ← 大量实现代码
└── build/
    └── lib/libnvm.so     ← 编译产物
```

### 4.2 移植难点

1. **/dev/libnvmX 设备驱动：** BaM 依赖用户态 NVMe 驱动（可能是 SPDK 或自定义），Iluvatar 环境下是否有等价物未知
2. **GPU MMIO 访问：** BaM 的 `nvm_cmd.h` 中 GPU kernel 直接写 NVMe 寄存器，需要 Iluvatar GPU 支持跨 PCIe 设备的 MMIO 写入
3. **PCIe BAR 映射：** BaM 需要将 NVMe 设备的 PCIe BAR 空间映射到 GPU 地址空间
4. **代码量巨大：** BaM 框架本身就有数千行 C++/CUDA 代码

---

## 5. 推荐方案

### 🏆 推荐方案 A（cuFile）

**理由：**
1. Iluvatar 已完整支持 cuFile API（`ixdrvFileRead/Write/HandleRegister/BufRegister/RDMA`）
2. cuFile 是 NVIDIA 官方推荐的 GPU Direct Storage 接口，Iluvatar 的兼容实现是成熟方案
3. BaM 框架的裸 NVMe 方式虽然延迟更低，但移植风险极高（硬件依赖不确定）
4. GIDS 的三项核心优化（Window Buffering、CPU Buffer、Accumulator）在 cuFile 方案下都可以保留
5. 数据无需重新准备（从裸设备改为文件），可用 `dd` 或 `tensor_write.py` 转换

### 实施步骤

```
Phase 1: 核心存储层 (ixFeatureStore)
├── 创建 ixfeaturestore.cu（替代 gids_nvme.cu + gids_kernel.cu）
├── 实现基于 ixdrvFileRead 的 read_feature
├── 保留 CPU buffer kernel（原样移植，API 兼容）
├── pybind11 绑定（保持与 BAM_Feature_Store 相同的 Python 接口）
└── CMakeLists.txt：ixc 编译器 + Iluvatar 架构目标

Phase 2: Python 层适配
├── GIDS.py：GIDS 类适配（接口不变，底层替换）
├── GIDS_DGLDataLoader：基本不变
└── 训练脚本：s/cuda/ix/ 设备字符串替换

Phase 3: 数据迁移
├── tensor_write.py 适配：写入普通文件而非裸设备
└── 数据转换脚本（裸设备 → 文件）

Phase 4: 编译验证
├── ixdriver SDK 环境配置
├── PyTorch Iluvatar 版本安装
└── 端到端测试
```

---

## 6. 数据迁移说明

如果之前用 BaM 的 `readwrite_stripe` 把数据写入了 `/dev/libnvmX` 裸设备：

```bash
# 从裸设备导出为普通文件
dd if=/dev/libnvm0 of=/mnt/nvme0/node_feat.bin bs=1M status=progress

# 或使用 GIDS 的 tensor_write.py 重新写入
python tensor_write.py --path node_feat.npy --output /mnt/nvme0/node_feat.bin ...
```

迁移后的 cuFile 方案使用普通文件路径，与 BaM 的裸设备路径不兼容。

---

## 7. 总结

| 维度 | 方案 A: cuFile | 方案 B: BaM 移植 |
|------|:---:|:---:|
| API 兼容性 | ✅ 已验证 | ⚠️ 未验证 |
| 移植工作量 | 🟢 低 (~500 行) | 🔴 高 (~5000 行) |
| 性能 | 🟡 接近（差 ~20%） | 🟢 理论上更优 |
| 稳定性 | 🟢 高（kernel 驱动） | 🔴 需大量测试 |
| 可维护性 | 🟢 好 | 🔴 需持续维护移植版 BaM |
| 推荐度 | ⭐⭐⭐⭐⭐ | ⭐⭐ |