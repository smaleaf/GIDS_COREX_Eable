# Corex Enable BAM — 文档总目录

> **目标：** 在 Iluvatar Corex IX GPU 上原生启用 BaM（GPU-Initiated NVMe）框架，  
> 使 GIDS GNN Dataloader 的 `bam_ptr<T>` 直接在 Corex 上运行，保留 GPU 自主发起 NVMe I/O 的全部性能优势。

**最后更新：** 2026-06-12  
**KMD 来源：** `/home/corex/sw_home_1/sw_home/sdk/ixdriver/kmd/`

---

## 核心结论（先读这里）

| 项目 | 结论 |
|---|---|
| **IX GPU 是否支持 GPU 显存 → NVMe DMA？** | ✅ 已生产验证（`itrfs.ko` GDS 驱动每次都在做） |
| **`nvidia_p2p_*` 是否有 Corex 等价物？** | ✅ 完整 1:1 对应（`rdma_itr_p2p_*`，EXPORT_SYMBOL） |
| **PCIe peer write / `ixHostRegisterIoMemory`？** | ⚠️ API 已声明（`0x04`），运行时需 `test_io_memory.cu` 验证 |
| **整体可行性** | ✅ **高**，6 大适配项仅 1 项需运行时测试，其余路径全部清晰 |
| **总工期** | 约 4~5 周 |

---

## 目录结构

```
corex_enable_bam/
│
├── README.md                              ← 本文件（总入口）
│
├── docs/                                  ← 技术分析文档
│   │
│   ├── 【重要】01-KMD源码分析报告.md      ★ Corex KMD P2P API 完整分析
│   ├── 【重要】02-bam_ptr适配完整方案.md  ★ 6大适配项逐项方案（v2.0）
│   │
│   ├── GIDS-架构总览.md                   GIDS 整体架构描述
│   ├── GIDS-与Iluvatar-GPU兼容性分析.md   早期兼容性分析
│   ├── GIDS-cuFile适配Corex方案总结.md    cuFile 路线参考（备选方案）
│   ├── GIDS-IX-依赖清单.md               依赖项完整清单
│   ├── GIDS-移植报告-端到端.md            端到端移植情况
│   ├── GIDS-移植方案分析-cuFile-vs-GDS.md 两种路线对比分析
│   ├── GIDS-源码分析-CUDA核心模块.md      BaM CUDA 核心源码分析
│   ├── GIDS-源码分析-Python接口.md        Python/pybind11 层分析
│   ├── GIDS-问题记录与解决方案.md         已知问题和解决方案
│   └── GIDS-使用指南.md                   GIDS 使用方法
│
├── code/                                  ← 可用代码
│   ├── libnvm_corex_map.c                ★ libnvm.ko GPU DMA 适配实现
│   │   （直接替换 BaM module/map.c 的 #ifdef _CUDA 部分）
│   └── test_io_memory.cu                 ★ ixHostRegisterIoMemory 验证程序
│       （D1-D3 必须运行，决定门铃写入路径）
│
└── ppt/                                   ← 汇报 PPT
    ├── bam_ptr-Corex适配-v2-KMD分析更新.pptx  ★ 最新（7页，含KMD结论）
    ├── GIDS-bam_ptr原生适配Corex-完整分析PPT.pptx  完整技术分析（17页）
    ├── GIDS-Corex适配方案决策-汇报PPT.pptx         领导决策用（16页）
    └── GIDS-架构分析与Corex适配-PPT.pptx           架构全景（28页）
```

---

## 6 大适配项快速参考

| # | 适配内容 | 状态 | 工期 | 关键证据 |
|---|---|---|---|---|
| ① | `simt::atomic` → `cuda::atomic` | ✅ 可行 | 1天 | `cuda/std/atomic` 已验证 |
| ② | `<<<>>>` 语法 + CUDA Runtime API | ✅ 可行 | 0 | `ixc` 编译器原生支持 |
| ③ | `cuda_err_chk` 宏重定义 | ✅ 可行 | 0.5天 | 宏映射 |
| ④ | `cudaHostRegisterIoMemory`（NVMe 门铃）| ⚠️ 需测试 | 3天 | API 声明存在，`test_io_memory.cu` |
| ⑤ | `libnvm.ko` Part A+B（纯 Linux PCI）| ✅ 可行 | 2天 | 无 NVIDIA 依赖 |
| ⑥ | `libnvm.ko` GPU DMA（替换 `nv-p2p.h`）| ⚠️ 路径清晰 | 1周 | `rdma_itr_p2p_*`，`itrfs.ko` 已验证 |

---

## Corex P2P API 对照表（KMD 分析结论）

| BaM 使用（NVIDIA `nv-p2p.h`）| Corex 等价（EXPORT_SYMBOL）| 实现文件 |
|---|---|---|
| `nvidia_p2p_get_pages()` | `rdma_itr_p2p_get_pages()` | `kmd/itr_peer_mem/itr_peer_mem_user.c` |
| `nvidia_p2p_dma_map_pages()` | `rdma_itr_p2p_dma_map_pages()` | 同上 |
| `nvidia_p2p_dma_unmap_pages()` | `rdma_itr_p2p_dma_unmap_pages()` | 同上 |
| `nvidia_p2p_put_pages()` | `rdma_itr_p2p_put_pages()` | 同上 |
| 底层实现 | `itr_lib_p2p_get_dev_pages()` | `kmd/itr/kmdlib/itr_lib_peer_mem.c` |

**`itrfs.ko`（GDS 驱动）在生产中使用上述 API，证明 IX GPU 显存 → NVMe DMA 链路已实际运行。**

---

## 立即行动（D1~D3）

```bash
# Step 1：编译并运行 IoMemory 验证测试（最高优先级）
cd /root/GIDS_cufile/corex_enable_bam/code
ixc -o test_io_memory test_io_memory.cu -lcuda
./test_io_memory /dev/libnvm0 0
# 通过 → 继续 libnvm_corex.ko 实现
# 失败 → 改用 itr_xfer_dma_map_resource 门铃备用路径

# Step 2：基础适配（当日完成）
find /root/GIDS_cufile/bam/include -name "*.h" | xargs \
  sed -i 's|simt::|cuda::|g; s|<simt/atomic>|<cuda/std/atomic>|g'

# Step 3：参考 KMD 工程示例
# GPU VA → PCIe DMA 地址：
/home/corex/sw_home_1/sw_home/sdk/ixdriver/kmd/itr_peer_mem/fpga_test/cuda_va2pa_test.c
# GDS 端到端测试：
/home/corex/sw_home_1/sw_home/sdk/ixdriver/kmd/itr_fs/test/itr_gds_test.c
```

---

## PPT 选择指南

| 场景 | 推荐 PPT |
|---|---|
| 向领导汇报决策 | `GIDS-Corex适配方案决策-汇报PPT.pptx` |
| 技术评审（含KMD分析）| `bam_ptr-Corex适配-v2-KMD分析更新.pptx` |
| 完整技术深潜 | `GIDS-bam_ptr原生适配Corex-完整分析PPT.pptx` |
| 整体架构介绍 | `GIDS-架构分析与Corex适配-PPT.pptx` |
