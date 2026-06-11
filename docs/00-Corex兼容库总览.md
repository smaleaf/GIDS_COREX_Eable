# GIDS Corex 兼容库总览

> **目录：** `/root/GIDS_cufile/`
> **最后更新：** 2026-06-11

---

## 目录结构

```
/root/GIDS_cufile/
├── dgl/                           # DGL v1.1.3 (Deep Graph Library)
├── HugeCTR/                       # NVIDIA-Merlin/HugeCTR (master)
├── docs/                          # 兼容分析文档
│   ├── 00-Corex兼容库总览.md       # 本文
│   ├── 01-DGL-Corex兼容分析.md     # DGL 编译兼容
│   ├── 02-HugeCTR-gpu_cache-Corex兼容分析.md  # GPU Cache 编译器兼容
│   ├── 03-gpu_cache-PTX指令清单与Corex兼容状态.md  # PTX 指令级分析
│   └── 04-Corex-ivcorex内建指令与copylib用法分析.md  # Corex builtin 指令分析
└── patches/                       # 补丁文件
```

---

## 依赖关系图

```
GIDS (cuFile方案)
├── PyTorch (torch-2.10.0+corex.4.5.0)  ✅ 已适配
├── DGL v1.1.3                             🔧 需源码编译
│   ├── libdgl.so (主库)                   ✅ cmake + make
│   ├── CUDA.cmake (-Xcompiler 格式)        ✅ 已修复
│   └── cooperative_groups (gpu_cache)      🔧 依赖 Corex 编译器
├── HugeCTR gpu_cache                      ❌ 编译器后端阻塞
│   └── cooperative_groups 生成 %laneid     ❌ 待 Corex llc 支持
└── IXFeatureStore (cuFile → SSD)          ✅ 编译通过
```

---

## 兼容状态矩阵

| 库 | 许可证 | 版本 | 兼容状态 | 阻塞点 | 优先级 |
|----|--------|------|----------|--------|--------|
| DGL | Apache 2.0 | v1.1.3 | 🟡 主库可编译，gpu_cache 禁 | `-Xcompiler` 格式已修，`cooperative_groups` 头文件已补 | P0 |
| HugeCTR gpu_cache | Apache 2.0 | (DGL内置) | 🔴 编译器后端不支持 | Corex llc 不支持 `%laneid` PTX 寄存器 | P1 |
| HugeCTR (完整) | Apache 2.0 | master | ⬜ 未评估 | 独立框架，非 GIDS 当前依赖 | P2 |

---

## 问题层级分类

| 层级 | 问题 | 修复位置 | 状态 |
|------|------|----------|------|
| **头文件 API** | `meta_group_rank()` 缺失 | `cooperative_groups.h` (C++) | ✅ 已修复 (SWPM-918-gids) |
| **编译参数** | `-Xcompiler` 逗号分隔不兼容 | `CUDA.cmake` (cmake) | ✅ 已修复 |
| **编译器后端** | `%laneid` PTX 寄存器不支持 | Corex LLVM fork (llc) | ❌ 待 Corex 团队 |
| **编译器后端** | `%warpid` 等其他 warp 寄存器 | Corex LLVM fork (llc) | ⬜ 未确认 |

---

## 短期策略（当前）

1. **DGL 主库**：禁 `gpu_cache`，`make -j$(nproc)` 编译通过 → `setup.py install`
2. **GPU 缓存**：GIDS 使用自己的 IXFeatureStore (cuFile → SSD)，不依赖 HugeCTR gpu_cache
3. **两级缓存优化**：待 Corex 编译器后端支持后启用

## 长期策略

1. Corex 编译器团队补充 `%laneid` / `%warpid` PTX 寄存器映射
2. 启用 DGL gpu_cache 编译
3. 实现 GIDS 两级缓存 (GPU HBM + NVMe SSD)