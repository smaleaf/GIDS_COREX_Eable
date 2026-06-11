# GIDS-IX 完整依赖清单

> **最后更新：** 2026-06-11 | **状态：** ✅ 全部依赖已安装并验证通过

---

## 环境匹配关系

| 组件 | 版本要求 | 实际安装 |
|------|---------|---------|
| 驱动 | Corex 4.5.0 | 4.5.0 ✅ |
| PyTorch wheel | `torch-2.10.0+corex.4.5.0` | 已安装 ✅ |
| Corex SDK 库 | **4.5.0 daily build** | 已全部安装 ✅ |

> ⚠️ PyTorch 4.5.0 wheel 依赖 4.5.0 版本的 SDK 库，版本不匹配会导致 `out of memory` 等虚假错误。

---

## Corex 4.5.0 SDK 包安装清单

以下按 PyTorch `import torch` 报错顺序排列，每次报一个缺失符号对应一个包。

| # | 报错符号 | 对应库 | 包名 | 安装命令 |
|---|---------|--------|------|---------|
| 1 | `ixEventCreateWithFlags` | `libcudart.so` | ixdriver | `repo-manager dl ixdriver -f $URL` |
| 2 | `libcuinfer.so.7` | `libcuinfer.so.7` | ixinfer | `repo-manager dl ixinfer -f $URL` |
| 3 | `ixptiActivityEnable` | `libcupti.so` | ixpti | `repo-manager dl ixpti -f $URL` |
| 4 | `ixblasCreate_v2` | `libcublas.so` | ixblas | `repo-manager dl ixblas -f $URL` |
| 5 | `ixdnnGetRNNLinLayerMatrixParams` | `libcudnn.so` | ixdnn | `repo-manager dl ixdnn -f $URL` |
| 6 | `ixsparseScsrgeam2_bufferSizeExt` | `libcusparse.so` | ixsparse | `repo-manager dl ixsparse -f $URL` |
| 7 | `ixfftDestroy` | `libcufft.so` | ixfft | `repo-manager dl ixfft -f $URL` |
| 8 | `ncclCommWindowDeregister` | `libnccl.so` | ixccl | `repo-manager dl ixccl -f $URL` |
| 9 | `ixAttnBkdFlashAttnBackward` | `libixattnbkd.so` | ixattention | `repo-manager dl ixattention -f $URL` |
| 10 | `ixsolverDn*` | `libcusolver.so` | ixsolver | `repo-manager dl ixsolver -f $URL` |
| — | (额外) | `libixml.so` | ixml | `repo-manager dl ixml -f $URL` |

> URL = `http://sw.iluvatar.ai/download/corex/daily_packages/ivcore11/x86_64/20260524/daily-20260524-ivcore11.yaml`

## 一键安装（Docker 容器中执行）

```bash
cd /home/corex/sw_home_1/sw_home
URL="http://sw.iluvatar.ai/download/corex/daily_packages/ivcore11/x86_64/20260524/daily-20260524-ivcore11.yaml"

for pkg in ixdriver ixinfer ixpti ixblas ixdnn ixsparse ixfft ixccl ixattention ixsolver ixml; do
    echo "=== Installing $pkg ==="
    repo-manager dl $pkg -f "$URL"
done
```

---

## Python 依赖

### 必需包

| 包名 | 版本 | 安装方式 | 备注 |
|------|------|---------|------|
| `torch` | `2.10.0+corex.4.5.0.20260524` | pip (whl) | Iluvatar 定制 PyTorch |
| `dgl` | `2.1.0` | pip | Deep Graph Library |
| `ogb` | `1.3.6` | pip | Open Graph Benchmark（数据集下载） |
| `pybind11` | `2.13.6` | pip | C++/Python 绑定编译 |
| `numpy` | `>=1.14.0` | pip | 数值计算 |

### 版本兼容性注意事项

| 注意项 | 说明 |
|--------|------|
| **torchdata 版本** | DGL 2.1.0 需要 `torchdata>=0.5.0` 但 `torchdata==0.11.0` 不兼容。**必须降级：`pip3 install torchdata==0.7.1`** |
| **Corex SDK 版本** | **必须与 PyTorch wheel 版本匹配**（4.5.0）。驱动版本也必须一致（`ixsmi` 显示 Driver Version: 4.5.0） |
| **设备名称** | Corex PyTorch 使用 `cuda:0`（不是 `ix:0`）。Iluvatar GPU 通过标准 CUDA 设备名访问。 |

### Python 包一键安装

```bash
# PyTorch (Iluvatar 定制版)
pip3 install torch-2.10.0+corex.4.5.0.20260524-cp310-cp310-linux_x86_64.whl

# 基础依赖
pip3 install pybind11 numpy

# 图神经网络框架（注意 torchdata 版本！）
pip3 install torchdata==0.7.1 dgl ogb
```

---

## 安装完成后验证

```bash
source /home/corex/sw_home_1/sw_home/enable
export LD_LIBRARY_PATH="/home/corex/sw_home_1/sw_home/local/corex/lib64:/usr/local/lib64:$LD_LIBRARY_PATH"

# 1. GPU 状态
ixsmi
# 期望: Driver Version: 4.5.0, GPU Memory-Usage 正常

# 2. PyTorch + GPU（设备名是 cuda:0，不是 ix:0）
python3 -c "import torch; t=torch.zeros(3,3,device='cuda:0'); print('✅ PyTorch OK')"

# 3. 编译 GIDS-IX
cd /home/corex/sw_home_1/GIDS_enable/GIDS/gids_module_ix && bash build_ix.sh

# 4. 加载模块
cd /home/corex/sw_home_1/GIDS_enable/GIDS/GIDS_Setup/GIDS
python3 -c "import IXFeatureStore; fs=IXFeatureStore.IXFeatureStore_float(); print('✅ IXFeatureStore OK')"

# 5. 端到端训练（一键脚本）
cd /home/corex/sw_home_1/GIDS_enable/GIDS
bash run.sh all
```

---

## 完整运行时依赖总表

```
Corex 4.5.0 SDK:
  libcudart.so      — CUDA Runtime (ixdriver)
  libcufile.so      — GPU Direct Storage (ixdriver)
  libcuda.so        — CUDA Driver (ixdriver)
  libcupti.so       — CUPTI Profiling (ixpti)
  libcuinfer.so.7   — Inference Engine (ixinfer)
  libcublas.so      — BLAS (ixblas)
  libcudnn.so       — DNN (ixdnn)
  libcusparse.so    — Sparse (ixsparse)
  libcufft.so       — FFT (ixfft)
  libnccl.so        — NCCL通信 (ixccl)
  libixattnbkd.so   — FlashAttention (ixattention)
  libcusolver.so    — Solver (ixsolver)
  libixml.so        — 机器学习 (ixml)

Python:
  torch==2.10.0+corex.4.5.0.20260524
  dgl==2.1.0
  ogb==1.3.6
  torchdata==0.7.1        ← 必须降级！0.11.0 与 DGL 2.1.0 不兼容
  pybind11==2.13.6
  numpy>=1.14.0
  scipy, pandas, tqdm, requests 等 (dgl/ogb 自动安装)

硬件:
  Iluvatar GPU (MR-V100 等)
  NVMe SSD (挂载如 /mnt/nvme0/)

设备名:
  PyTorch 使用 "cuda:0"（不是 "ix:0"）
  Corex PyTorch 将 Iluvatar GPU 注册为 CUDA 设备
```