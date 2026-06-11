# GIDS 使用指南

> **最后更新：** 2026-06-11 | 支持 NVIDIA CUDA 和 Iluvatar Corex 双平台

---

## 平台选择

| 平台 | 设备名 | 编译 | 存储方案 | 状态 |
|------|--------|------|---------|------|
| NVIDIA CUDA | `cuda:0` | nvcc | BaM 裸 NVMe | 原始版 |
| Iluvatar Corex | `cuda:0` | ixc/g++ | cuFile（标准 API） | GIDS-IX 移植版 |

> **注意：** Iluvatar Corex PyTorch 也使用 `cuda:0` 设备名（不是 `ix:0`）。

---

## 一、Iluvatar Corex 平台（GIDS-IX）

### 1. 环境准备

#### 1.1 硬件要求

| 组件 | 要求 |
|------|------|
| GPU | Iluvatar MR-V100 或更高 |
| 存储 | NVMe SSD × 1+ |
| 内存 | 64 GB+ |

#### 1.2 安装 Corex SDK 包

```bash
source /home/corex/sw_home_1/sw_home/enable
export LD_LIBRARY_PATH="/home/corex/sw_home_1/sw_home/local/corex/lib64:/usr/local/lib64:$LD_LIBRARY_PATH"

cd /home/corex/sw_home_1/sw_home
URL="http://sw.iluvatar.ai/download/corex/daily_packages/ivcore11/x86_64/20260524/daily-20260524-ivcore11.yaml"

for pkg in ixdriver ixinfer ixpti ixblas ixdnn ixsparse ixfft ixccl ixattention ixsolver ixml; do
    repo-manager dl $pkg -f "$URL"
done
```

#### 1.3 安装 Python 包

```bash
# PyTorch (Iluvatar 定制版)
pip3 install torch-2.10.0+corex.4.5.0.20260524-cp310-cp310-linux_x86_64.whl

# ⚠️ torchdata 必须降级！DGL 2.1.0 不兼容 torchdata 0.11.0
pip3 install torchdata==0.7.1 dgl ogb pybind11 numpy
```

#### 1.4 验证环境

```bash
# GPU 状态
ixsmi

# PyTorch GPU 测试（设备名是 cuda:0）
python3 -c "import torch; t=torch.zeros(3,3,device='cuda:0'); print('OK')"
```

---

### 2. 一键使用（推荐）

```bash
source /home/corex/sw_home_1/sw_home/enable
export LD_LIBRARY_PATH="/home/corex/sw_home_1/sw_home/local/corex/lib64:/usr/local/lib64:$LD_LIBRARY_PATH"

cd /home/corex/sw_home_1/GIDS_enable/GIDS
bash run.sh all
```

`run.sh` 命令：

| 命令 | 功能 |
|------|------|
| `bash run.sh all` | 一键构建+数据准备+训练 |
| `bash run.sh setup` | 仅编译 IXFeatureStore 模块 |
| `bash run.sh prepare-data` | 仅下载并准备数据集 |
| `bash run.sh train` | 仅运行训练 |
| `bash run.sh verify` | 仅验证环境 |

环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GPU_IDX` | 0 | GPU 设备索引 |
| `EPOCHS` | 10 | 训练 epoch 数 |
| `BATCH_SIZE` | 1024 | 批次大小 |
| `DATASET` | ogbn-products | 数据集名称 |
| `FEAT_FILE` | /mnt/nvme0/node_feat.bin | 特征文件路径 |

---

### 3. 手动编译

```bash
cd /home/corex/sw_home_1/GIDS_enable/GIDS/gids_module_ix
bash build_ix.sh
```

编译产物：`../GIDS_Setup/GIDS/IXFeatureStore.cpython-310-x86_64-linux-gnu.so`

编译器自动检测：
- **真机模式：** 检测到 `ixc` 编译器 → 编译 `.cu` 文件（含 GPU kernel）
- **Sandbox/Docker 模式：** 未检测到 `ixc` → 使用 g++ 编译 `.cpp` 文件

---

### 3.1 OGB 数据集加速下载

OGB 数据集从 Stanford 服务器下载（~1.38GB），默认单线程很慢。用 `aria2c` 多线程加速：

```bash
apt-get install -y aria2
mkdir -p /root/.ogb/dataset/nodeproppred
cd /root/.ogb/dataset/nodeproppred

aria2c -x 16 -s 16 http://snap.stanford.edu/ogb/data/nodeproppred/products.zip
unzip products.zip
```

OGB 检测到缓存目录已存在则跳过下载。

---

### 4. 数据准备

```bash
# 准备特征文件（写入 NVMe SSD 的文件系统）
python GIDS_Setup/GIDS/tensor_write_ix.py \
    --mode write \
    --input ogbn_products_feat.npy \
    --output /mnt/nvme0/node_feat.bin
```

---

### 5. 训练

#### 同构图训练（ogbn-products）

```bash
python evaluation/homogenous_train_ix.py \
    --dataset ogbn-products \
    --num_ssd 1 \
    --file_paths /mnt/nvme0/node_feat.bin \
    --epochs 10 \
    --batch_size 1024 \
    --fanout 10 \
    --n_layers 3 \
    --n_hidden 256 \
    --accumulator
```

#### 关键参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num_ssd` | SSD 数量 | 1 |
| `--page_size` | 页大小（字节） | 4096 |
| `--cache_size` | GPU 页缓存页数 | 10 |
| `--file_paths` | 特征文件路径（空格分隔多个） | 必填 |
| `--accumulator` | 启用存储访问累加器 | False |
| `--wb_size` | 窗口缓冲大小（批次数） | 40 |
| `--cpu_buffer_size` | CPU buffer 大小（0=禁用） | 0 |
| `--gpu` | GPU 设备索引 | 0 |

#### Python API

```python
from GIDS_IX import GIDS, GIDS_DGLDataLoader

gids = GIDS(
    page_size=4096,
    cache_dim=1024,
    num_ssd=4,
    cache_size=10,
    ctrl_idx=0,
    window_buffer=True,
    wb_size=8,
    accumulator_flag=True,
    file_paths=["/mnt/nvme0/node_feat.bin"],
)

loader = GIDS_DGLDataLoader(
    graph, train_idx,
    MultiLayerNeighborSampler([10, 10, 10]),
    batch_size=1024, dim=100,
    GIDS=gids, device="cuda:0",
)
```

---

### 6. 故障排查

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `ix:0` 报 `RuntimeError` | Corex PyTorch 使用 `cuda:0` | 全部改为 `cuda:0` |
| `torch.AcceleratorError: out of memory` 但 ixsmi 显存充足 | SDK/驱动版本不匹配 | 统一升级到 4.5.0 |
| `cuFile driver init failed (code=5001)` | Docker 无真实 cuFile 驱动 | 正常，自动降级 POSIX |
| `No module named 'torchdata.datapipes'` | torchdata 0.11 不兼容 DGL | `pip3 install torchdata==0.7.1` |
| `undefined symbol: ixHostGetDevicePointer` | C++ name mangling | API 声明放入 `extern "C"` |
| `libcuinfer.so.7: cannot open` | 缺少 ixinfer 包 | `repo-manager dl ixinfer -f $URL` |

---

## 二、NVIDIA CUDA 平台（原始 GIDS）

### 1. 环境准备

#### 1.1 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | NVIDIA GPU, CC ≥ 7.0 | A100 / V100 |
| CPU | x86_64 | 多核服务器 CPU |
| 内存 | 64 GB | 256 GB+ |
| 存储 | NVMe SSD × 1 | NVMe SSD × 4 (条带化) |

#### 1.2 软件依赖

```bash
CUDA Toolkit >= 10.2
Python >= 3.8
cmake >= 3.3.0
g++ (支持 C++11)
torch, dgl, pybind11, numpy
```

---

### 2. 安装

```bash
cd GIDS
git submodule init
git submodule update --recursive

# 编译 BaM
cd bam && mkdir build && cd build
cmake .. && make -j$(nproc)

# 编译 GIDS
cd ../../gids_module && mkdir build && cd build
cmake .. && make -j$(nproc)
```

---

### 3. 数据准备

```bash
# 使用 BaM 的 readwrite_stripe 写入裸 NVMe 设备
cd bam/build
./benchmarks/readwrite_stripe --input /path/to/features.bin --output /dev/libnvm0
```

---

### 4. 训练

```bash
CUDA_VISIBLE_DEVICES=0 python homogenous_train.py \
    --data IGB --dataset_size full \
    --path /mnt/nvme14/IGB260M \
    --model_type sage --epochs 10 --batch_size 1024 \
    --fan_out '10,15,15' --num_layers 3 --emb_size 1024 \
    --GIDS --num_ssd 1 --cache_size $((4*1024)) \
    --num_ele $((550*1000*1000*1024)) --page_size 4096 \
    --window_buffer --wb_size 8 --accumulator \
    --cpu_buffer --cpu_buffer_percent 0.2
```

---

### 5. 关键参数（NVIDIA 原版）

#### 存储相关

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num_ssd` | SSD 设备数量 | 1 |
| `--page_size` | 页大小（字节） | 4096 |
| `--num_ele` | 数据集总元素数 | 根据数据集 |
| `--cache_size` | GPU 页缓存页数 | 4*1024 |

#### 优化参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--window_buffer` | 启用窗口缓冲 | False |
| `--wb_size` | 窗口缓冲大小 | 8 |
| `--accumulator` | 启用存储访问累加器 | False |
| `--cpu_buffer` | 启用 CPU 特征缓冲 | False |
| `--cpu_buffer_percent` | CPU 缓冲节点比例 | 0.2 |
| `--bw` | SSD 峰值带宽 (GB/s) | 5.8 |
| `--l_ssd` | SSD 延迟 (μs) | 11.0 |