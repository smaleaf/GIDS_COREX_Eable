# GIDS-IX 完整依赖清单

> **最后更新：** 2026-06-11 | **状态：** ✅ 全部依赖已安装并验证通过
> 
> **一键安装：** `bash run.sh deps` 自动完成下面所有步骤
> 
> **兼容分析：** `/root/GIDS_cufile/docs/` 下有各开源库的 Corex 兼容分析文档

---

## 环境匹配关系

| 组件 | 版本要求 | 实际安装 |
|------|---------|---------|
| 驱动 | Corex 4.5.0 | 4.5.0 ✅ |
| PyTorch wheel | `torch-2.10.0+corex.4.5.0` | 已安装 ✅ |
| Corex SDK 库 | **4.5.0 daily build** | 已全部安装 ✅ |
| DGL | 1.1.3 **CUDA 版** (源码编译) | 需编译 🔧 |
| cooperative_groups.h | 含 `meta_group_rank()` | SWPM-918-gids 分支 ✅ |

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
| `dgl` | `1.1.3` | **源码编译** | 必须 CUDA 版，链接 Corex libcudart.so |
| `ogb` | `1.3.6` | pip | Open Graph Benchmark（数据集下载） |
| `pybind11` | `2.13.6` | pip | C++/Python 绑定编译 |
| `numpy` | `>=1.14.0` | pip | 数值计算 |

### 版本兼容性注意事项

| 注意项 | 说明 |
|--------|------|
| **torchdata 版本** | DGL 2.1.0 需要 `torchdata>=0.5.0` 但 `torchdata==0.11.0` 不兼容。**必须降级：`pip3 install torchdata==0.7.1`** |
| **DGL 必须 CUDA 版** | `pip install dgl` 装的是 CPU 版，`pin_memory_()` 和 `graph.to(device)` 都不可用。必须源码编译，见下方 |
| **Corex SDK 版本** | **必须与 PyTorch wheel 版本匹配**（4.5.0）。驱动版本也必须一致（`ixsmi` 显示 Driver Version: 4.5.0） |
| **设备名称** | Corex PyTorch 使用 `cuda:0`（不是 `ix:0`）。Iluvatar GPU 通过标准 CUDA 设备名访问。 |

### Python 包一键安装

```bash
# PyTorch (Iluvatar 定制版)
pip3 install torch-2.10.0+corex.4.5.0.20260524-cp310-cp310-linux_x86_64.whl

# 基础依赖
pip3 install pybind11 numpy

# 图神经网络框架（注意 torchdata 版本！）
pip3 install torchdata==0.7.1 ogb
# DGL 不通过 pip 安装，需源码编译（见下方）
```

---

## DGL CUDA 版源码编译 (Corex)

### 为什么必须编译

`pip install dgl` 安装的是 CPU 版 `libdgl.so`，不包含 CUDA Device API。GIDS 的 `GIDS_DGLDataLoader` 需要 `pin_memory_()` 和 UVA 采样，必须使用 CUDA 版 DGL。

### 前置条件

编译前需确保 Corex SDK 的 `cooperative_groups.h` 包含 `meta_group_rank()` 和 `meta_group_size()`：

```bash
# 检查是否已修复
grep "meta_group_rank" /home/corex/sw_home_1/sw_home/local/corex/include/cooperative_groups.h
# 若无输出 → 需部署 SWPM-918-gids 分支的补丁
```

### 编译步骤

```bash
COREX_DIR=/home/corex/sw_home_1/sw_home/local/corex
DGL_VER=1.1.3

git clone --depth 1 --branch v${DGL_VER} https://github.com/dmlc/dgl.git /tmp/dgl_corex
cd /tmp/dgl_corex
git submodule update --init --recursive

# 修复 -Xcompiler 格式（ixc 不认 nvcc 的逗号分隔）
sed -i 's/string(REGEX REPLACE "\[ \\t\\n\\r\]" "," CXX_HOST_FLAGS "${CMAKE_CXX_FLAGS}")/string(REGEX REPLACE "[ \\t\\n\\r]" " " CXX_HOST_FLAGS "${CMAKE_CXX_FLAGS}")\n  separate_arguments(CXX_HOST_FLAGS_LIST UNIX_COMMAND "${CXX_HOST_FLAGS}")\n  list(APPEND CUDA_NVCC_FLAGS ${CXX_HOST_FLAGS_LIST})/' cmake/modules/CUDA.cmake
sed -i '/list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "${CXX_HOST_FLAGS}")/d' cmake/modules/CUDA.cmake

mkdir build && cd build
cmake .. \
    -DUSE_CUDA=ON \
    -DUSE_OPENMP=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=${COREX_DIR} \
    -DCUDA_NVCC_EXECUTABLE=${COREX_DIR}/bin/ixc \
    -DBUILD_CPP_TEST=OFF

make -j$(nproc)

cd ../python
python3 setup.py install --user
```

### 验证

```bash
python3 -c "
import dgl
g = dgl.rand_graph(10, 20)
g = g.to('cuda:0')
print('DGL', dgl.__version__, 'CUDA OK')
"
```

---

## 安装完成后验证

```bash
source /home/corex/sw_home_1/sw_home/enable
export LD_LIBRARY_PATH="/home/corex/sw_home_1/sw_home/local/corex/lib64:/usr/local/lib64:$LD_LIBRARY_PATH"

# 1. GPU 状态
ixsmi
# 期望: Driver Version: 4.5.0, GPU Memory-Usage 正常

# 2. cuFile GDS 驱动（内核模块 + 设备节点）
lsmod | grep itrfs                    # 期望: itrfs
dmesg | grep -i itrfs | tail -1      # 期望: "Itrfs is ok."
ls -la /dev/itrfs                     # 期望: crw-rw-rw- 237,0
# 若无 /dev/itrfs: mknod /dev/itrfs c $(grep itrfs /proc/devices | awk '{print $1}') 0 && chmod 666 /dev/itrfs

# 3. PyTorch + GPU（设备名是 cuda:0，不是 ix:0）
python3 -c "import torch; t=torch.zeros(3,3,device='cuda:0'); print('✅ PyTorch OK')"

# 4. DGL CUDA 版
python3 -c "import dgl; g=dgl.rand_graph(10,20).to('cuda:0'); print('✅ DGL CUDA OK')"

# 5. 编译 GIDS-IX
cd /home/corex/sw_home_1/GIDS_enable/GIDS/gids_module_ix && bash build_ix.sh

# 6. 加载模块
cd /home/corex/sw_home_1/GIDS_enable/GIDS/GIDS_Setup/GIDS
python3 -c "import IXFeatureStore; fs=IXFeatureStore.IXFeatureStore_float(); print('✅ IXFeatureStore OK')"

# 7. 端到端训练（一键脚本）
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

Corex cuFile 内核驱动 (GDS 加速必需):
  itrfs.ko          — Iluvatar corex GDS driver (v4.5.0)
                      ↳ 位于 ixdriver/kmd/itr_fs/
                      ↳ 等价于 NVIDIA nvidia-fs.ko
  /dev/itrfs        — cuFile 设备节点
                      ↳ 主设备号从 /proc/devices 获取
                      ↳ 需在宿主机创建并映射到容器 (--device=/dev/itrfs)

Corex 头文件补丁 (DGL gpu_cache 编译必需):
  cooperative_groups.h — 需含 meta_group_rank() / meta_group_size()
                      ↳ 来自 SWPM-918-gids 分支
                      ↳ 路径: include/IX/ixrt/cooperative_groups.h

Python:
  torch==2.10.0+corex.4.5.0.20260524
  dgl==1.1.3 (源码编译, CUDA 版)  ← 不能 pip install dgl！
  ogb==1.3.6
  torchdata==0.7.1                ← 必须降级！0.11.0 与 DGL 2.1.0 不兼容
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