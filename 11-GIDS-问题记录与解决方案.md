# GIDS cuFile 方案 — 问题记录与解决方案

> **最后更新：** 2026-06-11 | 记录从环境搭建到训练跑通全流程遇到的问题及解决方案

---

## 目录

1. [数据集下载与解压](#1-数据集下载与解压)
2. [DGL CUDA 兼容性](#2-dgl-cuda-兼容性)
3. [训练脚本 Bug 修复](#3-训练脚本-bug-修复)
4. [run.sh 参数传递错误](#4-runsh-参数传递错误)
5. [cuFile 驱动降级](#5-cufile-驱动降级)

---

## 1. 数据集下载与解压

### 问题 1.1：OGB 下载慢且交互式确认

**现象：** `bash run.sh train` 执行时提示 `This will download 1.38GB. Will you proceed? (y/N)`，阻塞自动化流程。

**根因：** OGB 库从 Stanford 服务器单线程下载 `products.zip`（~1.38GB），且需要交互式确认。

**解决：**

```bash
# 提前用 aria2c 多线程下载（16 线程）
apt-get install -y aria2
mkdir -p /root/.ogb/dataset/nodeproppred
cd /root/.ogb/dataset/nodeproppred
aria2c -x 16 -s 16 http://snap.stanford.edu/ogb/data/nodeproppred/products.zip
```

---

### 问题 1.2：解压后训练仍提示下载

**现象：** 数据已下载到 `/root/.ogb/dataset/nodeproppred/products.zip`，训练仍提示下载。

**根因：**
1. **目录名不匹配**：解压得到 `products/` 目录，OGB 内部通过 `'_'.join(name.split('-'))` 将 `ogbn-products` 转换为 `ogbn_products`（下划线），期望目录名为 `ogbn_products/`。
2. **`--data_dir` 路径与实际目录不一致**：数据解压到了错误路径。

**OGB 目录检查关键源码**（`ogb/nodeproppred/dataset_dgl.py:L25-L30`）：

```python
self.dir_name = '_'.join(name.split('-'))   # ogbn-products → ogbn_products
if osp.exists(osp.join(root, self.dir_name + '_dgl')):
    self.dir_name = self.dir_name + '_dgl'
self.root = osp.join(root, self.dir_name)   # 最终查找路径
```

**解决：**

```bash
# 方式 A：Python zipfile 解压 + 重命名（推荐，不依赖 unzip 命令）
python3 -c "
import zipfile, os
d = '/path/to/GIDS/data'
with zipfile.ZipFile('/root/.ogb/dataset/nodeproppred/products.zip') as zf:
    zf.extractall(d)
os.rename(os.path.join(d, 'products'), os.path.join(d, 'ogbn_products'))
print('OK:', os.listdir(d))
"

# 方式 B：手动重命名
cd /path/to/GIDS/data
mv products ogbn_products
```

**注意：**
- Docker 容器中 `/root/.ogb` 可能因 overlay 文件系统只读而无法直接解压，建议解压到 `/tmp` 然后复制。
- Docker 中 `unzip` 命令可能未安装，推荐用 Python `zipfile` 模块。

---

### 问题 1.3：unzip 未安装且 sandbox 无 sudo 权限

**现象：** `bash: unzip: command not found`，`apt-get install -y unzip` 权限不足。

**根因：** Docker trae-sandbox 环境限制。

**解决：** 使用 Python 内置 `zipfile` 模块替代 `unzip` 命令。

---

## 2. DGL CUDA 兼容性

### 问题 2.1：`pin_memory_()` 报 CUDA 不支持

**现象：**

```
dgl._ffi.base.DGLError: Device API cuda is not enabled.
Please install the cuda version of dgl.
```

发生在 `GIDS_IX.py:L130` → `self.graph._graph.pin_memory_()`

**根因：** `pip install dgl` 安装的是 CPU 版本（`libdgl.so` 未链接 CUDA 库）。Corex 官方 SDK 仓库（`daily-20260524-ivcore11.yaml`）中**没有 DGL 包**。

**Corex SDK 现有包（不含 DGL）：**

| 类别 | 包 |
|------|---|
| 基础 | ixdriver, ixinfer, ixpti, ixblas, ixdnn, ixsparse, ixfft, ixccl, ixattention, ixsolver, ixml |
| 应用 | torch, diffusers, lmdeploy, mmcv, triton, onnxruntime, vllm, tensorrt-llm, ... |

### 问题 2.2：GPU 驻留图降级方案也失败

**尝试方案：** 修改 `GIDS_IX.py` 捕获 `pin_memory_()` 异常后调用 `self.graph.to(self.device)` 把图搬上 GPU。

**失败原因：** `graph.to(device)` 同样调用 DGL C++ 的 `HeteroGraph::CopyTo()`，需要 CUDA Device API。CPU 版 DGL 任何 GPU 操作（`pin_memory_`、`copy_to`、`NDArray::Empty` 等）都不可用。

**结论：** 必须源码编译 CUDA 版 DGL，链接 Corex 的 `libcudart.so`。

### 问题 2.3：源码编译 DGL — Corex ixc 编译器兼容

**方案：** 源码编译 DGL v1.1.3，使用 Corex `ixc` 编译器 + `libcudart.so`。

**编译环境：**

| 组件 | 路径/版本 |
|------|----------|
| CMake | 3.25.2-corex |
| CUDA Toolkit | `/home/corex/sw_home_1/sw_home/local/corex` (v10.2) |
| CUDA 编译器 | `/home/corex/sw_home_1/sw_home/local/corex/bin/ixc` (clang++ 封装) |
| libcudart | `/home/corex/sw_home_1/sw_home/local/corex/lib64/libcudart.so.10.2` |

**编译步骤：**

```bash
git clone --depth 1 --branch v1.1.3 https://github.com/dmlc/dgl.git /tmp/dgl_corex
cd /tmp/dgl_corex
git submodule update --init --recursive

# 关键 CMake 参数
mkdir build && cd build
cmake .. \
    -DUSE_CUDA=ON \
    -DUSE_OPENMP=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=/home/corex/sw_home_1/sw_home/local/corex \
    -DCUDA_NVCC_EXECUTABLE=/home/corex/sw_home_1/sw_home/local/corex/bin/ixc \
    -DBUILD_CPP_TEST=OFF
```

**遇到的编译问题及修复：**

| 问题 | 现象 | 修复 |
|------|------|------|
| **gpu_cache 编译失败** | `ixc` 不识别 nvcc 风格的 `.cu` 编译 | 见下方 [gpu_cache 模块说明](#gpu_cache-模块说明) |

**gpu_cache 模块说明：**

`gpu_cache` 是 DGL 内置的 **HugeCTR GPU 嵌入缓存**，编译进 `libdgl.so`。它将热门 embedding 缓存在 GPU 显存中，加速查表操作。

- **与 GIDS 的关系：** GIDS 有自己的 `IXFeatureStore`（cuFile 后端），直接从 SSD 读取特征。`gpu_cache` 是 DGL 的独立模块，用于 DGL 原生 embedding 场景。
- **是否保留：** ✅ **保留**。后续可作为 GIDS 的补充——`gpu_cache` 缓存在 GPU 显存（热数据），`IXFeatureStore` 从 SSD 读取（冷数据），形成两级缓存架构，提升整体性能。
- **编译兼容性：** `ixc` 编译器（clang++ 封装）不识别 `nvcc` 风格的 `.cu` 编译命令。修复 `-Xcompiler` 格式后仍遇到 `meta_group_rank` 未定义错误。
- **根因：** `meta_group_rank()` 是 CUDA 11.0+ 的 `cooperative_groups` API，Corex 旧版 SDK 头文件未实现。
- **修复：** ✅ 已在 Corex SDK 源码树 `SWPM-918-gids` 分支修复 —— `include/IX/ixrt/cooperative_groups.h` 中给 `thread_block_tile<Size>` 补了 `meta_group_rank()` 和 `meta_group_size()`。新增验证用例 `apps/cudasamples/cooperativeGroupsMetaGroupTest.cu`。待合入主线后，已安装 SDK 的 `cooperative_groups.h` 需同步更新。
- **新问题（2026-06-11）：** SDK 头文件补丁后，gpu_cache 编译触发了编译器后端错误 `unknown token in expression: mov.u32 v4, %laneid`。这是 Corex LLVM fork 的 llc 不支持 `%laneid` PTX 寄存器导致的，与源码无关（源码全部使用标准 `cooperative_groups` C++ API，无裸 PTX 汇编）。临时方案：禁掉 gpu_cache 模块编译。长期需要 Corex 编译器团队支持 `%laneid`。
- **当前状态：** ✅ SDK 头文件层已修复，❌ 编译器后端 `%laneid` 阻塞，gpu_cache 暂禁用。
- **详细分析：** `/root/GIDS_cufile/docs/01-DGL-Corex兼容分析.md` 和 `/root/GIDS_cufile/docs/02-HugeCTR-gpu_cache-Corex兼容分析.md`
| **`-Xcompiler` 格式不兼容** | `clang++: error: unknown argument: '-fopenmp,-O2,-Wall,...'` | nvcc 用逗号分隔 host flag，ixc 需要空格分隔。修改 `cmake/modules/CUDA.cmake` L239-243。补丁见 `/root/GIDS_cufile/patches/dgl_cuda_cmake_fix.patch` |

**`-Xcompiler` 修复细节：** 原代码 `CUDA.cmake:L239` 把 CXX flags 用逗号拼成 `-Xcompiler -fopenmp,-O2,-Wall,...`，ixc 把整个字符串当单个参数。需改为：

```cmake
# 原代码（cmake/modules/CUDA.cmake:239-243）
string(REGEX REPLACE "[ \t\n\r]" "," CXX_HOST_FLAGS "${CMAKE_CXX_FLAGS}")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "${CXX_HOST_FLAGS}")

# 修复（直接传递 CXX flags，不用 -Xcompiler 包裹）
string(REGEX REPLACE "[ \t\n\r]" " " CXX_HOST_FLAGS "${CMAKE_CXX_FLAGS}")
separate_arguments(CXX_HOST_FLAGS_LIST UNIX_COMMAND "${CXX_HOST_FLAGS}")
list(APPEND CUDA_NVCC_FLAGS ${CXX_HOST_FLAGS_LIST})
```

**注意：** `CMAKE_CUDA_COMPILER` 和 `CMAKE_CUDA_FLAGS` 对 DGL 的 `find_package(CUDA)` 旧式 API 不生效，必须用 `CUDA_NVCC_EXECUTABLE`。

---

## 3. 训练脚本 Bug 修复

### 问题 3.1：`NameError: name 'd' is not defined`

**现象：** `homogenous_train_ix.py:L74` → `split_idx = d.get_idx_split()` 报 `d` 未定义。

**根因：** `load_graph_data()` 内部创建 `d = DglNodePropPredDataset(...)`，但只返回 `(graph, labels)`，未返回 `d`。

**修复：** `homogenous_train_ix.py` 两处修改：

```python
# 修改 1：load_graph_data 增加返回值
def load_graph_data(dataset_name, data_dir):
    ...
    return graph, labels, d       # 原来是 return graph, labels

# 修改 2：train() 接收并使用
graph, labels, dataset = load_graph_data(args.dataset, args.data_dir)  # 原来是 graph, labels
...
split_idx = dataset.get_idx_split()  # 原来是 d.get_idx_split()
```

---

## 4. run.sh 参数传递错误

### 问题 4.1：`--page_size` 不被识别

**现象：** `homogenous_train_ix.py: error: unrecognized arguments: --page_size 4096 True`

**根因：**
1. `run.sh` 传递了 `--page_size 4096`，但 `homogenous_train_ix.py` 的参数解析器未定义 `--page_size`。
2. `--accumulator True` 格式错误，应使用 `--accumulator` flag。

**修复：** 修改 `run.sh`：

```bash
# 移除 --page_size
# 改为条件添加 --accumulator flag
local cmd=(
    python3 evaluation/homogenous_train_ix.py
    ...
    # 不传 --page_size
)
[ "${ACCUMULATOR}" = "True" ] && cmd+=(--accumulator)

# fanout 格式修复
--fanout "${FANOUT},${FANOUT},${FANOUT}"   # 原来是 --fanout ${FANOUT}
```

---

## 5. cuFile 驱动降级

### 问题 5.1：cuFile 驱动初始化失败（非阻塞）

**现象：** `[IXFeatureStore] WARNING: cuFile driver init failed (code=5001), falling back to POSIX read`

**根因：** Corex SDK **已提供** cuFile 内核驱动（`itrfs.ko`），但因 Docker 容器运行环境无法加载内核模块或访问 `/dev/itrfs` 设备节点，导致 cuFile 用户态库初始化失败。

**Corex cuFile 驱动架构（NVIDIA 对标）：**

| 组件 | NVIDIA | Corex（Iluvatar） | 位置 |
|------|--------|-------------------|------|
| 内核驱动 | `nvidia-fs.ko` | `itrfs.ko` | `ixdriver/kmd/itr_fs/` |
| 设备节点 | `/dev/nvidia-fs` | `/dev/itrfs` | 加载内核模块后自动创建 |
| 用户态库 | `libcufile.so` | `libcufile.so` | `ixdriver/cufileapi/` |
| ioctl 接口 | `NVFS_*` | `ITRFS_REG_BUF` / `ITRFS_READ` / `ITRFS_WRITE` | `ixdriver/common/cufile_ioctl.h` |

**关键源码文件：**

| 文件 | 说明 |
|------|------|
| `ixdriver/kmd/itr_fs/itrfs_drv.c` | 内核驱动主入口，注册 NVMe/vmem DMA ops |
| `ixdriver/kmd/itr_fs/itrfs_fops.c` | 文件操作实现（open/close/mmap/ioctl） |
| `ixdriver/kmd/itr_fs/version.json` | 版本描述："Iluvatar corex GDS driver." v4.5.0 |
| `ixdriver/cufileapi/cufile.cpp` | 用户态 cuFile API 封装（`cixdrvFileDriverOpen` 等） |
| `ixdriver/cufileapi/ixnvcufile.h` | 对标 NVIDIA cuFile 的导出 API |
| `ixdriver/common/cufile_ioctl.h` | 内核 ioctl 协议定义 |

**真机启用 GDS 的步骤：**

```bash
# 1. 加载内核模块
modprobe iluvatar                        # 先加载主 GPU 驱动
insmod /path/to/ixdriver/kmd/itr_fs/itrfs.ko   # 加载 GDS 驱动

# 2. 确认模块已加载
lsmod | grep itrfs                       # 应看到 itrfs
dmesg | grep -i itrfs                    # 应看到 "Itrfs is ok."

# 3. 获取主设备号并创建设备节点（若未自动创建）
MAJOR=$(grep itrfs /proc/devices | awk '{print $1}')   # 例如 237
mknod /dev/itrfs c $MAJOR 0
chmod 666 /dev/itrfs
ls -la /dev/itrfs                        # 确认存在

# 4. Docker 容器需要映射设备节点
docker run --device=/dev/itrfs ...
```

**当前环境验证（2026-06-11）：**

| 检查项 | 状态 |
|--------|------|
| `lsmod \| grep itrfs` | ✅ 已加载（主设备号 237） |
| `dmesg \| grep itrfs` | ✅ "Itrfs is ok." |
| `/dev/itrfs` 设备节点 | ✅ 已创建 `crw-rw-rw- 237,0` |

> cuFile GDS 路径已就绪，重新运行训练可验证是否不再降级到 POSIX read。

**影响：** 自动降级为 POSIX `pread()` 读取，功能正常但无 GPU Direct Storage 加速。

**状态：** 非阻塞警告，不影响训练正确性。`itrfs.ko` 已在宿主机加载，只需创建 `/dev/itrfs` 设备节点并映射到容器即可启用 GDS。

---

## 快速排查清单

| 症状 | 检查项 | 解决 |
|------|--------|------|
| `Will you proceed? (y/N)` | 数据集未解压或目录名错误 | 确认 `data/ogbn_products/` 存在且含 `raw/` `processed/` |
| `Device API cuda is not enabled` | DGL 是 CPU 版 | 当前方案：自动降级 GPU 驻留图 |
| `NameError: name 'd' is not defined` | 训练脚本 bug | 见 §3.1 修复 |
| `unrecognized arguments: --page_size` | run.sh 参数错误 | 见 §4.1 修复 |
| `cuFile driver init failed (code=5001)` | Docker 内无法加载 `itrfs.ko` 或 `/dev/itrfs` 未映射 | 正常降级，非阻塞；真机加载 `itrfs.ko` 后可启用 GDS |