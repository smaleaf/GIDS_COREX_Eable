# TODO: DGL GraphBolt 对 Corex PyTorch 的兼容方案

> **状态：** 待实施 | **优先级：** 低（GIDS 当前不依赖 graphbolt） | **创建日期：** 2026-06-11

---

## 1. 背景

DGL 2.x 引入 GraphBolt（高性能 GPU 图采样 C++ 加速库），编译产物为 `libgraphbolt_pytorch_2.X.X.so`。该 .so 文件名硬编码了标准 PyTorch 版本号，Corex 定制 PyTorch（`torch==2.10.0+corex.4.5.0`）无法直接使用 DGL 预编译的 graphbolt wheel。

当前解决方案：降级到 `dgl==1.1.3`（无 graphbolt 依赖，使用纯 Python dataloading）。

---

## 2. 兼容方案

### 方案 A：从 DGL 源码编译 graphbolt（推荐）

#### 原理

GraphBolt 仅使用标准 PyTorch C++ API（Tensor、Stream、CUDA 等），不依赖 NVIDIA 特定符号。Corex cudart 兼容层完全覆盖这些 API。从源码编译时，CMake 检测到 Corex PyTorch，即可生成匹配的 .so。

#### 步骤

```bash
# 1. 克隆 DGL 源码
git clone --recursive https://github.com/dmlc/dgl.git
cd dgl
git checkout v2.1.0  # 或最新稳定版

# 2. 配置编译（关键：确保 CMake 找到 Corex PyTorch）
mkdir build && cd build
cmake .. \
    -DUSE_CUDA=ON \
    -DUSE_GRAPHBOLT=ON \
    -DTORCH_CUDA_ARCH_LIST="7.0" \
    -DCUDA_TOOLKIT_ROOT_DIR=/home/corex/sw_home_1/sw_home/local/corex \
    -DPYTHON_EXECUTABLE=$(which python3)

make -j$(nproc)

# 3. 安装
cd ../python
python setup.py install
```

#### 验证

```python
import dgl
print(dgl.__version__)           # 期望: 2.x
import dgl.graphbolt             # 期望: 不报 FileNotFoundError
```

#### 风险点

| 风险 | 说明 | 缓解 |
|------|------|------|
| Corex CUDA toolkit 路径 | CMake 需正确找到 ixdriver 头文件和 lib | 通过 `CUDA_TOOLKIT_ROOT_DIR` 指定 |
| nvcc → ixc | GraphBolt 源码可能包含 .cu 文件 | 设置 `CMAKE_CUDA_COMPILER=ixc` |
| PyTorch 版本字符串 | .so 文件名含版本号，Corex 版 `2.10.0+corex` 可能与 DGL 期望不一致 | 软链接或修改 CMake 版本匹配逻辑 |

---

### 方案 B：Binary Patch（快捷但不推荐）

对预编译的 `libgraphbolt_pytorch_2.10.0.so`（标准 PyTorch 版）做符号替换：

```bash
# 检查缺失符号
ldd libgraphbolt_pytorch_2.10.0.so 2>&1 | grep "not found"

# 如果有 cuFile/cuBLAS 等缺失，设置 LD_PRELOAD
LD_PRELOAD="/home/corex/sw_home_1/sw_home/local/corex/lib64/libcufile.so" \
    python3 -c "import dgl.graphbolt"
```

**不推荐：** 标准 PyTorch 编译的 graphbolt .so 链接的是 NVIDIA `libcublas.so`/`libcudart.so`，与 Corex 的 `libcudart.so` 符号可能不兼容。Binary patch 不稳定。

---

### 方案 C：提交 PR 到 DGL 上游

向 DGL 提交 PR，添加 Corex PyTorch 后端支持：
- 扩展 `python/setup.py` 中的 PyTorch 版本检测
- 添加 CMake 选项 `-DUSE_COREX=ON`
- 测试 CI 集成（需 Corex 硬件资源）

长期最优，但周期较长。

---

## 3. 当前状态

| 项目 | 状态 |
|------|------|
| GIDS 运行 | ✅ 使用 `dgl==1.1.3`，无需 graphbolt |
| GraphBolt 兼容 | ⏳ 待实施（方案 A） |
| 对 GIDS 影响 | 无 — graphbolt 是可选的加速组件，不影响 GIDS 核心功能 |

---

## 4. 参考

- DGL 源码：https://github.com/dmlc/dgl
- GraphBolt 文档：https://docs.dgl.ai/guide/minibatch-graphbolt.html
- GraphBolt CMake：`dgl/cmake/modules/FindTorch.cmake`
- Corex PyTorch 设备测试结果：
  ```
  cuda available: True
  cuda count: 1
  privateuse1 backend name: 'privateuseone'
  privateuseone:0: No module named 'torch.privateuseone'  (C++ 已注册，Python 未暴露)
  ```