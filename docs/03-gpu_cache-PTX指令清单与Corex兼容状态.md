# DGL gpu_cache 涉及的 PTX 指令与 Corex 兼容状态

> **源码：** `/root/GIDS_cufile/dgl/third_party/HugeCTR/gpu_cache/`
> **编译器：** Corex clang++ (ixc), 基于 LLVM fork
> **测试日期：** 2026-06-11
> **测试方法：** 逐条编写 CUDA kernel，使用 `ixc -x ivcore --cuda-gpu-arch=ivcore11 -c` 编译到目标文件
> **上游参考：** [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

---

## 1. 总览

gpu_cache 源码 **没有手写 PTX 内联汇编**。所有 PTX 指令均由编译器后端（llc/lowering）从标准 CUDA C++ API 自动生成。

| 类别 | 数量 | Corex 状态 |
|------|------|-----------|
| 特殊寄存器 (Special Registers) | 9 个 | `%laneid` ❌，其余 ✅ |
| Warp 级操作 (Warp-level) | 4 条 | ⚠️ 裸 API ✅，CG 封装 ❌ |
| 同步/内存屏障 (Sync/Fence) | 4 条 | ✅ 全部通过 |
| 原子操作 (Atomic) | 6 条 | ✅ 全部通过 |
| 标准指令 (Standard) | ~20 条 | ✅ 全部通过 |

**关键发现：** `__shfl_down_sync`、`__ballot_sync`、`__any_sync`、`__all_sync` 作为裸 API 调用时编译通过。只有 `cooperative_groups::tiled_partition<>` 的 C++ 封装触发 `%laneid` 错误。Corex 的 shfl/vote 底层支持已经存在，问题仅在于 CG 包装层。

---

## 2. 特殊寄存器 (Special Registers)

### 2.1 `%laneid` ❌ 唯一阻塞

```
mov.u32 v0, %laneid;
```

| 属性 | 值 |
|------|-----|
| **含义** | 线程在其 warp 内的位置索引 (lane index)，范围 0-31 |
| **触发来源** | `cooperative_groups::tiled_partition<Size>(block)` — 仅此 API |
| **对应 CUDA C++** | `threadIdx.x % warpSize` |
| **Corex 状态** | ❌ `unknown token in expression` |
| **影响范围** | 所有使用 `tiled_partition` 的 kernel |
| **等效替代** | `threadIdx.x % 32` (需编译器端实现) |

### 2.2 `%tid` ✅ 通过

| 属性 | 值 |
|------|-----|
| **含义** | 线程在 block 内的线性索引 (= threadIdx.x) |
| **触发来源** | `threadIdx.x` / `cooperative_groups::this_thread_block().thread_rank()` |
| **测试** | `test_cg_basic` — `block.thread_rank()` 编译通过 |

### 2.3 `%ntid` ✅ 通过

| 属性 | 值 |
|------|-----|
| **含义** | Block 内线程总数 (= blockDim.x) |
| **触发来源** | `blockDim.x` / `block.size()` |
| **测试** | `test_special_regs` — `blockDim.x` 编译通过 |

### 2.4 `%ctaid` ✅ 通过

| 属性 | 值 |
|------|-----|
| **含义** | 当前 block 在 grid 中的索引 (= blockIdx.x) |
| **触发来源** | `blockIdx.x` |
| **测试** | `test_special_regs` — `blockIdx.x` 编译通过 |

### 2.5 `%nctaid` ✅ 通过

| 属性 | 值 |
|------|-----|
| **含义** | Grid 中 block 总数 (= gridDim.x) |
| **触发来源** | `gridDim.x` |
| **测试** | `test_special_regs` — `gridDim.x` 编译通过 |

### 2.6 `%clock` ✅ 通过

| 属性 | 值 |
|------|-----|
| **含义** | GPU 时钟周期计数器 |
| **触发来源** | `clock64()` |
| **测试** | `test_special_regs` — `clock64()` 编译通过 |

### 2.7 `%smid` / `%nsmid` ⬜ 未测试

| 属性 | 值 |
|------|-----|
| **含义** | SM 硬件 ID / SM 总数 |
| **Corex 状态** | 非 gpu_cache 所需，未测试 |

### 2.8 `%warpid` ⬜ 被 `%laneid` 阻塞

| 属性 | 值 |
|------|-----|
| **含义** | 当前 warp 在 block 内的索引 |
| **触发来源** | `tiled_partition` → `meta_group_rank()` |
| **Corex 状态** | 被 `%laneid` 阻塞，无法独立测试。修复 `%laneid` 后大概率通过（`meta_group_rank()` 已修复为 `threadIdx.x/32`） |

---

## 3. Warp 级操作 (Warp-Level Instructions)

### 3.1 核心发现：裸 API 通过，CG 封装失败

| 测试 | 调用方式 | 结果 |
|------|---------|------|
| `__shfl_down_sync(0xffffffff, val, 1)` | 裸 API | ✅ **PASS** |
| `__shfl_sync(0xffffffff, val, 0)` | 裸 API | ✅ **PASS** |
| `__shfl_xor_sync(0xffffffff, val, 1)` | 裸 API | ✅ **PASS** |
| `tile.shfl_down(val, 1)` | CG 封装 | ❌ 触发 `%laneid` |

**结论：** Corex 编译器已支持 `shfl.sync` 指令！`__shfl_*_sync` 系列裸 API 全部编译通过，生成的 PTX 被 llc 正确翻译。问题不在 shfl 指令本身，而在 `cooperative_groups::tiled_partition<>` 的构造函数中使用了 `%laneid` 来初始化 tile 内部数据。

### 3.2 `vote.sync` ✅ 裸 API 通过

| 测试 | 调用方式 | 结果 |
|------|---------|------|
| `__ballot_sync(0xffffffff, pred)` | 裸 API | ✅ **PASS** |
| `__any_sync(0xffffffff, pred)` | 裸 API | ✅ **PASS** |
| `__all_sync(0xffffffff, pred)` | 裸 API | ✅ **PASS** |

### 3.3 `match.sync` ⬜ 未测试（SM 7.0+）

| 属性 | 值 |
|------|-----|
| **含义** | Warp 内值匹配 |
| **Corex 状态** | Corex CUDA 10.2 兼容，SM 7.0+ 的 `__match_any_sync` 不可用（`__CUDA_ARCH__ < 700`），gpu_cache 源码未使用 |

### 3.4 `bar.warp.sync` ⬜ 被 `%laneid` 阻塞

| 触发 API | 作用 |
|----------|------|
| `__syncwarp(mask)` / `tile.sync()` | Warp 内收敛同步 |

**Corex 状态：** 被 `%laneid` 阻塞。修复 `%laneid` 后大概率通过。

---

## 4. 同步与内存屏障 ✅ 全部通过

| 测试 | 触发 API | 结果 |
|------|---------|------|
| `bar.sync` | `__syncthreads()` | ✅ **PASS** |
| `membar.gl` | `__threadfence()` | ✅ **PASS** |

---

## 5. 原子操作 ✅ 全部通过

| 测试 | 触发 API | 结果 |
|------|---------|------|
| `atom.global.add.u32` | `atomicAdd(p, v)` | ✅ **PASS** |
| `atom.global.cas.b32` | `atomicCAS(p, cmp, val)` | ✅ **PASS** |
| `atom.global.cas.b64` | `atomicCAS(p, cmp, val)` (64-bit) | ✅ **PASS** |
| `atom.global.exch.b32` | `atomicExch(p, v)` | ✅ **PASS** |
| `atom.shared.add.u32` | `atomicAdd(&smem, v)` | ✅ **PASS** |
| `cuda::atomic::fetch_add()` | libcu++ 包装 | ✅ 间接验证通过 |

---

## 6. 全局内存加载（缓存控制） ✅ 通过

| 测试 | 触发 API | 结果 |
|------|---------|------|
| `ld.global.nc` | `__ldg(ptr)` | ✅ **PASS** |

---

## 7. 标准计算指令 ✅ 全部通过

DGL 主库编译通过已证明这些指令全部可用：

| PTX 指令 | 作用 |
|----------|------|
| `mov.u32` / `mov.b32` | 寄存器移动 |
| `add.u32` / `sub.u32` | 整数加减 |
| `mul.lo.u32` / `mul.hi.u32` | 整数乘法 |
| `mad.lo.u32` | 乘加 |
| `div.u32` / `rem.u32` | 除法和取余 |
| `and.b32` / `or.b32` / `xor.b32` | 位运算 |
| `shr.u32` / `shl.b32` | 移位 |
| `setp.eq/ne/lt/gt/ge/le.u32` | 比较 |
| `selp.b32` | 条件选择 |
| `bra` / `brx` | 跳转 |
| `call` / `ret` | 函数调用 |
| `cvt.*` | 类型转换 |
| `ld.global.*` / `st.global.*` | 全局内存读写 |
| `ld.shared.*` / `st.shared.*` | 共享内存读写 |

---

## 8. 修复优先级（修正后）

| 优先级 | 问题 | 修复难度 | 状态 |
|--------|------|---------|------|
| 🔴 **P0** | `%laneid` — CG `tiled_partition` 构造函数 | 低 | ❌ 唯一阻塞 |
| 🟢 **P2** | CG `tiled_partition` 封装层 | 中 | 可绕过（用裸 API） |

**结论修正：** 之前认为 `shfl.sync` 和 `vote.sync` 都不支持，实际测试证明裸 API 全部通过。**真正阻塞的只有一条：`%laneid`**，且仅影响 `cooperative_groups::tiled_partition<>` 的 C++ 封装层。

---

## 9. 绕过方案（无需修改编译器）

既然 `__shfl_down_sync`、`__ballot_sync` 等裸 API 都可用，可以在 gpu_cache 源码层面绕过 `cooperative_groups::tiled_partition`，直接用裸 warp 原语重写 tile 操作：

```cpp
// 替代 cooperative_groups::tiled_partition<16>(block)
// 方案：直接使用裸 warp 原语
constexpr int WARP_SIZE = 32;
constexpr int TILE_SIZE = 16;

int lane_id = threadIdx.x % WARP_SIZE;          // 替代 %laneid
int warp_id = threadIdx.x / WARP_SIZE;           // 替代 meta_group_rank()
int tile_id = lane_id / TILE_SIZE;               // 替代 meta_group_rank() per tile
int tile_lane = lane_id % TILE_SIZE;             // 替代 tile.thread_rank()

// 替代 tile.shfl_down(val, offset)
float result = __shfl_down_sync(0xffffffff, val, offset);

// 替代 tile.ballot(pred)
unsigned mask = __ballot_sync(0xffffffff, pred);
```

**工作量：** 需修改 gpu_cache 4 个源文件中所有 `tiled_partition` 调用，约 20-30 处。

---

## 10. 结论

| 类别 | 测试数 | 通过 | 失败 | 未测试 |
|------|--------|------|------|--------|
| 特殊寄存器 | 8 | 5 | 1 (`%laneid`) | 2 |
| 同步/屏障 | 2 | 2 | 0 | 0 |
| 原子操作 | 5 | 5 | 0 | 0 |
| 缓存控制 | 1 | 1 | 0 | 0 |
| Warp 裸 API | 4 | 4 | 0 | 0 |
| Warp CG 封装 | 1 | 0 | 1 | 0 |
| **总计** | **21** | **17** | **1** | **2** |

**唯一阻塞：`%laneid`，仅来自 `cooperative_groups::tiled_partition<>`。**

**两条修复路径：**
1. **编译器端（推荐）：** 在 Corex LLVM 后端添加 `%laneid` 寄存器，一行映射即可
2. **源码端（绕过）：** 用 `__shfl_*_sync` / `__ballot_sync` 裸 API 重写 gpu_cache 中的 `tiled_partition` 调用

---

## 11. NVIDIA PTX ↔ Corex ISA 指令对应关系

> **获取 Corex 汇编方法：**
> ```bash
> # Step 1: 生成 LLVM IR（设备端）
> clang++ --cuda-device-only -S -emit-llvm kernel.cu -o kernel.ll
> # Step 2: llc 生成 Corex GPU 汇编
> llc -mcpu=ivcore11 -mtriple=bi-iluvatar-ilurt -filetype=asm -o kernel.s kernel.ll
> ```
>
> Corex ISA 为类 AMDGCN 风格，使用两级指令系统：
> - **`sl_*`** = Scalar ALU（标量单元，每条指令 1 个 warp 共享结果）
> - **`ml_*`** = Vector ALU（向量单元，每个线程独立）
> - **LSA** (Load/Store Address): 地址级全局内存访问
> - **SLB** (Scalar Load Buffer): 缓冲式内存访问，先经过 SLB 再到达 LSA

### 11.1 特殊寄存器

| NVIDIA PTX | CUDA C++ 来源 | LLVM Intrinsic | Corex ISA | 说明 |
|------------|--------------|----------------|-----------|------|
| `%tid.x` | `threadIdx.x` | `@llvm.nvvm.read.ptx.sreg.tid.x()` | `tid` (特殊寄存器) | 线程在 block 内的索引 |
| `%ntid.x` | `blockDim.x` | `@llvm.nvvm.read.ptx.sreg.ntid.x()` | kernel header SGPR | block 线程总数 |
| `%ctaid.x` | `blockIdx.x` | `@llvm.nvvm.read.ptx.sreg.ctaid.x()` | SGPR `s7` | block 在 grid 中的索引 |
| `%nctaid.x` | `gridDim.x` | `@llvm.nvvm.read.ptx.sreg.nctaid.x()` | SGPR `s6` | grid 中 block 总数 |
| `%clock` / `%clock64` | `clock64()` | `@llvm.nvvm.read.ptx.sreg.clock64()` | 支持（编译通过） | GPU 时钟周期计数器 |
| `%laneid` | CG `tiled_partition` 内部 | CG 层内部生成 | ❌ **不支持** | 线程在 warp 内的位置 0-31 |
| `%warpid` | `meta_group_rank()` | CG 层内部生成 | 被 `%laneid` 阻塞 | warp 在 block 内的索引 |

### 11.2 数据移动指令

| NVIDIA PTX | CUDA C++ 来源 | LLVM Intrinsic | Corex ISA | 说明 |
|------------|--------------|----------------|-----------|------|
| `mov.u32` | 赋值 `=` | LLVM IR `store`/`load` | `sl_mov_b32` / `ml_mov_b32` | 32-bit 寄存器移动 |
| `mov.b32` | bitcast | LLVM IR `bitcast` | `sl_mov_b32` | bitwise 移动 |

### 11.3 整数运算指令

| NVIDIA PTX | CUDA C++ 来源 | LLVM Intrinsic | Corex ISA | 说明 |
|------------|--------------|----------------|-----------|------|
| `add.u32` | `a + b` | LLVM IR `add` | `sl_add_u32` / `ml_add_i32` | 32-bit 加法 |
| `sub.u32` | `a - b` | LLVM IR `sub` | `sl_sub_u32` | 32-bit 减法 |
| `mul.lo.u32` | `a * b` | LLVM IR `mul` | `sl_mull_u32` / `ml_mul_u64_u32` | 32-bit 乘法（取低 32 位） |
| `mul.hi.u32` | (64-bit 结果高位) | LLVM IR `mul` (扩展) | `ml_mul_u64_u32` (取高 32 位) | 32-bit 乘法（取高 32 位） |
| `mad.lo.u32` | `a * b + c` | LLVM IR `mul`+`add` | `sl_mull_u32` + `sl_add_u32` | 乘加 |
| `div.u32` | `a / b` | LLVM IR `udiv` | (展开为多条) | 32-bit 无符号除法 |
| `rem.u32` | `a % b` | LLVM IR `urem` | (展开为多条) | 32-bit 取余 |

### 11.4 位运算指令

| NVIDIA PTX | CUDA C++ 来源 | LLVM Intrinsic | Corex ISA | 说明 |
|------------|--------------|----------------|-----------|------|
| `and.b32` | `a & b` | LLVM IR `and` | `sl_and_b32` | 32-bit 按位与 |
| `or.b32` | `a \| b` | LLVM IR `or` | `ml_or_b32` | 32-bit 按位或 |
| `xor.b32` | `a ^ b` | LLVM IR `xor` | `sl_xor_b64` | 32/64-bit 按位异或 |
| `shl.b32` | `a << b` | LLVM IR `shl` | `sl_shl_b32` / `ml_shl_b32` | 左移 |
| `shr.u32` | `a >> b` (unsigned) | LLVM IR `lshr` | `ml_srl_b32` | 逻辑右移 |
| `shr.s32` | `a >> b` (signed) | LLVM IR `ashr` | `ml_sra_b32` | 算术右移 |

### 11.5 比较与分支指令

| NVIDIA PTX | CUDA C++ 来源 | LLVM Intrinsic | Corex ISA | 说明 |
|------------|--------------|----------------|-----------|------|
| `setp.eq.u32` | `a == b` | LLVM IR `icmp eq` | `ml_cmp_eq_u32` | 相等比较 |
| `setp.ne.u32` | `a != b` | LLVM IR `icmp ne` | `ml_cmp_ne_u32` (组合) | 不等比较 |
| `setp.lt.u32` | `a < b` | LLVM IR `icmp ult` | `ml_cmp_lt_u32` (组合) | 小于比较 |
| `setp.gt.u32` | `a > b` | LLVM IR `icmp ugt` | `ml_cmp_gt_u32` (组合) | 大于比较 |
| `selp.b32` | `cond ? a : b` | LLVM IR `select` | `sl_csel_b32` / `sl_cbr_tmskaz` | 条件选择 |
| `bra` / `brx` | `if`/`goto`/循环 | LLVM IR `br` | `sl_cbr_tmskaz` (条件) / `sl_pc_set_b64` (跳转) | 分支跳转 |

### 11.6 同步与内存屏障指令

| NVIDIA PTX | CUDA C++ 来源 | LLVM Intrinsic | Corex ISA | 说明 |
|------------|--------------|----------------|-----------|------|
| `bar.sync` | `__syncthreads()` | `@llvm.nvvm.barrier0()` | `sl_barrier` | block 内所有线程同步 |
| `membar.gl` | `__threadfence()` | `@__nv_fence` | `ml_lsa_wbinv` + `sl_wait vmcnt(0) smcnt(0) lmcnt(0)` | 全局内存屏障 |
| `membar.cta` | `__threadfence_block()` | `@__nv_fence_block` | `sl_wait lmcnt(0)` | block 内内存屏障 |

### 11.7 原子操作指令

| NVIDIA PTX | CUDA C++ 来源 | LLVM Intrinsic | Corex ISA | 说明 |
|------------|--------------|----------------|-----------|------|
| `atom.global.add.u32` | `atomicAdd(p, v)` | `@__nv_atomic_add` | **两步:** `ml_slb_add_rtn_i32` → `ml_lsa_atomic_add_a64_i32_rtn` (kop=1) | 全局 32-bit 原子加（返回旧值） |
| `atom.global.cas.b32` | `atomicCAS(p, cmp, val)` | `@__nv_atomic_cas` | `ml_lsa_atomic_cmpswap_a64_b32_rtn` (kop=1) | 全局 32-bit CAS |
| `atom.global.cas.b64` | `atomicCAS(p, cmp, val)` | `@__nv_atomic_cas` | 64-bit LSA atomic cmpswap | 全局 64-bit CAS |
| `atom.global.exch.b32` | `atomicExch(p, v)` | `@__nv_atomic_exch` | `ml_lsa_atomic_swap_a64_b32_rtn` (kop=1) | 全局 32-bit 原子交换 |
| `atom.shared.add.u32` | `atomicAdd(&smem, v)` | LLVM IR `atomicrmw add` | 通过 SLB/LSA 到共享内存 | 共享内存原子加 |

> **Corex 原子操作两阶段模型：**
> ```
> ml_slb_add_rtn_i32   v2, v2, v0    ← SLB 阶段：准备操作数
> ml_lsa_atomic_add_a64_i32_rtn  v2, v[0:1], v2, kop=1   ← LSA 阶段：执行原子操作
> ml_lsa_wbinv                         ← 写回+失效缓存
> sl_wait vmcnt(0) smcnt(0) lmcnt(0)   ← 等待完成
> ```
> - `kop=0`: 不返回旧值
> - `kop=1`: 返回旧值

### 11.8 Warp 级操作指令

| NVIDIA PTX | CUDA C++ 来源 | LLVM Intrinsic | Corex ISA | 说明 |
|------------|--------------|----------------|-----------|------|
| `shfl.sync.down.b32` | `__shfl_down_sync(mask, val, offset)` | `@_Z16__shfl_down_syncjfji` | `ml_slb_shuffle_rtn_b32 vDst, vSrc, vOffset` | warp 内数据下移 |
| `shfl.sync.idx.b32` | `__shfl_sync(mask, val, lane)` | `@_Z11__shfl_syncjfji` | `ml_slb_shuffle_rtn_b32` | 从指定 lane 读取 |
| `shfl.sync.bfly.b32` | `__shfl_xor_sync(mask, val, mask)` | `@_Z11__shfl_xor_syncjfji` | `ml_slb_shuffle_rtn_b32` | 蝴蝶式交换 |
| `vote.sync.ballot.b32` | `__ballot_sync(mask, pred)` | **无专用 intrinsic** → 展开 | **无专用指令** → 展开为 `load`/`store`/`cmp` 序列 | 返回 warp 内 predicate 掩码 |
| `vote.sync.any.pred` | `__any_sync(mask, pred)` | **无专用 intrinsic** → 展开 | **无专用指令** → 展开为 `load`/`store`/`cmp` 序列 | warp 内任一 predicate 为 true |
| `vote.sync.all.pred` | `__all_sync(mask, pred)` | **无专用 intrinsic** → 展开 | **无专用指令** → 展开为 `load`/`store`/`cmp` 序列 | warp 内全部 predicate 为 true |
| `match.sync` | `__match_any_sync` (SM 7.0+) | 不可用（`__CUDA_ARCH__ < 700`） | 不可用 | warp 内值匹配 |

> **重要发现：** Corex 有专用硬件 shuffle 指令 `ml_slb_shuffle_rtn_b32`，但 **没有** 专用 ballot/vote/match 硬件指令（这些通过标准指令序列模拟）。`shfl` 操作在 Corex 上可以高效执行，但 ballot/vote 需要通过更复杂的指令序列实现。

### 11.9 全局内存加载（缓存控制）

| NVIDIA PTX | CUDA C++ 来源 | LLVM Intrinsic | Corex ISA | 说明 |
|------------|--------------|----------------|-----------|------|
| `ld.global.nc` | `__ldg(ptr)` | `@__nv_ldg` | 常规 `ml_lsa_load_a64_dword*` 序列（无 `.nc` 等价语义） | 绕过 L1 缓存读全局内存 |
| `ld.global.*` | `*ptr` | LLVM IR `load` | `ml_lsa_load_a64_dword` / `dwordx2` / `dwordx4` | 全局内存加载 |
| `st.global.*` | `*ptr = val` | LLVM IR `store` | `ml_lsa_store_a64_dword` / `dwordx2` / `dwordx4` | 全局内存存储 |
| `ld.shared.*` | shared memory read | LLVM IR `load addrspace(3)` | 通过 SLB 访问 | 共享内存加载 |
| `st.shared.*` | shared memory write | LLVM IR `store addrspace(3)` | 通过 SLB 访问 | 共享内存存储 |

### 11.10 指令前缀与命名约定

| Corex 前缀 | 含义 | 执行单元 |
|-----------|------|---------|
| `sl_` | Scalar ALU（标量） | 每条指令 1 个 warp 共享 1 个结果 |
| `ml_` | Vector ALU（向量） | 每个线程独立执行 |
| `sl_lsa_*` | Scalar Load/Store Address | 标量地址计算 |
| `ml_lsa_*` | Vector Load/Store Address | 向量地址级内存操作 |
| `ml_slb_*` | Vector Scalar Load Buffer | 缓冲式内存操作（SLB → LSA 管线） |

### 11.11 条件码与线程掩码寄存器

| Corex 特殊寄存器 | 对应 NVIDIA 概念 | 说明 |
|-----------------|-----------------|------|
| `tcr` | Condition Code (CC) | 线程条件寄存器（进位/溢出/零标志） |
| `tmsk` | Active Mask | 线程活跃掩码（处理 warp 内分支发散） |
| `wcr` | — | Warp 条件寄存器 |
| `tid` | `%tid` / `threadIdx.x` | 线程 ID |
| `exec` | (implicit in PTX predicates) | 执行掩码 |

---

## 参考资料

- [NVIDIA PTX ISA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [CUDA Cooperative Groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)
- gpu_cache 源码：`/root/GIDS_cufile/dgl/third_party/HugeCTR/gpu_cache/src/`
- 测试文件：`/tmp/ptx_test_basic.cu`、`/tmp/ptx_test_warp.cu`、`/tmp/ptx_shfl.cu` 等
- Corex 反汇编流水线：`clang++ --cuda-device-only -S -emit-llvm → llc -filetype=asm`