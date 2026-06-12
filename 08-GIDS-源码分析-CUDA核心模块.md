# GIDS 源码分析：CUDA Kernel 与 NVMe 接口层

## 1. 文件概览

| 文件 | 角色 |
|------|------|
| [gids_kernel.cu](file:///home/corex/sw_home_1/GIDS_enable/GIDS/gids_module/gids_kernel.cu) | GPU 端 CUDA Kernel 实现（设备端执行） |
| [gids_nvme.cu](file:///home/corex/sw_home_1/GIDS_enable/GIDS/gids_module/gids_nvme.cu) | Host 端 C++ 管理逻辑 + pybind11 绑定 |
| [bam_nvme.h](file:///home/corex/sw_home_1/GIDS_enable/GIDS/gids_module/include/bam_nvme.h) | 核心类和结构体声明 |

---

## 2. GPU Kernel 深度分析 (gids_kernel.cu)

### 2.1 read_feature_kernel — 基础特征读取

```cuda
template <typename T = float>
__global__ void read_feature_kernel(
    array_d_t<T> *dr,           // BaM 数组（GPU 端数据映射）
    T *out_tensor_ptr,          // 输出特征 tensor
    int64_t *index_ptr,         // 需要读取的节点 ID 列表
    int dim,                    // 特征维度
    int64_t num_idx,            // 节点数量
    int cache_dim,              // 缓存维度（用于地址计算）
    uint64_t key_off            // key 偏移（异构图多类型节点）
) {
    uint64_t bid = blockIdx.x;
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int idx_idx = bid * num_warps + warp_id;  // 当前处理的节点索引

    if (idx_idx < num_idx) {
        bam_ptr<T> ptr(dr);                   // GPU 端智能指针
        uint64_t row_index = index_ptr[idx_idx] + key_off;
        uint64_t tid = threadIdx.x % 32;       // warp 内线程 ID

        // 32 个线程协作读取一行特征
        for (; tid < dim; tid += 32) {
            const size_t idx = (row_index) * cache_dim + tid;
            out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = ptr.read(idx);
        }
    }
}
```

**设计要点：**
- **Warp-per-row 调度：** 每个 warp（32 线程）处理一行特征数据，实现 coalesced access
- **bam_ptr 透明页故障：** `ptr.read(idx)` 内部处理 page cache 查找/NVMe 读取
- **参数化缓存维度：** `cache_dim` 可能与实际 `dim` 不同，用于处理 padding

**Grid/Block 维度计算：**
```cpp
uint64_t b_size = blkSize;          // 默认 128 线程/block
uint64_t n_warp = b_size / 32;      // 4 warps/block
uint64_t g_size = (num_index + n_warp - 1) / n_warp; // grid 大小
```

---

### 2.2 read_feature_kernel_with_cpu_backing_memory — CPU 缓冲增强

```cuda
template <typename T = float>
__global__ void read_feature_kernel_with_cpu_backing_memory(
    array_d_t<T> *dr, range_d_t<T> *range,
    T *out_tensor_ptr, int64_t *index_ptr,
    int dim, int64_t num_idx, int cache_dim,
    GIDS_CPU_buffer<T> CPU_buffer,    // CPU 缓冲结构体
    bool cpu_seq,                      // true=顺序模式, false=哈希模式
    unsigned int* d_cpu_access,        // CPU 命中计数器
    uint64_t key_off
) {
    // ... 同上获取 idx_idx, row_index, tid ...

    uint32_t cpu_off = range->get_cpu_offset(row_index);

    if (cpu_seq) {
        // 顺序模式：row_index < CPU_buffer.cpu_buffer_len 即为命中
        if (row_index < CPU_buffer.cpu_buffer_len) {
            if (tid == 0) atomicAdd(d_cpu_access, 1);
            for (; tid < dim; tid += 32) {
                T temp = CPU_buffer.device_cpu_buffer[row_index * cache_dim + tid];
                out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
            }
        } else {
            // 未命中：回退到 SSD 读取
            for (; tid < dim; tid += 32) {
                T temp = ptr.read(row_index * cache_dim + tid);
                out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
            }
        }
    } else {
        // 哈希模式：cpu_off 低位为 1 表示命中
        if ((cpu_off & 0x1) == 1) {
            // cpu_off >> 1 得到 CPU 缓冲中的索引
            for (; tid < dim; tid += 32) {
                T temp = CPU_buffer.device_cpu_buffer[(cpu_off >> 1) * cache_dim + tid];
                out_tensor_ptr[...] = temp;
            }
        } else {
            // 未命中
            for (; tid < dim; tid += 32) {
                T temp = ptr.read(row_index * cache_dim + tid);
                out_tensor_ptr[...] = temp;
            }
        }
    }
}
```

**两种 CPU 缓冲模式：**

| 模式 | 判断逻辑 | 适用场景 |
|------|---------|---------|
| 顺序模式 (`cpu_seq=true`) | `row_index < cpu_buffer_len` | 节点 ID 连续排列 |
| 哈希模式 (`cpu_seq=false`) | `cpu_off & 0x1 == 1` | 节点 ID 随机分布 |

---

### 2.3 set_cpu_buffer_kernel — 设置 CPU 缓冲标记

```cuda
template <typename T = float>
__global__ void set_cpu_buffer_kernel(
    range_d_t<T> *d_range, 
    uint64_t* idx_ptr,    // 需要缓存的节点 ID 列表
    int num, 
    uint32_t pageSize
) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num) {
        d_range->set_cpu_buffer(idx_ptr[idx], idx);
    }
}
```

在 BaM 的 range 结构中标记哪些节点已被预取到 CPU 缓冲。

---

### 2.4 set_cpu_buffer_data_kernel — 实际预取数据到 CPU

```cuda
template <typename T = float>
__global__ void set_cpu_buffer_data_kernel(
    array_d_t<T> *dr,     // SSD 数据源
    T* CPU_buffer,         // CPU 缓冲目标
    uint64_t* idx_ptr,     // 节点列表
    uint64_t dim, 
    int num
) {
    uint64_t bid = blockIdx.x;
    bam_ptr<T> ptr(dr);
    if (bid < num) {
        uint64_t idx = idx_ptr[bid];
        for (uint64_t i = threadIdx.x; i < dim; i += blockDim.x) {
            CPU_buffer[bid * dim + i] = ptr[idx * dim + i];
        }
    }
}
```

从 SSD 读取指定节点的特征数据到 CPU 缓冲。

---

### 2.5 set_window_buffering_kernel — 窗口缓冲预取

```cuda
template <typename T = float>
__global__ void set_window_buffering_kernel(
    array_d_t<T>* dr, 
    uint64_t *index_ptr,   // 预取节点的页面索引
    uint64_t page_size, 
    int hash_off
) {
    bam_ptr<T> ptr(dr);
    if (threadIdx.x == 0) {
        uint64_t page_idx = index_ptr[blockIdx.x] + hash_off;
        // 通知 page cache 增加该页面的预取优先级
        ptr.set_window_buffer_counter(page_idx * page_size / sizeof(T), 1);
    }
}
```

每个 block 处理一个页面，由 thread 0 调用 BaM 的 `set_window_buffer_counter` 通知 page cache 提前预取。

---

### 2.6 write_feature_kernel2 — 数据写入 SSD

```cuda
template <typename T = float>
__global__ void write_feature_kernel2(
    Controller** ctrls, 
    page_cache_d_t* pc, 
    array_d_t<T> *dr, 
    T* in_tensor_ptr,      // 输入数据
    uint64_t dim, 
    uint32_t num_ctrls, 
    uint64_t offset
) {
    bam_ptr<T> ptr(dr);
    uint64_t row_index = blockIdx.x;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        ptr[row_index * dim + i] = in_tensor_ptr[row_index * dim + i + offset];
    }
}
```

将特征数据写入 BaM 管理的存储空间（最终 flush 到 SSD）。

---

## 3. Host 端管理逻辑 (gids_nvme.cu)

### 3.1 GIDS_Controllers — NVMe 控制器管理

```cpp
struct GIDS_Controllers {
    // 预定义的 SSD 设备路径
    const char *const ctrls_paths[6] = {
        "/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2",
        "/dev/libnvm3", "/dev/libnvm4", "/dev/libnvm5"
    };
    std::vector<Controller *> ctrls;
    uint32_t n_ctrls = 1;
    uint64_t queueDepth = 1024;
    uint64_t numQueues = 128;
    uint32_t cudaDevice = 0;      // CUDA 设备 ID
    uint32_t nvmNamespace = 1;     // NVMe namespace

    void init_GIDS_controllers(uint32_t num_ctrls, 
                                uint64_t q_depth, uint64_t num_q,
                                const std::vector<int>& ssd_list) {
        for (size_t i = 0; i < n_ctrls; i++) {
            ctrls.push_back(new Controller(
                ctrls_paths[ssd_list[i]],  // 设备路径
                nvmNamespace,               // namespace
                cudaDevice,                 // GPU 设备
                queueDepth,                 // 队列深度
                numQueues                   // 队列数量
            ));
        }
    }
};
```

**关键参数：**
- 最多支持 **6 块** NVMe SSD
- 每块 SSD 拥有 **128 个** 命令队列、每个队列深度 **1024**
- `ssd_list` 允许用户指定使用的 SSD 子集和顺序（用于条带化）

---

### 3.2 BAM_Feature_Store::init_controllers — 初始化

```cpp
void BAM_Feature_Store<TYPE>::init_controllers(
    GIDS_Controllers GIDS_ctrl, 
    uint32_t ps,           // 页大小
    uint64_t read_off,     // 读取偏移
    uint64_t cache_size,   // 缓存大小 (MB)
    uint64_t num_ele,      // 元素总数
    uint64_t num_ssd       // SSD 数量
) {
    numElems = num_ele;
    read_offset = read_off;
    n_ctrls = num_ssd;
    this->pageSize = ps;
    this->dim = ps / sizeof(TYPE);

    ctrls = GIDS_ctrl.ctrls;

    uint64_t page_size = pageSize;
    uint64_t n_pages = cache_size * 1024LL * 1024 / page_size;
    this->numPages = n_pages;

    // 创建 GPU 端页缓存
    this->h_pc = new page_cache_t(
        page_size, n_pages, cudaDevice, 
        ctrls[0][0],           // 第一个控制器的第一个队列
        (uint64_t)64,          // 关联性？
        ctrls                  // 所有控制器
    );

    uint64_t t_size = numElems * sizeof(TYPE);

    // 创建数据范围（条带化模式）
    this->h_range = new range_t<TYPE>(
        (uint64_t)0,                              // 起始地址
        (uint64_t)numElems,                        // 元素数
        (uint64_t)read_off,                        // 读取偏移
        (uint64_t)(t_size / page_size),           // 总页数
        (uint64_t)0,                               // 
        (uint64_t)page_size,                       // 页大小
        h_pc,                                      // 页缓存
        cudaDevice,                                // GPU 设备
        STRIPE                                     // 条带化模式
    );

    this->d_range = (range_d_t<TYPE> *)h_range->d_range_ptr;
    this->vr.push_back(nullptr);
    this->vr[0] = h_range;
    this->a = new array_t<TYPE>(numElems, 0, vr, cudaDevice);

    // 分配 CPU 访问计数器
    cudaMalloc(&d_cpu_access, sizeof(unsigned int));
    cudaMemset(d_cpu_access, 0, sizeof(unsigned));
}
```

---

### 3.3 read_feature — 单批次读取

```cpp
void BAM_Feature_Store<TYPE>::read_feature(
    uint64_t i_ptr,         // 输出 tensor 指针
    uint64_t i_index_ptr,   // 节点 ID 指针
    int64_t num_index,      // 节点数
    int dim,                // 特征维度
    int cache_dim,          // 缓存维度
    uint64_t key_off        // key 偏移
) {
    TYPE *tensor_ptr = (TYPE *)i_ptr;
    int64_t *index_ptr = (int64_t *)i_index_ptr;

    uint64_t b_size = blkSize;
    uint64_t n_warp = b_size / 32;
    uint64_t g_size = (num_index + n_warp - 1) / n_warp;

    cudaDeviceSynchronize();
    auto t1 = Clock::now();

    if (cpu_buffer_flag == false) {
        read_feature_kernel<TYPE><<<g_size, b_size>>>(
            a->d_array_ptr, tensor_ptr, index_ptr, 
            dim, num_index, cache_dim, key_off);
    } else {
        read_feature_kernel_with_cpu_backing_memory<<<g_size, b_size>>>(
            a->d_array_ptr, d_range, tensor_ptr, index_ptr,
            dim, num_index, cache_dim, CPU_buffer, seq_flag,
            d_cpu_access, key_off);
    }

    cudaDeviceSynchronize();
    // 统计 kernel 执行时间
    auto t2 = Clock::now();
    kernel_time += ms_fractional;
    total_access += num_index;

    // 读取 CPU 缓冲命中计数
    cudaMemcpy(&cpu_access_count, d_cpu_access, 
               sizeof(unsigned int), cudaMemcpyDeviceToHost);
}
```

---

### 3.4 read_feature_hetero / read_feature_merged — 多流并发

**异构图多流读取：** 每种节点类型使用独立的 CUDA Stream：

```cpp
void BAM_Feature_Store<TYPE>::read_feature_hetero(
    int num_iter,  // 异构图节点类型数
    const std::vector<uint64_t>& i_ptr_list,
    const std::vector<uint64_t>& i_index_ptr_list,
    const std::vector<uint64_t>& num_index,
    int dim, int cache_dim,
    const std::vector<uint64_t>& key_off
) {
    cudaStream_t streams[num_iter];
    for (int i = 0; i < num_iter; i++) {
        cudaStreamCreate(&streams[i]);
    }

    for (uint64_t i = 0; i < num_iter; i++) {
        // 每种节点类型独立的 kernel launch，使用独立 stream
        read_feature_kernel<TYPE><<<g_size, b_size, 0, streams[i]>>>(
            a->d_array_ptr, tensor_ptr, index_ptr,
            dim, num_index[i], cache_dim, key_off[i]);
    }

    // 等待所有 stream 完成
    for (int i = 0; i < num_iter; i++) {
        cudaStreamSynchronize(streams[i]);
    }
}
```

**read_feature_merged** 同构图多 batch 合并：accumulator 模式下使用。

---

### 3.5 cpu_backing_buffer — CPU 零拷贝缓冲

```cpp
void BAM_Feature_Store<TYPE>::cpu_backing_buffer(uint64_t dim, uint64_t len) {
    TYPE* cpu_buffer_ptr;
    TYPE* d_cpu_buffer_ptr;

    // cudaHostAllocMapped: 分配 page-locked 内存并映射到 GPU 地址空间
    cudaHostAlloc((TYPE **)&cpu_buffer_ptr, 
                  sizeof(TYPE) * dim * len, 
                  cudaHostAllocMapped);
    
    // 获取 GPU 端虚拟地址（zero-copy）
    cudaHostGetDevicePointer((TYPE **)&d_cpu_buffer_ptr, 
                             (TYPE *)cpu_buffer_ptr, 0);

    CPU_buffer.cpu_buffer_dim = dim;
    CPU_buffer.cpu_buffer_len = len;
    CPU_buffer.cpu_buffer = cpu_buffer_ptr;
    CPU_buffer.device_cpu_buffer = d_cpu_buffer_ptr;
    cpu_buffer_flag = true;
}
```

**关键 CUDA API：**
- `cudaHostAllocMapped`：分配 Unified Memory 可映射的主机内存
- `cudaHostGetDevicePointer`：获取设备端虚拟地址，GPU kernel 可直接访问

---

## 4. pybind11 绑定

```cpp
PYBIND11_MODULE(BAM_Feature_Store, m) {
    namespace py = pybind11;

    // float 类型绑定
    py::class_<BAM_Feature_Store<float>>(m, "BAM_Feature_Store_float")
        .def(py::init<>())
        .def("init_controllers", ...)
        .def("read_feature", ...)
        .def("read_feature_hetero", ...)
        .def("read_feature_merged", ...)
        .def("read_feature_merged_hetero", ...)
        .def("set_window_buffering", ...)
        .def("cpu_backing_buffer", ...)
        .def("set_cpu_buffer", ...)
        .def("flush_cache", ...)
        .def("store_tensor", ...)
        .def("read_tensor", ...)
        .def("get_array_ptr", ...)
        .def("get_offset_array", ...)
        .def("set_offsets", ...)
        .def("get_cpu_access_count", ...)
        .def("flush_cpu_access_count", ...)
        .def("print_stats", ...);

    // int64_t 类型绑定（用于图结构数据）
    py::class_<BAM_Feature_Store<int64_t>>(m, "BAM_Feature_Store_long")
        .def(py::init<>())
        // ... 同 float 绑定

    // 控制器绑定
    py::class_<GIDS_Controllers>(m, "GIDS_Controllers")
        .def(py::init<>())
        .def("init_GIDS_controllers", ...);
}
```

编译产物：`BAM_Feature_Store.cpython-*.so`

---

## 5. CMake 构建配置关键点

```cmake
# GPU 架构支持：Volta (SM 7.0) + Ampere (SM 8.0)
target_compile_options(BAM_Feature_Store PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_70,code=sm_70>"
)
target_compile_options(BAM_Feature_Store PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_80,code=sm_80>"
)

# 链接 BaM 的 libnvm.so
target_link_libraries(BAM_Feature_Store PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../bam/build/lib/libnvm.so)

# 头文件路径
target_include_directories(BAM_Feature_Store PRIVATE 
    ./include ../bam/include ../bam/include/freestanding/include/)
```

---

## 6. 关键数据结构

### GIDS_CPU_buffer
```cpp
template <typename TYPE>
struct GIDS_CPU_buffer {
    TYPE* cpu_buffer;           // CPU 端指针
    TYPE* device_cpu_buffer;    // GPU 端虚拟地址 (zero-copy)
    uint64_t cpu_buffer_dim;    // 特征维度
    uint64_t cpu_buffer_len;    // 缓冲节点数量
};
```

### 设备路径映射
| ssd_list 索引 | 设备路径 |
|--------------|---------|
| 0 | /dev/libnvm0 |
| 1 | /dev/libnvm1 |
| 2 | /dev/libnvm2 |
| 3 | /dev/libnvm3 |
| 4 | /dev/libnvm4 |
| 5 | /dev/libnvm5 |