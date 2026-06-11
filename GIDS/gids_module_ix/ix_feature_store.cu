#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>

#include "ix_feature_store.h"

typedef std::chrono::high_resolution_clock Clock;

// ============================================================
// cuFile API declarations (standard NVIDIA cuFile, available on Iluvatar)
// ============================================================
extern "C" {
    int cuFileDriverOpen();
    int cuFileDriverClose();
    int cuFileHandleRegister(void** fh, void* descr);
    int cuFileHandleDeregister(void* fh);
    int cuFileBufRegister(const void* devPtr_base, size_t size, int flags);
    int cuFileBufDeregister(const void* devPtr_base);
    ssize_t cuFileRead(void* fh, void* devPtr_base, size_t size,
                       int64_t file_offset, int64_t devPtr_offset);
    ssize_t cuFileWrite(void* fh, const void* devPtr_base, size_t size,
                        int64_t file_offset, int64_t devPtr_offset);
}

// ============================================================
// IX API declarations (from ixdriver SDK / mapping_cudart.h)
// All are C-linkage in libcudart.so
// ============================================================
extern "C" {
extern int ixMalloc(void** devPtr, size_t size);
extern int ixFree(void* devPtr);
extern int ixHostAlloc(void** pHost, size_t size, unsigned int flags);
extern int ixFreeHost(void* ptr);
extern int ixHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags);
extern int ixMemcpy(void* dst, const void* src, size_t count, int kind);
extern int ixDeviceSynchronize(void);
extern int ixMemset(void* devPtr, int value, size_t count);
extern int ixStreamCreate(int* pStream);
extern int ixStreamDestroy(int stream);
extern int ixStreamSynchronize(int stream);
extern int ixGetDeviceCount(int* count);
extern int ixSetDevice(int device);
extern int ixGetLastError(void);
extern const char* ixGetErrorString(int error);
}

enum {
    ixMemcpyHostToDevice = 1,
    ixMemcpyDeviceToHost = 2,
    ixMemcpyDeviceToDevice = 3,
    ixHostAllocMapped = 2,
    ixHostAllocWriteCombined = 4,
    ixHostAllocDefault = 0,
};

static bool g_cufile_initialized = false;
static bool g_cufile_available = false;

static void ensure_cufile_init() {
    if (g_cufile_initialized) return;
    g_cufile_initialized = true;
    int ret = cuFileDriverOpen();
    if (ret == 0) {
        g_cufile_available = true;
        std::cout << "[IXFeatureStore] cuFile driver initialized successfully" << std::endl;
    } else {
        std::cerr << "[IXFeatureStore] WARNING: cuFile driver init failed (code="
                  << ret << "), falling back to POSIX read" << std::endl;
        g_cufile_available = false;
    }
}

// Helper to read from file (cuFile preferred, POSIX fallback)
static ssize_t file_read_to_gpu(void* gpu_dev_ptr, size_t size,
                                 int64_t file_offset, int fd,
                                 void* cuFile_handle = nullptr) {
    if (g_cufile_available && cuFile_handle) {
        return cuFileRead(cuFile_handle, gpu_dev_ptr, size, file_offset, 0);
    }
    // Fallback: read to pinned host buffer then copy to GPU
    void* host_buf = nullptr;
    ixHostAlloc(&host_buf, size, ixHostAllocDefault);
    ssize_t ret = pread(fd, host_buf, size, file_offset);
    if (ret > 0) {
        ixMemcpy(gpu_dev_ptr, host_buf, ret, ixMemcpyHostToDevice);
    }
    ixFreeHost(host_buf);
    return ret;
}

static ssize_t file_write_from_gpu(const void* gpu_dev_ptr, size_t size,
                                    int64_t file_offset, int fd,
                                    void* cuFile_handle = nullptr) {
    if (g_cufile_available && cuFile_handle) {
        return cuFileWrite(cuFile_handle, gpu_dev_ptr, size, file_offset, 0);
    }
    void* host_buf = nullptr;
    ixHostAlloc(&host_buf, size, ixHostAllocDefault);
    ixMemcpy(host_buf, gpu_dev_ptr, size, ixMemcpyDeviceToHost);
    ssize_t ret = pwrite(fd, host_buf, size, file_offset);
    ixFreeHost(host_buf);
    return ret;
}

// ============================================================
// IXFeatureStore implementation
// ============================================================

template <typename TYPE>
void IXFeatureStore<TYPE>::init_controllers(
    uint32_t num_ssd,
    const std::vector<std::string>& file_paths,
    uint32_t ps, uint64_t r_off, uint64_t num_ele, uint64_t cache_size) {

    ensure_cufile_init();

    numElems = num_ele;
    read_offset = r_off;
    n_ctrls = num_ssd;
    pageSize = ps;
    dim = ps / sizeof(TYPE);
    total_access = 0;

    uint64_t n_pages = cache_size * 1024LL * 1024 / pageSize;
    numPages = n_pages;

    std::cout << "[IXFeatureStore] n pages: " << (int)numPages << std::endl;
    std::cout << "[IXFeatureStore] page size: " << (int)pageSize << std::endl;
    std::cout << "[IXFeatureStore] num elements: " << numElems << std::endl;
    std::cout << "[IXFeatureStore] num SSDs: " << n_ctrls << std::endl;

    // Open files for each SSD
    for (uint32_t i = 0; i < n_ctrls; i++) {
        IXFileHandle* h = new IXFileHandle();
        if (i < file_paths.size()) {
            h->path = file_paths[i];
        } else {
            h->path = file_paths[0]; // fallback
        }
        h->fd = open(h->path.c_str(), O_RDWR | O_DIRECT);
        if (h->fd < 0) {
            h->fd = open(h->path.c_str(), O_RDWR);
        }
        if (h->fd < 0) {
            std::cerr << "[IXFeatureStore] ERROR: Cannot open " << h->path << std::endl;
            throw std::runtime_error("Cannot open feature file: " + h->path);
        }
        h->file_size = lseek(h->fd, 0, SEEK_END);
        h->fh = nullptr;

        if (g_cufile_available) {
            struct { int type; int fd; } descr = { 0, h->fd };
            int ret = cuFileHandleRegister(&h->fh, &descr);
            if (ret != 0) {
                std::cerr << "[IXFeatureStore] WARNING: cuFile register failed for "
                          << h->path << " (code=" << ret << ")" << std::endl;
            }
        }

        file_handles.push_back(h);
        std::cout << "[IXFeatureStore] Opened " << h->path
                  << " (size=" << h->file_size << "B)" << std::endl;
    }

    // Pre-allocate GPU read buffer (4MB for batched reads)
    gpu_buffer_size = 4 * 1024 * 1024;
    ixMalloc(&gpu_read_buffer, gpu_buffer_size);
    if (g_cufile_available) {
        cuFileBufRegister(gpu_read_buffer, gpu_buffer_size, 0);
    }

    // CPU access counter
    ixMalloc((void**)&d_cpu_access, sizeof(unsigned int));
    ixMemset(d_cpu_access, 0, sizeof(unsigned int));
}

template <typename TYPE>
int IXFeatureStore<TYPE>::get_file_idx(uint64_t row_index) {
    if (n_ctrls <= 1) return 0;
    return (row_index / (pageSize / sizeof(TYPE))) % n_ctrls;
}

template <typename TYPE>
uint64_t IXFeatureStore<TYPE>::get_file_offset(uint64_t row_index, int cache_dim) {
    uint64_t elem_offset = row_index * cache_dim;
    if (n_ctrls <= 1) {
        return read_offset + elem_offset * sizeof(TYPE);
    }
    // Striped: interleave pages across SSDs
    uint64_t elements_per_page = pageSize / sizeof(TYPE);
    uint64_t page_idx = row_index * cache_dim / elements_per_page;
    uint64_t stripe_idx = page_idx / n_ctrls;
    uint64_t page_offset_in_stripe = (row_index * cache_dim) % elements_per_page;
    return read_offset + (stripe_idx * pageSize) + page_offset_in_stripe * sizeof(TYPE);
}

template <typename TYPE>
void IXFeatureStore<TYPE>::do_cufile_read(int file_idx, void* dst_ptr,
                                           size_t size, uint64_t file_offset) {
    if (file_idx >= (int)file_handles.size()) file_idx = 0;
    IXFileHandle* h = file_handles[file_idx];
    ssize_t ret = file_read_to_gpu(dst_ptr, size, file_offset, h->fd, h->fh);
    if (ret < 0 || (size_t)ret < size) {
        std::cerr << "[IXFeatureStore] WARNING: short read " << ret << "/" << size << std::endl;
    }
}

// ============================================================
// read_feature - Single batch, single stream
// ============================================================
template <typename TYPE>
void IXFeatureStore<TYPE>::read_feature(uint64_t i_ptr, uint64_t i_index_ptr,
                                         int64_t num_index, int dim,
                                         int cache_dim, uint64_t key_off) {

    TYPE* tensor_ptr = (TYPE*)i_ptr;
    int64_t* index_ptr = (int64_t*)i_index_ptr;
    size_t row_bytes = dim * sizeof(TYPE);

    ixDeviceSynchronize();
    auto t1 = Clock::now();

    if (cpu_buffer_flag) {
        // For CPU buffer mode: iterate nodes, use CPU buffer for hot ones
        void* host_idx = nullptr;
        ixHostAlloc(&host_idx, num_index * sizeof(int64_t), ixHostAllocDefault);
        ixMemcpy(host_idx, index_ptr, num_index * sizeof(int64_t), ixMemcpyDeviceToHost);

        int64_t* host_index = (int64_t*)host_idx;
        for (int64_t i = 0; i < num_index; i++) {
            uint64_t row_idx = host_index[i] + key_off;
            void* dst = (char*)tensor_ptr + i * row_bytes;

            if (seq_flag && row_idx < CPU_buffer.cpu_buffer_len) {
                // Hot node: read from CPU buffer via device pointer
                ixMemcpy(dst, CPU_buffer.device_cpu_buffer + row_idx * cache_dim,
                         row_bytes, ixMemcpyDeviceToDevice);
            } else {
                int fidx = get_file_idx(row_idx);
                uint64_t foff = get_file_offset(row_idx, cache_dim);
                do_cufile_read(fidx, dst, row_bytes, foff);
            }
        }
        ixFreeHost(host_idx);
    } else {
        // Direct SSD read for all nodes
        void* host_idx = nullptr;
        ixHostAlloc(&host_idx, num_index * sizeof(int64_t), ixHostAllocDefault);
        ixMemcpy(host_idx, index_ptr, num_index * sizeof(int64_t), ixMemcpyDeviceToHost);

        int64_t* host_index = (int64_t*)host_idx;
        // Batch reads by SSD
        for (int fidx = 0; fidx < (int)n_ctrls; fidx++) {
            std::vector<int64_t> batch_nodes;
            std::vector<void*> batch_dsts;
            for (int64_t i = 0; i < num_index; i++) {
                uint64_t row_idx = host_index[i] + key_off;
                if (get_file_idx(row_idx) == fidx) {
                    batch_nodes.push_back(row_idx);
                    batch_dsts.push_back((char*)tensor_ptr + i * row_bytes);
                }
            }
            // Sequential reads to same SSD
            for (size_t j = 0; j < batch_nodes.size(); j++) {
                uint64_t foff = get_file_offset(batch_nodes[j], cache_dim);
                do_cufile_read(fidx, batch_dsts[j], row_bytes, foff);
            }
        }
        ixFreeHost(host_idx);
    }

    ixDeviceSynchronize();
    auto t2 = Clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    kernel_time += static_cast<float>(us.count()) / 1000;
    total_access += num_index;
}

// ============================================================
// read_feature_hetero - Multi-type nodes, concurrent CUDA streams
// ============================================================
template <typename TYPE>
void IXFeatureStore<TYPE>::read_feature_hetero(
    int num_iter,
    const std::vector<uint64_t>& i_ptr_list,
    const std::vector<uint64_t>& i_index_ptr_list,
    const std::vector<uint64_t>& num_index,
    int dim, int cache_dim,
    const std::vector<uint64_t>& key_off) {

    // Create IX streams for concurrent execution
    std::vector<int> streams(num_iter);
    for (int i = 0; i < num_iter; i++) {
        ixStreamCreate(&streams[i]);
    }

    ixDeviceSynchronize();
    auto t1 = Clock::now();

    // Process each node type
    for (int i = 0; i < num_iter; i++) {
        TYPE* tensor_ptr = (TYPE*)i_ptr_list[i];
        int64_t* index_ptr = (int64_t*)i_index_ptr_list[i];
        int64_t nidx = num_index[i];
        uint64_t koff = key_off[i];
        size_t row_bytes = dim * sizeof(TYPE);

        void* host_idx = nullptr;
        ixHostAlloc(&host_idx, nidx * sizeof(int64_t), ixHostAllocDefault);
        ixMemcpy(host_idx, index_ptr, nidx * sizeof(int64_t), ixMemcpyDeviceToHost);

        int64_t* host_index = (int64_t*)host_idx;
        for (int64_t j = 0; j < nidx; j++) {
            uint64_t row_idx = host_index[j] + koff;
            void* dst = (char*)tensor_ptr + j * row_bytes;
            int fidx = get_file_idx(row_idx);
            uint64_t foff = get_file_offset(row_idx, cache_dim);
            do_cufile_read(fidx, dst, row_bytes, foff);
        }
        ixFreeHost(host_idx);
        total_access += nidx;
    }

    // Sync all streams
    for (int i = 0; i < num_iter; i++) {
        ixStreamSynchronize(streams[i]);
        ixStreamDestroy(streams[i]);
    }
    ixDeviceSynchronize();

    auto t2 = Clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    kernel_time += static_cast<float>(us.count()) / 1000;
}

// ============================================================
// read_feature_merged - Batch multiple reads together
// ============================================================
template <typename TYPE>
void IXFeatureStore<TYPE>::read_feature_merged(
    int num_iter,
    const std::vector<uint64_t>& i_ptr_list,
    const std::vector<uint64_t>& i_index_ptr_list,
    const std::vector<uint64_t>& num_index,
    int dim, int cache_dim) {

    ixDeviceSynchronize();
    auto t1 = Clock::now();

    for (int i = 0; i < num_iter; i++) {
        TYPE* tensor_ptr = (TYPE*)i_ptr_list[i];
        int64_t* index_ptr = (int64_t*)i_index_ptr_list[i];
        int64_t nidx = num_index[i];
        size_t row_bytes = dim * sizeof(TYPE);

        void* host_idx = nullptr;
        ixHostAlloc(&host_idx, nidx * sizeof(int64_t), ixHostAllocDefault);
        ixMemcpy(host_idx, index_ptr, nidx * sizeof(int64_t), ixMemcpyDeviceToHost);

        int64_t* host_index = (int64_t*)host_idx;
        for (int64_t j = 0; j < nidx; j++) {
            uint64_t row_idx = host_index[j];
            void* dst = (char*)tensor_ptr + j * row_bytes;
            int fidx = get_file_idx(row_idx);
            uint64_t foff = get_file_offset(row_idx, cache_dim);
            do_cufile_read(fidx, dst, row_bytes, foff);
        }
        ixFreeHost(host_idx);
        total_access += nidx;
    }
    ixDeviceSynchronize();

    auto t2 = Clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    kernel_time += static_cast<float>(us.count()) / 1000;
}

// ============================================================
// read_feature_merged_hetero
// ============================================================
template <typename TYPE>
void IXFeatureStore<TYPE>::read_feature_merged_hetero(
    int num_iter,
    const std::vector<uint64_t>& i_ptr_list,
    const std::vector<uint64_t>& i_index_ptr_list,
    const std::vector<uint64_t>& num_index,
    int dim, int cache_dim,
    const std::vector<uint64_t>& key_off) {

    ixDeviceSynchronize();
    auto t1 = Clock::now();

    for (int i = 0; i < num_iter; i++) {
        TYPE* tensor_ptr = (TYPE*)i_ptr_list[i];
        int64_t* index_ptr = (int64_t*)i_index_ptr_list[i];
        int64_t nidx = num_index[i];
        uint64_t koff = key_off[i];
        size_t row_bytes = dim * sizeof(TYPE);

        void* host_idx = nullptr;
        ixHostAlloc(&host_idx, nidx * sizeof(int64_t), ixHostAllocDefault);
        ixMemcpy(host_idx, index_ptr, nidx * sizeof(int64_t), ixMemcpyDeviceToHost);

        int64_t* host_index = (int64_t*)host_idx;
        for (int64_t j = 0; j < nidx; j++) {
            uint64_t row_idx = host_index[j] + koff;
            void* dst = (char*)tensor_ptr + j * row_bytes;
            int fidx = get_file_idx(row_idx);
            uint64_t foff = get_file_offset(row_idx, cache_dim);
            do_cufile_read(fidx, dst, row_bytes, foff);
        }
        ixFreeHost(host_idx);
        total_access += nidx;
    }
    ixDeviceSynchronize();

    auto t2 = Clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    kernel_time += static_cast<float>(us.count()) / 1000;
}

// ============================================================
// CPU Buffer management (zero-copy via ixHostAllocMapped)
// ============================================================
template <typename TYPE>
void IXFeatureStore<TYPE>::cpu_backing_buffer(uint64_t dim, uint64_t len) {
    TYPE* cpu_buffer_ptr = nullptr;
    TYPE* d_cpu_buffer_ptr = nullptr;

    ixHostAlloc((void**)&cpu_buffer_ptr, sizeof(TYPE) * dim * len, ixHostAllocMapped);
    ixHostGetDevicePointer((void**)&d_cpu_buffer_ptr, cpu_buffer_ptr, 0);

    CPU_buffer.cpu_buffer_dim = dim;
    CPU_buffer.cpu_buffer_len = len;
    CPU_buffer.cpu_buffer = cpu_buffer_ptr;
    CPU_buffer.device_cpu_buffer = d_cpu_buffer_ptr;
    cpu_buffer_flag = true;

    std::cout << "[IXFeatureStore] CPU buffer allocated: "
              << len << " nodes x " << dim << " dims ("
              << (sizeof(TYPE) * dim * len / (1024.0 * 1024)) << " MB)" << std::endl;
}

template <typename TYPE>
void IXFeatureStore<TYPE>::set_cpu_buffer(uint64_t idx_buffer, int num) {
    // idx_buffer: GPU pointer to node indices
    // Read these nodes from SSD into CPU buffer
    uint64_t* host_idx = new uint64_t[num];
    ixMemcpy(host_idx, (void*)idx_buffer, num * sizeof(uint64_t), ixMemcpyDeviceToHost);

    for (int i = 0; i < num; i++) {
        uint64_t row_idx = host_idx[i];
        int fidx = get_file_idx(row_idx);
        uint64_t foff = get_file_offset(row_idx, dim);
        size_t row_bytes = dim * sizeof(TYPE);

        // Read directly into CPU buffer (pinned memory)
        if (g_cufile_available && file_handles[fidx]->fh) {
            cuFileRead(file_handles[fidx]->fh,
                          CPU_buffer.device_cpu_buffer + i * dim,
                          row_bytes, foff, 0);
        } else {
            ssize_t ret_rb = pread(file_handles[fidx]->fd,
                  CPU_buffer.cpu_buffer + i * dim,
                  row_bytes, foff);
        }
    }
    delete[] host_idx;
    ixDeviceSynchronize();
    seq_flag = false;

    std::cout << "[IXFeatureStore] CPU buffer populated with " << num << " nodes" << std::endl;
}

// ============================================================
// Window buffering (no-op in cuFile, kept for API compatibility)
// ============================================================
template <typename TYPE>
void IXFeatureStore<TYPE>::set_window_buffering(uint64_t, int64_t, int) {
    // cuFile reads are CPU-initiated, GPU-side page cache doesn't exist
    // Window buffering is handled at Python level (batch prefetching)
}

// ============================================================
// Store tensor (write features to SSD file)
// ============================================================
template <typename TYPE>
void IXFeatureStore<TYPE>::store_tensor(uint64_t tensor_ptr, uint64_t num, uint64_t offset) {
    TYPE* t_ptr = (TYPE*)tensor_ptr;
    uint64_t total_bytes = num * sizeof(TYPE);

    std::cout << "[IXFeatureStore] Writing " << num << " elements ("
              << (total_bytes / (1024.0 * 1024)) << " MB)"
              << " to " << file_handles[0]->path << std::endl;

    file_write_from_gpu(t_ptr, total_bytes, offset,
                        file_handles[0]->fd, file_handles[0]->fh);
    ixDeviceSynchronize();
}

template <typename TYPE>
void IXFeatureStore<TYPE>::read_tensor(uint64_t num, uint64_t offset) {
    TYPE* tmp = new TYPE[num];
    file_read_to_gpu(tmp, num * sizeof(TYPE), offset,
                     file_handles[0]->fd, file_handles[0]->fh);
    std::cout << "[IXFeatureStore] Read " << num << " elements from offset " << offset << std::endl;
    delete[] tmp;
}

// ============================================================
// Misc
// ============================================================
template <typename TYPE>
void IXFeatureStore<TYPE>::flush_cache() {
    ixDeviceSynchronize();
    // cuFile doesn't have a GPU-side page cache to flush
    // File writes are immediate (O_DIRECT)
}

template <typename TYPE>
void IXFeatureStore<TYPE>::print_stats() {
    std::cout << "[IXFeatureStore] Kernel Time: " << kernel_time << " ms" << std::endl;
    std::cout << "[IXFeatureStore] Total Access: " << total_access << std::endl;
    std::cout << "[IXFeatureStore] cuFile mode: "
              << (g_cufile_available ? "enabled" : "fallback (POSIX)") << std::endl;
    kernel_time = 0;
    total_access = 0;
}

template <typename TYPE>
void IXFeatureStore<TYPE>::print_stats_no_ctrl() {
    print_stats();
}

template <typename TYPE>
uint64_t IXFeatureStore<TYPE>::get_array_ptr() {
    return 0; // Not applicable in cuFile mode
}

template <typename TYPE>
uint64_t IXFeatureStore<TYPE>::get_offset_array() {
    return 0;
}

template <typename TYPE>
void IXFeatureStore<TYPE>::set_offsets(uint64_t, uint64_t, uint64_t) {
    // Not needed in cuFile mode
}

template <typename TYPE>
unsigned int IXFeatureStore<TYPE>::get_cpu_access_count() {
    return cpu_access_count;
}

template <typename TYPE>
void IXFeatureStore<TYPE>::flush_cpu_access_count() {
    cpu_access_count = 0;
    ixMemset(d_cpu_access, 0, sizeof(unsigned int));
}

// ============================================================
// pybind11 module definition
// ============================================================
PYBIND11_MODULE(IXFeatureStore, m) {
    m.doc() = "GIDS Feature Store for Iluvatar GPU (cuFile backend)";

    namespace py = pybind11;

    // float type
    py::class_<IXFeatureStore<float>>(m, "IXFeatureStore_float")
        .def(py::init<>())
        .def("init_controllers", &IXFeatureStore<float>::init_controllers)
        .def("read_feature", &IXFeatureStore<float>::read_feature)
        .def("read_feature_hetero", &IXFeatureStore<float>::read_feature_hetero)
        .def("read_feature_merged", &IXFeatureStore<float>::read_feature_merged)
        .def("read_feature_merged_hetero", &IXFeatureStore<float>::read_feature_merged_hetero)
        .def("set_window_buffering", &IXFeatureStore<float>::set_window_buffering)
        .def("cpu_backing_buffer", &IXFeatureStore<float>::cpu_backing_buffer)
        .def("set_cpu_buffer", &IXFeatureStore<float>::set_cpu_buffer)
        .def("flush_cache", &IXFeatureStore<float>::flush_cache)
        .def("store_tensor", &IXFeatureStore<float>::store_tensor)
        .def("read_tensor", &IXFeatureStore<float>::read_tensor)
        .def("get_array_ptr", &IXFeatureStore<float>::get_array_ptr)
        .def("get_offset_array", &IXFeatureStore<float>::get_offset_array)
        .def("set_offsets", &IXFeatureStore<float>::set_offsets)
        .def("get_cpu_access_count", &IXFeatureStore<float>::get_cpu_access_count)
        .def("flush_cpu_access_count", &IXFeatureStore<float>::flush_cpu_access_count)
        .def("print_stats", &IXFeatureStore<float>::print_stats);

    // int64_t type (for graph structure data)
    py::class_<IXFeatureStore<int64_t>>(m, "IXFeatureStore_long")
        .def(py::init<>())
        .def("init_controllers", &IXFeatureStore<int64_t>::init_controllers)
        .def("read_feature", &IXFeatureStore<int64_t>::read_feature)
        .def("read_feature_hetero", &IXFeatureStore<int64_t>::read_feature_hetero)
        .def("read_feature_merged", &IXFeatureStore<int64_t>::read_feature_merged)
        .def("read_feature_merged_hetero", &IXFeatureStore<int64_t>::read_feature_merged_hetero)
        .def("set_window_buffering", &IXFeatureStore<int64_t>::set_window_buffering)
        .def("cpu_backing_buffer", &IXFeatureStore<int64_t>::cpu_backing_buffer)
        .def("set_cpu_buffer", &IXFeatureStore<int64_t>::set_cpu_buffer)
        .def("flush_cache", &IXFeatureStore<int64_t>::flush_cache)
        .def("store_tensor", &IXFeatureStore<int64_t>::store_tensor)
        .def("read_tensor", &IXFeatureStore<int64_t>::read_tensor)
        .def("get_array_ptr", &IXFeatureStore<int64_t>::get_array_ptr)
        .def("get_offset_array", &IXFeatureStore<int64_t>::get_offset_array)
        .def("set_offsets", &IXFeatureStore<int64_t>::set_offsets)
        .def("get_cpu_access_count", &IXFeatureStore<int64_t>::get_cpu_access_count)
        .def("flush_cpu_access_count", &IXFeatureStore<int64_t>::flush_cpu_access_count)
        .def("print_stats", &IXFeatureStore<int64_t>::print_stats);
}