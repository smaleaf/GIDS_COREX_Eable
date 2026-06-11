#ifndef IX_FEATURE_STORE_H
#define IX_FEATURE_STORE_H

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

inline void ix_err_chk(int err, const char* msg = "") {
    if (err != 0) {
        std::cerr << "[IX ERROR] " << msg << " (code=" << err << ")" << std::endl;
        throw std::runtime_error(msg);
    }
}

template <typename TYPE>
struct IX_CPU_buffer {
    TYPE* cpu_buffer;
    TYPE* device_cpu_buffer;
    uint64_t cpu_buffer_dim;
    uint64_t cpu_buffer_len;
};

struct IXFileHandle {
    int fd;
    void* fh;
    size_t file_size;
    std::string path;

    ~IXFileHandle() {
        if (fh) {
            // ixdrvFileHandleDeregister(fh);
        }
        if (fd >= 0) {
            close(fd);
        }
    }
};

template <typename TYPE>
struct IXFeatureStore {

    IX_CPU_buffer<TYPE> CPU_buffer;
    bool cpu_buffer_flag = false;
    bool seq_flag = true;
    uint64_t* offset_array;

    int dim;
    uint64_t total_access;
    unsigned int cpu_access_count = 0;
    unsigned int* d_cpu_access;

    uint32_t ixDevice = 0;
    size_t numPages = 262144 * 8;
    uint32_t n_ctrls = 1;
    size_t blkSize = 128;
    uint32_t pageSize = 4096;
    uint64_t numElems = 300LL * 1000 * 1000 * 1024;
    uint64_t read_offset = 0;

    std::vector<IXFileHandle*> file_handles;
    size_t gpu_buffer_size;
    void* gpu_read_buffer;

    float kernel_time = 0;

    void init_controllers(uint32_t num_ssd, const std::vector<std::string>& file_paths,
                          uint32_t ps, uint64_t r_off, uint64_t num_ele, uint64_t cache_size);

    void read_feature(uint64_t tensor_ptr, uint64_t index_ptr,
                      int64_t num_index, int dim, int cache_dim, uint64_t key_off);
    void read_feature_hetero(int num_iter,
                             const std::vector<uint64_t>& i_ptr_list,
                             const std::vector<uint64_t>& i_index_ptr_list,
                             const std::vector<uint64_t>& num_index,
                             int dim, int cache_dim,
                             const std::vector<uint64_t>& key_off);
    void read_feature_merged(int num_iter,
                             const std::vector<uint64_t>& i_ptr_list,
                             const std::vector<uint64_t>& i_index_ptr_list,
                             const std::vector<uint64_t>& num_index,
                             int dim, int cache_dim);
    void read_feature_merged_hetero(int num_iter,
                                    const std::vector<uint64_t>& i_ptr_list,
                                    const std::vector<uint64_t>& i_index_ptr_list,
                                    const std::vector<uint64_t>& num_index,
                                    int dim, int cache_dim,
                                    const std::vector<uint64_t>& key_off);

    void cpu_backing_buffer(uint64_t dim, uint64_t len);
    void set_cpu_buffer(uint64_t idx_buffer, int num);

    void set_window_buffering(uint64_t id_idx, int64_t num_pages, int hash_off);
    void print_stats();
    void print_stats_no_ctrl();

    uint64_t get_array_ptr();
    uint64_t get_offset_array();
    void set_offsets(uint64_t in_off, uint64_t index_off, uint64_t data_off);
    void store_tensor(uint64_t tensor_ptr, uint64_t num, uint64_t offset);
    void read_tensor(uint64_t num, uint64_t offset);
    void flush_cache();
    unsigned int get_cpu_access_count();
    void flush_cpu_access_count();

private:
    int get_file_idx(uint64_t row_index);
    uint64_t get_file_offset(uint64_t row_index, int cache_dim);
    void do_cufile_read(int file_idx, void* dst_ptr, size_t size, uint64_t file_offset);
};

#endif