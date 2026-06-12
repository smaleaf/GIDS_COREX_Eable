# cooperative_groups meta group interface plan

## Background

DGL HugeCTR gpu_cache uses `cooperative_groups::thread_block_tile<32>::meta_group_rank()`.
The ixdriver CUDA compatibility header implements `thread_block_tile` basics such as
`thread_rank()` and `size()`, but does not provide the CUDA-compatible meta group
query APIs. As a result, third-party CUDA sources fail at compile time with:

```text
error: no member named 'meta_group_rank' in 'cooperative_groups::__v1::thread_block_tile<32>'
```

## Scope

1. Add a minimal cudasample that compiles a kernel using:
   - `meta_group_rank()`
   - `meta_group_size()`
2. Implement both APIs in `include/IX/ixrt/cooperative_groups.h`.
3. Validate with a focused cudasample build using at most 4 build jobs.

## Design

For a tile partitioned from a CTA, `meta_group_rank()` is the rank of the current
tile among equal-sized tiles in the parent CTA. `meta_group_size()` is the number
of such tiles needed to cover the CTA.

The compatibility implementation derives those values from the CTA linear thread
rank and CTA size:

```cpp
meta_group_rank = cta_thread_rank / tile_size
meta_group_size = ceil(cta_size / tile_size)
```

This is sufficient for the DGL HugeCTR compile failure and matches the intended
CUDA cooperative groups behavior for CTA tile partitions.

## Test Plan

Build the new `cooperativeGroupsMetaGroupTest` cudasample target and confirm the
compiler accepts the new cooperative groups API calls.

## Status

✅ **已完成** — 2026-06-11 在 SWPM-918-gids 分支实现并验证通过。
SDK 头文件已部署到 `/home/corex/sw_home_1/sw_home/local/corex/include/cooperative_groups.h`。

**遗留问题：** 编译器后端 `%laneid` PTX 寄存器不支持，HugeCTR gpu_cache 仍无法编译。
详见 `/root/GIDS_cufile/docs/02-HugeCTR-gpu_cache-Corex兼容分析.md`。

## 相关文档

- DGL Corex 兼容分析：`/root/GIDS_cufile/docs/01-DGL-Corex兼容分析.md`
- HugeCTR gpu_cache 分析：`/root/GIDS_cufile/docs/02-HugeCTR-gpu_cache-Corex兼容分析.md`
- 兼容库总览：`/root/GIDS_cufile/docs/00-Corex兼容库总览.md`
