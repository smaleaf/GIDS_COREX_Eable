/*
 * libnvm_corex_map.c
 *
 * BaM libnvm.ko 的 Corex 适配版本
 * 将 BaM module/map.c 中 #ifdef _CUDA 部分替换为 Corex itr_p2p_* API
 *
 * 原始依赖 (NVIDIA):
 *   #include <nv-p2p.h>
 *   nvidia_p2p_get_pages()
 *   nvidia_p2p_dma_map_pages()
 *   nvidia_p2p_dma_unmap_pages()
 *   nvidia_p2p_put_pages()
 *   nvidia_p2p_free_page_table()
 *
 * 替换依赖 (Corex):
 *   从 iluvatar.ko 导入:
 *     itr_p2p_get_dev_pages()  [EXPORT_SYMBOL_GPL]
 *     itr_p2p_put_dev_pages()  [EXPORT_SYMBOL_GPL]
 *   从 itr_peer_mem.ko 导入:
 *     rdma_itr_p2p_get_pages()        [EXPORT_SYMBOL]
 *     rdma_itr_p2p_put_pages()        [EXPORT_SYMBOL]
 *     rdma_itr_p2p_dma_map_pages()    [EXPORT_SYMBOL]
 *     rdma_itr_p2p_dma_unmap_pages()  [EXPORT_SYMBOL]
 *
 * 编译时需要:
 *   KBUILD_EXTRA_SYMBOLS += $(ILUVATAR_KO_DIR)/Module.symvers
 *   KBUILD_EXTRA_SYMBOLS += $(ITR_PEER_MEM_KO_DIR)/Module.symvers
 */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/pci.h>
#include <linux/scatterlist.h>

/* ---- Corex P2P API 外部符号声明 ---- */
/* 来源：kmd/itr/itr_peer_mem.h + itr_peer_mem_user.h */
extern int rdma_itr_p2p_get_pages(unsigned long addr, size_t size, void **ctx);
extern void rdma_itr_p2p_put_pages(void *ctx);
extern int rdma_itr_p2p_dma_map_pages(struct device *dma_device, void *sg_head, void *ctx);
extern int rdma_itr_p2p_dma_unmap_pages(struct device *dma_device, void *sg_head, void *ctx);

/* BaM 中 GPU_PAGE_SIZE = 64KB，与 Corex ITRFS_GPU_PAGE_SIZE 一致 */
#define GPU_PAGE_SHIFT  16
#define GPU_PAGE_SIZE   (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_MASK   ~(GPU_PAGE_SIZE - 1)

/*
 * Corex GPU 内存区域描述符
 * 替换 BaM 中的 struct gpu_region { nvidia_p2p_page_table_t*, ... }
 */
struct gpu_region_corex {
	void            *ctx;    /* itr_peer_mem_ctx，由 rdma_itr_p2p_get_pages 分配 */
	struct sg_table  sgt;    /* DMA 地址 scatterlist，由 rdma_itr_p2p_dma_map_pages 填充 */
	u32              n_ctrl; /* 已 DMA 映射的 NVMe 控制器数量 */
};

/* 前向声明，BaM 中的 map/ctrl/list 结构 */
struct map;
struct ctrl;
struct list;

/* BaM map 结构的部分定义（简化版，实际使用时包含 BaM 头文件）*/
struct map_minimal {
	u64             vaddr;       /* GPU 虚拟地址 */
	u64            *addrs;       /* 输出：NVMe PRP bus 地址数组 */
	unsigned long   n_addrs;     /* 页数 */
	u64             page_size;   /* 页大小（GPU_PAGE_SIZE） */
	void           *data;        /* 保存 gpu_region_corex* */
	void          (*release)(struct map_minimal *map);
};

/* ------------------------------------------------------------------ */
/* 释放函数                                                             */
/* ------------------------------------------------------------------ */

static void release_gpu_memory_corex(struct map_minimal *map)
{
	struct gpu_region_corex *gd = (struct gpu_region_corex *)map->data;

	if (!gd)
		return;

	/* 无需传 dma_device，sg_table 内部已记录 */
	if (gd->sgt.sgl)
		sg_free_table(&gd->sgt);

	if (gd->ctx)
		rdma_itr_p2p_put_pages(gd->ctx);

	kfree(gd);
	map->data = NULL;
}

/* ------------------------------------------------------------------ */
/* 核心函数：GPU 内存 → NVMe DMA 映射                                  */
/* 替换 BaM map.c 中的 map_gpu_memory()                               */
/* ------------------------------------------------------------------ */

/**
 * map_gpu_memory_corex - 将 GPU 显存页注册为 NVMe DMA 目标
 * @map:       BaM map 描述符（含 vaddr/n_addrs/addrs）
 * @nvme_pdev: NVMe 控制器 PCI 设备（获取 DMA 地址空间）
 *
 * 成功后 map->addrs[] 填充可供 NVMe PRP 使用的 bus 地址。
 * 相当于 BaM 中 nvidia_p2p_get_pages() + nvidia_p2p_dma_map_pages()。
 *
 * 返回 0 成功，负值失败。
 */
int map_gpu_memory_corex(struct map_minimal *map, struct pci_dev *nvme_pdev)
{
	struct gpu_region_corex *gd;
	struct scatterlist *sg;
	unsigned long i;
	int err;

	gd = kzalloc(sizeof(*gd), GFP_KERNEL);
	if (!gd) {
		pr_err("[libnvm_corex] kmalloc gpu_region_corex failed\n");
		return -ENOMEM;
	}

	map->data    = gd;
	map->release = (void (*)(struct map_minimal *))release_gpu_memory_corex;

	/*
	 * Step 1: 通过 Corex P2P 接口锁定 GPU 物理页
	 * 等价于: nvidia_p2p_get_pages(0, 0, map->vaddr, size, &pages, cb, map)
	 *
	 * itr_peer_mem_ctx 内部会：
	 *   - 调用 itr_lib_p2p_get_dev_pages()
	 *   - 获取 GPU 物理页的 itr_address_range[]（含 DMA bus 地址）
	 */
	err = rdma_itr_p2p_get_pages(
		(unsigned long)map->vaddr,
		(size_t)(map->n_addrs * GPU_PAGE_SIZE),
		&gd->ctx);
	if (err) {
		pr_err("[libnvm_corex] rdma_itr_p2p_get_pages(va=%llx, size=%lu) failed: %d\n",
		       map->vaddr, map->n_addrs * GPU_PAGE_SIZE, err);
		kfree(gd);
		map->data = NULL;
		return err;
	}

	/*
	 * Step 2: DMA 映射 GPU 页到 NVMe 控制器设备地址空间
	 * 等价于: nvidia_p2p_dma_map_pages(nvme_pdev, pages, &dma_mapping)
	 *
	 * 结果存在 gd->sgt，每个 sg entry 的 dma_address 即 NVMe PRP 所用地址
	 */
	err = rdma_itr_p2p_dma_map_pages(&nvme_pdev->dev, &gd->sgt, gd->ctx);
	if (err) {
		pr_err("[libnvm_corex] rdma_itr_p2p_dma_map_pages failed: %d\n", err);
		goto err_put_pages;
	}

	/*
	 * Step 3: 填充 map->addrs[] 供 BaM 使用
	 * 等价于: dma_mapping->dma_addresses[i]
	 */
	i = 0;
	for_each_sg(gd->sgt.sgl, sg, gd->sgt.nents, i) {
		if (i >= map->n_addrs) {
			pr_warn("[libnvm_corex] sg entries (%u) > expected pages (%lu)\n",
				gd->sgt.nents, map->n_addrs);
			break;
		}
		map->addrs[i] = sg_dma_address(sg);
		pr_debug("[libnvm_corex] GPU page[%lu]: bus_addr=0x%llx\n",
			 i, map->addrs[i]);
	}
	map->n_addrs = i;

	pr_info("[libnvm_corex] Mapped %lu GPU pages (va=0x%llx) for NVMe DMA\n",
		map->n_addrs, map->vaddr);
	return 0;

err_put_pages:
	rdma_itr_p2p_put_pages(gd->ctx);
	kfree(gd);
	map->data = NULL;
	return err;
}
EXPORT_SYMBOL_GPL(map_gpu_memory_corex);

/* ------------------------------------------------------------------ */
/* 释放函数：unmap + put_pages                                          */
/* ------------------------------------------------------------------ */

/**
 * unmap_gpu_memory_corex - 释放 GPU DMA 映射
 * 等价于: nvidia_p2p_dma_unmap_pages() + nvidia_p2p_put_pages()
 */
void unmap_gpu_memory_corex(struct map_minimal *map, struct pci_dev *nvme_pdev)
{
	struct gpu_region_corex *gd = (struct gpu_region_corex *)map->data;

	if (!gd)
		return;

	if (gd->sgt.sgl)
		rdma_itr_p2p_dma_unmap_pages(&nvme_pdev->dev, &gd->sgt, gd->ctx);

	rdma_itr_p2p_put_pages(gd->ctx);
	kfree(gd);
	map->data = NULL;
}
EXPORT_SYMBOL_GPL(unmap_gpu_memory_corex);

/* ------------------------------------------------------------------ */
/* 使用示例（集成到 libnvm.ko 的 map.c 时）                             */
/* ------------------------------------------------------------------ */

/*
 * 原 BaM map.c 的 map_ioctl() 中 NVM_MAP_DEVICE_MEMORY case：
 *
 * #ifdef _CUDA
 *     map = map_device_memory(...);  // 调用 map_gpu_memory()
 * #endif
 *
 * 替换为：
 *
 * #ifdef _COREX
 *     map = map_device_memory_corex(...);  // 调用 map_gpu_memory_corex()
 * #endif
 *
 * map_device_memory_corex() 与原 map_device_memory() 接口相同，
 * 内部调用 map_gpu_memory_corex() 而非 map_gpu_memory()。
 */

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("GIDS-Corex Porting Team");
MODULE_DESCRIPTION("BaM libnvm GPU memory DMA mapping for Corex (replaces nv-p2p.h)");
