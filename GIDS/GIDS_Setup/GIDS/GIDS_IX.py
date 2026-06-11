import math
import time
import torch
import numpy as np

# GIDS Iluvatar port - uses IXFeatureStore (cuFile backend) instead of BAM_Feature_Store
try:
    import IXFeatureStore
    _IX_AVAILABLE = True
except ImportError:
    IXFeatureStore = None
    _IX_AVAILABLE = False
    print("[GIDS-IX] WARNING: IXFeatureStore module not found. "
          "Build it with: cd gids_module_ix && mkdir -p build_ix && cd build_ix && cmake .. && make -j")

import dgl
from torch.utils.data import DataLoader
from collections.abc import Mapping

from dgl.dataloading import create_tensorized_dataset, WorkerInitWrapper, remove_parent_storage_columns
from dgl.utils import (
    recursive_apply, ExceptionWrapper, recursive_apply_pair, set_num_threads, get_num_threads,
    get_numa_nodes_cores, context_of, dtype_of)

from dgl import DGLHeteroGraph
from dgl.frame import LazyFeature
from dgl.storages import wrap_storage
from dgl.dataloading.base import BlockSampler, as_edge_prediction_sampler
from dgl import backend as F
from dgl.distributed import DistGraph
from dgl.multiprocessing import call_once_and_share

def _get_device(device):
    device = torch.device(device)
    if device.type in ('cuda', 'ix') and device.index is None:
        if device.type == 'ix':
            device = torch.device('cuda', 0)
        else:
            device = torch.device('cuda', torch.cuda.current_device())
    return device

class CollateWrapper(object):
    def __init__(self, sample_func, g, device):
        self.sample_func = sample_func
        self.g = g
        self.device = device

    def __call__(self, items):
        graph_device = getattr(self.g, 'device', None)
        items = recursive_apply(items, lambda x: x.to(self.device))
        batch = self.sample_func(self.g, items)
        return recursive_apply(batch, remove_parent_storage_columns, self.g)

class _PrefetchingIter(object):
    def __init__(self, dataloader, dataloader_it, GIDS_Loader=None):
        self.dataloader_it = dataloader_it
        self.dataloader = dataloader
        self.graph_sampler = self.dataloader.graph_sampler
        self.GIDS_Loader = GIDS_Loader

    def __iter__(self):
        return self

    def __next__(self):
        cur_it = self.dataloader_it
        batch = self.GIDS_Loader.fetch_feature(self.dataloader.dim, cur_it, self.GIDS_Loader.gids_device)
        return batch

class GIDS_DGLDataLoader(torch.utils.data.DataLoader):

    def __init__(self, graph, indices, graph_sampler, batch_size, dim, GIDS, device=None, use_ddp=False,
                 ddp_seed=0, drop_last=False, shuffle=False,
                 use_alternate_streams=None,
                 **kwargs):

        use_uva = False
        self.GIDS_Loader = GIDS
        self.dim = dim

        if isinstance(kwargs.get('collate_fn', None), CollateWrapper):
            assert batch_size is None
            self.graph = graph
            self.indices = indices
            self.graph_sampler = graph_sampler
            self.device = device
            self.use_ddp = use_ddp
            self.ddp_seed = ddp_seed
            self.shuffle = shuffle
            self.drop_last = drop_last
            self.use_alternate_streams = use_alternate_streams
            self.use_uva = use_uva
            kwargs['batch_size'] = None
            super().__init__(**kwargs)
            return

        if isinstance(graph, DistGraph):
            raise TypeError(
                'Please use dgl.dataloading.DistNodeDataLoader or '
                'dgl.datalaoding.DistEdgeDataLoader for DistGraphs.')

        self.graph = graph
        self.indices = indices
        num_workers = kwargs.get('num_workers', 0)

        indices_device = None
        try:
            if isinstance(indices, Mapping):
                indices = {k: (torch.tensor(v) if not torch.is_tensor(v) else v)
                           for k, v in indices.items()}
                indices_device = next(iter(indices.values())).device
            else:
                indices = torch.tensor(indices) if not torch.is_tensor(indices) else indices
                indices_device = indices.device
        except:
            pass

        if indices_device is None:
            if not hasattr(indices, 'device'):
                raise AttributeError('Custom indices dataset requires a \"device\" '
                                     'attribute indicating where the indices is.')
            indices_device = indices.device

        if device is None:
            device = torch.device('cuda', 0)
        self.device = _get_device(device)

        if isinstance(self.graph, DGLHeteroGraph):
            self.graph.create_formats_()
            if not self.graph._graph.is_pinned():
                self.graph._graph.pin_memory_()

            if use_alternate_streams is None:
                use_alternate_streams = (
                    self.device.type in ('cuda', 'ix') and self.graph.device.type == 'cpu' and
                    not use_uva)

        if (torch.is_tensor(indices) or (
                isinstance(indices, Mapping) and
                all(torch.is_tensor(v) for v in indices.values()))):
            self.dataset = create_tensorized_dataset(
                indices, batch_size, drop_last, use_ddp, ddp_seed, shuffle,
                kwargs.get('persistent_workers', False))
        else:
            self.dataset = indices

        self.ddp_seed = ddp_seed
        self.use_ddp = use_ddp
        self.use_uva = use_uva
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.graph_sampler = graph_sampler
        self.use_alternate_streams = use_alternate_streams

        self.cpu_affinity_enabled = False

        worker_init_fn = WorkerInitWrapper(kwargs.get('worker_init_fn', None))

        self.other_storages = {}

        super().__init__(
            self.dataset,
            collate_fn=CollateWrapper(
                self.graph_sampler.sample, graph, self.device),
            batch_size=None,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
            **kwargs)

    def __iter__(self):
        if self.shuffle:
            self.dataset.shuffle()
        num_threads = torch.get_num_threads() if self.num_workers > 0 else None
        return _PrefetchingIter(
            self, super().__iter__(), GIDS_Loader=self.GIDS_Loader)

    def print_stats(self):
        self.GIDS_Loader.print_stats()

    def print_timer(self):
        self.sample_time = 0.0
        self.graph_travel_time = 0.0

class GIDS():
    def __init__(self, page_size=4096, off=0, cache_dim=1024, num_ele=300*1000*1000*1024,
                 num_ssd=1, ssd_list=None, cache_size=10,
                 ctrl_idx=0,
                 window_buffer=False, wb_size=8,
                 accumulator_flag=False,
                 long_type=False,
                 heterograph=False,
                 heterograph_map=None,
                 file_paths=None):

        if not _IX_AVAILABLE:
            raise RuntimeError(
                "IXFeatureStore module not available. "
                "Build it: cd gids_module_ix && mkdir -p build_ix && cd build_ix && cmake .. && make -j")

        if long_type:
            self.IX_FS = IXFeatureStore.IXFeatureStore_long()
        else:
            self.IX_FS = IXFeatureStore.IXFeatureStore_float()

        self.accumulator_flag = accumulator_flag
        self.required_accesses = 0
        self.prev_cpu_access = 0
        self.return_torch_buffer = []
        self.index_list = []

        self.window_buffering_flag = window_buffer
        self.window_buffer = []
        self.wb_init = False
        self.wb_size = wb_size

        self.page_size = page_size

        if file_paths is None:
            default_path = "/mnt/nvme0/node_feat.bin"
            self.file_paths = [default_path]
            if num_ssd > 1:
                self.file_paths = [
                    "/mnt/nvme{}/node_feat_part_{}.bin".format(i, i)
                    for i in range(num_ssd)
                ]
        else:
            self.file_paths = file_paths

        self.off = math.ceil(math.ceil(off / page_size) / num_ssd)
        self.num_ele = num_ele
        self.cache_size = cache_size

        self.heterograph = heterograph
        self.heterograph_map = heterograph_map
        self.graph_GIDS = None

        self.cache_dim = cache_dim
        self.gids_device = "cuda:" + str(ctrl_idx)

        print("[GIDS-IX] Initializing with cuFile backend")
        print("[GIDS-IX] SSD paths:", self.file_paths)
        print("[GIDS-IX] page_size:", page_size, "cache_size:", cache_size, "num_ssd:", num_ssd)

        self.IX_FS.init_controllers(num_ssd, self.file_paths, page_size, self.off, num_ele, cache_size)

        self.GIDS_time = 0.0
        self.WB_time = 0.0

    def init_graph_GIDS(self, page_size, off, cache_size, num_ele, num_ssd):
        self.graph_GIDS = IXFeatureStore.IXFeatureStore_long()
        self.graph_GIDS.init_controllers(num_ssd, self.file_paths, page_size, off, num_ele, cache_size)

    def get_offset_array(self):
        if self.graph_GIDS:
            return self.graph_GIDS.get_offset_array()
        return 0

    def get_array_ptr(self):
        if self.graph_GIDS:
            return self.graph_GIDS.get_array_ptr()
        return 0

    def cpu_backing_buffer(self, dim, length):
        self.IX_FS.cpu_backing_buffer(dim, length)

    def set_cpu_buffer(self, ten, N):
        topk_ten = ten[:N]
        topk_len = len(topk_ten)
        d_ten = topk_ten.to(self.gids_device)
        self.IX_FS.set_cpu_buffer(d_ten.data_ptr(), topk_len)

    def window_buffering(self, batch):
        s_time = time.time()
        if self.heterograph:
            for key, value in batch[0].items():
                if len(value) == 0:
                    continue
                input_tensor = value.to(self.gids_device)
                key_off = 0
                if self.heterograph_map is not None:
                    if key in self.heterograph_map:
                        key_off = self.heterograph_map[key]
                    else:
                        print("Cannot find key:", key, "in the heterograph map!")
                num_pages = len(input_tensor)
                self.IX_FS.set_window_buffering(input_tensor.data_ptr(), num_pages, key_off)
                e_time = time.time()
                self.WB_time += e_time - s_time
                s_time = e_time
        else:
            input_tensor = batch[0].to(self.gids_device)
            num_pages = len(input_tensor)
            self.IX_FS.set_window_buffering(input_tensor.data_ptr(), num_pages, 0)
            e_time = time.time()
            self.WB_time += e_time - s_time

    def fill_wb(self, it, num):
        for i in range(num):
            batch = next(it)
            self.window_buffer.append(batch)
            self.window_buffering(batch)

    def set_required_storage_access(self, bw, l_ssd, l_system, num_ssd, p):
        accesses = (p * bw * 1024 / self.page_size * (l_ssd + l_system) * num_ssd) / (1 - p)
        self.required_accesses = accesses
        print("Number of required storage accesses:", accesses)

    def fetch_feature(self, dim, it, device):
        GIDS_time_start = time.time()

        if self.window_buffering_flag:
            if not self.wb_init:
                self.fill_wb(it, self.wb_size)
                self.wb_init = True

        next_batch = next(it)

        self.window_buffer.append(next_batch)
        if self.window_buffering_flag:
            self.window_buffering(next_batch)

        if self.accumulator_flag:
            index_size_list = []
            index_ptr_list = []
            return_torch_list = []
            key_list = []

            if len(self.return_torch_buffer) != 0:
                return_ten = self.return_torch_buffer.pop(0)
                return_batch = self.window_buffer.pop(0)
                return_batch.append(return_ten)
                self.GIDS_time += time.time() - GIDS_time_start
                return return_batch

            buffer_size = len(self.window_buffer)
            current_access = 0
            num_iter = 0
            required_accesses = self.required_accesses

            if self.heterograph:
                while True:
                    if num_iter >= buffer_size:
                        batch = next(it)
                        for k, v in batch[0].items():
                            current_access += len(v)
                        self.window_buffer.append(batch)
                        if self.window_buffering_flag:
                            self.window_buffering(batch)
                    else:
                        batch = self.window_buffer[num_iter]
                        for k, v in batch[0].items():
                            current_access += len(v)
                    num_iter += 1
                    required_accesses += self.prev_cpu_access
                    if current_access > required_accesses:
                        break

                num_concurrent_iter = 0
                for i in range(num_iter):
                    batch = self.window_buffer[i]
                    ret_ten = {}
                    for k, v in batch[0].items():
                        if len(v) == 0:
                            empty_t = torch.empty((0, dim)).to(self.gids_device)
                            ret_ten[k] = empty_t
                        else:
                            key_off = 0
                            if self.heterograph_map is not None:
                                if k in self.heterograph_map:
                                    key_off = self.heterograph_map[k]
                                else:
                                    print("Cannot find key:", k, "in the heterograph map!")
                            v = v.to(self.gids_device)
                            index_size = len(v)
                            index_size_list.append(index_size)
                            return_torch = torch.zeros([index_size, dim], dtype=torch.float, device=self.gids_device)
                            index_ptr_list.append(v.data_ptr())
                            ret_ten[k] = return_torch
                            return_torch_list.append(return_torch.data_ptr())
                            key_list.append(key_off)
                            num_concurrent_iter += 1
                    self.return_torch_buffer.append(ret_ten)

                self.IX_FS.read_feature_merged_hetero(num_concurrent_iter, return_torch_list,
                                                       index_ptr_list, index_size_list, dim,
                                                       self.cache_dim, key_list)

                return_ten = self.return_torch_buffer.pop(0)
                return_b = self.window_buffer.pop(0)
                if isinstance(return_b, tuple):
                    return_batch = (*return_b, return_ten)
                else:
                    return_batch = return_b
                    return_batch.append(return_ten)
                self.GIDS_time += time.time() - GIDS_time_start

                cpu_access_count = self.IX_FS.get_cpu_access_count()
                self.prev_cpu_access = int(cpu_access_count / num_iter)
                self.IX_FS.flush_cpu_access_count()

                return return_batch
            else:
                while True:
                    if num_iter >= buffer_size:
                        batch = next(it)
                        current_access += len(batch[0])
                        self.window_buffer.append(batch)
                        if self.window_buffering_flag:
                            self.window_buffering(batch)
                    else:
                        batch = self.window_buffer[num_iter]
                        current_access += len(batch[0])
                    num_iter += 1
                    required_accesses += self.prev_cpu_access
                    if current_access > required_accesses:
                        break

                for i in range(num_iter):
                    batch = self.window_buffer[i]
                    index = batch[0].to(self.gids_device)
                    index_size = len(index)
                    index_size_list.append(index_size)
                    return_torch = torch.zeros([index_size, dim], dtype=torch.float, device=self.gids_device)
                    index_ptr_list.append(index.data_ptr())
                    return_torch_list.append(return_torch.data_ptr())
                    self.return_torch_buffer.append(return_torch)

                self.IX_FS.read_feature_merged(num_iter, return_torch_list, index_ptr_list,
                                                index_size_list, dim, self.cache_dim)
                return_ten = self.return_torch_buffer.pop(0)
                return_b = self.window_buffer.pop(0)
                if isinstance(return_b, tuple):
                    return_batch = (*return_b, return_ten)
                else:
                    return_batch = return_b
                    return_batch.append(return_ten)

                self.GIDS_time += time.time() - GIDS_time_start

                cpu_access_count = self.IX_FS.get_cpu_access_count()
                self.prev_cpu_access = int(cpu_access_count / num_iter)
                self.IX_FS.flush_cpu_access_count()

                return return_batch

        else:
            if self.heterograph:
                batch = self.window_buffer.pop(0)
                ret_ten = {}
                index_size_list = []
                index_ptr_list = []
                return_torch_list = []
                key_list = []

                num_keys = 0
                for key, v in batch[0].items():
                    if len(v) == 0:
                        empty_t = torch.empty((0, dim)).to(self.gids_device).contiguous()
                        ret_ten[key] = empty_t
                    else:
                        key_off = 0
                        if self.heterograph_map is not None:
                            if key in self.heterograph_map:
                                key_off = self.heterograph_map[key]
                            else:
                                print("Cannot find key:", key, "in the heterograph map!")

                        g_index = v.to(self.gids_device)
                        index_size = len(g_index)
                        index_ptr = g_index.data_ptr()

                        return_torch = torch.zeros([index_size, dim], dtype=torch.float,
                                                    device=self.gids_device).contiguous()
                        return_torch_list.append(return_torch.data_ptr())
                        ret_ten[key] = return_torch
                        num_keys += 1
                        index_ptr_list.append(index_ptr)
                        index_size_list.append(index_size)
                        key_list.append(key_off)

                self.IX_FS.read_feature_hetero(num_keys, return_torch_list, index_ptr_list,
                                                index_size_list, dim, self.cache_dim, key_list)

                self.GIDS_time += time.time() - GIDS_time_start
                if isinstance(batch, tuple):
                    return (*batch, ret_ten)
                else:
                    batch.append(ret_ten)
                    return batch

            else:
                batch = self.window_buffer.pop(0)
                index = batch[0].to(self.gids_device)
                index_size = len(index)
                index_ptr = index.data_ptr()
                return_torch = torch.zeros([index_size, dim], dtype=torch.float,
                                            device=self.gids_device).contiguous()
                self.IX_FS.read_feature(return_torch.data_ptr(), index_ptr, index_size,
                                         dim, self.cache_dim, 0)
                self.GIDS_time += time.time() - GIDS_time_start

                if isinstance(batch, tuple):
                    return (*batch, return_torch)
                else:
                    batch.append(return_torch)
                    return batch

    def print_stats(self):
        print("GIDS time:", self.GIDS_time)
        wbtime = self.WB_time
        print("WB time:", wbtime)
        self.WB_time = 0.0
        self.GIDS_time = 0.0
        self.IX_FS.print_stats()

        if self.graph_GIDS is not None:
            self.graph_GIDS.print_stats_no_ctrl()

    def store_tensor(self, in_ten, offset):
        num_e = len(in_ten)
        self.IX_FS.store_tensor(in_ten.data_ptr(), num_e, offset)