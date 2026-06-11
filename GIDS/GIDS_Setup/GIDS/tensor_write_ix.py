import argparse
import os
import time
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import IXFeatureStore
    HAS_IXFS = True
except ImportError:
    HAS_IXFS = False


def tensor_write_to_file(data, output_path, page_size=4096, ctrl_idx=0):
    """
    Write tensor data to a regular file for cuFile access.
    No raw NVMe device required - filesystem-based storage.
    """
    if isinstance(data, np.ndarray):
        num_elements = data.size
        data_np = data
    elif HAS_TORCH and isinstance(data, torch.Tensor):
        num_elements = data.numel()
        data_np = data.numpy()
    else:
        raise TypeError("data must be numpy.ndarray or torch.Tensor")

    file_size = num_elements * data_np.itemsize
    file_size_aligned = ((file_size + page_size - 1) // page_size) * page_size

    print(f"[tensor_write] Writing {num_elements} elements ({file_size / 1024**2:.1f} MB)")
    print(f"[tensor_write] File size aligned: {file_size_aligned / 1024**2:.1f} MB")
    print(f"[tensor_write] Output: {output_path}")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    t_start = time.time()

    if HAS_IXFS and HAS_TORCH:
        fs = IXFeatureStore.IXFeatureStore_float()
        fs.init_controllers(1, [output_path], page_size, 0, num_elements, 0)
        fs.store_tensor(torch.tensor(data_np).to(f"ix:{ctrl_idx}").data_ptr(),
                        num_elements, 0)
    else:
        data_np.tofile(output_path)

    t_elapsed = time.time() - t_start
    bw = file_size / t_elapsed / 1e9 if t_elapsed > 0 else 0
    print(f"[tensor_write] Done in {t_elapsed:.2f}s ({bw:.2f} GB/s)")


def prepare_graph_csr(data_dir, output_dir, num_partitions=1):
    """
    Convert DGL graph CSR data to cuFile-compatible files.
    """
    indptr_path = os.path.join(data_dir, "indptr.dat")
    indices_path = os.path.join(data_dir, "indices.dat")

    if not os.path.exists(indptr_path) or not os.path.exists(indices_path):
        print(f"[prepare_graph_csr] CSR files not found in {data_dir}")
        print("[prepare_graph_csr] Looking for indptr.dat and indices.dat")
        return False

    os.makedirs(output_dir, exist_ok=True)

    for src, name in [(indptr_path, "indptr"), (indices_path, "indices")]:
        data = np.fromfile(src, dtype=np.int64)
        dst = os.path.join(output_dir, f"{name}.bin")
        tensor_write_to_file(data, dst)
        print(f"[prepare_graph_csr] {name}: {len(data)} elements -> {dst}")

    return True


def prepare_node_features(feat_dir, num_nodes, feat_dim, output_dir,
                          page_size=4096, num_partitions=1):
    """
    Prepare node features for cuFile access.

    If partitions > 1, splits features across multiple files for SSD striping.
    """
    feat_files = sorted([
        f for f in os.listdir(feat_dir)
        if f.endswith('.npy') or f.endswith('.bin')
    ])

    if feat_files:
        all_features = []
        for f in feat_files:
            fp = os.path.join(feat_dir, f)
            if f.endswith('.npy'):
                arr = np.load(fp)
            else:
                arr = np.fromfile(fp, dtype=np.float32)
            all_features.append(arr)

        if all_features:
            features = np.concatenate(all_features) if len(all_features) > 1 else all_features[0]
        else:
            features = np.zeros((num_nodes, feat_dim), dtype=np.float32)
    else:
        features = np.zeros((num_nodes, feat_dim), dtype=np.float32)

    features = features.reshape(-1, feat_dim).astype(np.float32)

    os.makedirs(output_dir, exist_ok=True)

    if num_partitions == 1:
        output_path = os.path.join(output_dir, "node_feat.bin")
        tensor_write_to_file(features.ravel(), output_path, page_size)
    else:
        # Page-level striping across files
        elements_per_page = page_size // 4  # float32 = 4 bytes
        total_elements = features.size
        total_rows = total_elements // feat_dim

        for pid in range(num_partitions):
            output_path = os.path.join(output_dir, f"node_feat_part_{pid}.bin")
            print(f"[prepare_node_features] Writing partition {pid} -> {output_path}")

            # Collect pages belonging to this partition
            pages_for_partition = []
            for page_idx in range((total_elements + elements_per_page - 1) // elements_per_page):
                if page_idx % num_partitions == pid:
                    start = page_idx * elements_per_page
                    end = min(start + elements_per_page, total_elements)
                    pages_for_partition.append(features.ravel()[start:end])
            if pages_for_partition:
                partition_data = np.concatenate(pages_for_partition)
                tensor_write_to_file(partition_data, output_path, page_size)

    print(f"[prepare_node_features] Done: {num_nodes} nodes x {feat_dim} dims")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GIDS-IX data preparation for cuFile")
    parser.add_argument("--mode", choices=["write", "prepare_graph", "prepare_feat"], required=True,
                        help="Operation mode")
    parser.add_argument("--input", required=True, help="Input numpy file or data directory")
    parser.add_argument("--output", required=True, help="Output file or directory")
    parser.add_argument("--num_nodes", type=int, default=0, help="Number of graph nodes")
    parser.add_argument("--feat_dim", type=int, default=0, help="Feature dimension")
    parser.add_argument("--page_size", type=int, default=4096, help="Page size in bytes")
    parser.add_argument("--num_partitions", type=int, default=1, help="Number of SSD partitions")
    parser.add_argument("--ctrl_idx", type=int, default=0, help="GPU device index")
    args = parser.parse_args()

    if args.mode == "write":
        data = np.load(args.input) if args.input.endswith('.npy') else np.fromfile(args.input, dtype=np.float32)
        tensor_write_to_file(data, args.output, args.page_size, args.ctrl_idx)
    elif args.mode == "prepare_graph":
        prepare_graph_csr(args.input, args.output, args.num_partitions)
    elif args.mode == "prepare_feat":
        prepare_node_features(args.input, args.num_nodes, args.feat_dim,
                              args.output, args.page_size, args.num_partitions)