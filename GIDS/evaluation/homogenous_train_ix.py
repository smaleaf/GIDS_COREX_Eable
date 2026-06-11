import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.dataloading import MultiLayerNeighborSampler

# Import GIDS Iluvatar port
from GIDS_IX import GIDS, GIDS_DGLDataLoader

# ============================================================
# Model definition (same as original GDS training)
# ============================================================
class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dgl.nn.SAGEConv(in_feats, n_hidden, 'mean'))
        for _ in range(1, n_layers - 1):
            self.layers.append(dgl.nn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dgl.nn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[:block.number_of_dst_nodes()]
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


def load_graph_data(dataset_name, data_dir):
    if dataset_name == "ogbn-products":
        from ogb.nodeproppred import DglNodePropPredDataset
        d = DglNodePropPredDataset(name="ogbn-products", root=data_dir)
        graph, labels = d[0]
    elif dataset_name == "ogbn-papers100M":
        from ogb.nodeproppred import DglNodePropPredDataset
        d = DglNodePropPredDataset(name="ogbn-papers100M", root=data_dir)
        graph, labels = d[0]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return graph, labels, d


def train(args):
    device = torch.device(f"cuda:{args.gpu}")

    # ============================================================
    # Load data
    # ============================================================
    print(f"[Train-IX] Loading dataset: {args.dataset}")
    graph, labels, dataset = load_graph_data(args.dataset, args.data_dir)

    n_classes = (labels.max() + 1).item()
    n_feats = graph.ndata["feat"].shape[1]
    labels = labels.squeeze().to(device)

    # Split
    idx_split = None
    if args.dataset == "ogbn-products":
        split_idx = dataset.get_idx_split()
        train_idx = split_idx["train"]
        val_idx = split_idx["valid"]
        test_idx = split_idx["test"]

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)

    # Remove node features from memory (they're on SSD)
    del graph.ndata["feat"]

    # ============================================================
    # Initialize GIDS feature store (cuFile backend)
    # ============================================================
    feat_dim = n_feats
    cache_dim = feat_dim * args.pad_factor

    gids = GIDS(
        page_size=4096,
        off=0,
        cache_dim=cache_dim,
        num_ele=graph.number_of_nodes() * cache_dim,
        num_ssd=args.num_ssd,
        cache_size=args.cache_size,
        ctrl_idx=args.gpu,
        window_buffer=True,
        wb_size=args.wb_size,
        accumulator_flag=args.accumulator,
        file_paths=args.file_paths,
    )

    # CPU buffer for hot nodes
    if args.cpu_buffer_size > 0:
        gids.cpu_backing_buffer(cache_dim, args.cpu_buffer_size)

    # ============================================================
    # Model and optimizer
    # ============================================================
    model = SAGE(n_feats, args.n_hidden, n_classes, args.n_layers,
                 nn.ReLU(), args.dropout)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ============================================================
    # DataLoader with GIDS
    # ============================================================
    train_loader = GIDS_DGLDataLoader(
        dgl.add_self_loop(graph), train_idx,
        MultiLayerNeighborSampler([args.fanout] * args.n_layers),
        batch_size=args.batch_size,
        dim=n_feats,
        GIDS=gids,
        device=device,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    # ============================================================
    # Training loop
    # ============================================================
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            if len(batch) == 0:
                continue

            if isinstance(batch[-1], torch.Tensor):
                input_nodes, output_nodes, blocks = batch[:3]
                feat = batch[-1]
            else:
                input_nodes, output_nodes, blocks = batch

            blocks = [b.to(device) for b in blocks]
            feat = feat.to(device)

            logits = model(blocks, feat)

            target = labels[output_nodes].long()
            loss = loss_fn(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (logits.argmax(1) == target).sum().item()
            total_samples += len(output_nodes)

            if step % args.log_every == 0:
                acc = total_correct / max(total_samples, 1)
                print(f"[Epoch {epoch:3d} Step {step:4d}] "
                      f"Loss: {total_loss / max(step, 1):.4f} "
                      f"Acc: {acc:.4f}")

        epoch_time = time.time() - epoch_start
        acc = total_correct / max(total_samples, 1)
        print(f"[Epoch {epoch:3d}] Loss: {total_loss / max(step, 1):.4f} "
              f"Acc: {acc:.4f} Time: {epoch_time:.1f}s")

        gids.print_stats()

    print("[Train-IX] Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GIDS-IX Training Script")
    parser.add_argument("--dataset", type=str, default="ogbn-products", help="Dataset name")
    parser.add_argument("--data_dir", type=str, default="./data", help="Dataset directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--n_hidden", type=int, default=256, help="Hidden dim")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--fanout", type=str, default="10,10,10", help="Sampler fanout")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    # GIDS-specific
    parser.add_argument("--num_ssd", type=int, default=1, help="Number of SSDs")
    parser.add_argument("--cache_size", type=int, default=10, help="Cache size in MB")
    parser.add_argument("--pad_factor", type=int, default=1, help="Feature padding factor")
    parser.add_argument("--wb_size", type=int, default=8, help="Window buffer size")
    parser.add_argument("--accumulator", action="store_true", help="Enable storage access accumulator")
    parser.add_argument("--cpu_buffer_size", type=int, default=0, help="CPU buffer node count (0=disabled)")

    # File paths for cuFile
    parser.add_argument("--file_paths", type=str, nargs="+", default=None,
                        help="Paths to feature files (e.g. /mnt/nvme0/node_feat.bin)")

    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers")
    parser.add_argument("--log_every", type=int, default=10, help="Log every N steps")

    args = parser.parse_args()

    # Parse fanout
    if isinstance(args.fanout, str):
        args.fanout = [int(x) for x in args.fanout.split(",")]

    train(args)