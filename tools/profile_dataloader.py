#!/usr/bin/env python3
"""
tools/profile_dataloader.py: High-Precision DataLoader & GPU Pipeline Profiler.
Measures pure disk I/O, multi-worker collate, GPU transfer, forward, and backward times
over 100 iterations to pinpoint the exact binding bottleneck in training throughput.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.multimodal_dataset import MultimodalSequenceDataset, collate_multimodal_batch
from models.proposed_model import ProposedXMISTModel


def parse_args():
    parser = argparse.ArgumentParser(description="Profile DataLoader & Training Pipeline")
    parser.add_argument("--data_dir", type=str, default="data/processed_pie", help="Processed data directory")
    parser.add_argument("--split", type=str, default="train", help="Split to profile")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_batches", type=int, default=100, help="Number of batches to profile")
    parser.add_argument("--workers_test", type=str, default="2,4,8", help="Comma-separated list of worker counts to benchmark")
    return parser.parse_args()


def benchmark_workers(dataset, batch_size: int, num_workers: int, num_batches: int, device: torch.device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_multimodal_batch,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    times_load = []
    times_gpu = []
    t_prev = time.time()

    for idx, batch in enumerate(loader):
        if idx >= num_batches:
            break
        t_batch_ready = time.time()
        times_load.append(t_batch_ready - t_prev)

        # GPU transfer time
        t_gpu0 = time.time()
        batch_gpu = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_gpu.append(time.time() - t_gpu0)

        t_prev = time.time()

    # Discard warmup (first 5 batches)
    times_load = times_load[5:]
    times_gpu = times_gpu[5:]

    avg_load_ms = np.mean(times_load) * 1000.0
    avg_gpu_ms = np.mean(times_gpu) * 1000.0
    return avg_load_ms, avg_gpu_ms


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print("DATALOADER & PIPELINE PROFILING REPORT")
    print(f"Device: {device} | Batch Size: {args.batch_size} | Split: {args.split}")
    print("=" * 75)

    dataset = MultimodalSequenceDataset(
        data_dir=args.data_dir,
        split=args.split,
        window_size=32
    )
    print(f"Dataset Size: {len(dataset):,} windows")

    # Benchmark worker configurations
    worker_counts = [int(w.strip()) for w in args.workers_test.split(",") if w.strip()]
    print("\n--- 1. Multi-Worker DataLoader Throughput ---")
    results = {}
    for w in worker_counts:
        load_ms, gpu_ms = benchmark_workers(dataset, args.batch_size, w, args.num_batches, device)
        total_ms = load_ms + gpu_ms
        est_epoch_sec = (len(dataset) / args.batch_size) * (total_ms / 1000.0)
        results[w] = (load_ms, gpu_ms, est_epoch_sec)
        print(f"num_workers={w:<2d} | Load Time: {load_ms:6.1f} ms/batch | GPU Transfer: {gpu_ms:5.1f} ms | Est. Load Overhead/Epoch: {est_epoch_sec/60.0:5.1f} min")

    # Benchmark Model Forward & Backward Pass
    print("\n--- 2. Model Compute Throughput (ProposedXMISTModel) ---")
    model = ProposedXMISTModel(feature_dim=256, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    from training.losses import MultiTaskBehaviorLoss
    criterion = MultiTaskBehaviorLoss()

    # Dummy batch
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multimodal_batch, num_workers=2)
    sample_batch = next(iter(loader))
    sample_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample_batch.items()}

    fwd_times = []
    bwd_times = []
    for _ in range(50):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        out = model(sample_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        fwd_times.append(time.time() - t0)

        loss_dict = criterion(out, sample_batch)
        loss = loss_dict["loss_total"]

        t1 = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        bwd_times.append(time.time() - t1)

    # Discard warmup
    fwd_ms = np.mean(fwd_times[10:]) * 1000.0
    bwd_ms = np.mean(bwd_times[10:]) * 1000.0
    total_compute_ms = fwd_ms + bwd_ms
    est_compute_epoch = (len(dataset) / args.batch_size) * (total_compute_ms / 1000.0)

    print(f"Forward Pass:     {fwd_ms:6.1f} ms/batch")
    print(f"Backward Pass:    {bwd_ms:6.1f} ms/batch")
    print(f"Total GPU Compute:{total_compute_ms:6.1f} ms/batch ({est_compute_epoch/60.0:5.1f} min/epoch pure compute)")

    print("\n" + "=" * 75)
    print("PROFILING SUMMARY & OPTIMIZATION VERDICT:")
    best_w = min(results.keys(), key=lambda k: results[k][0])
    print(f"- Optimal DataLoader workers: num_workers={best_w}")
    print(f"- Total Epoch Duration breakdown: Compute ≈ {est_compute_epoch/60.0:.1f} min + Best I/O ≈ {results[best_w][0]*len(dataset)/(args.batch_size*60000.0):.1f} min")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
