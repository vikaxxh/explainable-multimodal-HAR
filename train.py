"""
Main Training Entrypoint for Single-GPU and Distributed Multi-GPU HPC Clusters.
Usage:
  # 1. Quick test on CPU/GPU using synthetic data
  python train.py --model proposed --synthetic --epochs 2 --batch_size 4

  # 2. Train IntentFormer reproduction baseline
  python train.py --model intentformer --epochs 30 --batch_size 16

  # 3. Train full Proposed X-MIST architecture
  python train.py --model proposed --epochs 50 --batch_size 16

  # 4. Distributed Multi-GPU (Slurm / torchrun)
  torchrun --nproc_per_node=4 train.py --model proposed --distributed
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from configs.config_loader import load_config
from datasets.synthetic_dataset import SyntheticMultimodalDataset
from datasets.multimodal_dataset import MultimodalSequenceDataset, collate_multimodal_batch
from models.baselines import (
    RGBBaselineModel,
    TrajectoryBaselineModel,
    PoseBaselineModel,
    IntentFormerReproductionModel
)
from models.proposed_model import ProposedXMISTModel
from training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train X-MIST / EMIT-HAR Models")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default="proposed",
                        choices=["proposed", "baseline_rgb", "baseline_traj", "baseline_pose", "intentformer"],
                        help="Model architecture variant")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to processed sequences")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data for fast HPC smoke testing")
    parser.add_argument("--distributed", action="store_true", help="Enable multi-GPU DistributedDataParallel")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for torchrun DDP")
    return parser.parse_args()


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world_size, local_rank = 0, 1, 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        dist.init_process_group(backend="gloo", init_method="env://")

    return rank, world_size, local_rank


def build_model(model_type: str, config: dict, device: torch.device) -> torch.nn.Module:
    feat_dim = config.get("model", {}).get("feature_dim", 256)
    if model_type == "proposed":
        model = ProposedXMISTModel(feature_dim=feat_dim)
    elif model_type == "baseline_rgb":
        model = RGBBaselineModel(feature_dim=feat_dim)
    elif model_type == "baseline_traj":
        model = TrajectoryBaselineModel(feature_dim=feat_dim)
    elif model_type == "baseline_pose":
        model = PoseBaselineModel(feature_dim=feat_dim)
    elif model_type == "intentformer":
        model = IntentFormerReproductionModel(feature_dim=feat_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.to(device)
    return model


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI Overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["lr"] = args.lr

    # Distributed setup
    is_distributed = args.distributed or "RANK" in os.environ
    if is_distributed:
        rank, world_size, local_rank = init_distributed()
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"[Init] Launching training with model '{args.model}' on {world_size} rank(s)...")

    # Datasets
    batch_size = config["training"]["batch_size"]
    window_size = config["data"]["window_size"]

    if args.synthetic:
        if rank == 0:
            print("[Init] Using SyntheticMultimodalDataset for fast dry-run / verification.")
        train_ds = SyntheticMultimodalDataset(num_samples=128, window_size=window_size, seed=42)
        val_ds = SyntheticMultimodalDataset(num_samples=32, window_size=window_size, seed=100)
    else:
        train_ds = MultimodalSequenceDataset(data_dir=args.data_dir, split="train", window_size=window_size)
        val_ds = MultimodalSequenceDataset(data_dir=args.data_dir, split="val", window_size=window_size)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank) if is_distributed else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_multimodal_batch,
        num_workers=0 if device.type == "cpu" else 2,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_multimodal_batch,
        num_workers=0 if device.type == "cpu" else 2,
        pin_memory=(device.type == "cuda")
    )

    # Build Model
    model = build_model(args.model, config, device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=True)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        is_distributed=is_distributed,
        rank=rank
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Fit Model
    trainer.fit()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
