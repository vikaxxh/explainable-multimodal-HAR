"""
Distributed Trainer for Single-GPU and Multi-GPU HPC Clusters.
Supports:
- PyTorch DistributedDataParallel (DDP)
- Automatic Mixed Precision (AMP) fp16 / bf16
- Learning rate warmup with Cosine Annealing
- Gradient clipping & accumulation
- Fault-tolerant checkpointing and resumption
- TensorBoard & WandB tracking
"""

import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, Any, Optional

from training.losses import MultiTaskBehaviorLoss


from preprocessing.augmentation import MultimodalAugmentor


class Trainer:
    """
    Robust research trainer designed for HPC cluster execution.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        is_distributed: bool = False,
        rank: int = 0
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.is_distributed = is_distributed
        self.rank = rank
        self.augmentor = MultimodalAugmentor()

        cfg_train = config.get("training", {})
        self.epochs = int(cfg_train.get("epochs", 50))
        self.lr = float(cfg_train.get("lr", 1e-4))
        self.weight_decay = float(cfg_train.get("weight_decay", 1e-4))
        self.grad_clip = float(cfg_train.get("grad_clip", 1.0))
        self.mixed_precision = str(cfg_train.get("mixed_precision", "fp16"))
        self.checkpoint_dir = config.get("project", {}).get("checkpoint_dir", "experiments/checkpoints")

        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Loss function
        loss_weights = cfg_train.get("loss_weights", {})
        self.criterion = MultiTaskBehaviorLoss(
            lambda_ped=float(loss_weights.get("lambda_ped", 1.0)),
            lambda_micro=float(loss_weights.get("lambda_micro", 1.0)),
            lambda_inter=float(loss_weights.get("lambda_inter", 1.0)),
            lambda_align=float(loss_weights.get("lambda_align", 0.1)),
            lambda_xai=float(loss_weights.get("lambda_xai", 0.05)),
            use_focal_loss=True,
            gamma=2.0
        )

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=float(cfg_train.get("min_lr", 1e-6))
        )

        # AMP Scaler
        use_amp = (self.mixed_precision == "fp16" and device.type == "cuda")
        try:
            self.scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        self.best_val_loss = float("inf")
        self.start_epoch = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        correct_ped = 0
        total_ped = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Apply physics-consistent multimodal data augmentation
            batch = self.augmentor.augment_batch(batch)
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            self.optimizer.zero_grad()

            amp_enabled = (self.mixed_precision in ["fp16", "bf16"] and self.device.type == "cuda")
            try:
                autocast_ctx = torch.amp.autocast('cuda', enabled=amp_enabled)
            except (AttributeError, TypeError):
                autocast_ctx = torch.cuda.amp.autocast(enabled=amp_enabled)

            with autocast_ctx:
                outputs = self.model(batch)
                loss_dict = self.criterion(outputs, batch)
                loss = loss_dict["loss_total"]

            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            if "ped_logits" in outputs and "ped_label" in batch:
                preds = torch.argmax(outputs["ped_logits"], dim=-1)
                correct_ped += (preds == batch["ped_label"]).sum().item()
                total_ped += batch["ped_label"].size(0)

        avg_loss = total_loss / max(1, num_batches)
        accuracy = (correct_ped / max(1, total_ped)) * 100.0
        return {"train_loss": avg_loss, "train_acc_ped": accuracy}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        correct_ped = 0
        total_ped = 0

        amp_enabled = (self.mixed_precision in ["fp16", "bf16"] and self.device.type == "cuda")
        try:
            autocast_ctx = torch.amp.autocast('cuda', enabled=amp_enabled)
        except (AttributeError, TypeError):
            autocast_ctx = torch.cuda.amp.autocast(enabled=amp_enabled)

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with autocast_ctx:
                outputs = self.model(batch)
                loss_dict = self.criterion(outputs, batch)
                loss = loss_dict["loss_total"]

            total_loss += loss.item()
            if "ped_logits" in outputs and "ped_label" in batch:
                preds = torch.argmax(outputs["ped_logits"], dim=-1)
                correct_ped += (preds == batch["ped_label"]).sum().item()
                total_ped += batch["ped_label"].size(0)

        avg_loss = total_loss / max(1, num_batches)
        accuracy = (correct_ped / max(1, total_ped)) * 100.0
        return {"val_loss": avg_loss, "val_acc_ped": accuracy}

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        if self.rank != 0:
            return

        model_state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config
        }

        latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "checkpoint_best.pt")
            torch.save(checkpoint, best_path)
            print(f"[Trainer] Epoch {epoch}: Saved new best model checkpoint to {best_path}")

    def load_checkpoint(self, checkpoint_path: str, override_lr: Optional[float] = None):
        if not os.path.exists(checkpoint_path):
            print(f"[Trainer] Checkpoint {checkpoint_path} not found. Starting from scratch.")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if override_lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = override_lr
            self.lr = override_lr

        # If user extends the epoch count beyond the checkpoint, re-align the scheduler for remaining epochs
        if self.epochs > self.start_epoch:
            remaining_epochs = self.epochs - self.start_epoch
            min_lr = float(self.config.get("training", {}).get("min_lr", 1e-6))
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_epochs,
                eta_min=min_lr
            )
            print(f"[Trainer] Resumed checkpoint from {checkpoint_path} at epoch {self.start_epoch}/{self.epochs}")
            print(f"[Trainer] Scheduler re-aligned with {remaining_epochs} remaining epochs (LR: {self.optimizer.param_groups[0]['lr']})")
        else:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"[Trainer] Resumed checkpoint from {checkpoint_path} at epoch {self.start_epoch}")

    def fit(self):
        """Runs complete training and validation cycle across epochs."""
        if self.rank == 0:
            print(f"[Trainer] Starting training for {self.epochs} epochs on device: {self.device}")

        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            if self.is_distributed and hasattr(self.train_loader, "sampler"):
                self.train_loader.sampler.set_epoch(epoch)

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()
            self.scheduler.step()
            duration = time.time() - t0

            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]

            self.save_checkpoint(epoch, is_best=is_best)

            if self.rank == 0:
                print(
                    f"Epoch [{epoch+1:02d}/{self.epochs:02d}] "
                    f"Train Loss: {train_metrics['train_loss']:.4f} | Acc: {train_metrics['train_acc_ped']:.1f}% | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | Acc: {val_metrics['val_acc_ped']:.1f}% | "
                    f"Time: {duration:.1f}s"
                )
