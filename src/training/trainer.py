"""
GNN Training Loop — 训练循环

Trains the SchoolGraphScorer on the Phase 1 dataset using MSE loss
plus constraint-specific differentiable losses.

Usage:
    python -m training.trainer
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.multitask_scorer import MultiTaskScorer
from models.ranking_loss import multitask_ranking_loss
from training.data_loader import SchoolDataLoader


class Trainer:
    """
    Trains the MultiTaskScorer model with per-task quality labels.

    Loss = L_MSE + lambda_rank * L_rank
    Training uses per-graph (batch_size=1) with accumulated batch for ranking.
    """

    def __init__(
        self,
        dataset_dir: str = 'outputs/dataset_200_new',
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        lr: float = 1e-3,
        device: str = None,
        lambda_rank: float = 0.1,
        rank_accumulate: int = 16,
    ):
        self.dataset_dir = dataset_dir
        self.lambda_rank = lambda_rank
        self.rank_accumulate = rank_accumulate

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Model
        self.model = MultiTaskScorer(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        # Data
        self.train_loader = SchoolDataLoader(dataset_dir, 'train')
        self.val_loader = SchoolDataLoader(dataset_dir, 'val', shuffle=False)

        # Tracking
        self.train_losses: list = []
        self.val_losses: list = []
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch: int) -> float:
        """Train one epoch with ranking loss accumulation."""
        self.model.train()
        total_loss = 0.0
        n_graphs = 0

        # Accumulate predictions for pairwise ranking
        acc_preds = {name: [] for name in self.model.get_task_names()}
        acc_targets = {name: [] for name in self.model.get_task_names()}

        for data, target_dict, _metadata in self.train_loader.iter_all():
            data = data.to(self.device)
            targets = {k: v.to(self.device) for k, v in target_dict.items()
                      if isinstance(v, torch.Tensor)}
            predictions = self.model(data)

            # MSE loss
            loss_mse = self.model.compute_loss(predictions, targets)

            # Accumulate for ranking
            for name in self.model.get_task_names():
                if name in predictions and name in targets:
                    acc_preds[name].append(predictions[name].detach())
                    acc_targets[name].append(targets[name])

            # Apply ranking loss every rank_accumulate steps
            loss_rank = torch.tensor(0.0, device=self.device)
            if len(acc_preds[list(acc_preds.keys())[0]]) >= self.rank_accumulate:
                batch_preds = {k: torch.stack(v) for k, v in acc_preds.items()}
                batch_targets = {k: torch.stack(v) for k, v in acc_targets.items()}
                loss_rank = multitask_ranking_loss(batch_preds, batch_targets)
                # Clear accumulators
                acc_preds = {name: [] for name in self.model.get_task_names()}
                acc_targets = {name: [] for name in self.model.get_task_names()}

            loss = loss_mse + self.lambda_rank * loss_rank

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_graphs += 1

        avg_loss = total_loss / max(1, n_graphs)
        self.train_losses.append(avg_loss)
        self.train_loader.on_epoch_end()
        return avg_loss

    @torch.no_grad()
    def validate(self) -> float:
        """Validate on val set. Returns average multi-task loss."""
        self.model.eval()
        total_loss = 0.0
        n_graphs = 0

        for data, target_dict, _metadata in self.val_loader.iter_all():
            data = data.to(self.device)
            targets = {k: v.to(self.device) for k, v in target_dict.items()
                      if isinstance(v, torch.Tensor)}
            predictions = self.model(data)
            loss = self.model.compute_loss(predictions, targets)
            total_loss += loss.item()
            n_graphs += 1

        avg_loss = total_loss / max(1, n_graphs)
        self.val_losses.append(avg_loss)
        return avg_loss

    def train(
        self,
        num_epochs: int = 80,
        save_path: str = 'outputs/model_checkpoint.pt',
        print_every: int = 10,
    ):
        """
        Full training loop.

        Args:
            num_epochs: Maximum number of epochs.
            save_path: Where to save the best model.
            print_every: Print progress every N epochs.
        """
        print(f"Training on {self.device}")
        print(f"  Train graphs: {len(self.train_loader)}")
        print(f"  Val graphs:   {len(self.val_loader)}")
        print(f"  Hidden dim:   {self.model.encoder.hidden_dim}")
        print(f"  Layers:       {self.model.encoder.num_layers}")
        print()

        t_start = time.time()

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }, save_path)

            if epoch % print_every == 0 or epoch == 1:
                elapsed = time.time() - t_start
                print(
                    f"  Epoch {epoch:3d}/{num_epochs} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"val_loss: {val_loss:.4f} | "
                    f"lr: {self.optimizer.param_groups[0]['lr']:.2e} | "
                    f"elapsed: {elapsed:.0f}s"
                )

        elapsed = time.time() - t_start
        print(f"\nTraining complete in {elapsed:.0f}s")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Model saved to: {save_path}")

    def evaluate_test(self, test_dir: str = None) -> dict:
        """Evaluate on the test set. Reports per-task R2."""
        if test_dir is None:
            test_loader = SchoolDataLoader(self.dataset_dir, 'test', shuffle=False)
        else:
            test_loader = SchoolDataLoader(test_dir, 'test', shuffle=False)

        self.model.eval()
        task_preds = {}
        task_targets = {}

        for data, target_dict, meta in test_loader.iter_all():
            data = data.to(self.device)
            with torch.no_grad():
                predictions = self.model(data)
            for k, v in predictions.items():
                task_preds.setdefault(k, []).append(v.cpu().item())
            for k, v in target_dict.items():
                if isinstance(v, torch.Tensor):
                    task_targets.setdefault(k, []).append(v.item())

        results = {'num_graphs': len(task_preds.get('overall_quality', []))}
        print(f"\nTest Results ({results['num_graphs']} graphs):")
        print(f"  {'Task':<28} {'R2':>8} {'MAE':>8} {'PredMean':>10} {'TargetMean':>10}")
        print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

        for task_name in self.model.get_task_names():
            if task_name not in task_preds or task_name not in task_targets:
                continue
            p = torch.tensor(task_preds[task_name])
            t = torch.tensor(task_targets[task_name])
            mae = (p - t).abs().mean().item()
            ss_res = ((t - p) ** 2).sum()
            ss_tot = ((t - t.mean()) ** 2).sum()
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            results[f'{task_name}_r2'] = r2
            results[f'{task_name}_mae'] = mae
            print(f"  {task_name:<28} {r2:>8.4f} {mae:>8.4f} {p.mean():>10.4f} {t.mean():>10.4f}")

        return results


# ==========================================================================
# Main entry point
# ==========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  Phase 2: GNN School Graph Scorer Training")
    print("=" * 60)
    print()

    trainer = Trainer(
        dataset_dir='outputs/dataset_200_new',
        hidden_dim=128,
        num_layers=3,
        lr=1e-3,
    )

    trainer.train(num_epochs=80)
    trainer.evaluate_test()
